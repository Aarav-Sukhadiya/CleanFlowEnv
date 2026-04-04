"""
Gradio dashboard for CleanFlowEnv.
Provides a visual, interactive demo for judges and users.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import gradio as gr
import pandas as pd

from cleanflow_env.baseline.rule_agent import RuleBasedAgent
from cleanflow_env.env.environment import CleanFlowEnv, build_observation
from cleanflow_env.env.grader import final_score, score_breakdown_report
from cleanflow_env.tasks.task_custom import (
    analyze_dataset,
    generate_custom_task,
)
from cleanflow_env.tasks.task_easy import generate_easy_task
from cleanflow_env.tasks.task_expert import generate_expert_task
from cleanflow_env.tasks.task_hard import generate_hard_task
from cleanflow_env.tasks.task_medium import generate_medium_task

TASK_REGISTRY = {
    "task_easy": generate_easy_task,
    "task_medium": generate_medium_task,
    "task_hard": generate_hard_task,
    "task_expert": generate_expert_task,
}

TASK_LABELS = {
    "task_easy": "Task 1 — Basic Cleaning (Easy)",
    "task_medium": "Task 2 — Schema Normalization (Medium)",
    "task_hard": "Task 3 — Advanced Cleaning (Hard)",
    "task_expert": "Task 4 — Budget-Constrained (Expert)",
}

TASK_DESCRIPTIONS = {
    "task_easy": "Employee survey data (200 rows). Issues: missing values in age/salary, 12 duplicate rows.",
    "task_medium": "Transaction records (300 rows). Issues: mixed date formats, currency strings, mixed booleans, trailing whitespace.",
    "task_hard": "Medical trial data (400 rows). Issues: outliers in blood_pressure/cholesterol, mixed patient_id formats, year typos.",
    "task_expert": "E-commerce catalog (500 rows). All issue types combined + 5 distractor columns. Tight budget of 15 credits.",
}


def _describe_action(action, rows_before: int, rows_after: int, env_state) -> str:
    """Generate a plain-English description of what an action did."""
    a = action.action_type
    col = action.column or ""

    if a == "fill_null":
        method = action.method
        if method == "median":
            try:
                val = env_state.prev_table[col].median()
                if isinstance(val, (int, float)) and not pd.isna(val):
                    return f"Filled {col} nulls with median ({val:.1f})"
            except Exception:
                pass
            return f"Filled {col} nulls with median"
        elif method == "mean":
            try:
                val = env_state.prev_table[col].mean()
                if isinstance(val, (int, float)) and not pd.isna(val):
                    return f"Filled {col} nulls with mean ({val:.1f})"
            except Exception:
                pass
            return f"Filled {col} nulls with mean"
        elif method == "mode":
            mode_vals = env_state.prev_table[col].mode()
            val = mode_vals.iloc[0] if len(mode_vals) > 0 else "?"
            return f"Filled {col} nulls with mode ({val})"
        elif method == "constant":
            return f"Filled {col} nulls with constant value ({action.constant_value})"
        else:
            return f"Filled {col} nulls using {method}"

    elif a == "drop_duplicates":
        removed = rows_before - rows_after
        return f"Removed {removed} duplicate rows ({rows_before} -> {rows_after} rows)"

    elif a == "strip_whitespace":
        return f"Stripped leading/trailing whitespace from {col}"

    elif a == "replace_substring":
        old = action.old_value or ""
        new = action.new_value or ""
        if new == "":
            return f"Removed '{old}' from {col}"
        return f"Replaced '{old}' with '{new}' in {col}"

    elif a == "map_values":
        return f"Mapped categorical values in {col} (e.g. yes/no -> True/False)"

    elif a == "convert_type":
        return f"Converted {col} to {action.target_type}"

    elif a == "normalize":
        return f"Normalized {col} values to 0-1 range"

    elif a == "remove_outliers":
        removed = rows_before - rows_after
        method = action.outlier_method or "iqr"
        return f"Removed {removed} outlier rows from {col} using {method.upper()} ({rows_before} -> {rows_after} rows)"

    return f"{a} on {col}"


def _distribution_comparison(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> str:
    """Generate a before/after distribution comparison for numeric columns."""
    lines = ["\n---\n**Distribution Check (Before → After):**\n"]
    numeric_cols = raw_df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        return ""

    lines.append("| Column | Stat | Before | After | Δ |")
    lines.append("|--------|------|--------|-------|---|")

    for col in numeric_cols:
        raw_s = raw_df[col].dropna()
        clean_s = pd.to_numeric(cleaned_df[col], errors="coerce").dropna() if col in cleaned_df.columns else pd.Series(dtype=float)
        if len(raw_s) == 0 and len(clean_s) == 0:
            continue

        for stat_name, func in [("median", "median"), ("mean", "mean"), ("std", "std"), ("skew", "skew")]:
            before = getattr(raw_s, func)() if len(raw_s) > 0 else float("nan")
            after = getattr(clean_s, func)() if len(clean_s) > 0 else float("nan")
            delta = after - before if pd.notna(before) and pd.notna(after) else float("nan")
            b_str = f"{before:.2f}" if pd.notna(before) else "—"
            a_str = f"{after:.2f}" if pd.notna(after) else "—"
            d_str = f"{delta:+.2f}" if pd.notna(delta) else "—"
            lines.append(f"| {col} | {stat_name} | {b_str} | {a_str} | {d_str} |")

    return "\n".join(lines)


def run_episode_visual(task_id: str):
    """
    Run a full episode and return all the data needed for the dashboard.
    """
    env = CleanFlowEnv(task_registry=TASK_REGISTRY)
    obs = env.reset(task_id)
    agent = RuleBasedAgent()

    # Capture initial state
    raw_df = env._state.raw_table.copy()
    initial_nulls = sum(obs.null_counts.values())
    initial_dups = obs.duplicate_count
    initial_budget = obs.budget_remaining

    # Step log
    steps_log: List[Dict[str, Any]] = []
    quality_over_time = [0.0]
    budget_over_time = [initial_budget]

    done = False
    step_num = 0

    while not done:
        action = agent.act(obs)
        if action is None:
            break

        rows_before = len(env._state.current_table)
        obs, reward = env.step(action.model_dump())
        rows_after = len(env._state.current_table)
        done = reward.done
        step_num += 1

        description = _describe_action(action, rows_before, rows_after, env._state)

        steps_log.append({
            "Step": step_num,
            "Action": action.action_type,
            "Column": action.column or "—",
            "Detail": description,
            "Reward": round(reward.reward, 3),
            "Quality": round(reward.cumulative_quality, 3),
            "Budget Left": obs.budget_remaining,
        })

        quality_over_time.append(round(reward.cumulative_quality, 3))
        budget_over_time.append(obs.budget_remaining)

    # Final results
    result = final_score(env._state)
    cleaned_df = env._state.current_table.copy()
    gt_df = env._state.ground_truth.copy()
    report = score_breakdown_report(env._state)

    return (
        raw_df,
        cleaned_df,
        gt_df,
        steps_log,
        result,
        quality_over_time,
        budget_over_time,
        initial_nulls,
        initial_dups,
        initial_budget,
        report,
    )


def format_score_html(result: Dict[str, float]) -> str:
    """Generate an HTML score card."""
    score = result["score"]

    # Color based on score
    if score >= 0.8:
        color = "#22c55e"  # green
        grade = "Excellent"
    elif score >= 0.6:
        color = "#eab308"  # yellow
        grade = "Good"
    elif score >= 0.4:
        color = "#f97316"  # orange
        grade = "Fair"
    else:
        color = "#ef4444"  # red
        grade = "Needs Work"

    quality_overall = result.get("quality_overall", 0.0)
    validation = result.get("validation", 0.0)
    schema = result.get("schema_accuracy", 0.0)

    return f"""
    <div style="text-align:center; padding:20px;">
        <div style="font-size:64px; font-weight:bold; color:{color};">{score:.3f}</div>
        <div style="font-size:20px; color:{color}; margin-bottom:16px;">{grade}</div>
        <div style="display:flex; justify-content:center; gap:24px; flex-wrap:wrap;">
            <div style="text-align:center;">
                <div style="font-size:24px; font-weight:bold;">{quality_overall:.3f}</div>
                <div style="font-size:12px; color:#888;">Quality (40%)</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:24px; font-weight:bold;">{validation:.3f}</div>
                <div style="font-size:12px; color:#888;">Validation (20%)</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:24px; font-weight:bold;">{result['efficiency']:.3f}</div>
                <div style="font-size:12px; color:#888;">Efficiency (15%)</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:24px; font-weight:bold;">{result['action_quality']:.3f}</div>
                <div style="font-size:12px; color:#888;">Action Quality (10%)</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:24px; font-weight:bold;">{schema:.3f}</div>
                <div style="font-size:12px; color:#888;">Schema (15%)</div>
            </div>
        </div>
    </div>
    """


def format_initial_stats_html(nulls: int, dups: int, budget: int, task_id: str) -> str:
    desc = TASK_DESCRIPTIONS.get(task_id, "")
    return f"""
    <div style="padding:12px;">
        <div style="font-size:14px; color:#888; margin-bottom:12px;">{desc}</div>
        <div style="display:flex; gap:24px; flex-wrap:wrap;">
            <div style="text-align:center; padding:12px; background:#1e1e2e; border-radius:8px; min-width:100px;">
                <div style="font-size:28px; font-weight:bold; color:#f97316;">{nulls}</div>
                <div style="font-size:12px; color:#888;">Missing Values</div>
            </div>
            <div style="text-align:center; padding:12px; background:#1e1e2e; border-radius:8px; min-width:100px;">
                <div style="font-size:28px; font-weight:bold; color:#ef4444;">{dups}</div>
                <div style="font-size:12px; color:#888;">Duplicate Rows</div>
            </div>
            <div style="text-align:center; padding:12px; background:#1e1e2e; border-radius:8px; min-width:100px;">
                <div style="font-size:28px; font-weight:bold; color:#22c55e;">{budget}</div>
                <div style="font-size:12px; color:#888;">Budget Credits</div>
            </div>
        </div>
    </div>
    """


def run_and_display(task_id: str):
    """Main callback: run episode and return all display elements."""
    (
        raw_df,
        cleaned_df,
        gt_df,
        steps_log,
        result,
        quality_over_time,
        budget_over_time,
        initial_nulls,
        initial_dups,
        initial_budget,
        report,
    ) = run_episode_visual(task_id)

    # DataFrames for display (limit to first 15 rows for readability)
    raw_display = raw_df.head(15).fillna("NULL").reset_index(drop=True)
    cleaned_display = cleaned_df.head(15).fillna("NULL").reset_index(drop=True)
    gt_display = gt_df.head(15).fillna("NULL").reset_index(drop=True)

    # Steps log as DataFrame
    steps_df = pd.DataFrame(steps_log) if steps_log else pd.DataFrame(
        columns=["Step", "Action", "Column", "Detail", "Reward", "Quality", "Budget Left"]
    )

    # Score HTML
    score_html = format_score_html(result)

    # Initial stats HTML
    stats_html = format_initial_stats_html(initial_nulls, initial_dups, initial_budget, task_id)

    # Quality chart data
    quality_chart = pd.DataFrame({
        "Step": list(range(len(quality_over_time))),
        "Quality": quality_over_time,
    })

    # Budget chart data
    budget_chart = pd.DataFrame({
        "Step": list(range(len(budget_over_time))),
        "Budget": budget_over_time,
    })

    # Summary stats with human-readable action descriptions
    final_rows = len(cleaned_df)
    rows_changed = len(raw_df) - final_rows

    summary_lines = [
        f"**{TASK_LABELS[task_id]}**\n",
        f"Steps taken: **{len(steps_log)}** | "
        f"Budget used: **{initial_budget - (steps_log[-1]['Budget Left'] if steps_log else initial_budget)}** / {initial_budget} | "
        f"Final score: **{result['score']:.3f}**\n",
        "---",
        "**What the agent did:**\n",
    ]
    for s in steps_log:
        summary_lines.append(f"- Step {s['Step']}: {s['Detail']}")

    if rows_changed > 0:
        summary_lines.append(f"\n**Rows:** {len(raw_df)} -> {final_rows} ({rows_changed} removed by dedup/outlier removal)")
    else:
        summary_lines.append(f"\n**Rows:** {len(raw_df)} (no rows removed)")

    # Validation violations
    v_details = result.get("validation_details", {})
    violations = v_details.get("violations", [])
    if violations:
        summary_lines.append("\n**Validation Issues:**")
        for v in violations:
            summary_lines.append(f"- {v}")

    # Distribution comparison
    dist_comp = _distribution_comparison(raw_df, cleaned_df)
    if dist_comp:
        summary_lines.append(dist_comp)

    summary = "\n".join(summary_lines)

    return (
        stats_html,
        raw_display,
        steps_df,
        score_html,
        cleaned_display,
        gt_display,
        quality_chart,
        budget_chart,
        report,
        summary,
    )


def run_all_tasks():
    """Run baseline on all 4 tasks and return summary."""
    env = CleanFlowEnv(task_registry=TASK_REGISTRY)
    agent = RuleBasedAgent()

    rows = []
    for task_id in ["task_easy", "task_medium", "task_hard", "task_expert"]:
        obs = env.reset(task_id)
        agent.reset()
        done = False
        steps = 0
        while not done:
            action = agent.act(obs)
            if action is None:
                break
            obs, reward = env.step(action.model_dump())
            done = reward.done
            steps += 1

        result = final_score(env._state)
        budget_used = env._state.initial_budget - env._state.budget_remaining
        rows.append({
            "Task": TASK_LABELS[task_id],
            "Score": round(result["score"], 3),
            "Correctness": round(result["correctness"], 3),
            "Completeness": round(result["completeness"], 3),
            "Efficiency": round(result["efficiency"], 3),
            "Steps": steps,
            "Budget Used": budget_used,
        })

    df = pd.DataFrame(rows)
    avg_score = df["Score"].mean()

    summary_html = f"""
    <div style="text-align:center; padding:16px;">
        <div style="font-size:18px; color:#888;">Average Score Across All Tasks</div>
        <div style="font-size:56px; font-weight:bold; color:#3b82f6;">{avg_score:.3f}</div>
    </div>
    """

    return df, summary_html


def analyze_and_display(file_obj):
    """Analyze an uploaded CSV and return issue report + preview."""
    if file_obj is None:
        return "", pd.DataFrame(), ""

    df = pd.read_csv(file_obj)
    analysis = analyze_dataset(df)

    # Build issue report HTML
    issues = analysis["issues_summary"]
    if issues:
        issue_items = "".join(f"<li>{iss}</li>" for iss in issues)
        issue_color = "#f97316" if len(issues) < 5 else "#ef4444"
        issues_html = f"""
        <div style="padding:12px;">
            <div style="font-size:16px; font-weight:bold; color:{issue_color}; margin-bottom:8px;">
                {len(issues)} Issue(s) Detected
            </div>
            <ul style="margin:0; padding-left:20px;">{issue_items}</ul>
        </div>
        """
    else:
        issues_html = """
        <div style="padding:12px;">
            <div style="font-size:16px; font-weight:bold; color:#22c55e;">
                No issues detected — dataset looks clean!
            </div>
        </div>
        """

    stats_html = f"""
    <div style="padding:12px;">
        <div style="display:flex; gap:24px; flex-wrap:wrap;">
            <div style="text-align:center; padding:12px; background:#1e1e2e; border-radius:8px; min-width:100px;">
                <div style="font-size:28px; font-weight:bold; color:#3b82f6;">{analysis['rows']}</div>
                <div style="font-size:12px; color:#888;">Rows</div>
            </div>
            <div style="text-align:center; padding:12px; background:#1e1e2e; border-radius:8px; min-width:100px;">
                <div style="font-size:28px; font-weight:bold; color:#3b82f6;">{analysis['columns']}</div>
                <div style="font-size:12px; color:#888;">Columns</div>
            </div>
            <div style="text-align:center; padding:12px; background:#1e1e2e; border-radius:8px; min-width:100px;">
                <div style="font-size:28px; font-weight:bold; color:#f97316;">{sum(analysis['null_columns'].values())}</div>
                <div style="font-size:12px; color:#888;">Missing Values</div>
            </div>
            <div style="text-align:center; padding:12px; background:#1e1e2e; border-radius:8px; min-width:100px;">
                <div style="font-size:28px; font-weight:bold; color:#ef4444;">{analysis['duplicate_count']}</div>
                <div style="font-size:12px; color:#888;">Duplicates</div>
            </div>
            <div style="text-align:center; padding:12px; background:#1e1e2e; border-radius:8px; min-width:100px;">
                <div style="font-size:28px; font-weight:bold; color:#eab308;">{len(analysis['outlier_columns'])}</div>
                <div style="font-size:12px; color:#888;">Columns with Outliers</div>
            </div>
        </div>
    </div>
    """

    preview = df.head(15).fillna("NULL").reset_index(drop=True)
    return stats_html, preview, issues_html


def run_custom_episode(file_obj, gt_file_obj, budget, difficulty):
    """Run the agent on a user-uploaded dataset."""
    if file_obj is None:
        return ("", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                pd.DataFrame(), "", pd.DataFrame(), pd.DataFrame(), "")

    budget = int(budget)
    difficulty = difficulty or "hard"
    raw_df = pd.read_csv(file_obj)

    gt_df = None
    if gt_file_obj is not None:
        gt_df = pd.read_csv(gt_file_obj)

    raw_table, ground_truth, budget_val, col_desc = generate_custom_task(
        raw_df, ground_truth_df=gt_df, budget=budget, difficulty=difficulty
    )

    # Create a temporary env with the custom task
    def _custom_gen():
        return raw_table, ground_truth, budget_val, col_desc

    env = CleanFlowEnv(task_registry={"custom": _custom_gen})
    obs = env.reset("custom")
    agent = RuleBasedAgent()

    initial_nulls = sum(obs.null_counts.values())
    initial_dups = obs.duplicate_count

    steps_log = []
    quality_over_time = [0.0]
    budget_over_time = [budget_val]

    done = False
    step_num = 0
    while not done:
        action = agent.act(obs)
        if action is None:
            break
        rows_before = len(env._state.current_table)
        obs, reward = env.step(action.model_dump())
        rows_after = len(env._state.current_table)
        done = reward.done
        step_num += 1

        description = _describe_action(action, rows_before, rows_after, env._state)

        steps_log.append({
            "Step": step_num,
            "Action": action.action_type,
            "Column": action.column or "—",
            "Detail": description,
            "Reward": round(reward.reward, 3),
            "Quality": round(reward.cumulative_quality, 3),
            "Budget Left": obs.budget_remaining,
        })
        quality_over_time.append(round(reward.cumulative_quality, 3))
        budget_over_time.append(obs.budget_remaining)

    result = final_score(env._state)
    cleaned_df = env._state.current_table
    report = score_breakdown_report(env._state)

    # Format outputs
    score_html = format_score_html(result)

    steps_df = pd.DataFrame(steps_log) if steps_log else pd.DataFrame(
        columns=["Step", "Action", "Column", "Detail", "Reward", "Quality", "Budget Left"]
    )

    quality_chart = pd.DataFrame({
        "Step": list(range(len(quality_over_time))),
        "Quality": quality_over_time,
    })
    budget_chart = pd.DataFrame({
        "Step": list(range(len(budget_over_time))),
        "Budget": budget_over_time,
    })

    raw_display = raw_df.head(15).fillna("NULL").reset_index(drop=True)
    cleaned_display = cleaned_df.head(15).fillna("NULL").reset_index(drop=True)
    gt_display = ground_truth.head(15).fillna("NULL").reset_index(drop=True)

    diff_label = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}.get(difficulty, difficulty)
    final_rows = len(cleaned_df)
    rows_changed = raw_df.shape[0] - final_rows

    # Build human-readable summary
    summary_lines = [
        f"**Custom Dataset — {diff_label}** ({raw_df.shape[0]} rows, {raw_df.shape[1]} columns)\n",
        f"Steps taken: **{len(steps_log)}** | "
        f"Budget used: **{budget_val - (steps_log[-1]['Budget Left'] if steps_log else budget_val)}** / {budget_val} | "
        f"Final score: **{result['score']:.3f}**\n",
        "---",
        "**What the agent did:**\n",
    ]
    for s in steps_log:
        summary_lines.append(f"- Step {s['Step']}: {s['Detail']}")

    if rows_changed > 0:
        summary_lines.append(f"\n**Rows:** {raw_df.shape[0]} -> {final_rows} ({rows_changed} removed by dedup/outlier removal)")
    elif rows_changed < 0:
        summary_lines.append(f"\n**Rows:** {raw_df.shape[0]} -> {final_rows}")
    else:
        summary_lines.append(f"\n**Rows:** {raw_df.shape[0]} (no rows removed)")

    # Validation violations
    v_details = result.get("validation_details", {})
    violations = v_details.get("violations", [])
    if violations:
        summary_lines.append("\n**Validation Issues:**")
        for v in violations:
            summary_lines.append(f"- {v}")

    # Distribution comparison
    dist_comp = _distribution_comparison(raw_df, cleaned_df)
    if dist_comp:
        summary_lines.append(dist_comp)

    summary = "\n".join(summary_lines)

    return (
        score_html,
        steps_df,
        quality_chart,
        budget_chart,
        raw_display,
        cleaned_display,
        gt_display,
        report,
        summary,
    )


def create_dashboard() -> gr.Blocks:
    """Create the Gradio Blocks dashboard."""

    with gr.Blocks(
        title="CleanFlowEnv — Data Cleaning Agent Environment",
    ) as demo:

        gr.Markdown(
            """
            # CleanFlowEnv
            ### An OpenEnv-compliant environment for evaluating AI agents on real-world data cleaning workflows

            Select a task below and click **Run Episode** to watch the rule-based agent clean a messy dataset in real-time.
            The agent decides which cleaning actions to apply based on column descriptions and data statistics.
            """
        )

        with gr.Tabs():

            # --- Tab 1: Single Task Runner ---
            with gr.Tab("Run Episode"):

                with gr.Row():
                    task_dropdown = gr.Dropdown(
                        choices=[
                            ("Task 1 — Basic Cleaning (Easy)", "task_easy"),
                            ("Task 2 — Schema Normalization (Medium)", "task_medium"),
                            ("Task 3 — Advanced Cleaning (Hard)", "task_hard"),
                            ("Task 4 — Budget-Constrained (Expert)", "task_expert"),
                        ],
                        value="task_easy",
                        label="Select Task",
                        scale=3,
                    )
                    run_btn = gr.Button("Run Episode", variant="primary", scale=1)

                summary_md = gr.Markdown("")
                stats_html = gr.HTML(label="Dataset Overview")

                with gr.Row():
                    with gr.Column(scale=1):
                        quality_chart = gr.LinePlot(
                            x="Step",
                            y="Quality",
                            title="Quality Over Time",
                            height=250,
                        )
                    with gr.Column(scale=1):
                        budget_chart = gr.LinePlot(
                            x="Step",
                            y="Budget",
                            title="Budget Remaining",
                            height=250,
                        )

                score_html = gr.HTML(label="Final Score")

                gr.Markdown("### Agent Action Log")
                steps_table = gr.Dataframe(
                    label="Step-by-Step Actions",
                    interactive=False,
                    wrap=True,
                )

                with gr.Accordion("Data Tables", open=False):
                    gr.Markdown("#### Raw Data (first 15 rows)")
                    raw_table = gr.Dataframe(
                        label="Raw (Messy) Data",
                        interactive=False,
                        wrap=True,
                    )
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Cleaned by Agent")
                            cleaned_table = gr.Dataframe(
                                label="Agent's Cleaned Data",
                                interactive=False,
                                wrap=True,
                            )
                        with gr.Column():
                            gr.Markdown("#### Ground Truth")
                            gt_table = gr.Dataframe(
                                label="Ground Truth",
                                interactive=False,
                                wrap=True,
                            )

                with gr.Accordion("Grader Report (Text)", open=False):
                    report_text = gr.Textbox(
                        label="Full Grader Report",
                        lines=12,
                        interactive=False,
                    )

                run_btn.click(
                    fn=run_and_display,
                    inputs=[task_dropdown],
                    outputs=[
                        stats_html,
                        raw_table,
                        steps_table,
                        score_html,
                        cleaned_table,
                        gt_table,
                        quality_chart,
                        budget_chart,
                        report_text,
                        summary_md,
                    ],
                )

            # --- Tab 2: All Tasks Overview ---
            with gr.Tab("Baseline Benchmark"):

                gr.Markdown(
                    """
                    ### Run Baseline Agent on All 4 Tasks
                    Click below to run the rule-based agent across all difficulty levels and see comparative scores.
                    """
                )

                benchmark_btn = gr.Button("Run All Tasks", variant="primary")
                benchmark_summary = gr.HTML()
                benchmark_table = gr.Dataframe(
                    label="Baseline Results",
                    interactive=False,
                    wrap=True,
                )

                benchmark_btn.click(
                    fn=run_all_tasks,
                    inputs=[],
                    outputs=[benchmark_table, benchmark_summary],
                )

            # --- Tab 3: Custom Dataset ---
            with gr.Tab("Custom Dataset"):

                gr.Markdown(
                    """
                    ### Upload Your Own Dataset
                    Upload a messy CSV file and watch the AI agent clean it automatically.
                    The environment will auto-detect issues (nulls, duplicates, type mismatches, outliers)
                    and the rule-based agent will attempt to fix them.

                    **Difficulty levels control what the agent cleans:**
                    - **Easy** — Fill nulls + drop duplicates only
                    - **Medium** — Easy + convert types + strip whitespace + map categoricals
                    - **Hard** — Medium + remove outliers (full cleaning)

                    Optionally upload a ground truth CSV for precise scoring, or let the system auto-generate one.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        csv_upload = gr.File(
                            label="Upload Messy CSV",
                            file_types=[".csv"],
                            type="filepath",
                        )
                    with gr.Column(scale=2):
                        gt_upload = gr.File(
                            label="Upload Ground Truth CSV (optional)",
                            file_types=[".csv"],
                            type="filepath",
                        )
                    with gr.Column(scale=1):
                        difficulty_dropdown = gr.Dropdown(
                            choices=[
                                ("Easy — Nulls & Duplicates", "easy"),
                                ("Medium — + Types & Whitespace", "medium"),
                                ("Hard — Full Cleaning", "hard"),
                            ],
                            value="hard",
                            label="Difficulty",
                        )
                        budget_slider = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=20,
                            step=1,
                            label="Budget (action credits)",
                        )

                analyze_btn = gr.Button("Analyze Dataset", variant="secondary")
                custom_stats_html = gr.HTML()
                custom_issues_html = gr.HTML()
                custom_preview = gr.Dataframe(
                    label="Data Preview (first 15 rows)",
                    interactive=False,
                    wrap=True,
                )

                analyze_btn.click(
                    fn=analyze_and_display,
                    inputs=[csv_upload],
                    outputs=[custom_stats_html, custom_preview, custom_issues_html],
                )

                gr.Markdown("---")
                clean_btn = gr.Button("Run Agent on This Dataset", variant="primary")
                custom_summary_md = gr.Markdown("")
                custom_score_html = gr.HTML()

                with gr.Row():
                    with gr.Column(scale=1):
                        custom_quality_chart = gr.LinePlot(
                            x="Step",
                            y="Quality",
                            title="Quality Over Time",
                            height=250,
                        )
                    with gr.Column(scale=1):
                        custom_budget_chart = gr.LinePlot(
                            x="Step",
                            y="Budget",
                            title="Budget Remaining",
                            height=250,
                        )

                gr.Markdown("### Agent Action Log")
                custom_steps_table = gr.Dataframe(
                    label="Step-by-Step Actions",
                    interactive=False,
                    wrap=True,
                )

                with gr.Accordion("Data Comparison", open=False):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Original (Messy)")
                            custom_raw_table = gr.Dataframe(interactive=False, wrap=True)
                        with gr.Column():
                            gr.Markdown("#### Cleaned by Agent")
                            custom_cleaned_table = gr.Dataframe(interactive=False, wrap=True)
                        with gr.Column():
                            gr.Markdown("#### Ground Truth")
                            custom_gt_table = gr.Dataframe(interactive=False, wrap=True)

                with gr.Accordion("Grader Report", open=False):
                    custom_report = gr.Textbox(lines=12, interactive=False)

                clean_btn.click(
                    fn=run_custom_episode,
                    inputs=[csv_upload, gt_upload, budget_slider, difficulty_dropdown],
                    outputs=[
                        custom_score_html,
                        custom_steps_table,
                        custom_quality_chart,
                        custom_budget_chart,
                        custom_raw_table,
                        custom_cleaned_table,
                        custom_gt_table,
                        custom_report,
                        custom_summary_md,
                    ],
                )

            # --- Tab 4: About ---
            with gr.Tab("About"):
                gr.Markdown(
                    """
                    ## About CleanFlowEnv

                    **CleanFlowEnv** is an OpenEnv-compliant environment that simulates real-world
                    data cleaning and ETL workflows. AI agents iteratively transform messy datasets
                    into clean, analysis-ready tables.

                    ### Why Data Cleaning?
                    Data cleaning accounts for **60-80% of real-world data work**, yet there are very
                    few standardized environments to evaluate how well AI agents perform these tasks.

                    ### Key Design Features

                    | Feature | Description |
                    |---------|-------------|
                    | **1-Step Observation Lag** | Stats reflect the *previous* step, forcing agents to reason about action effects |
                    | **High-Water Mark Reward** | Prevents reward hacking via apply-undo-apply oscillation |
                    | **Budget Mechanic** | Each action costs credits (1-3), mirroring real ETL billing constraints |
                    | **Semantic Hints** | Column descriptions force genuine NLP reasoning, not just pattern matching |
                    | **Deterministic Grading** | Fixed IQR x 1.5 rule, seeded data generation, fully reproducible |
                    | **Gap-Aware Sequential Fill** | Detects ID patterns (e.g. Employee_001) and fills gaps before extending past max |

                    ### Action Types

                    | Action | Cost | Description |
                    |--------|------|-------------|
                    | `fill_null` | 1 | Fill missing values (mean/median/mode/constant/ffill/bfill/**sequential**) |
                    | `drop_duplicates` | 1 | Remove all fully duplicate rows |
                    | `strip_whitespace` | 1 | Strip leading/trailing whitespace from string column |
                    | `replace_substring` | 1 | Replace a substring in string values (e.g. remove "$") |
                    | `convert_type` | 2 | Convert column dtype (int/float/datetime/string) |
                    | `map_values` | 2 | Map categorical values (e.g. "yes"/"no" → True/False) |
                    | `normalize` | 2 | Scale column values (minmax/zscore) |
                    | `remove_outliers` | 3 | Remove outliers using IQR x 1.5 rule |

                    ### Scoring Formula
                    ```
                    score = 0.40 * quality_overall
                          + 0.20 * validation
                          + 0.15 * efficiency
                          + 0.10 * action_quality
                          + 0.15 * schema_accuracy
                    ```
                    `quality_overall = 0.6 * correctness + 0.3 * completeness + 0.1 * schema_accuracy`

                    ### API Endpoints
                    This Space also exposes a full OpenEnv-compliant API:
                    - `POST /reset` — Start a new episode
                    - `POST /step` — Apply an action
                    - `GET /state` — Get current state
                    - `GET /grader` — Get final score
                    - `GET /tasks` — List available tasks
                    - `POST /baseline` — Run baseline agent

                    ### Tech Stack
                    Python 3.10+ · FastAPI · Pydantic v2 · Pandas · NumPy · Gradio
                    """
                )

    return demo
