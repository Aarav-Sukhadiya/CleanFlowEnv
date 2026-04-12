# CleanFlowEnv — Project Summary

## What This Project Is

CleanFlowEnv is an **OpenEnv-compliant environment** for the OpenEnv hackathon (Meta + Hugging Face). It is NOT an AI model — it is a game board that AI agents interact with. Agents clean messy datasets by issuing structured actions, and the environment scores them.

Data cleaning accounts for 60–80% of real-world data work, yet few standardized environments exist to evaluate how well AI agents perform these tasks. CleanFlowEnv fills that gap.

## Architecture

```
cleanflow_env/
  api/          → FastAPI + Gradio dashboard
  env/          → Core environment (reset/step/state/grader/rewards/budget/validation)
  models/       → Pydantic v2 typed models (Action, Observation, Reward)
  tasks/        → 6 built-in tasks + custom dataset support
  baseline/     → Rule-based agent + LLM-based inference

Top-level:
  run.py                → Entry point (uvicorn on :7860, auto-opens dashboard)
  inference.py          → LLM-based agent for hackathon submission
  openenv.yaml          → OpenEnv spec metadata (v2.0)
  validate_submission.py→ Submission validation script
  simulate.py           → Simulation helper
  smoke_test.py         → Quick smoke tests
  Dockerfile            → python:3.10-slim, exposes port 7860
  requirements.txt      → Dependencies
  tests/                → 80 tests (models, actions, integration)
  Workflow/             → Project docs (Summary, Constraints, Prompts, SSTXMeta)
```

## 11 Action Types

| Action | Cost | What It Does |
|--------|------|-------------|
| `fill_null` | 1 | Fill missing values (mean/median/mode/constant/ffill/bfill/sequential) |
| `drop_duplicates` | 1 | Remove fully duplicate rows |
| `strip_whitespace` | 1 | Strip leading/trailing whitespace from string column |
| `replace_substring` | 1 | Replace substring in string values (e.g. "$" → "", "2033" → "2023") |
| `convert_type` | 2 | Convert column dtype (int/float/datetime/string) |
| `map_values` | 2 | Map categorical values via dict (e.g. "yes" → True, "no" → False) |
| `normalize` | 2 | Min-max or z-score normalization |
| `standardize_format` | 2 | Standardize mixed-format ID columns to consistent prefix+number pattern |
| `validate_foreign_key` | 2 | Remove rows with orphan FK references (multi-table) |
| `lookup_fill` | 2 | Fill nulls via FK lookup from another table (multi-table) |
| `remove_outliers` | 3 | Remove rows outside IQR x 1.5 bounds |

## 6 Built-in Tasks

| Task | Difficulty | Rows | Cols | Issues | Budget |
|------|-----------|------|------|--------|--------|
| task_easy | Easy | 200 | 5 | Nulls in name/age/salary/department/start_date, 12 duplicates | 20 |
| task_medium | Medium | 300 | 6 | Mixed date formats, "$" currency strings, mixed booleans, trailing whitespace | 20 |
| task_hard | Hard | 400 | 8 | Outliers in blood_pressure/cholesterol, mixed patient_id formats, year typos (2033→2023) | 20 |
| task_expert | Expert | 500 | 10 | All issues combined + 5 distractor columns, tight budget | 15 |
| task_multi | Expert+ | 100+300 | — | Two linked tables (customers + orders) with FK relationships, orphan keys, mixed formats | 25 |
| task_messy_contacts | Medium-Hard | 250 | 7 | Whitespace in names, mixed phone formats, $ salary strings, mixed date formats, nulls, dups | 20 |

## Custom Dataset Support

Users can upload their own CSV in the Custom Dataset tab. The system:
1. Auto-detects issues (nulls, duplicates, type mismatches, whitespace, categoricals, outliers)
2. Generates ground truth based on selected difficulty level:
   - **Easy**: drop duplicates + fill nulls + convert dates to datetime
   - **Medium**: + convert other types + strip whitespace + map booleans
   - **Hard**: + remove outliers (full cleaning)
3. Runs the rule-based agent and shows step-by-step results

## Observation Space

What the agent sees after each `reset()` or `step()`:

| Field | Source | Description |
|-------|--------|-------------|
| `table_preview` | Previous step | First 5 rows (for before/after UI display) |
| `table_schema` | Current | Column → dtype mapping |
| `null_counts` | Current | Nulls per column |
| `duplicate_count` | Current | Duplicate row count |
| `stats` | Current | Mean/std per numeric column |
| `distribution` | Current | Min/q1/median/q3/max/skew per numeric column |
| `step_count` | Current | Steps taken so far (0–20) |
| `budget_remaining` | Current | Action credits left |
| `task_id` | Current | Active task identifier |
| `column_descriptions` | Static | Semantic hints per column (contain cleaning instructions) |

Most fields reflect **current state** so the agent can react immediately (e.g. detect new duplicates from null-filling). Only `table_preview` is lagged for UI comparison.

## Scoring System

### Per-Step Reward (during episode)
```
reward = quality_delta * REWARD_SCALE - penalty - normalized_cost
```
- **REWARD_SCALE = 10.0** — amplifies quality signal so useful actions are clearly positive
- **quality_delta** = improvement over best_quality_so_far (high-water mark, prevents oscillation exploits)
- **penalty**: 0.5 for invalid actions, 0.3 + proportional damage for harmful actions, 0.1 for redundant
- **normalized_cost** = budget_cost / initial_budget (always 0–1 range)

### Quality Metric (used by both reward and final score)
```
overall = 0.6 * correctness + 0.3 * completeness + 0.1 * schema_accuracy
```
- **Correctness**: column-by-column sorted comparison with numeric tolerance (1e-6), row-ratio penalty
- **Completeness**: quadratic penalty for nulls and duplicates — `max(0, 1 - fraction/0.20)^2`
- **Schema accuracy**: fraction of columns with matching dtype (uses compatibility groups)

### Final Score (at episode end)
```
score = 0.40 * quality_overall
      + 0.20 * validation
      + 0.15 * efficiency
      + 0.10 * action_quality
      + 0.15 * schema_accuracy
```
- **Validation**: fraction of data validation rules passed (NULL_REMAINING, DUPLICATES, TYPE_MISMATCH, RANGE_VIOLATION, ROW_COUNT)
- **Efficiency**: 1 - (budget_used / total_budget)
- **Action quality**: 1 - (redundant_actions / total_actions)

## Validation Rules (5 rules checked at episode end)

| Rule | Description |
|------|-------------|
| NULL_REMAINING | Columns mentioned in descriptions as having nulls should have 0 nulls |
| DUPLICATES | No fully duplicate rows should remain |
| TYPE_MISMATCH | All column dtypes must match ground truth (or be compatible) |
| RANGE_VIOLATION | Numeric values within ±10% of ground truth range |
| ROW_COUNT | Row count within 80–120% of ground truth |

## Baseline Agent (Rule-Based)

**RuleBasedAgent** in `baseline/rule_agent.py` — deterministic, greedy 8-priority decision:

1. **Validate FK** (multi-table) — remove orphan rows before further work
2. **Drop duplicates** whenever `duplicate_count > 0` (checked every step — null-filling creates new dups)
3. **Strip whitespace** on string columns that mention it
4. **Standardize format** on mixed-format ID columns
5. **Replace substring** (e.g. "$" → "", "," → "", "2033" → "2023")
6. **Convert types** based on column descriptions (datetime, float)
7. **Fill nulls** using description-aware method selection:
   - Numeric → median | Date-like → forward_fill | Identifier/categorical → constant "Unknown"
8. **Map categorical values** (e.g. "yes"/"no" → True/False) and **remove outliers** on columns that mention outliers

## Null-Filling Strategy

| Column type | Fill method | Example |
|-------------|------------|---------|
| Numeric (int/float) | Median | age: 44.0 |
| Date-like string | forward_fill + bfill fallback | start_date: propagate adjacent dates |
| Identifier/name | Constant "Unknown" | name, department |
| Other string | Constant "Unknown" | categorical columns |

## Baseline Scores (80 tests passing)

```
task_easy:            0.940 (7 steps, 8 budget)
task_medium:          0.916 (6 steps, 9 budget)
task_hard:            0.917 (5 steps, 11 budget)
task_expert:          0.876 (7 steps, 11 budget)
task_multi:           0.881 (13 steps, 17 budget)
task_messy_contacts:  0.905 (9 steps, 11 budget)
Average:              0.906
```

## Key Design Decisions

1. **Observations use current state** — null_counts, duplicate_count, stats, distribution all reflect the CURRENT table so the agent can react immediately. Only table_preview is lagged (for UI before/after display).
2. **Reactive dedup** — Agent checks for duplicates at the top of EVERY step because null-filling can create new duplicates (rows differing only by NaN become identical after fill).
3. **Dedup → fill → dedup** — GT and agent both dedup first (so ffill isn't position-skewed by duplicate rows), fill nulls, then dedup again (to catch fill-created duplicates).
4. **Upfront date detection** — Date columns are identified BEFORE null-filling. After filling, `_is_date_column()` may fail if the column was filled with "Unknown". Detection result is cached and reused.
5. **Column-by-column sorted comparison** — Correctness compares each column's sorted values independently, avoiding row-alignment issues from outlier removal, dedup, or any row-reordering operation.
6. **GT built from messy data** — All task GTs start from the RAW messy table, not from pre-generated clean arrays. This ensures IQR bounds, medians, and other statistics match what the agent computes.
7. **Budget mechanic** — Variable costs (1–3 credits) force strategic action selection, mirroring real-world ETL billing constraints.
8. **Semantic column descriptions** — Agent must reason about what columns mean. Descriptions include cleaning hints appropriate to the difficulty level.
9. **High-water mark reward** — Uses `max(0, quality_now - best_so_far)` to prevent reward hacking via apply-undo-apply oscillation.

## Files That Matter

| File | Purpose |
|------|---------|
| `cleanflow_env/env/actions.py` | All 11 action implementations (IQR x 1.5 outlier rule) |
| `cleanflow_env/models/action.py` | ActionModel with outlier_method/outlier_threshold fields |
| `cleanflow_env/env/environment.py` | reset(), step(), build_observation() — obs uses current_table |
| `cleanflow_env/env/state.py` | EnvironmentState dataclass (raw, gt, current, prev tables) |
| `cleanflow_env/env/rewards.py` | compute_quality() with sorted comparison, compute_reward() with REWARD_SCALE |
| `cleanflow_env/env/grader.py` | final_score() — 5-component formula with validation |
| `cleanflow_env/env/validation.py` | 5 validation rules (NULL, DUPLICATES, TYPE, RANGE, ROW_COUNT) |
| `cleanflow_env/env/budget.py` | Action cost table + per-task budget defaults |
| `cleanflow_env/baseline/rule_agent.py` | Rule-based agent — reactive dedup, smart fill method selection |
| `cleanflow_env/tasks/task_easy.py` | Easy task: 200 rows, nulls + duplicates |
| `cleanflow_env/tasks/task_medium.py` | Medium task: 300 rows, mixed formats + whitespace |
| `cleanflow_env/tasks/task_hard.py` | Hard task: 400 rows, outliers + typos |
| `cleanflow_env/tasks/task_expert.py` | Expert task: 500 rows, all issues + distractors + tight budget |
| `cleanflow_env/tasks/task_multi.py` | Multi-table task: customers + orders with FK relationships |
| `cleanflow_env/tasks/task_messy_contacts.py` | Messy contacts: 250 rows, mixed phones, $ salaries, whitespace |
| `cleanflow_env/tasks/task_custom.py` | Custom dataset — upfront date detection, dedup→fill→dedup→convert |
| `cleanflow_env/api/dashboard.py` | Gradio dashboard (6 tabs) with interactive HITL mode |
| `cleanflow_env/api/main.py` | FastAPI endpoints (reset, step, preview, undo, state, grader, baseline) |
| `cleanflow_env/api/mcp_server.py` | MCP tool server — 7 tools for LLM agent integration |
| `inference.py` | LLM-based agent for hackathon submission |
| `openenv.yaml` | OpenEnv spec metadata (v2.0) |
| `run.py` | Entry point — uvicorn on :7860, auto-opens dashboard |

## How to Run

```bash
# Start server (opens dashboard at http://localhost:7860/dashboard)
python run.py

# Run tests
python -m pytest tests/ -v

# Run baseline
python -m cleanflow_env.baseline.run_baseline

# Run LLM inference (needs HF_TOKEN)
HF_TOKEN=your_token python inference.py

# Validate submission
python validate_submission.py
```

## What the Hackathon Judges

They judge the **environment quality**, not any AI model:
- Real-world utility (30%) — data cleaning is genuine, underserved
- Task & grader quality (25%) — 6 tasks (incl. multi-table), deterministic, difficulty progression
- Environment design (20%) — budget mechanic, validation rules, preview/undo, distribution stats
- Code quality & spec (15%) — typed models, 80 tests, Dockerfile, openenv.yaml, MCP server
- Creativity & novelty (10%) — budget mechanic, semantic hints, multi-table FK actions, HITL dashboard

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 3.0.1 | Data manipulation |
| numpy | 2.4.3 | Numerical computing |
| pydantic | 2.12.5 | Data validation (v2) |
| fastapi | 0.135.2 | Web framework |
| uvicorn | 0.42.0 | ASGI server |
| gradio | 6.10.0 | UI dashboard |
| pytest | 9.0.2 | Testing |

## Known Pitfalls (for future self)

1. **forward_fill is position-dependent** — Two duplicate rows at different positions get different fill values, breaking their duplicate status. Always dedup BEFORE ffill.
2. **Null-filling creates duplicates** — Rows differing only by NaN become identical after fill. Always dedup AFTER filling too.
3. **`_is_date_column` needs original data** — After filling nulls with "Unknown", date detection fails. Detect dates BEFORE filling.
4. **`format="mixed"` may not exist** — Older pandas versions don't support it. Always wrap in try/except with fallback to `pd.to_datetime(s, errors="coerce")`.
5. **GT must use messy data as starting point** — If GT is built from pre-generated clean arrays, IQR bounds and medians differ from what the agent computes on messy data. Always `gt = raw.copy()` then apply operations.
6. **Column descriptions trigger agent behavior** — Keywords like "date", "format", "datetime" in descriptions trigger type conversion. Use "No type conversion needed" to suppress. The column name itself is NOT checked, only the description value.
7. **`_dtype_compatible` groups** — Numeric (int/float variants), datetime (ns/us/ms), string (object/string/str/pyarrow). Cross-group comparison always fails. datetime64[us] vs object = TYPE_MISMATCH.
8. **`pd.duplicated()` treats NaN == NaN** — Two rows identical except both have NaN in the same column ARE considered duplicates by pandas.
