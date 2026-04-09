---
title: CleanFlowEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# CleanFlowEnv

**An OpenEnv-compliant environment for evaluating AI agents on real-world data cleaning and ETL workflows.**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Pydantic v2](https://img.shields.io/badge/pydantic-v2-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)

## Overview

Data cleaning accounts for **60-80% of real-world data work**, yet there are very few standardized environments to evaluate how well AI agents perform these tasks. CleanFlowEnv fills this gap with a fully deterministic, exploit-resistant, budget-constrained environment that scores agents on correctness, completeness, efficiency, and schema accuracy.

## Key Features

- **10 structured cleaning actions** — fill nulls, drop duplicates, strip whitespace, replace substrings, convert types, map categorical values, normalize columns, remove outliers, validate foreign keys, and lookup fill
- **5 difficulty-tiered tasks** — from basic null-filling (Easy) to multi-table cross-FK cleaning (Expert+)
- **Custom dataset support** — upload any CSV and auto-generate a cleaning task with ground truth at easy/medium/hard difficulty
- **Interactive Gradio dashboard** — visual step-by-step demo at `/dashboard` with live table previews, quality progress tracking, and one-click baseline execution
- **Exploit-resistant reward design** — high-water mark prevents apply-undo-apply oscillation; harmful and redundant actions are penalized
- **1-step observation lag** — table preview reflects the previous step, forcing agents to reason about whether their action worked instead of trivially reading and repeating
- **Budget-constrained actions** — each action type costs 1-3 credits, mirroring real ETL billing; agents must reason about ROI per action
- **5-component grading** — final score combines quality, validation, efficiency, action quality, and schema accuracy
- **Fully deterministic** — seeded data generation (seed 42), IQR x 1.5 outlier rule, reproducible across runs
- **Comprehensive test suite** — 60+ unit, integration, and smoke tests
- **Rule-based baseline agent** — deterministic 7-priority agent achieving ~0.90 average score, no API keys needed

## Environment Description

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `table_preview` | `List[TablePreviewRow]` | First 5 rows (1-step lagged from previous state) |
| `table_schema` | `Dict[str, str]` | Column → dtype mapping (current) |
| `null_counts` | `Dict[str, int]` | Nulls per column (current) |
| `duplicate_count` | `int` | Duplicate row count (current) |
| `stats` | `Dict[str, float]` | Mean/std per numeric column (current) |
| `distribution` | `Dict[str, Dict]` | min/q1/median/q3/max/skew per numeric column (current) |
| `step_count` | `int` | Steps taken so far |
| `budget_remaining` | `int` | Remaining action credits |
| `task_id` | `str` | Active task identifier |
| `column_descriptions` | `Dict[str, str]` | Semantic hints per column (cleaning instructions) |

### Action Space

| Action Type | Required Fields | Cost | Description |
|-------------|----------------|------|-------------|
| `fill_null` | `column`, `method` | 1 | Fill missing values (mean/median/mode/constant/forward\_fill/backward\_fill/sequential) |
| `drop_duplicates` | — | 1 | Remove all fully duplicate rows |
| `strip_whitespace` | `column` | 1 | Strip leading/trailing whitespace from string column |
| `replace_substring` | `column`, `old_value`, `new_value` | 1 | Replace occurrences of a substring (e.g. remove `$` or `,`) |
| `convert_type` | `column`, `target_type` | 2 | Convert column dtype (int/float/datetime/string) |
| `map_values` | `column`, `mapping` | 2 | Remap categorical values (e.g. yes/no/1/0 → True/False) |
| `normalize` | `column`, `method` | 2 | Scale column values (minmax/zscore) |
| `validate_foreign_key` | `column`, `table`, `foreign_key_column`, `lookup_table`, `lookup_key_column` | 2 | Remove rows with orphan FK references |
| `lookup_fill` | `column`, `table`, `foreign_key_column`, `lookup_table`, `lookup_key_column`, `lookup_value_column` | 2 | Fill nulls via FK lookup from another table |
| `remove_outliers` | `column` | 3 | Remove outliers using IQR x 1.5 rule |

### Reward Function

The per-step reward uses a high-water mark to prevent oscillation:

```
quality       = 0.6 * correctness + 0.3 * completeness + 0.1 * schema_accuracy
quality_delta = max(0, quality_now - best_quality_so_far)
reward        = quality_delta * 10.0 - penalty - normalized_cost
```

Penalties: invalid action (0.5), harmful action (0.3 + proportional damage), redundant action (0.1).

### Final Grading Formula

The grader evaluates the final cleaned dataset across 5 weighted components:

```
score = 0.40 * quality_overall
      + 0.20 * validation
      + 0.15 * efficiency
      + 0.10 * action_quality
      + 0.15 * schema_accuracy
```

| Component | Weight | Description |
|-----------|--------|-------------|
| Quality Overall | 40% | Cell-level correctness, completeness, and schema accuracy |
| Validation | 20% | Fraction of data validation rules passed (nulls, duplicates, types, ranges, row count) |
| Efficiency | 15% | `1 - (budget_used / total_budget)` — rewards conservative budget usage |
| Action Quality | 10% | `1 - (redundant_actions / total_actions)` — penalizes repeated actions |
| Schema Accuracy | 15% | Fraction of columns with correct dtype matching ground truth |

### Episode Lifecycle

```
reset(task_id) → step(action) → step(action) → ... → done=True → grader scores final state
```

Termination conditions: max 20 steps, budget exhausted, or perfect quality (1.0).

## Tasks

| Task ID | Difficulty | Dataset | Key Issues | Budget | Baseline Score |
|---------|-----------|---------|------------|--------|---------------|
| `task_easy` | Easy | Employee survey (200 rows, 5 cols) | Missing values, duplicates | 20 | ~0.94 |
| `task_medium` | Medium | Transactions (300 rows, 6 cols) | Mixed date formats, currency strings `$1,234.56`, mixed booleans (yes/no/1/0/True/False) | 20 | ~0.92 |
| `task_hard` | Hard | Medical trials (400 rows, 8 cols) | Outliers (IQR), mixed ID formats, year typos (2033→2023) | 20 | ~0.88 |
| `task_expert` | Expert | E-commerce catalog (500 rows, 10 cols) | All issues combined + 5 distractor columns that are already clean | 15 | ~0.88 |
| `task_multi` | Expert+ | Customers (100 rows) + Orders (300 rows) | Two linked tables with FK relationships, orphan keys, `$` amounts, nulls, duplicates, whitespace | 25 | ~0.86 |
| `task_custom` | Custom | User-uploaded CSV | Auto-detected issues with configurable difficulty (easy/medium/hard) | 20 | — |

## Interactive Dashboard

CleanFlowEnv includes a **Gradio-powered interactive dashboard** accessible at `/dashboard` that provides:

- **Task selector** — choose any built-in task or upload a custom CSV
- **Step-by-step visualization** — watch the table transform as actions are applied
- **Live metrics** — null counts, duplicate counts, quality progress, and budget usage update in real time
- **One-click baseline** — run the rule-based agent and see its full action sequence
- **Score breakdown** — view the final grading report with all 5 components

The dashboard is mounted alongside the API, so both are available when the server is running.

## Quickstart

### Docker

```bash
docker build -t cleanflow-env .
docker run -p 7860:7860 cleanflow-env
```

### Python (direct)

```bash
pip install -r requirements.txt
uvicorn cleanflow_env.api.main:app --host 0.0.0.0 --port 7860
```

### Python (in-process, no server)

```bash
python simulate.py
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start a new episode with a task |
| `POST` | `/step` | Apply a cleaning action |
| `GET` | `/state` | Get current internal state |
| `GET` | `/grader` | Get final score after episode |
| `GET` | `/tasks` | List available tasks and action schema |
| `POST` | `/grade/{task_id}` | Stateless per-task grading |
| `POST` | `/baseline` | Run rule-based baseline on all tasks |
| `GET` | `/dashboard` | Interactive Gradio UI |

### Example API Calls

```bash
# Reset environment with easy task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'

# Apply a fill_null action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "fill_null", "column": "age", "method": "mean"}}'

# Apply a map_values action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "map_values", "column": "is_active", "mapping": {"yes": true, "no": false}}}'

# Get final score
curl http://localhost:7860/grader

# List available tasks
curl http://localhost:7860/tasks

# Run baseline agent on all tasks
curl -X POST http://localhost:7860/baseline
```

## Baseline Scores

The rule-based baseline agent uses a deterministic 7-priority decision strategy (no LLM, no API keys):

| Task | Score | Steps | Budget Used |
|------|-------|-------|-------------|
| Task 1 (Easy) | ~0.94 | 7 | 8 / 20 |
| Task 2 (Medium) | ~0.92 | 6 | 9 / 20 |
| Task 3 (Hard) | ~0.88 | 4 | 9 / 20 |
| Task 4 (Expert) | ~0.88 | 7 | 11 / 15 |
| Task 5 (Multi-Table) | ~0.86 | 13 | 17 / 25 |
| **Average** | **~0.89** | | |

## Project Structure

```
cleanflow_env/
├── api/
│   ├── main.py              # FastAPI endpoints (reset, step, state, grader, tasks, baseline)
│   └── dashboard.py          # Gradio interactive dashboard (mounted at /dashboard)
├── baseline/
│   ├── rule_agent.py         # Rule-based deterministic agent (7-priority strategy)
│   └── run_baseline.py       # Baseline episode runner
├── env/
│   ├── actions.py            # 10 cleaning functions + O(1) dispatcher
│   ├── budget.py             # Budget cost table per action type
│   ├── environment.py        # CleanFlowEnv core (reset/step/state)
│   ├── grader.py             # Final 5-component scoring
│   ├── rewards.py            # Per-step quality + reward computation
│   ├── state.py              # EnvironmentState dataclass
│   └── validation.py         # Data validation rules (5 checks)
├── models/
│   ├── action.py             # ActionModel (Pydantic v2, type-validated)
│   ├── observation.py        # ObservationModel + TablePreviewRow (Pydantic v2)
│   └── reward.py             # RewardModel (Pydantic v2, consistency-checked)
├── tasks/
│   ├── task_easy.py          # Basic cleaning (200 rows, 5 cols)
│   ├── task_medium.py        # Schema normalization (300 rows, 6 cols)
│   ├── task_hard.py          # Advanced cleaning (400 rows, 8 cols)
│   ├── task_expert.py        # Budget-constrained (500 rows, 10 cols)
│   ├── task_multi.py         # Multi-table cleaning (customers + orders, FK relationships)
│   └── task_custom.py        # Custom CSV task generator with auto ground truth
tests/
├── test_actions.py           # Unit tests for all 8 cleaning functions
├── test_models.py            # Pydantic model validation tests
└── test_integration.py       # Full episode end-to-end tests
smoke_test.py                 # 5-check pre-submission smoke test
simulate.py                   # In-process simulation (no HTTP)
inference.py                  # LLM-based agent (HF Inference API)
Dockerfile                    # Python 3.11-slim, FastAPI on port 7860
openenv.yaml                  # OpenEnv compliance spec
requirements.txt              # Runtime dependencies
```

## Design Decisions

### 1-Step Observation Lag
`table_preview` reflects the *previous* step's state, not the current one. `null_counts`, `duplicate_count`, `stats`, and `distribution` are current. This prevents trivial read-act-repeat exploitation and forces agents to reason about whether their action actually worked.

### Hardcoded IQR x 1.5
Outlier removal uses IQR x 1.5 exclusively with no configurable threshold. This ensures fully deterministic, reproducible grading across all runs.

### Budget Mechanic
Each action type has a different credit cost (1-3 credits), with total budgets ranging from 15 (expert) to 20 (other tasks). This adds strategic depth — agents must reason about ROI per action, mirroring real-world ETL billing constraints.

### High-Water Mark Reward
The reward formula uses `max(0, quality_now - best_quality_so_far)` instead of raw quality delta. This prevents reward hacking via apply-undo-apply oscillation cycles.

### Gap-Aware Sequential Fill
For columns with sequential IDs (e.g. `Employee_001`, `Employee_002`, ...), the `sequential` fill method detects the prefix+number pattern and fills gaps first (e.g. filling `Employee_003` between `002` and `004`) before extending past the max. Falls back to `"Unknown"` if no sequential pattern is detected.

### Double Deduplication
Null-filling can create new duplicates (rows that differed only by NaN become identical after fill). The baseline agent checks for duplicates at every step, and the ground truth pipeline applies dedup both before and after filling.

### Sorted Column Comparison
Correctness is measured by comparing sorted column values rather than row-aligned comparison. This eliminates false penalties from row reordering caused by dedup or outlier removal.

## Future Extensions

- Streaming data cleaning (mini-batch episodes)
- Noisy real-world datasets from public data portals
- Human-in-the-loop feedback mode
- Leaderboard integration via HF Spaces
