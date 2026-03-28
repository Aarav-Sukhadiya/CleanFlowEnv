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

## Overview

Data cleaning accounts for **60-80% of real-world data work**, yet there are very few standardized environments to evaluate how well AI agents perform these tasks. CleanFlowEnv fills this gap.

- **Realistic datasets** with column-level semantics and human-readable descriptions
- **Budget-constrained actions** that mirror real ETL billing costs
- **Exploit-resistant reward design** using a high-water mark to prevent oscillation

## Environment Description

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `table_preview` | `List[TablePreviewRow]` | First 5 rows (1-step lagged) |
| `schema` | `Dict[str, str]` | Column → dtype mapping (current) |
| `null_counts` | `Dict[str, int]` | Nulls per column (1-step lagged) |
| `duplicate_count` | `int` | Duplicate row count (1-step lagged) |
| `stats` | `Dict[str, float]` | Mean/std per numeric column (1-step lagged) |
| `step_count` | `int` | Steps taken so far |
| `budget_remaining` | `int` | Remaining action credits |
| `task_id` | `str` | Active task identifier |
| `column_descriptions` | `Dict[str, str]` | Semantic hints per column |

### Action Space

| Action Type | Required Fields | Cost | Description |
|-------------|----------------|------|-------------|
| `fill_null` | `column`, `method` | 1 | Fill missing values (mean/median/mode/constant/ffill/bfill) |
| `drop_duplicates` | — | 1 | Remove all fully duplicate rows |
| `convert_type` | `column`, `target_type` | 2 | Convert column dtype (int/float/datetime/string) |
| `normalize` | `column` | 2 | Scale column values (minmax/zscore) |
| `remove_outliers` | `column` | 3 | Remove outliers using IQR x 1.5 rule |

### Reward Function

```
quality_now = 0.5 * correctness + 0.3 * completeness + 0.2 * schema_accuracy
net_improvement = max(0, quality_now - best_quality_so_far)
reward = net_improvement - penalties - budget_cost
```

### Episode Lifecycle

```
reset(task_id) → step(action) → step(action) → ... → done=True → grader scores final state
```

Termination: max 20 steps, budget exhausted, or perfect quality (1.0).

## Tasks

| Task ID | Difficulty | Dataset | Key Issues | Budget | Baseline Score |
|---------|-----------|---------|------------|--------|---------------|
| `task_easy` | Easy | Employee survey (200 rows) | Missing values, duplicates | 20 | ~0.90 |
| `task_medium` | Medium | Transactions (300 rows) | Mixed date formats, currency strings, mixed booleans | 20 | ~0.70 |
| `task_hard` | Hard | Medical trials (400 rows) | Outliers, mixed ID formats, year typos | 20 | ~0.45 |
| `task_expert` | Expert | E-commerce catalog (500 rows) | All issues + distractor columns | 15 | ~0.30 |

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

# Get final score
curl http://localhost:7860/grader

# List available tasks
curl http://localhost:7860/tasks

# Run baseline agent
curl -X POST http://localhost:7860/baseline
```

## Baseline Scores

| Task | Expected Score |
|------|---------------|
| Task 1 (Easy) | ~0.90 |
| Task 2 (Medium) | ~0.70 |
| Task 3 (Hard) | ~0.45 |
| Task 4 (Expert) | ~0.30 |
| **Average** | **~0.59** |

## Project Structure

```
cleanflow_env/
├── api/
│   └── main.py              # FastAPI endpoints
├── baseline/
│   ├── rule_agent.py         # Rule-based deterministic agent
│   └── run_baseline.py       # Baseline runner (rule-based + OpenAI)
├── env/
│   ├── actions.py            # Cleaning functions + dispatcher
│   ├── budget.py             # Budget cost table
│   ├── environment.py        # CleanFlowEnv (reset/step/state)
│   ├── grader.py             # Final scoring
│   ├── rewards.py            # Quality + reward computation
│   └── state.py              # EnvironmentState dataclass
├── models/
│   ├── action.py             # ActionModel (Pydantic v2)
│   ├── observation.py        # ObservationModel (Pydantic v2)
│   └── reward.py             # RewardModel (Pydantic v2)
├── tasks/
│   ├── task_easy.py          # Basic cleaning (200 rows)
│   ├── task_medium.py        # Schema normalization (300 rows)
│   ├── task_hard.py          # Advanced cleaning (400 rows)
│   └── task_expert.py        # Budget-constrained (500 rows)
├── Dockerfile
├── openenv.yaml
├── requirements.txt
└── README.md
```

## Design Decisions

### 1-Step Observation Lag
Stats, null counts, and table preview reflect the *previous* step's state, not the current one. This prevents trivial read-act-repeat exploitation and forces agents to reason about whether their action actually worked.

### Hardcoded IQR x 1.5
Outlier removal uses IQR x 1.5 exclusively with no configurable threshold. This ensures fully deterministic, reproducible grading across all runs.

### Budget Mechanic
Each action type has a different credit cost (1-3 credits), with total budgets ranging from 15 (expert) to 20 (other tasks). This adds strategic depth — agents must reason about ROI per action, mirroring real-world ETL billing constraints.

### High-Water Mark Reward
The reward formula uses `max(0, quality_now - best_quality_so_far)` instead of raw quality delta. This prevents reward hacking via apply-undo-apply oscillation cycles.

## Future Extensions

- Multi-table joins with referential integrity checks
- Streaming data cleaning (mini-batch episodes)
- Noisy real-world datasets from public data portals
- Human-in-the-loop feedback mode
- Leaderboard integration via HF Spaces
