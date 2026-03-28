# **🧾 CleanFlowEnv — Product Requirements Document (PRD) v2.0**

> **Changelog from v1.0:** Pydantic typed models added (spec compliance); reward oscillation bug fixed; Task 3 grader tightened with explicit IQR definition; Task 4 (bonus) added; `openenv.yaml` spec defined; baseline script updated to use OpenAI client; budget-constrained mechanic introduced; observation stats lagged by 1 step to prevent trivial solve.

---

## **1. Overview**

### **Name**

**CleanFlowEnv**

### **Summary**

CleanFlowEnv is an OpenEnv-compliant environment that simulates real-world **data cleaning and ETL workflows**, enabling AI agents to iteratively transform messy datasets into clean, analysis-ready tables.

The environment exposes a structured API (`reset()`, `step()`, `state()`) and evaluates agents using deterministic graders across **four tasks** of increasing complexity. Agents must reason about column semantics, manage a step budget, and apply minimal, effective transformation pipelines.

---

## **2. Motivation**

Data cleaning accounts for **60–80% of real-world data work**. Despite this, there are very few standardized environments to evaluate how well AI agents perform:

- Missing value imputation
- Schema normalization
- Deduplication
- Outlier handling

CleanFlowEnv fills this gap by providing:

- Realistic datasets with column-level semantics
- Structured, deterministic transformations
- Reproducible, deterministic evaluation
- Budget-aware reward signals that map to real-world ETL cost constraints

---

## **3. Goals**

### **Primary Goals**

- Simulate realistic data preprocessing workflows
- Enable reproducible agent evaluation
- Provide meaningful reward signals throughout execution

### **Secondary Goals**

- Support multi-step reasoning over column semantics
- Encourage efficient, minimal action sequences
- Penalize harmful, redundant, or budget-exceeding operations

---

## **4. Typed Models (OpenEnv Spec Compliance)**

> **NEW in v2.0** — All observation, action, and reward objects are typed Pydantic models. Required for `openenv validate` to pass.

### **4.1 Observation Model**

```python
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

class TablePreviewRow(BaseModel):
    row_index: int
    values: Dict[str, Optional[Any]]

class ObservationModel(BaseModel):
    table_preview: List[TablePreviewRow]       # First 5 rows (1-step lagged)
    schema: Dict[str, str]                     # column → inferred dtype
    null_counts: Dict[str, int]                # 1-step lagged null counts
    duplicate_count: int                       # 1-step lagged duplicate count
    stats: Dict[str, float]                    # mean/std per numeric column
    step_count: int
    budget_remaining: int                      # NEW: remaining action budget
    task_id: str
    column_descriptions: Dict[str, str]        # NEW: semantic hints per column
```

> **Key change:** `null_counts`, `duplicate_count`, and `table_preview` reflect the state **from the previous step**, not the current one. This 1-step lag prevents trivial read-and-act exploitation on Tasks 1 and 2 and forces agents to reason about whether an action actually worked.

### **4.2 Action Model**

```python
from pydantic import BaseModel
from typing import Literal, Optional

class ActionModel(BaseModel):
    action_type: Literal[
        "fill_null",
        "drop_duplicates",
        "convert_type",
        "normalize",
        "remove_outliers"
    ]
    column: Optional[str] = None
    method: Optional[str] = None          # e.g. "mean", "median", "mode", "constant"
    target_type: Optional[str] = None     # e.g. "datetime", "int", "float"
    constant_value: Optional[Any] = None  # used when method = "constant"
```

### **4.3 Reward Model**

```python
from pydantic import BaseModel

class RewardModel(BaseModel):
    reward: float                  # net reward for this step
    quality_delta: float           # improvement over best_quality_so_far
    penalty: float                 # penalties incurred this step
    budget_cost: float             # cost deducted from budget this step
    cumulative_quality: float      # quality of current_table vs ground_truth
    done: bool
    info: dict
```

---

## **5. Environment Design**

### **5.1 Internal State**

```python
{
  "raw_table": DataFrame,
  "current_table": DataFrame,
  "prev_table": DataFrame,             # NEW: for 1-step lag in observation
  "ground_truth": DataFrame,
  "operations_history": List[Action],
  "step_count": int,
  "task_id": str,
  "best_quality_so_far": float,        # NEW: tracks peak quality achieved
  "budget_remaining": int,             # NEW: action budget (see Section 8)
  "column_descriptions": Dict[str, str]  # NEW: semantic metadata per column
}
```

### **Key Design Insight**

We maintain **both raw and ground truth datasets**:

- `raw_table`: messy input
- `ground_truth`: perfectly cleaned version

This enables **exact, deterministic grading**.

We also track `best_quality_so_far` to prevent reward oscillation (see Section 7.1).

---

### **5.2 Observation Space**

Agents receive a **partial, structured view** with a 1-step lag on summary stats:

```json
{
  "table_preview": [
    {"row_index": 0, "values": {"age": null, "salary": 45000, "dob": "01-Jan-1990"}},
    {"row_index": 1, "values": {"age": 34.0, "salary": null, "dob": "1990/01/15"}}
  ],
  "schema": {
    "age": "float",
    "salary": "float",
    "dob": "string"
  },
  "null_counts": {
    "age": 5,
    "salary": 3
  },
  "duplicate_count": 3,
  "stats": {
    "age_mean": 29.4,
    "age_std": 6.1,
    "salary_mean": 52000
  },
  "step_count": 2,
  "budget_remaining": 18,
  "task_id": "task_easy",
  "column_descriptions": {
    "age": "Respondent age in years. Should be integer 18–90.",
    "salary": "Annual salary in USD. Nulls should use median.",
    "dob": "Date of birth. Target format: YYYY-MM-DD."
  }
}
```

### **Design Rationale**

- **1-step lag on stats**: Prevents trivial read-act-repeat exploitation
- **Column descriptions**: Forces semantic reasoning — e.g. agent must understand `dob` needs a specific date format, not just any type conversion
- **Budget remaining**: Visible to agent so it can plan efficient pipelines
- **Partial preview only**: Prevents full-table memorization

---

### **5.3 Action Space**

Actions are **structured and deterministic**.

#### **Fill Missing Values**

```json
{
  "action_type": "fill_null",
  "column": "age",
  "method": "mean"
}
```

Supported methods: `mean`, `median`, `mode`, `constant` (requires `constant_value`), `forward_fill`, `backward_fill`

#### **Remove Duplicates**

```json
{
  "action_type": "drop_duplicates"
}
```

#### **Convert Data Types**

```json
{
  "action_type": "convert_type",
  "column": "dob",
  "target_type": "datetime"
}
```

#### **Normalize Column**

```json
{
  "action_type": "normalize",
  "column": "salary"
}
```

#### **Remove Outliers**

```json
{
  "action_type": "remove_outliers",
  "column": "price"
}
```

> **Grader Note:** Outlier removal uses **IQR × 1.5 rule exclusively** (see Section 6.3). This is hardcoded — no ambiguity.

---

### **5.4 Constraints**

- Max steps per episode: **20**
- Invalid actions incur penalties
- Repeated identical actions on the same column are penalized
- Each action deducts from `budget_remaining` (see Section 8)

---

## **6. Tasks**

### **🟢 Task 1 — Basic Cleaning (Easy)**

**Dataset:** Employee survey data, 200 rows, 5 columns (`name`, `age`, `salary`, `department`, `start_date`)

**Issues introduced:**
- 15 missing values in `age` (fill with mean)
- 8 missing values in `salary` (fill with median)
- 12 duplicate rows

**Objective:**
- Fill nulls correctly per column semantics
- Remove all duplicates

**Evaluation:**
- % of nulls resolved with correct method
- Duplicate count reduced to zero

**Expected baseline score:** ~0.90

---

### **🟡 Task 2 — Schema Normalization (Medium)**

**Dataset:** Customer transaction records, 300 rows, 6 columns (`customer_id`, `amount`, `transaction_date`, `category`, `country_code`, `is_active`)

**Issues introduced:**
- `transaction_date` stored as inconsistent strings (`01-Jan-2023`, `2023/01/01`, `Jan 1 2023`)
- `amount` stored as string with currency symbols (`$1,200.50`)
- `is_active` stored as mixed booleans (`"yes"`, `"no"`, `1`, `0`)
- `country_code` has trailing whitespace

**Objective:**
- Convert all columns to correct types
- Standardize formats to match `ground_truth` schema

**Evaluation:**
- Schema dtype match vs ground truth
- Value-level accuracy after conversion

**Expected baseline score:** ~0.70

---

### **🔴 Task 3 — Advanced Cleaning (Hard)**

**Dataset:** Medical trial records, 400 rows, 8 columns

**Issues introduced:**
- Outliers in `blood_pressure` and `cholesterol` (defined as values outside IQR × 1.5 — **hardcoded rule, no ambiguity**)
- Mixed-type `patient_id` column (`"P001"`, `1`, `"001"`) — must normalize to zero-padded string format `"P001"`
- Subtle date inconsistency: `visit_date` contains plausible but wrong year entries (e.g. `2033` instead of `2023`)
- Some operations can degrade data quality if applied in wrong order

**Objective:**
- Detect and remove outliers using IQR × 1.5
- Normalize `patient_id` to `"P{zero_padded_3_digit_int}"` format
- Fix year anomalies in `visit_date`

**Difficulty Factors:**
- Outlier rule is fixed (IQR × 1.5), but agent must choose the right columns
- Wrong order of operations (e.g. normalizing before type conversion) costs quality points
- Multi-step reasoning required across dependent columns

**Expected baseline score:** ~0.45

---

### **⚫ Task 4 — Budget-Constrained Cleaning (Bonus / Expert)**

> **NEW in v2.0**

**Dataset:** E-commerce product catalog, 500 rows, 10 columns

**Issues introduced:**
- All issues from Tasks 1–3 combined
- Each action has a **cost** (see Section 8)
- Total budget: **15 action credits**
- Some columns are intentionally irrelevant (cleaning them wastes budget)

**Objective:**
- Maximize quality score under a strict action budget
- Prioritize high-impact cleaning actions

**Difficulty Factors:**
- Agent must reason about ROI per action
- Irrelevant columns act as distractors
- Budget exhaustion before completion is penalized

**Expected baseline score:** ~0.30

---

## **7. Reward Design**

### **Philosophy**

Reward should reflect **incremental progress above the best seen so far**, not just final success and not raw delta (which enables oscillation).

---

### **7.1 Reward Formula**

> **Fixed in v2.0:** Previous formula used raw `quality_delta` which allowed oscillation (apply → undo → apply = double reward). Now uses `best_quality_so_far` as a high-water mark.

```python
quality_now = compute_quality(current_table, ground_truth)

net_improvement = max(0, quality_now - best_quality_so_far)

reward = net_improvement - penalties - budget_cost

best_quality_so_far = max(best_quality_so_far, quality_now)
```

This ensures:
- **No reward for undoing and redoing** the same transformation
- **Continuous learning signal** — every genuine improvement is rewarded
- **Prevents reward hacking** via oscillation

---

### **7.2 Quality Score Calculation**

```python
quality = (
  0.5 * correctness +
  0.3 * completeness +
  0.2 * schema_accuracy
)
```

---

### **7.3 Reward Table**

| Action | Reward |
|--------|--------|
| Correct null fill (matching ground truth method) | +0.10 |
| Correct type conversion | +0.15 |
| Removing all duplicates | +0.10 |
| Correct outlier removal | +0.12 |
| Invalid action | -0.20 |
| Harmful transformation (degrades quality) | -0.30 |
| Redundant action (same column, same op) | -0.05 |
| Budget exceeded | -0.50 (episode ends) |

---

### **7.4 Key Properties**

- Reward is based on **improvement above best-so-far**, not absolute quality
- Oscillation is impossible — `net_improvement = max(0, ...)` floors at zero
- Budget cost is deducted per action (see Section 8)
- Ensures continuous, non-exploitable learning signal

---

## **8. Budget System (NEW)**

> Each action consumes credits from `budget_remaining`. This maps to real-world ETL billing (compute, API calls, cloud function costs).

### **Action Costs**

| Action | Credit Cost |
|--------|------------|
| `fill_null` | 1 |
| `drop_duplicates` | 1 |
| `convert_type` | 2 |
| `normalize` | 2 |
| `remove_outliers` | 3 |
| Invalid action | 1 (still consumed) |

### **Budget per Task**

| Task | Budget |
|------|--------|
| Task 1 (Easy) | 20 |
| Task 2 (Medium) | 20 |
| Task 3 (Hard) | 20 |
| Task 4 (Expert) | 15 |

### **Budget Termination**

If `budget_remaining` reaches 0 before the episode ends naturally, the episode terminates immediately and the final quality score is computed on the current state.

---

## **9. Grading System**

### **Final Score Range**

**0.0 → 1.0**

---

### **9.1 Scoring Formula**

```python
score = (
    0.4 * correctness +
    0.3 * completeness +
    0.2 * efficiency +
    0.1 * action_quality
)
```

---

### **9.2 Components**

**Correctness**

```
correct_cells / total_cells
```

**Completeness**

- All nulls handled
- Duplicates removed

**Efficiency**

```python
efficiency = 1 - (budget_used / total_budget)
```

> **Changed in v2.0:** Efficiency now uses `budget_used / total_budget` instead of `steps_used / max_steps`, aligning with the budget mechanic.

**Action Quality**

- Penalizes unnecessary transformations on already-clean columns
- Rewards minimal, effective pipelines

---

### **9.3 Determinism Guarantees**

- No randomness in grading
- Same state → same score
- Outlier removal uses **IQR × 1.5 exclusively** (no configurable threshold)
- All random dataset generation uses `np.random.seed(42)`
- Fully reproducible across runs

---

## **10. Episode Lifecycle**

```
reset()
 → step()
 → step()
 → ...
 → done = True
 → grader evaluates final state
```

### **Termination Conditions**

- Max steps reached (20)
- Budget exhausted
- Dataset quality reaches 1.0 (perfect match with ground truth)

---

## **11. Baseline Agent**

### **Strategy**

1. Fill missing values using column description hints
2. Remove duplicates
3. Convert types
4. Remove outliers on numeric columns

### **Implementation**

> **Updated in v2.0** — Uses `openai` Python client with `OPENAI_API_KEY` from environment variables, as required by OpenEnv spec.

```python
import os
from openai import OpenAI
import requests

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

def run_baseline(task_id: str) -> float:
    obs = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}).json()
    done = False

    while not done:
        prompt = f"""
You are a data cleaning agent. Given this observation:
{obs}

Return a single JSON action from the allowed action types:
fill_null, drop_duplicates, convert_type, normalize, remove_outliers.

Use column_descriptions to guide your choices.
Respond with only valid JSON matching the ActionModel schema.
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        action = response.choices[0].message.content
        result = requests.post(f"{BASE_URL}/step", json={"action": action}).json()
        obs = result["observation"]
        done = result["done"]

    score = requests.get(f"{BASE_URL}/grader").json()["score"]
    return score

if __name__ == "__main__":
    for task in ["task_easy", "task_medium", "task_hard", "task_expert"]:
        score = run_baseline(task)
        print(f"{task}: {score:.3f}")
```

### **Expected Scores**

| Task | Expected Score |
|------|---------------|
| Task 1 (Easy) | ~0.90 |
| Task 2 (Medium) | ~0.70 |
| Task 3 (Hard) | ~0.45 |
| Task 4 (Expert) | ~0.30 |
| Average | ~0.59 |

---

## **12. API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Initialize environment, accepts `{"task_id": str}` |
| `/step` | POST | Apply action, returns `ObservationModel + RewardModel` |
| `/state` | GET | Return full internal state |
| `/tasks` | GET | List tasks and action schema |
| `/grader` | GET | Return final score after episode ends |
| `/baseline` | POST | Run baseline agent, returns scores for all tasks |

---

## **13. `openenv.yaml` (NEW)**

> **Required for `openenv validate` to pass.**

```yaml
name: CleanFlowEnv
version: "2.0"
description: >
  An OpenEnv-compliant environment for evaluating AI agents on
  real-world data cleaning and ETL workflows.
tags:
  - openenv
  - data-cleaning
  - etl
  - tabular

endpoints:
  reset: /reset
  step: /step
  state: /state
  tasks: /tasks
  grader: /grader
  baseline: /baseline

tasks:
  - id: task_easy
    name: Basic Cleaning
    difficulty: easy
  - id: task_medium
    name: Schema Normalization
    difficulty: medium
  - id: task_hard
    name: Advanced Cleaning
    difficulty: hard
  - id: task_expert
    name: Budget-Constrained Cleaning
    difficulty: expert

score_range: [0.0, 1.0]
deterministic: true
random_seed: 42
```

---

## **14. Deployment**

### **Docker Requirements**

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### **Hugging Face Space**

- Container-based deployment
- Must respond to `/reset` with 200
- Tagged with `openenv` in Space metadata
- `README.md` must include `tags: [openenv]` in YAML front matter

---

## **15. Tech Stack**

- Python 3.10+
- FastAPI
- Pydantic v2
- Pandas
- NumPy
- Uvicorn
- OpenAI Python SDK (baseline only)

---

## **16. Project Structure**

```
cleanflow_env/
│
├── env/
│   ├── environment.py       # reset(), step(), state() logic
│   ├── state.py             # Internal state dataclass
│   ├── actions.py           # Action execution + validation
│   ├── rewards.py           # Reward computation (best_quality_so_far logic)
│   ├── grader.py            # Final scoring
│   ├── budget.py            # NEW: Budget tracking and cost table
│
├── tasks/
│   ├── task_easy.py
│   ├── task_medium.py
│   ├── task_hard.py
│   ├── task_expert.py       # NEW: Budget-constrained task
│
├── models/
│   ├── observation.py       # NEW: ObservationModel (Pydantic)
│   ├── action.py            # NEW: ActionModel (Pydantic)
│   ├── reward.py            # NEW: RewardModel (Pydantic)
│
├── api/
│   ├── main.py
│
├── baseline/
│   ├── run_baseline.py      # Updated to use OpenAI client
│
├── openenv.yaml             # NEW: Required for spec compliance
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## **17. Optimization Strategy**

### **Key Principles**

**1. Vectorized Operations**

Use Pandas vectorized ops. No row-level Python loops.

**2. Cached Metrics with Lag**

Null counts and stats are computed once per step and cached. The observation serves the *previous step's* cached values (1-step lag).

**3. Deterministic Seeds**

```python
np.random.seed(42)
```

Applied at dataset generation time only — not during grading.

**4. Minimal State Copies**

Only `prev_table` and `current_table` are full DataFrame copies. `raw_table` and `ground_truth` are read-only references.

---

## **18. Success Criteria**

- `openenv validate` passes
- Docker builds and runs cleanly
- HF Space deploys and responds to `/reset`
- Baseline script produces reproducible scores across all 4 tasks
- All tasks graded within 0.0–1.0
- No reward oscillation possible (verified via unit test)

---

## **19. Key Strengths**

- **Deterministic grading** — IQR threshold hardcoded, seeds fixed
- **Real-world relevance** — data cleaning is genuinely the most common data task
- **Semantic reasoning** — column descriptions force agents to reason about *meaning*, not just shape
- **Budget mechanic** — maps to real ETL cost constraints, adds strategic depth
- **Reward oscillation-proof** — best-quality high-water mark prevents exploitation
- **Spec-compliant** — typed Pydantic models, `openenv.yaml`, OpenAI-based baseline

---

## **20. Future Extensions**

- Multi-table joins with referential integrity checks
- Streaming data cleaning (mini-batch episodes)
- Noisy real-world datasets from public data portals
- Human-in-the-loop feedback mode
- Leaderboard integration via HF Spaces

---

# **🚀 Conclusion**

CleanFlowEnv v2.0 provides a **practical, exploit-resistant, and fully spec-compliant** environment for evaluating AI agents on data cleaning — one of the most important and underserved real-world tasks.

The v2.0 additions — typed models, reward high-water mark, 1-step observation lag, explicit IQR grading, budget mechanic, and Task 4 — address every gap in the original PRD and directly target the competition's scoring rubric across all five dimensions: real-world utility, task quality, environment design, code compliance, and creativity.
