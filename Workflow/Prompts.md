---

**🚀 PHASE 1 — MODELS**

---

**Prompt 1 — Action Model**

**Create a Pydantic v2 model called ActionModel for a data cleaning environment.**

**Fields:**  
**\- action\_type: Literal\["fill\_null", "drop\_duplicates", "convert\_type", "normalize", "remove\_outliers"\]**  
**\- column: optional string**  
**\- method: optional string — one of: mean, median, mode, constant, forward\_fill, backward\_fill**  
**\- target\_type: optional string — one of: int, float, datetime, string**  
**\- constant\_value: optional Any**

**Validation rules (use @model\_validator):**  
**\- fill\_null → column and method are required**  
**\- method \== constant → constant\_value must be provided**  
**\- convert\_type → column and target\_type are required**  
**\- normalize → column is required**  
**\- remove\_outliers → column is required**

**Requirements:**  
**\- Use Pydantic v2 syntax (model\_config, @model\_validator, @field\_validator)**  
**\- Raise clear, descriptive ValueError messages on validation failure**  
**\- Add a docstring to the class explaining its purpose**

**After the code, explain:**  
**1\. Why each validator is structured the way it is**  
**2\. Why Pydantic v2 is preferred over v1 here**  
**3\. Any edge cases handled**

**Optimize for: minimal overhead, clear error messages, no redundant checks.**  
---

**Prompt 2 — Observation Model**

**Create two Pydantic v2 models: TablePreviewRow and ObservationModel for a data cleaning RL environment.**

**TablePreviewRow fields:**  
**\- row\_index: int**  
**\- values: Dict\[str, Optional\[Any\]\]**

**ObservationModel fields:**  
**\- table\_preview: List\[TablePreviewRow\]**  
**\- schema: Dict\[str, str\]**  
**\- null\_counts: Dict\[str, int\]**  
**\- duplicate\_count: int**  
**\- stats: Dict\[str, float\]**  
**\- step\_count: int**  
**\- budget\_remaining: int**  
**\- task\_id: str**  
**\- column\_descriptions: Dict\[str, str\]**

**Requirements:**  
**\- Use Pydantic v2 syntax**  
**\- Add Field(..., description="...") annotations on every field — these will serve as live API docs**  
**\- table\_preview should have a max length of 10 rows (add validator)**  
**\- budget\_remaining must be \>= 0**

**After the code, explain:**  
**1\. Why Field descriptions matter for OpenEnv spec compliance**  
**2\. Why table\_preview has a max length constraint**  
**3\. How this model maps to what an agent actually receives**

**Optimize for: minimal serialization cost, clean JSON output.**  
---

**Prompt 3 — Reward Model**

**Create a Pydantic v2 model called RewardModel.**

**Fields:**  
**\- reward: float**  
**\- quality\_delta: float  — improvement over best\_quality\_so\_far**  
**\- penalty: float        — total penalties this step**  
**\- budget\_cost: float    — credits consumed this step**  
**\- cumulative\_quality: float  — quality of current table vs ground truth**  
**\- done: bool**  
**\- info: Dict\[str, Any\]  — arbitrary metadata (e.g. which action was applied, why done)**

**Requirements:**  
**\- Use Pydantic v2**  
**\- Add Field descriptions on all fields**  
**\- reward should be computed as: quality\_delta \- penalty \- budget\_cost (add a @computed\_field or a @model\_validator that checks this relationship and warns if inconsistent — do not enforce strictly, just log)**  
**\- Add a class method from\_step(quality\_delta, penalty, budget\_cost, cumulative\_quality, done, info) that constructs the model cleanly**

**After the code, explain:**  
**1\. Why info: dict is important for debugging RL environments**  
**2\. What the from\_step factory method buys us**  
**3\. How this reward model prevents oscillation when used correctly**

**Optimize for: immutability where possible, fast construction.**  
---

**⚙️ PHASE 2 — CORE LOGIC**

---

**Prompt 4 — Environment State Class**

**Create a Python dataclass (or plain class) called EnvironmentState to manage all state for a data cleaning RL environment.**

**Fields:**  
**\- raw\_table: pd.DataFrame          — original messy data, never mutated**  
**\- current\_table: pd.DataFrame      — agent's working copy**  
**\- prev\_table: pd.DataFrame         — copy from previous step (for 1-step observation lag)**  
**\- ground\_truth: pd.DataFrame       — target clean dataset**  
**\- step\_count: int                  — increments each step**  
**\- budget\_remaining: int            — decrements by action cost each step**  
**\- best\_quality\_so\_far: float       — high-water mark for reward computation**  
**\- operations\_history: List\[dict\]   — log of all actions taken**  
**\- task\_id: str**  
**\- column\_descriptions: Dict\[str, str\]**

**Methods:**  
**\- \_\_init\_\_(task\_id, raw\_table, ground\_truth, budget, column\_descriptions)**  
**\- reset() → restores all fields to initial values without re-generating data**  
**\- snapshot() → returns a deep copy of current\_table (used before applying action)**  
**\- to\_dict() → serializable dict of all state fields (excluding DataFrames, replace with shape info)**

**Requirements:**  
**\- raw\_table and ground\_truth must NEVER be mutated — enforce this with a property or comment clearly**  
**\- prev\_table must always be updated BEFORE current\_table in the step loop**  
**\- Use copy.deepcopy carefully — only where necessary to avoid performance issues**

**After the code, explain:**  
**1\. Why prev\_table exists and how it creates the 1-step observation lag**  
**2\. Why raw\_table must be immutable and how you enforce it**  
**3\. The memory implications of storing multiple DataFrame copies and how to minimize them**

**Optimize for: minimal unnecessary copies, fast reset, low memory footprint.**  
---

**Prompt 5 — Actions Implementation**

**Implement the following data cleaning functions using pandas. Each function must return a NEW dataframe and never mutate the input.**

**Functions:**

**1\. fill\_null(df: pd.DataFrame, column: str, method: str, constant\_value=None) \-\> pd.DataFrame**  
   **\- Supported methods: mean, median, mode, constant, forward\_fill, backward\_fill**  
   **\- For mode, use the first mode value**  
   **\- Raise ValueError for unknown method**

**2\. drop\_duplicates(df: pd.DataFrame) \-\> pd.DataFrame**  
   **\- Drop all fully duplicate rows**  
   **\- Reset index after dropping**

**3\. convert\_type(df: pd.DataFrame, column: str, target\_type: str) \-\> pd.DataFrame**  
   **\- Supported target types: int, float, datetime, string**  
   **\- For datetime: use pd.to\_datetime with errors="coerce"**  
   **\- For int/float: use pd.to\_numeric with errors="coerce"**  
   **\- Raise ValueError for unknown type**

**4\. normalize(df: pd.DataFrame, column: str, method: str \= "minmax") \-\> pd.DataFrame**  
   **\- minmax: scale to \[0, 1\]**  
   **\- zscore: subtract mean, divide by std**  
   **\- Raise ValueError for unknown method**  
   **\- Handle edge case: std \== 0 for zscore (return column of zeros)**

**5\. remove\_outliers(df: pd.DataFrame, column: str) \-\> pd.DataFrame**  
   **\- Use IQR × 1.5 rule EXCLUSIVELY — no other method**  
   **\- Remove rows where value \< Q1 \- 1.5\*IQR or value \> Q3 \+ 1.5\*IQR**  
   **\- Reset index after removal**

**Requirements:**  
**\- Use vectorized pandas operations only — no row-level Python loops**  
**\- Add type hints to all functions**  
**\- Add a one-line docstring to each function**

**After the code, explain:**  
**1\. Why IQR × 1.5 is hardcoded and why this matters for deterministic grading**  
**2\. Why errors="coerce" is used in type conversion and what happens to coerced values**  
**3\. The edge case in zscore normalization and why it must be handled**

**Optimize for: vectorized ops only, no unnecessary copies, minimal passes over data.**  
---

**Prompt 6 — Action Dispatcher**

**Create a function apply\_action(df: pd.DataFrame, action: ActionModel) \-\> pd.DataFrame that dispatches to the correct cleaning function.**

**Requirements:**  
**\- Import and call: fill\_null, drop\_duplicates, convert\_type, normalize, remove\_outliers**  
**\- Match on action.action\_type using a dispatch dict (not if/elif chains)**  
**\- If action\_type is unrecognized, raise ValueError with a clear message listing valid types**  
**\- If a required field is missing (e.g. column is None for fill\_null), raise ValueError**  
**\- Wrap all calls in try/except and re-raise as a custom InvalidActionError with context**

**Also define:**  
**\- class InvalidActionError(Exception): pass**  
**\- A BUDGET\_COSTS dict: { "fill\_null": 1, "drop\_duplicates": 1, "convert\_type": 2, "normalize": 2, "remove\_outliers": 3 }**  
**\- A function get\_action\_cost(action: ActionModel) \-\> int that returns the cost**

**After the code, explain:**  
**1\. Why a dispatch dict is better than if/elif for this use case**  
**2\. Why InvalidActionError is its own exception class**  
**3\. How budget costs are designed and why remove\_outliers costs more**

**Optimize for: O(1) dispatch, clear error propagation, no redundant validation (Pydantic already validated the model).**  
---

**Prompt 7 — Quality \+ Reward Function**

**Implement two functions for a data cleaning RL environment:**

**\--- Function 1: compute\_quality(current: pd.DataFrame, ground\_truth: pd.DataFrame) \-\> dict \---**

**Returns a dict with:**  
**\- correctness: float  — fraction of cells matching ground truth (after aligning on index and columns)**  
**\- completeness: float — 1.0 if no nulls and no duplicates, scaled otherwise**  
**\- schema\_accuracy: float — fraction of columns with matching dtype**  
**\- overall: float — 0.5 \* correctness \+ 0.3 \* completeness \+ 0.2 \* schema\_accuracy**

**Edge cases to handle:**  
**\- Mismatched shapes (align on common columns/indices)**  
**\- All nulls in ground\_truth column (skip that column for correctness)**  
**\- Empty dataframe (return 0.0 for all)**

**\--- Function 2: compute\_reward(current, ground\_truth, best\_quality\_so\_far, action, budget\_cost) \-\> RewardModel \---**

**Steps:**  
**1\. Call compute\_quality → get quality dict**  
**2\. quality\_delta \= max(0, overall \- best\_quality\_so\_far)**  
**3\. penalty \= 0 (valid action), \-0.2 (invalid), \-0.3 (harmful \= quality dropped), \-0.05 (redundant)**  
**4\. reward \= quality\_delta \- abs(penalty) \- budget\_cost**  
**5\. new best\_quality\_so\_far \= max(best\_quality\_so\_far, overall)**  
**6\. Return RewardModel.from\_step(...)**

**Requirements:**  
**\- Use vectorized pandas comparison for correctness**  
**\- Never mutate inputs**  
**\- All logic must be deterministic — no randomness**

**After the code, explain:**  
**1\. Why quality\_delta uses max(0, ...) and how this prevents oscillation**  
**2\. How correctness handles dtype mismatches during cell comparison**  
**3\. The difference between penalty tiers (-0.05 vs \-0.2 vs \-0.3)**

**Optimize for: single-pass quality computation where possible, vectorized cell comparison.**  
---

**🔥 PHASE 3 — ENVIRONMENT**

---

**Prompt 8 — Environment Class**

**Create a class CleanFlowEnv that implements the full OpenEnv interface.**

**Methods:**

**1\. reset(task\_id: str) \-\> ObservationModel**  
   **\- Load dataset for task\_id (call load\_task(task\_id) — assume it returns raw\_table, ground\_truth, budget, column\_descriptions)**  
   **\- Initialize EnvironmentState**  
   **\- Return initial observation via build\_observation(state)**

**2\. step(action\_dict: dict) \-\> tuple\[ObservationModel, RewardModel\]**  
   **\- Parse action\_dict into ActionModel (catch ValidationError)**  
   **\- Get budget cost via get\_action\_cost(action)**  
   **\- Check if budget\_remaining \>= cost, else return penalty reward and end episode**  
   **\- Snapshot current state to prev\_table**  
   **\- Apply action via apply\_action — catch InvalidActionError**  
   **\- Update state: step\_count, budget\_remaining, operations\_history, best\_quality\_so\_far**  
   **\- Compute reward via compute\_reward**  
   **\- Check termination: max steps OR budget \== 0 OR quality \== 1.0**  
   **\- Return (build\_observation(state), reward\_model)**

**3\. state() \-\> dict**  
   **\- Return state.to\_dict()**

**Class-level:**  
**\- Store a single EnvironmentState instance**  
**\- Store task\_registry: Dict\[str, callable\] \= {"task\_easy": generate\_easy\_task, ...}**

**Requirements:**  
**\- step() must update prev\_table BEFORE modifying current\_table**  
**\- Handle edge case: step() called before reset() — raise RuntimeError**  
**\- All errors must be caught and returned as penalty rewards, not crashes**

**After the code, explain:**  
**1\. Why prev\_table is updated before current\_table and what breaks if you get this wrong**  
**2\. How budget exhaustion terminates an episode cleanly**  
**3\. Why step() should never crash the server — errors become penalty rewards**

**Optimize for: minimal state copies per step, fast action dispatch, clean episode lifecycle.**  
---

**Prompt 9 — Observation Builder**

**Create a function build\_observation(state: EnvironmentState) \-\> ObservationModel.**

**Requirements:**  
**\- Use state.prev\_table (NOT current\_table) for:**  
  **\- table\_preview (first 5 rows)**  
  **\- null\_counts**  
  **\- duplicate\_count**  
  **\- stats (mean, std for numeric columns only)**  
**\- Use state.current\_table for:**  
  **\- schema (dtype mapping) — always reflects latest state**  
**\- Include state.step\_count, state.budget\_remaining, state.task\_id, state.column\_descriptions**

**Implementation details:**  
**\- table\_preview: convert first 5 rows to List\[TablePreviewRow\], replacing NaN with None**  
**\- schema: map pandas dtypes to human-readable strings: float64 → "float", int64 → "int", object → "string", datetime64 → "datetime"**  
**\- null\_counts: use df.isnull().sum() as dict**  
**\- stats: for each numeric column include {col\_mean: float, col\_std: float}**  
**\- duplicate\_count: int(df.duplicated().sum())**

**After the code, explain:**  
**1\. Why prev\_table is used for stats/nulls — what the 1-step lag achieves for agent behaviour**  
**2\. Why schema uses current\_table (hint: schema changes must be immediately visible)**  
**3\. How NaN → None conversion works and why it matters for JSON serialization**

**Optimize for: single-pass stats computation, minimal DataFrame copies, fast serialization.**  
---

**📊 PHASE 4 — TASKS**

---

**Prompt 10 — Task Easy (Basic Cleaning)**

**Create a function generate\_easy\_task() that returns (raw\_table, ground\_truth, budget, column\_descriptions).**

**Dataset spec:**  
**\- 200 rows**  
**\- columns: name (string), age (float), salary (float), department (string), start\_date (string)**  
**\- Use np.random.seed(42) for reproducibility**

**Issues to inject into raw\_table:**  
**\- 15 random nulls in age**  
**\- 8 random nulls in salary**  
**\- 12 duplicate rows (copy 12 existing rows and append)**

**ground\_truth:**  
**\- age nulls filled with mean (rounded to 1 decimal)**  
**\- salary nulls filled with median (rounded to 2 decimals)**  
**\- duplicates removed**  
**\- index reset**

**column\_descriptions: dict with a clear human-readable description for each column including expected type and value range.**

**budget: 20**

**Requirements:**  
**\- ground\_truth must be computed programmatically from raw\_table — not hardcoded**  
**\- All random operations use np.random.seed(42)**  
**\- Return types must be exactly: pd.DataFrame, pd.DataFrame, int, Dict\[str, str\]**

**After the code, explain:**  
**1\. Why ground\_truth is computed from raw\_table programmatically (reproducibility \+ no drift)**  
**2\. How duplicate injection works without introducing obvious patterns**  
**3\. Why np.random.seed(42) is set at the top of the function, not globally**

**Optimize for: fast dataset generation, minimal memory, seed isolation.**  
---

**Prompt 11 — Tasks Medium \+ Hard \+ Expert**

**Create three functions following the same pattern as generate\_easy\_task():**

**\--- generate\_medium\_task() \---**  
**Dataset: 300 rows, columns: customer\_id, amount, transaction\_date, category, country\_code, is\_active**  
**Issues:**  
**\- transaction\_date: mixed formats ("01-Jan-2023", "2023/01/01", "Jan 1 2023") — ground truth is datetime**  
**\- amount: stored as string with "$" and "," (e.g. "$1,200.50") — ground truth is float**  
**\- is\_active: mixed ("yes", "no", 1, 0\) — ground truth is bool**  
**\- country\_code: trailing whitespace on 30% of rows**  
**Budget: 20**

**\--- generate\_hard\_task() \---**  
**Dataset: 400 rows, medical trial data: patient\_id, age, blood\_pressure, cholesterol, visit\_date, treatment, outcome, dosage**  
**Issues:**  
**\- blood\_pressure and cholesterol: outliers using IQR × 1.5 rule (inject values at Q3 \+ 2\*IQR)**  
**\- patient\_id: mixed ("P001", "1", "001") — normalize to "P{zero\_padded\_3\_digit}"**  
**\- visit\_date: 20 rows have year 2033 instead of 2023 (subtle typo)**  
**Budget: 20**

**\--- generate\_expert\_task() \---**  
**Dataset: 500 rows, e-commerce product catalog — combine all issues from easy \+ medium \+ hard**  
**Add 5 irrelevant clean columns (distractors — normalizing them wastes budget)**  
**Budget: 15**

**Requirements for all:**  
**\- np.random.seed(42) in each function**  
**\- ground\_truth always computed programmatically**  
**\- column\_descriptions included for every column**

**After the code, explain:**  
**1\. How mixed-format date injection works without breaking pandas parsing**  
**2\. Why the expert task has distractor columns and what this tests in the agent**  
**3\. How injected outliers are guaranteed to be outside IQR × 1.5**

**Optimize for: realistic-looking data, fast generation, correct ground truth computation.**  
---

**🌐 PHASE 5 — API**

---

**Prompt 12 — FastAPI Application**

**Create a FastAPI application for CleanFlowEnv.**

**Endpoints:**

**POST /reset**  
**\- Body: { "task\_id": str }**  
**\- Calls env.reset(task\_id)**  
**\- Returns ObservationModel as JSON**

**POST /step**  
**\- Body: { "action": dict }**  
**\- Calls env.step(action)**  
**\- Returns { "observation": ObservationModel, "reward": RewardModel }**

**GET /state**  
**\- Returns env.state()**

**GET /grader**  
**\- Calls grader.score(env.state)**   
**\- Returns { "score": float, "breakdown": dict }**

**GET /tasks**  
**\- Returns list of available task IDs \+ action schema (ActionModel.model\_json\_schema())**

**POST /baseline**  
**\- Runs the baseline agent against all 4 tasks**  
**\- Returns { task\_id: score } for each**

**Setup:**  
**\- Single global CleanFlowEnv instance**  
**\- Use lifespan context manager for startup**  
**\- All endpoints return proper HTTP error codes (400 for bad input, 500 for server errors)**  
**\- Add CORS middleware**

**Requirements:**  
**\- Use FastAPI's response\_model= on every endpoint**  
**\- Wrap all env calls in try/except → return HTTPException with detail**  
**\- Do not let unhandled exceptions reach the client**

**After the code, explain:**  
**1\. Why a global env instance is acceptable here (single-agent, single-session)**  
**2\. How the /baseline endpoint works without blocking the server (hint: consider background tasks or just sync for now)**  
**3\. Why /tasks exposes ActionModel.model\_json\_schema() — how this helps agents**

**Optimize for: fast response serialization, clean error handling, minimal boilerplate.**  
---

**Prompt 13 — Grader**

**Create a grader module with a function final\_score(state: EnvironmentState) \-\> dict.**

**Scoring formula:**  
**score \= 0.4 \* correctness \+ 0.3 \* completeness \+ 0.2 \* efficiency \+ 0.1 \* action\_quality**

**Components:**  
**\- correctness: correct\_cells / total\_cells (reuse compute\_quality logic)**  
**\- completeness: 1.0 if no nulls and no duplicates in current\_table, else scaled**  
**\- efficiency: 1 \- (budget\_used / total\_budget), where budget\_used \= initial\_budget \- budget\_remaining**  
**\- action\_quality: 1.0 \- (redundant\_actions / total\_actions), where redundant \= same op on same column twice**

**Return:**  
**{**  
  **"score": float,**  
  **"correctness": float,**  
  **"completeness": float,**  
  **"efficiency": float,**  
  **"action\_quality": float**  
**}**

**Requirements:**  
**\- Score must always be in \[0.0, 1.0\] — clamp if needed**  
**\- Fully deterministic — same state always returns same score**  
**\- No side effects**

**After the code, explain:**  
**1\. Why efficiency uses budget\_used instead of steps\_used**  
**2\. How action\_quality penalizes redundant ops without being too harsh**  
**3\. Why score is clamped to \[0, 1\] even though the formula should already be bounded**

**Optimize for: single-pass where possible, no redundant recomputation of quality.**  
---

**🤖 PHASE 6 — AGENT \+ TESTING**

---

**Prompt 14 — Baseline Agent (OpenAI)**

**Create a baseline inference script using the OpenAI Python client.**

**Requirements:**  
**\- Read OPENAI\_API\_KEY and ENV\_BASE\_URL from environment variables**  
**\- Use the openai Python client (not requests directly for the LLM call)**  
**\- For each task in \[task\_easy, task\_medium, task\_hard, task\_expert\]:**  
  **1\. POST /reset with task\_id**  
  **2\. Loop:**  
     **a. Build a prompt from the current observation**  
     **b. Call gpt-4o with response\_format={"type": "json\_object"}**  
     **c. Parse response into ActionModel**  
     **d. POST /step with action**  
     **e. Print step\_count, reward, cumulative\_quality**  
  **3\. GET /grader → print final score**  
**\- Print a summary table at the end**

**Prompt to the model should include:**  
**\- Full observation (formatted clearly)**  
**\- column\_descriptions**  
**\- Instruction to return valid JSON matching ActionModel schema**  
**\- Reminder about budget\_remaining**

**Requirements:**  
**\- Catch JSON parse errors → skip step with a log**  
**\- Catch API errors → retry once, then skip**  
**\- All scores printed to 3 decimal places**

**After the code, explain:**  
**1\. Why response\_format=json\_object is used and what it prevents**  
**2\. How the prompt includes budget context to encourage efficient behaviour**  
**3\. What happens when the model returns an invalid action**

**Optimize for: minimal API calls, clear prompt, robust error handling.**  
---

**Prompt 15 — End-to-End Simulation Loop**

**Write a Python script simulate.py that runs a complete end-to-end test of CleanFlowEnv without using the API (direct Python calls only — no HTTP).**

**Steps:**  
**1\. Import CleanFlowEnv, generate\_easy\_task, generate\_medium\_task**  
**2\. Instantiate env \= CleanFlowEnv()**  
**3\. For task\_id in \["task\_easy", "task\_medium"\]:**  
   **a. obs \= env.reset(task\_id)**  
   **b. Print initial observation summary (null counts, duplicate count, budget)**  
   **c. Loop until done:**  
      **\- Use the simple rule-based agent (from previous prompt) to choose action**  
      **\- obs, reward \= env.step(action.model\_dump())**  
      **\- Print: step | action\_type | column | reward.reward | reward.cumulative\_quality | budget\_remaining**  
   **d. score \= grader.final\_score(env.\_state)**  
   **e. Print final score breakdown**

**Requirements:**  
**\- Rule-based agent must avoid repeating the same action (track history)**  
**\- Print output must be readable — use tabular format (no external libs, just f-strings)**  
**\- Script must run to completion without errors**

**After the code, explain:**  
**1\. Why this direct-call simulation is valuable before adding the HTTP layer**  
**2\. How the rule-based agent avoids infinite loops**  
**3\. What the printed output tells you about agent behaviour**

**Optimize for: clarity over cleverness — this is a diagnostic tool.**

**Prompt 16 — Rule-Based Agent**

**Create a class RuleBasedAgent for a data cleaning environment.**

**It receives an ObservationModel and returns an ActionModel.**

**Decision logic (in priority order):**  
**1\. If duplicate\_count \> 0 and "drop\_duplicates" not in action\_history → return drop\_duplicates action**  
**2\. For each column in null\_counts where null\_counts\[col\] \> 0:**  
   **\- If not already filled (check action\_history) → return fill\_null**  
   **\- Use column\_descriptions to pick method:**  
     **\- If description mentions "age", "salary", "numeric" → use median**  
     **\- If description mentions "category", "name", "string" → use mode**  
     **\- Default → mean**  
**3\. For each column in schema where dtype is "string":**  
   **\- If column\_descriptions suggests it should be datetime → return convert\_type**  
   **\- If column\_descriptions suggests it should be int or float → return convert\_type**  
**4\. If no obvious action → return remove\_outliers on first numeric column not yet processed**

**Action history tracking:**  
**\- Store as List\[dict\] — each entry is {action\_type, column}**  
**\- is\_redundant(action) → returns True if same action\_type \+ column already in history**

**Requirements:**  
**\- Never return an action already in history (check before returning)**  
**\- If all actions exhausted → raise StopIteration**  
**\- Add type hints throughout**

**After the code, explain:**  
**1\. How priority ordering prevents wasted budget steps**  
**2\. How column\_descriptions drive method selection without hardcoding column names**  
**3\. What happens when the agent exhausts all useful actions**

**Optimize for: fewest steps to maximum quality, clear decision trace.**  
---

**🐳 PHASE 7 — DEPLOYMENT**

---

**Prompt 17 — Dockerfile**

**Write a production-ready Dockerfile for CleanFlowEnv.**

**Requirements:**  
**\- Base image: python:3.10-slim**  
**\- Working directory: /app**  
**\- Copy requirements.txt first (layer caching)**  
**\- Install dependencies with \--no-cache-dir**  
**\- Copy source code after dependencies**  
**\- Expose port 7860**  
**\- Start command: uvicorn api.main:app \--host 0.0.0.0 \--port 7860**

**Also write requirements.txt with pinned versions for:**  
**\- fastapi**  
**\- uvicorn\[standard\]**  
**\- pydantic (v2)**  
**\- pandas**  
**\- numpy**  
**\- openai**  
**\- httpx (for testing)**

**Requirements:**  
**\- Image must be minimal — no dev dependencies**  
**\- Add a .dockerignore file that excludes: \_\_pycache\_\_, .env, \*.pyc, .git, tests/**

**After the code, explain:**  
**1\. Why requirements.txt is copied before source code (Docker layer caching)**  
**2\. Why python:3.10-slim is chosen over python:3.10-full**  
**3\. What uvicorn\[standard\] adds over plain uvicorn**

**Optimize for: smallest possible image size, fast rebuild on code changes.**  
---

**Prompt 18 — openenv.yaml**

**Write the complete openenv.yaml file for CleanFlowEnv.**

**It must include:**

**Top-level metadata:**  
**\- name: CleanFlowEnv**  
**\- version: "2.0"**  
**\- description: one paragraph, clear and specific**  
**\- tags: \[openenv, data-cleaning, etl, tabular\]**  
**\- random\_seed: 42**  
**\- deterministic: true**

**Endpoints section:**  
**\- reset, step, state, tasks, grader, baseline — each with path and method**

**Tasks section — for each of the 4 tasks:**  
**\- id, name, difficulty, description, budget, expected\_baseline\_score**

**Score section:**  
**\- range: \[0.0, 1.0\]**  
**\- formula description**  
**\- components: correctness, completeness, efficiency, action\_quality with weights**

**Action schema section:**  
**\- List all 5 action types with required fields for each**

**Also write a validate\_openenv.py script that:**  
**\- Loads openenv.yaml**  
**\- Checks all required keys are present**  
**\- Pings each endpoint (using httpx) and checks for 200 response**  
**\- Prints PASS / FAIL for each check**

**After the code, explain:**  
**1\. Why deterministic: true is a competitive advantage for this submission**  
**2\. What openenv validate checks and why your yaml satisfies it**  
**3\. Why expected\_baseline\_score is included per task**

**Optimize for: human readability, complete spec compliance, no missing fields.**  
---

**Prompt 19 — README**

**Write a complete README.md for CleanFlowEnv.**

**Sections required (in this order):**

**1\. Header — name, one-line description, badges (OpenEnv compliant, Python 3.10, Docker)**

**2\. Overview — what the environment simulates, why data cleaning, 3 bullet points on what makes it unique**

**3\. Environment Description —**   
   **\- Observation space (table with field names, types, descriptions)**  
   **\- Action space (table with action\_type, required fields, cost, description)**  
   **\- Reward function (formula, plain English explanation)**  
   **\- Episode lifecycle (reset → step loop → grader)**

**4\. Tasks —**  
   **\- Table: Task ID | Difficulty | Dataset | Key Issues | Budget | Baseline Score**

**5\. Quickstart —**  
   **\- Docker: docker build \+ docker run commands**  
   **\- Python direct: pip install \+ uvicorn command**  
   **\- Example /reset and /step curl commands**

**6\. Baseline Scores — table of all 4 tasks with expected scores**

**7\. Project Structure — directory tree with one-line description per file**

**8\. Design Decisions — 3–5 short paragraphs on:**  
   **\- Why 1-step observation lag**  
   **\- Why IQR × 1.5 is hardcoded**  
   **\- Why budget mechanic**  
   **\- Why best\_quality\_so\_far prevents reward hacking**

**9\. Future Extensions — bullet list**

**Requirements:**  
**\- All curl examples must be copy-pasteable and correct**  
**\- No placeholder text — every section must be complete**  
**\- Markdown tables for observation space, action space, tasks**

**After writing, explain:**  
**1\. Which README sections judges will read first and why**  
**2\. How the Design Decisions section signals engineering maturity**  
**3\. Why curl examples must be exact**

**Optimize for: skimmability, judge-first impression, technical completeness.**  
---

**🧪 PHASE 8 — TESTING**

---

**Prompt 20 — Unit Tests: Models**

**Write pytest unit tests for ActionModel, ObservationModel, and RewardModel.**

**For ActionModel test:**  
**\- Valid fill\_null action passes**  
**\- fill\_null without column raises ValidationError**  
**\- fill\_null with method=constant but no constant\_value raises ValidationError**  
**\- convert\_type without target\_type raises ValidationError**  
**\- Invalid action\_type raises ValidationError**  
**\- Valid drop\_duplicates (no column needed) passes**

**For ObservationModel test:**  
**\- Valid full construction passes**  
**\- table\_preview with more than 10 rows raises ValidationError**  
**\- budget\_remaining below 0 raises ValidationError**  
**\- Empty null\_counts and stats are valid**

**For RewardModel test:**  
**\- from\_step() constructs correctly**  
**\- reward field equals quality\_delta \- penalty \- budget\_cost (within float tolerance)**  
**\- done=True accepted**  
**\- info accepts arbitrary dict**

**Requirements:**  
**\- Use pytest fixtures for reusable valid model instances**  
**\- Each test has a clear docstring**  
**\- Group tests in classes: TestActionModel, TestObservationModel, TestRewardModel**

**After the code, explain:**  
**1\. Why testing Pydantic models explicitly is worth it (they can have bugs in validators)**  
**2\. How float tolerance is handled in reward equality checks**  
**3\. What a fixture buys you vs inline construction in every test**

**Optimize for: fast test suite, no I/O, pure unit tests only.**  
---

**Prompt 21 — Unit Tests: Actions \+ Reward**

**Write pytest unit tests for the data cleaning functions and reward computation.**

**For fill\_null:**  
**\- mean fill produces correct value (verify against manual calculation)**  
**\- median fill on skewed data produces correct median**  
**\- mode fill on categorical column works**  
**\- constant fill with value 0 works**  
**\- forward\_fill propagates correctly**  
**\- unknown method raises ValueError**

**For drop\_duplicates:**  
**\- removes exact duplicates**  
**\- index is reset after dropping**  
**\- no duplicates → unchanged**

**For convert\_type:**  
**\- string "123" → int works**  
**\- mixed date formats → datetime works (use 3 known formats)**  
**\- invalid string → float produces NaN (errors=coerce)**  
**\- unknown target\_type raises ValueError**

**For remove\_outliers:**  
**\- injected outlier (Q3 \+ 2\*IQR) is removed**  
**\- non-outlier values are preserved**  
**\- result index is reset**

**For compute\_quality:**  
**\- perfect match returns 1.0 overall**  
**\- all nulls returns low completeness**  
**\- dtype mismatch returns low schema\_accuracy**  
**\- empty df returns 0.0**

**Requirements:**  
**\- Use small (10–20 row) hand-crafted DataFrames — no task generators**  
**\- All expected values computed manually, not from the function itself**  
**\- Each function gets at least 3 tests**

**After the code, explain:**  
**1\. Why test DataFrames are hand-crafted and not generated from task functions**  
**2\. How to test IQR removal deterministically**  
**3\. What "computed manually" means and why it prevents circular testing**

**Optimize for: deterministic, isolated, fast — no file I/O, no randomness.**  
---

**Prompt 22 — Integration Test**

**Write a pytest integration test that runs a full episode end-to-end using direct Python calls (no HTTP).**

**Test: test\_full\_episode\_easy\_task()**

**Steps:**  
**1\. Instantiate CleanFlowEnv**  
**2\. Call reset("task\_easy") — assert returns valid ObservationModel**  
**3\. Assert initial null\_counts \> 0 and duplicate\_count \> 0 in observation**  
**4\. Run up to 20 steps using RuleBasedAgent**  
**5\. Assert done=True is reached before step 20**  
**6\. Call grader.final\_score(env.\_state)**  
**7\. Assert score \>= 0.80 (easy task should score well with rule-based agent)**  
**8\. Assert operations\_history is non-empty**  
**9\. Assert budget\_remaining \>= 0**

**Also write: test\_reward\_no\_oscillation()**  
**\- Apply same action twice**  
**\- Assert second reward.quality\_delta \== 0.0 (high-water mark prevents double reward)**

**Also write: test\_budget\_exhaustion()**  
**\- Set budget to 3**  
**\- Run steps until budget \== 0**  
**\- Assert episode terminates with done=True**  
**\- Assert final reward includes budget\_cost deduction**

**Requirements:**  
**\- Use pytest, no mocking — real env, real data**  
**\- Each assertion has a comment explaining what it's checking and why**

**After the code, explain:**  
**1\. Why \>= 0.80 is the right threshold for easy task (not 1.0)**  
**2\. How test\_reward\_no\_oscillation proves the high-water mark works**  
**3\. Why budget\_exhaustion is a critical integration test**

**Optimize for: comprehensive coverage in minimal test count, clear failure messages.**  
---

**Prompt 23 — Final Validation Script**

**Write a script called validate\_submission.py that acts as a pre-submission checklist.**

**It must check and print PASS/FAIL for each of the following:**

**Environment checks (direct Python):**  
**\- \[ \] CleanFlowEnv instantiates without error**  
**\- \[ \] reset() returns valid ObservationModel for all 4 task IDs**  
**\- \[ \] step() returns valid ObservationModel \+ RewardModel**  
**\- \[ \] state() returns a dict**  
**\- \[ \] grader.final\_score() returns float in \[0.0, 1.0\]**  
**\- \[ \] Score is deterministic: run same episode twice, assert identical score**

**API checks (HTTP via httpx, assumes server running on localhost:7860):**  
**\- \[ \] GET / returns 200**  
**\- \[ \] POST /reset returns 200**  
**\- \[ \] POST /step returns 200**  
**\- \[ \] GET /grader returns score in \[0.0, 1.0\]**  
**\- \[ \] GET /tasks returns list with 4 tasks**  
**\- \[ \] POST /baseline completes and returns 4 scores**

**Spec checks:**  
**\- \[ \] openenv.yaml exists and has all required keys**  
**\- \[ \] ActionModel.model\_json\_schema() is non-empty**  
**\- \[ \] ObservationModel.model\_json\_schema() is non-empty**

**At the end print:**  
**\- Total: X/Y checks passed**  
**\- If any FAIL: print which ones and exit with code 1**  
**\- If all PASS: print "✅ Ready to submit"**

**Requirements:**  
**\- Catches all exceptions per check — one failure must not stop others**  
**\- Uses httpx for HTTP checks (sync client)**  
**\- Color output: green PASS, red FAIL (use ANSI codes, no external libs)**

**After the code, explain:**  
**1\. Why exit code 1 on failure matters for CI/CD pipelines**  
**2\. Why each check is isolated in try/except**  
**3\. How determinism check works — what "same episode" means exactly**

**Optimize for: fast execution, all checks independent, zero external dependencies beyond httpx.**

**Prompt 24 — Project Entry Point \+ Import Structure**

**Write the complete project entry point and import structure for CleanFlowEnv.**

**Create api/main.py that wires everything together cleanly.**

**Full import map (write this as a comment block at the top of main.py):**

**models/action.py        → ActionModel**  
**models/observation.py   → ObservationModel, TablePreviewRow**  
**models/reward.py        → RewardModel**  
**env/state.py            → EnvironmentState**  
**env/actions.py          → fill\_null, drop\_duplicates, convert\_type, normalize, remove\_outliers, InvalidActionError**  
**env/dispatcher.py       → apply\_action, get\_action\_cost, BUDGET\_COSTS**  
**env/rewards.py          → compute\_quality, compute\_reward**  
**env/grader.py           → final\_score**  
**env/observation.py      → build\_observation**  
**env/environment.py      → CleanFlowEnv**  
**tasks/task\_easy.py      → generate\_easy\_task**  
**tasks/task\_medium.py    → generate\_medium\_task**  
**tasks/task\_hard.py      → generate\_hard\_task**  
**tasks/task\_expert.py    → generate\_expert\_task**  
**baseline/run\_baseline.py → run\_baseline**

**Requirements for main.py:**  
**\- Register all 4 tasks in TASK\_REGISTRY dict at startup**  
**\- Use FastAPI lifespan to initialize global env instance**  
**\- Every endpoint imports only from env/environment.py and models/ — no direct task imports in API layer**  
**\- Add a GET / health check endpoint returning name, version, status: "ok"**

**Also write \_\_init\_\_.py files for:**  
**\- models/**  
**\- env/**  
**\- tasks/**  
**\- baseline/**

**Each \_\_init\_\_.py should export its public symbols cleanly.**

**Also write a run.py at project root:**  
**\- Calls uvicorn.run("api.main:app", host="0.0.0.0", port=7860, reload=False)**  
**\- Reads port from PORT env var if set**

**After the code, explain:**  
**1\. Why TASK\_REGISTRY lives in main.py and not inside CleanFlowEnv**  
**2\. Why API layer imports only from environment.py (separation of concerns)**  
**3\. Why reload=False in production and when you would use reload=True**

**Optimize for: clean import graph with no circular dependencies, fast startup.**  
---

**Prompt 25 — CleanFlowEnv Full Wiring (Environment Class Final Version)**

**Write the final, complete version of CleanFlowEnv in env/environment.py.**

**This is the integration prompt — wire together ALL previously built components.**

**The class must:**

**\_\_init\_\_(self, task\_registry: dict):**  
**\- Store task\_registry**  
**\- Set self.\_state \= None**  
**\- Set self.\_initial\_budget \= None**

**reset(self, task\_id: str) \-\> ObservationModel:**  
**\- Look up task\_id in task\_registry, raise ValueError if not found**  
**\- Call task\_fn() to get raw\_table, ground\_truth, budget, column\_descriptions**  
**\- Initialize EnvironmentState(task\_id, raw\_table, ground\_truth, budget, column\_descriptions)**  
**\- Set self.\_initial\_budget \= budget**  
**\- Set prev\_table \= raw\_table.copy() on first reset (no lag on first observation)**  
**\- Return build\_observation(self.\_state)**

**step(self, action\_dict: dict) \-\> tuple\[ObservationModel, RewardModel\]:**  
**\- Guard: if self.\_state is None raise RuntimeError("Call reset() first")**  
**\- Try parse action\_dict → ActionModel (catch ValidationError → penalty reward, done=False)**  
**\- Get cost \= get\_action\_cost(action)**  
**\- If cost \> budget\_remaining:**  
    **return current obs, RewardModel with reward=-0.5, done=True, info={"reason": "budget\_exhausted"}**  
**\- Update prev\_table \= current\_table.copy() BEFORE applying action**  
**\- Try apply\_action → catch InvalidActionError → penalty reward, do NOT update table**  
**\- Update state: step\_count \+= 1, budget\_remaining \-= cost, append to operations\_history**  
**\- Compute reward via compute\_reward(...)**  
**\- Update best\_quality\_so\_far**  
**\- Check done: step\_count \>= 20 OR budget\_remaining \== 0 OR cumulative\_quality \>= 1.0**  
**\- Return (build\_observation(self.\_state), reward\_model)**

**state(self) \-\> dict:**  
**\- Guard: if self.\_state is None return {"status": "not\_initialized"}**  
**\- Return self.\_state.to\_dict()**

**Properties:**  
**\- @property current\_quality → calls compute\_quality, returns overall float**  
**\- @property is\_done → returns bool based on termination conditions**

**Requirements:**  
**\- Every method must have a docstring**  
**\- step() must NEVER raise an unhandled exception — all errors become penalty rewards**  
**\- Log every action to operations\_history even if invalid**

**After the code, explain:**  
**1\. Why invalid actions are still logged to operations\_history**  
**2\. The exact order of operations in step() and why it cannot be reordered**  
**3\. Why current\_quality is a property and not stored in state**

**Optimize for: correctness of step order, zero crashes, minimal redundant computation.**  
---

**Prompt 26 — Grader Edge Cases**

**Extend the grader module (env/grader.py) to handle all edge cases correctly.**

**Start with the base final\_score(state: EnvironmentState) \-\> dict from earlier.**

**Now add robust handling for:**

**Correctness edge cases:**  
**\- current\_table has more columns than ground\_truth → only score on common columns**  
**\- current\_table has more rows than ground\_truth → align on index, missing rows score 0**  
**\- current\_table has fewer rows than ground\_truth → missing rows score 0 (not ignored)**  
**\- dtype mismatch during cell comparison (e.g. float vs int for same value) →**   
  **attempt numeric coercion before comparing, only fail if coercion fails**

**Completeness edge cases:**  
**\- ground\_truth itself has nulls in some columns → those columns excluded from null scoring**  
**\- All rows removed (empty dataframe) → completeness \= 0.0, do not divide by zero**

**Efficiency edge cases:**  
**\- initial\_budget not stored → fall back to max\_steps \= 20 as proxy**  
**\- zero steps taken → efficiency \= 1.0**

**Action quality edge cases:**  
**\- operations\_history is empty → action\_quality \= 1.0**  
**\- all actions were invalid → action\_quality \= 0.0**

**Also add a function score\_breakdown\_report(state) \-\> str that returns a human-readable**  
**multi-line string summary:**

**"""**  
**\=== CleanFlowEnv Grader Report \===**  
**Task: task\_easy**  
**Steps used: 8 / 20**  
**Budget used: 12 / 20**

**Correctness:    0.94  (weight: 40%)**  
**Completeness:   1.00  (weight: 30%)**  
**Efficiency:     0.60  (weight: 20%)**  
**Action Quality: 0.85  (weight: 10%)**

**FINAL SCORE: 0.893**  
**\==================================**  
**"""**

**After the code, explain:**  
**1\. Why row count mismatches penalize rather than ignore missing rows**  
**2\. How numeric coercion works in cell comparison and when it fails**  
**3\. Why action\_quality \= 1.0 for empty history (not 0.0)**

**Optimize for: no division by zero anywhere, all paths return float in \[0.0, 1.0\].**  
---

**Prompt 27 — /baseline Endpoint Full Implementation**

**Implement the POST /baseline endpoint in full detail.**

**This endpoint must:**  
**1\. Accept optional body: { "tasks": \["task\_easy", "task\_medium", "task\_hard", "task\_expert"\] }**  
   **\- Default: run all 4 tasks**  
**2\. For each task:**  
   **a. Call env.reset(task\_id)**  
   **b. Run RuleBasedAgent (not OpenAI — baseline must work without API key)**  
   **c. Loop until done or max 20 steps**  
   **d. Call final\_score(env.\_state)**  
**3\. Return:**  
**{**  
  **"results": {**  
    **"task\_easy":   { "score": 0.91, "steps": 7,  "budget\_used": 9  },**  
    **"task\_medium": { "score": 0.72, "steps": 12, "budget\_used": 16 },**  
    **"task\_hard":   { "score": 0.44, "steps": 18, "budget\_used": 19 },**  
    **"task\_expert": { "score": 0.31, "steps": 15, "budget\_used": 15 }**  
  **},**  
  **"average\_score": 0.595,**  
  **"timestamp": "2025-01-01T00:00:00Z"**  
**}**

**Requirements:**  
**\- Must complete synchronously (no background tasks) — judges will wait for it**  
**\- Each task resets env cleanly — no state leakage between tasks**  
**\- Catch per-task errors and return { "score": null, "error": "..." } for that task**  
**\- timestamp uses UTC ISO format**  
**\- average\_score excludes null scores from average**

**Also write a helper run\_episode(env, agent) \-\> dict that:**  
**\- Runs one full episode**  
**\- Returns steps, budget\_used, final\_score**  
**\- Can be reused by both /baseline and simulate.py**

**After the code, explain:**  
**1\. Why /baseline uses RuleBasedAgent not OpenAI (reproducibility without API key)**  
**2\. How state leakage between tasks is prevented**  
**3\. Why null scores are excluded from average (not zeroed)**

**Optimize for: clean episode isolation, fast execution, no shared mutable state between tasks.**  
---

**Prompt 28 — Error Handling \+ Logging Layer**

**Add a consistent error handling and logging layer across CleanFlowEnv.**

**Part 1 — Custom Exceptions (env/exceptions.py):**

**Define these exception classes with clear docstrings:**

**class CleanFlowError(Exception): pass           \# base**  
**class InvalidActionError(CleanFlowError): pass  \# bad action structure**  
**class BudgetExhaustedError(CleanFlowError): pass**  
**class EpisodeNotInitializedError(CleanFlowError): pass**  
**class TaskNotFoundError(CleanFlowError): pass**  
**class GraderError(CleanFlowError): pass**

**Each exception should accept message \+ optional context dict:**  
    **raise InvalidActionError("Unknown action type", context={"action\_type": "foo"})**

**Part 2 — Logging setup (env/logger.py):**

**\- Use Python stdlib logging (no external libs)**  
**\- Logger name: "cleanflow"**  
**\- Format: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"**  
**\- Log level: INFO by default, DEBUG if ENV=development**  
**\- Add a get\_logger(module\_name) helper**

**Part 3 — Apply logging throughout:**

**In CleanFlowEnv:**  
**\- INFO: "Episode reset for task {task\_id}"**  
**\- INFO: "Step {n}: action={action\_type} col={column} reward={reward:.3f}"**  
**\- WARNING: "Invalid action attempted: {reason}"**  
**\- WARNING: "Budget exhausted at step {n}"**  
**\- ERROR: "Unhandled exception in step(): {e}" (before returning penalty reward)**

**In grader:**  
**\- INFO: "Grading episode: task={task\_id} steps={n} final\_score={score:.3f}"**

**Requirements:**  
**\- Never log DataFrames — log shape only**  
**\- Never log raw table data (privacy)**  
**\- All loggers use get\_logger(\_\_name\_\_)**

**After the code, explain:**  
**1\. Why a CleanFlowError base class is better than using built-in exceptions directly**  
**2\. Why DataFrames must never be logged**  
**3\. How ENV=development enables debug logging without code changes**

**Optimize for: zero performance impact on hot path (step loop), structured log messages.**  
---

**Prompt 29 — Performance \+ Memory Audit**

**Review and optimize the following components of CleanFlowEnv for performance and memory efficiency.**

**Write an optimized version of each, then benchmark it.**

**Component 1 — build\_observation():**  
**Current issue: calls df.isnull().sum(), df.duplicated().sum(), and df.describe() separately.**  
**Optimized: compute all stats in a single pass using df.agg() where possible.**  
**Write both versions and add a benchmark using timeit (1000 runs, 200-row DataFrame).**

**Component 2 — compute\_quality():**  
**Current issue: cell-by-cell comparison may use applymap (slow).**  
**Optimized: use vectorized (df1 \== df2) with .sum().sum() for correctness.**  
**Handle NaN \== NaN as True (use df.equals() logic selectively).**  
**Write both versions, benchmark on 400-row DataFrame.**

**Component 3 — EnvironmentState.reset():**  
**Current issue: may re-copy DataFrames unnecessarily.**  
**Optimized: only copy current\_table and prev\_table on reset, keep raw\_table and ground\_truth as references.**  
**Show memory usage before/after using sys.getsizeof() on each DataFrame.**

**Component 4 — operations\_history:**  
**Current issue: storing full action dicts may bloat memory over 20 steps.**  
**Optimized: store only (action\_type, column, step\_count) tuples instead of full dicts.**  
**Show memory comparison for 20 steps.**

**Requirements:**  
**\- Use timeit for CPU benchmarks**  
**\- Use sys.getsizeof() for memory estimates**  
**\- Write clean before/after for each component**  
**\- All optimizations must preserve correctness**

**After the code, explain:**  
**1\. Why single-pass aggregation is faster than separate pandas calls**  
**2\. How vectorized \== handles mixed dtypes and NaN**  
**3\. The trade-off between storing full action dicts vs tuples**

**Optimize for: measurable improvement on 200–500 row DataFrames (the realistic episode size).**  
---

**Prompt 30 — Final Smoke Test \+ Submission Checklist**

**Write a final smoke test script called smoke\_test.py that verifies the entire**   
**CleanFlowEnv stack is working correctly before submission.**

**It must run 5 checks in sequence, printing PASS/FAIL with timing for each:**

**Check 1 — Direct Python stack (no HTTP):**  
**\- Import all modules cleanly (catch ImportError)**  
**\- Instantiate CleanFlowEnv**  
**\- Run one full easy episode with RuleBasedAgent**  
**\- Assert final score \>= 0.75**  
**\- Time: must complete in \< 5 seconds**

**Check 2 — All 4 tasks generate without error:**  
**\- Call each generate\_\*\_task()**  
**\- Assert raw\_table.shape\[0\] matches spec (200, 300, 400, 500\)**  
**\- Assert ground\_truth has no nulls and no duplicates**  
**\- Assert column\_descriptions has entry for every column**

**Check 3 — Reward system integrity:**  
**\- Run easy task, apply same action twice**  
**\- Assert second reward.quality\_delta \== 0.0**  
**\- Apply a harmful action (normalize already-normalized column)**  
**\- Assert reward.penalty \< 0**

**Check 4 — API stack (requires server on localhost:7860):**  
**\- POST /reset → assert 200 \+ valid ObservationModel shape**  
**\- POST /step → assert 200 \+ reward in \[-1.0, 1.0\]**  
**\- GET /grader → assert score in \[0.0, 1.0\]**  
**\- POST /baseline → assert all 4 task scores present**  
**\- Time: all API checks must complete in \< 10 seconds**

**Check 5 — Determinism:**  
**\- Run task\_easy episode twice with same RuleBasedAgent**  
**\- Assert both final scores are identical (== not just close)**  
**\- Assert operations\_history is identical in both runs**

**Final output:**  
**"""**  
**\============================**  
**CleanFlowEnv Smoke Test**  
**\============================**  
**Check 1 — Python stack:     PASS (1.2s)**  
**Check 2 — Task generation:  PASS (0.4s)**  
**Check 3 — Reward integrity: PASS (0.8s)**  
**Check 4 — API stack:        PASS (2.1s)**  
**Check 5 — Determinism:      PASS (1.6s)**  
**\============================**  
**ALL CHECKS PASSED — Ready to submit ✅**  
**\============================**  
**"""**

**Requirements:**  
**\- Check 4 skips gracefully if server not running (prints SKIP not FAIL)**  
**\- Each check is fully isolated — failure in one does not affect others**  
**\- Exit code 0 if all pass (or skip), exit code 1 if any fail**

**After the code, explain:**  
**1\. Why determinism check uses \== not math.isclose()**  
**2\. Why Check 4 skips instead of failing when server is down**  
**3\. What "harmful action" means in Check 3 and how you trigger it reliably**

**Optimize for: fast execution, clear output, zero false positives.**  
