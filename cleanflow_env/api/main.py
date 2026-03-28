"""
FastAPI application for CleanFlowEnv.

Import map:
  models/action.py        → ActionModel
  models/observation.py   → ObservationModel
  models/reward.py        → RewardModel
  env/environment.py      → CleanFlowEnv
  env/grader.py           → final_score
  tasks/                  → generate_*_task
  baseline/run_baseline.py → run_baseline_all, run_episode
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cleanflow_env.baseline.run_baseline import run_baseline_all
from cleanflow_env.env.environment import CleanFlowEnv
from cleanflow_env.env.grader import final_score
from cleanflow_env.models.action import ActionModel
from cleanflow_env.tasks.task_easy import generate_easy_task
from cleanflow_env.tasks.task_expert import generate_expert_task
from cleanflow_env.tasks.task_hard import generate_hard_task
from cleanflow_env.tasks.task_medium import generate_medium_task

# --- Task Registry ---
TASK_REGISTRY = {
    "task_easy": generate_easy_task,
    "task_medium": generate_medium_task,
    "task_hard": generate_hard_task,
    "task_expert": generate_expert_task,
}

# Global environment instance — initialized eagerly
env = CleanFlowEnv(task_registry=TASK_REGISTRY)

app = FastAPI(
    title="CleanFlowEnv",
    description="OpenEnv-compliant environment for data cleaning and ETL workflows.",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---


class ResetRequest(BaseModel):
    task_id: str


class StepRequest(BaseModel):
    action: Dict[str, Any]


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool


class GraderResponse(BaseModel):
    score: float
    correctness: float
    completeness: float
    efficiency: float
    action_quality: float


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str


class TasksResponse(BaseModel):
    tasks: List[TaskInfo]
    action_schema: Dict[str, Any]


class BaselineRequest(BaseModel):
    tasks: Optional[List[str]] = None


# --- Endpoints ---


@app.get("/")
def root():
    """Health check endpoint."""
    return {"name": "CleanFlowEnv", "version": "2.0", "status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest):
    """Initialize environment for a new episode."""
    try:
        obs = env.reset(req.task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """Apply an action and return observation + reward."""
    try:
        obs, reward = env.step(req.action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": reward.done,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    """Return current internal state."""
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/grader", response_model=GraderResponse)
def grader():
    """Return final score after episode ends."""
    try:
        if env._state is None:
            raise HTTPException(
                status_code=400, detail="No episode in progress. Call /reset first."
            )
        result = final_score(env._state)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks", response_model=TasksResponse)
def tasks():
    """Return list of available tasks and action schema."""
    task_info = [
        TaskInfo(id="task_easy", name="Basic Cleaning", difficulty="easy"),
        TaskInfo(id="task_medium", name="Schema Normalization", difficulty="medium"),
        TaskInfo(id="task_hard", name="Advanced Cleaning", difficulty="hard"),
        TaskInfo(id="task_expert", name="Budget-Constrained Cleaning", difficulty="expert"),
    ]
    return TasksResponse(
        tasks=task_info,
        action_schema=ActionModel.model_json_schema(),
    )


@app.post("/baseline")
def baseline(_req: BaselineRequest = BaselineRequest()):
    """Run the rule-based baseline agent on all tasks."""
    try:
        results = run_baseline_all(env)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


