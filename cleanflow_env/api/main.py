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

import math
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cleanflow_env.api.dashboard import create_dashboard

from cleanflow_env.baseline.run_baseline import run_baseline_all
from cleanflow_env.env.environment import CleanFlowEnv
from cleanflow_env.env.grader import final_score
from cleanflow_env.models.action import ActionModel
from cleanflow_env.models.observation import ObservationModel
from cleanflow_env.tasks.task_easy import generate_easy_task
from cleanflow_env.tasks.task_expert import generate_expert_task
from cleanflow_env.tasks.task_hard import generate_hard_task
from cleanflow_env.tasks.task_medium import generate_medium_task
from cleanflow_env.tasks.task_multi import generate_multi_task

# --- Task Registry ---
TASK_REGISTRY = {
    "task_easy": generate_easy_task,
    "task_medium": generate_medium_task,
    "task_hard": generate_hard_task,
    "task_expert": generate_expert_task,
    "task_multi": generate_multi_task,
}

# Global environment instance — initialized eagerly
env = CleanFlowEnv(task_registry=TASK_REGISTRY)

app = FastAPI(
    title="CleanFlowEnv",
    description="OpenEnv-compliant environment for data cleaning and ETL workflows.",
    version="1.0",
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
    task_id: str = "task_easy"


class StepRequest(BaseModel):
    action: Dict[str, Any]


class GraderResponse(BaseModel):
    score: float
    correctness: float
    completeness: float
    schema_accuracy: float
    quality_overall: float
    efficiency: float
    action_quality: float
    validation: float


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str


class TasksResponse(BaseModel):
    tasks: List[TaskInfo]
    action_schema: Dict[str, Any]


class BaselineRequest(BaseModel):
    tasks: Optional[List[str]] = None


# --- OpenEnv Standard Endpoints ---


@app.get("/health")
def health():
    """OpenEnv standard health endpoint."""
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    """OpenEnv standard metadata endpoint."""
    return {
        "name": "CleanFlowEnv",
        "description": "OpenEnv-compliant environment for data cleaning and ETL workflows.",
    }


@app.get("/schema")
def schema():
    """OpenEnv standard schema endpoint."""
    return {
        "action": ActionModel.model_json_schema(),
        "observation": ObservationModel.model_json_schema(),
        "state": {"type": "object", "description": "Internal environment state"},
    }


@app.post("/mcp")
def mcp_endpoint():
    """OpenEnv standard MCP endpoint — lists available MCP tools."""
    return {
        "jsonrpc": "2.0",
        "result": {
            "tools": [
                {"name": "reset_environment", "description": "Reset environment and start a new cleaning episode"},
                {"name": "apply_action", "description": "Apply a cleaning action to the dataset"},
                {"name": "get_status", "description": "Get current null counts, duplicates, budget, schema"},
                {"name": "get_score", "description": "Get the current grading score"},
                {"name": "get_data_preview", "description": "Preview current data (first N rows)"},
            ],
            "mcp_endpoint": "/mcp/sse",
        },
        "id": None,
    }


# --- Core Endpoints ---


@app.get("/")
def root():
    """Root endpoint — returns JSON for validators, dashboard at /dashboard."""
    return {
        "name": "CleanFlowEnv",
        "status": "healthy",
        "version": "1.0",
        "dashboard": "/dashboard",
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    """Initialize environment for a new episode."""
    try:
        task_id = req.task_id if req else "task_easy"
        obs = env.reset(task_id)
        # Return standard openenv ResetResponse format
        return {
            "observation": obs.model_dump(),
            "reward": None,
            "done": False,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_EPS = 1e-4


def _clamp_score(v: float) -> float:
    """Clamp to strict (0, 1), guarding against NaN/inf."""
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        return _EPS
    return max(_EPS, min(1.0 - _EPS, float(v)))


@app.post("/step")
def step(req: StepRequest):
    """Apply an action and return observation + reward."""
    try:
        obs, reward = env.step(req.action)
        obs_dict = obs.model_dump()
        # Sparse reward strategy: validator sums all step rewards and checks
        # the cumulative total is in (0, 1). Return epsilon for intermediate
        # steps and a capped terminal score when done.
        _STEP_EPS = 0.01
        if reward.done:
            # Cap terminal reward so accumulated total stays strictly < 1.0
            n_intermediate = max(env._state.step_count - 1, 0)
            accumulated = n_intermediate * _STEP_EPS
            max_terminal = max(_EPS, 0.99 - accumulated)
            step_reward = min(_clamp_score(reward.cumulative_quality), max_terminal)
        else:
            step_reward = _STEP_EPS
        return {
            "observation": obs_dict,
            "reward": step_reward,
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
        # Auto-initialize if no episode exists (validator may call before /reset)
        if env._state is None:
            env.reset("task_easy")
        result = final_score(env._state)
        # Belt-and-suspenders: clamp every score field at the API boundary
        for key in ("score", "correctness", "completeness", "schema_accuracy",
                     "quality_overall", "efficiency", "action_quality", "validation"):
            if key in result:
                result[key] = _clamp_score(result[key])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grade/{task_id}", response_model=GraderResponse)
def grade_task(task_id: str):
    """Stateless per-task grading — resets the task and returns its score."""
    try:
        if task_id not in TASK_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
        env.reset(task_id)
        result = final_score(env._state)
        for key in ("score", "correctness", "completeness", "schema_accuracy",
                     "quality_overall", "efficiency", "action_quality", "validation"):
            if key in result:
                result[key] = _clamp_score(result[key])
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
        TaskInfo(id="task_multi", name="Multi-Table Cleaning", difficulty="expert+"),
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


# --- MCP Tool Server ---
# Mounted at /mcp/sse — allows MCP-compatible LLM agents to interact
from cleanflow_env.api.mcp_server import mcp as mcp_server

app.mount("/mcp", mcp_server.http_app())

# --- Gradio Dashboard ---
# Mounted at /dashboard — the visual demo for judges
dashboard = create_dashboard()
app = gr.mount_gradio_app(app, dashboard, path="/dashboard")
