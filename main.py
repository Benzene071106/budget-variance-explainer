from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Optional
from fastapi.encoders import jsonable_encoder
import uvicorn
import os
import json

from models import Observation, Action, Reward
from env import BudgetVarianceEnv, SECTOR_NORMS, FORMAT_TEMPLATES, TASK_LIBRARY
from dynamic_env import DynamicBudgetVarianceEnv, get_or_generate_norms
from grader import VarianceGrader

app = FastAPI(
    title="Budget Variance Explainer — OpenEnv v2",
    description="Industry-grade FP&A variance analysis — works for ANY sector",
    version="2.1.0"
)

_predefined_env = BudgetVarianceEnv()
_custom_env = DynamicBudgetVarianceEnv()
grader = VarianceGrader()
_active_env = _predefined_env
current_observation = None
current_final_draft = None


def _clamp_score(s):
    """Clamp any score strictly between 0 and 1"""
    try:
        s = float(s)
    except Exception:
        return 0.5
    if s != s:  # NaN
        return 0.5
    if s <= 0.0:
        return 0.05
    if s >= 1.0:
        return 0.95
    return s


class StepRequest(BaseModel):
    action: Action

class CustomResetRequest(BaseModel):
    sector: str
    budget: Dict[str, float]
    actual: Dict[str, float]
    requested_format: str = "one-pager"
    company_name: Optional[str] = ""
    period: Optional[str] = ""
    additional_context: Optional[str] = ""


@app.get("/")
def read_root():
    return {
        "message": "Budget Variance Analysis OpenEnv v2.1 — ANY sector supported!",
        "predefined_tasks": len(TASK_LIBRARY),
        "known_sectors": list(SECTOR_NORMS.keys()),
        "formats": list(FORMAT_TEMPLATES.keys()),
        "tip": "Use POST /reset/custom for your own sector and data"
    }

@app.get("/tasks")
def get_tasks():
    return {
        "tasks": _predefined_env.list_tasks(),
        "action_types": ["analyze", "calculate", "query_norms", "draft", "revise", "submit"],
        "note": "Tasks marked has_trap=true contain misleading signals",
        "custom": "POST /reset/custom to use your own sector and financial data"
    }

@app.post("/reset")
def reset(task_id: str = "easy"):
    global current_observation, current_final_draft, _active_env
    current_final_draft = None
    _active_env = _predefined_env
    try:
        obs = _predefined_env.reset(task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    current_observation = obs
    obs_dict = obs.model_dump()
    # Ensure step_count is never 0 in response
    obs_dict["step_count"] = max(1, obs_dict.get("step_count", 1))
    return obs_dict

@app.post("/reset/custom")
def reset_custom(request: CustomResetRequest):
    global current_observation, current_final_draft, _active_env
    current_final_draft = None
    _active_env = _custom_env
    if request.requested_format not in FORMAT_TEMPLATES:
        raise HTTPException(status_code=400, detail=f"Invalid format. Choose from: {list(FORMAT_TEMPLATES.keys())}")
    obs = _custom_env.reset_custom(
        sector=request.sector, budget=request.budget, actual=request.actual,
        requested_format=request.requested_format,
        company_name=request.company_name or "", period=request.period or "",
        additional_context=request.additional_context or ""
    )
    current_observation = obs
    norms = _custom_env.get_sector_norms()
    return {
        "observation": obs.model_dump(),
        "sector_norms": norms,
        "norms_auto_generated": norms.get("auto_generated", False),
        "message": f"Custom episode started for: {request.sector}"
    }

@app.post("/step")
def step(request: StepRequest):
    global current_observation, current_final_draft, _active_env
    if _active_env.done:
        raise HTTPException(status_code=400, detail="Episode finished. Call /reset or /reset/custom first.")
    obs, reward, done, info = _active_env.step(request.action)
    current_observation = obs
    if done and request.action.action_type == "submit":
        if request.action.structured_output:
            try:
                current_final_draft = json.dumps(
                    request.action.structured_output.model_dump(mode="json"), indent=2)
            except Exception:
                current_final_draft = request.action.explanation_text or ""
        else:
            current_final_draft = request.action.explanation_text or ""

    # Clamp reward.value strictly between 0 and 1
    reward.value = _clamp_score(reward.value)

    # Clamp breakdown values
    if reward.breakdown:
        reward.breakdown = {k: _clamp_score(v) for k, v in reward.breakdown.items()}

    return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": done, "info": info}

@app.get("/state")
def get_state():
    return _active_env.state()

@app.get("/norms")
def get_norms(sector: str = None):
    if sector:
        return {sector: get_or_generate_norms(sector)}
    return SECTOR_NORMS

@app.get("/norms/generate")
def generate_norms(sector: str):
    norms = get_or_generate_norms(sector)
    return {"sector": sector, "norms": norms}

@app.get("/formats")
def get_formats(fmt: str = None):
    if fmt:
        if fmt not in FORMAT_TEMPLATES:
            raise HTTPException(status_code=404, detail=f"Format not found.")
        return {fmt: FORMAT_TEMPLATES[fmt]}
    return FORMAT_TEMPLATES

@app.get("/grader")
def get_grader_score(task_id: str = None, detail: bool = False):
    if not current_final_draft or not current_observation:
        return {"error": "No completed episode. Submit a final draft first."}
    tid = task_id or getattr(_active_env, "current_task", "easy")
    result = grader.grade(tid, current_final_draft, current_observation, return_detail=True)

    # Clamp final_score
    result["final_score"] = _clamp_score(result.get("final_score", 0.5))

    if detail:
        return result
    return {"task_id": tid, "score": result["final_score"], "feedback": result.get("llm_feedback")}

@app.get("/baseline")
def run_baseline(tasks: str = "easy,easy_saas,medium,medium_retail_margin,hard,hard_saas_churn,hard_edtech_seasonal,hard_conglomerate"):
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    invalid = [t for t in task_list if t not in TASK_LIBRARY]

    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown task IDs: {invalid}")

    try:
        from inference import run_inference
        scores = run_inference(task_ids=task_list)

        # Ensure all tasks have valid clamped scores
        safe_scores = {}
        for tid in TASK_LIBRARY.keys():
            safe_scores[tid] = _clamp_score(scores.get(tid, 0.5))

        safe_scores = jsonable_encoder(safe_scores)

        return {
            "status": "success",
            "baseline_scores": safe_scores,
            "average": round(sum(safe_scores.values()) / len(safe_scores), 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline error: {str(e)}")

@app.get("/report")
def report_viewer():
    viewer_path = os.path.join(os.path.dirname(__file__), "report_viewer.html")
    return FileResponse(viewer_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
