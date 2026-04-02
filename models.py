from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Literal


# ─────────────────────────────────────────────
# Sub-models for richer structured actions
# ─────────────────────────────────────────────

class Driver(BaseModel):
    """One identified root-cause driver with evidence"""
    name: str = Field(..., description="Short driver name e.g. 'holiday_volume_lift'")
    direction: Literal["favorable", "unfavorable"] = Field(...)
    estimated_impact: Optional[float] = Field(
        default=None, description="Estimated $ impact of this driver"
    )
    evidence: str = Field(..., description="One sentence citing data that proves this driver")


class VarianceCalculation(BaseModel):
    """Structured math verification — prevents hallucination"""
    metric: str
    budget_value: float
    actual_value: float
    absolute_variance: float
    pct_variance: float

    @field_validator("absolute_variance")
    @classmethod
    def check_absolute(cls, v, info):
        data = info.data
        if "budget_value" in data and "actual_value" in data:
            expected = round(data["actual_value"] - data["budget_value"], 2)
            if abs(v - expected) > 0.05:
                raise ValueError(
                    f"absolute_variance {v} doesn't match actual-budget={expected}"
                )
        return v

    @field_validator("pct_variance")
    @classmethod
    def check_pct(cls, v, info):
        data = info.data
        if "budget_value" in data and "actual_value" in data and data["budget_value"] != 0:
            expected = round(
                ((data["actual_value"] - data["budget_value"]) / data["budget_value"]) * 100, 2
            )
            if abs(v - expected) > 0.1:
                raise ValueError(
                    f"pct_variance {v} doesn't match expected={expected}"
                )
        return v


class NormQuery(BaseModel):
    """What the agent is querying from sector knowledge base"""
    sector: str
    metric: str
    question: str


class StructuredReport(BaseModel):
    """Enforced output format — one per format type"""
    executive_summary: str = Field(..., max_length=400)
    variance_table: Dict[str, float] = Field(
        ..., description="metric → variance% mapping, must match observation"
    )
    drivers: List[Driver] = Field(..., min_length=1)
    sector_norm_applied: str = Field(
        ..., description="Which norm was applied and why e.g. '±5-8% FMCG revenue = normal'"
    )
    recommendation: str = Field(..., max_length=300)
    risk_flag: bool = Field(..., description="True if any variance breaches sector red-flag threshold")
    risk_reason: Optional[str] = Field(
        default=None, description="Required when risk_flag=True"
    )


# ─────────────────────────────────────────────
# Core env types
# ─────────────────────────────────────────────

class Observation(BaseModel):
    """Observation returned after reset() or step()"""
    task_id: str
    sector: str
    budget: Dict[str, float]
    actual: Dict[str, float]
    variances: Dict[str, float]
    variance_pct: Dict[str, float]
    requested_format: str
    format_requirements: Optional[Dict] = None   # injected from FORMAT_TEMPLATES
    previous_drafts: List[str]
    step_count: int
    hint: Optional[str] = None

    class Config:
        json_encoders = {float: lambda v: round(v, 2)}


class Action(BaseModel):
    """Action the agent can take"""
    action_type: Literal["analyze", "calculate", "query_norms", "draft", "revise", "submit"]

    # analyze / calculate
    calculations: Optional[List[VarianceCalculation]] = Field(
        default=None, description="Math verification rows — one per metric"
    )

    # query_norms
    norm_query: Optional[NormQuery] = None

    # draft / revise / submit
    selected_drivers: Optional[List[str]] = None
    explanation_text: Optional[str] = None
    structured_output: Optional[StructuredReport] = None
    revision_instructions: Optional[str] = None

    class Config:
        extra = "forbid"


class Reward(BaseModel):
    """Reward given after each step"""
    value: float = Field(..., description="Reward value between -1.0 and 1.0")
    reason: str
    breakdown: Optional[Dict[str, float]] = None   # component-wise for debugging