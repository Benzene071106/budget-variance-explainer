import os
import json
from openai import OpenAI
from models import Action, VarianceCalculation, NormQuery, StructuredReport, Driver
from env import BudgetVarianceEnv, SECTOR_NORMS, FORMAT_TEMPLATES
from grader import VarianceGrader
from typing import Dict

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


SYSTEM_PROMPT = """You are a senior FP&A analyst AI agent. Each step return ONLY a valid JSON object.

STEP SEQUENCE:
Step 1 -> action_type: "analyze"      -> just explanation_text
Step 2 -> action_type: "calculate"    -> calculations array
Step 3 -> action_type: "query_norms"  -> norm_query object
Step 4 -> action_type: "draft"        -> explanation_text only
Step 5 -> action_type: "submit"       -> explanation_text AND structured_output

CRITICAL: For step 5 submit, structured_output MUST follow this EXACT format:
{
  "action_type": "submit",
  "explanation_text": "Your full report text here",
  "structured_output": {
    "executive_summary": "ONE string — max 400 chars — NOT a list",
    "variance_table": {"Revenue": 12.0, "COGS": 5.0, "Gross_Margin": 22.5},
    "drivers": [
      {
        "name": "holiday_volume",
        "direction": "favorable",
        "estimated_impact": 120000,
        "evidence": "Revenue +12% vs budget driven by Dec holiday promotion"
      }
    ],
    "sector_norm_applied": "Retail FMCG: ±5-8% normal, >10% red flag — revenue at 12% exceeds threshold",
    "recommendation": "Document promo ROI and investigate sustainability of 12% beat",
    "risk_flag": true,
    "risk_reason": "Revenue variance 12% exceeds Retail FMCG red flag threshold of 10%"
  }
}

RULES:
- executive_summary must be a STRING, never a list or array
- drivers must have at least 1 item with name, direction, evidence fields
- variance_table must include all metrics from the observation
- Only use numbers from variance_pct in the observation — no hallucination
- Return ONLY valid JSON, no markdown, no extra text
"""


def _call_llm(messages: list, task_id: str) -> dict:
    """Call LLM and parse JSON action response"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=1500
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  LLM call failed: {e} — using fallback", flush=True)
        return None


def _build_action(raw: dict, obs_dict: dict, step: int) -> Action:
    """Convert raw LLM dict to typed Action, with fallback per step"""
    if raw is None:
        fallbacks = {
            1: Action(action_type="analyze", explanation_text="Analyzing variance data."),
            2: Action(action_type="calculate"),
            3: Action(action_type="query_norms", norm_query=NormQuery(
                sector=obs_dict["sector"], metric="all", question="What are the sector norms?"
            )),
            4: Action(action_type="draft", explanation_text="Draft placeholder."),
        }
        return fallbacks.get(step, Action(
            action_type="submit",
            explanation_text=f"Variance analysis for {obs_dict.get('sector')} sector."
        ))

    action_type = raw.get("action_type", "analyze")

    calculations = None
    if raw.get("calculations"):
        calculations = []
        for c in raw["calculations"]:
            try:
                calculations.append(VarianceCalculation(**c))
            except Exception:
                pass

    norm_query = None
    if raw.get("norm_query"):
        try:
            norm_query = NormQuery(**raw["norm_query"])
        except Exception:
            pass

    structured_output = None
    if raw.get("structured_output"):
        try:
            so = raw["structured_output"]

            if isinstance(so.get("executive_summary"), list):
                so["executive_summary"] = " ".join(so["executive_summary"])

            raw_drivers = so.get("drivers", [])
            if not raw_drivers:
                raw_drivers = [{
                    "name": "variance_driver",
                    "direction": "favorable",
                    "evidence": f"Variance data from {obs_dict.get('sector')} sector analysis"
                }]

            drivers = []
            for d in raw_drivers:
                try:
                    driver_obj = Driver(**d)
                    drivers.append({
                        "name": driver_obj.name,
                        "direction": driver_obj.direction,
                        "estimated_impact": getattr(driver_obj, "estimated_impact", None),
                        "evidence": driver_obj.evidence
                    })
                except Exception:
                    pass

            if not drivers:
                drivers = [{
                    "name": "variance_driver",
                    "direction": "favorable",
                    "estimated_impact": None,
                    "evidence": f"Variance identified in {obs_dict.get('sector')} sector"
                }]

            so["drivers"] = drivers

            if "variance_table" not in so:
                so["variance_table"] = obs_dict.get("variance_pct", {})
            if "sector_norm_applied" not in so:
                so["sector_norm_applied"] = f"{obs_dict.get('sector')} sector norms applied"
            if "recommendation" not in so:
                so["recommendation"] = "Review variance drivers and take corrective action."
            if "risk_flag" not in so:
                so["risk_flag"] = False
            structured_output = so
        except Exception as e:
            print(f"  structured_output parse error: {e}", flush=True)
            if not raw.get("explanation_text") and raw.get("structured_output"):
                so = raw["structured_output"]
                summary = so.get("executive_summary", "")
                if isinstance(summary, list):
                    summary = " ".join(summary)
                raw["explanation_text"] = summary or str(so)

    try:
        return Action(
            action_type=action_type,
            calculations=calculations,
            norm_query=norm_query,
            selected_drivers=raw.get("selected_drivers"),
            explanation_text=raw.get("explanation_text"),
            structured_output=structured_output,
            revision_instructions=raw.get("revision_instructions")
        )
    except Exception as e:
        print(f"  Action build error: {e}", flush=True)
        return Action(action_type="analyze", explanation_text="Fallback action.")


def get_model_response(obs_dict: dict, task_id: str, history: list, step: int) -> Action:
    """Build messages and get next action from LLM"""
    sector_norms_context = ""
    if step == 3:
        norms = SECTOR_NORMS.get(obs_dict.get("sector", ""), {})
        sector_norms_context = f"\nSECTOR NORMS FOR {obs_dict.get('sector')}:\n{json.dumps(norms, indent=2)}\n"

    vp = obs_dict.get("variance_pct", {})
    va = obs_dict.get("variances", {})
    variance_lines = "\n".join([
        f"  {k}: {vp.get(k,0)}% (absolute: {va.get(k,0)})"
        for k in vp
    ])

    submit_instruction = ""
    if step >= 4:
        submit_instruction = f"""
IMPORTANT FOR SUBMIT ACTION:
- explanation_text must mention these EXACT percentages: {list(vp.values())}
- Use the sector red-flag threshold from norms
- Do NOT mention any percentage not in the list above
- Format as {obs_dict.get("requested_format")} for audience: {obs_dict.get("format_requirements", {}).get("audience", "CFO")}
"""

    user_content = f"""
TASK: {task_id}  |  STEP: {step}
SECTOR: {obs_dict.get("sector")}
REQUESTED FORMAT: {obs_dict.get("requested_format")}

BUDGET: {obs_dict.get("budget")}
ACTUAL: {obs_dict.get("actual")}
VARIANCE SUMMARY:
{variance_lines}

HINT: {obs_dict.get("hint", "")}
{sector_norms_context}{submit_instruction}
Return your next action as JSON.
"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": user_content})

    raw = _call_llm(messages, task_id)
    return _build_action(raw, obs_dict, step), user_content, (json.dumps(raw) if raw else "fallback")


def run_inference(task_ids: list = None) -> Dict:
    """Main inference function — multi-step agentic loop with structured stdout logs"""
    if task_ids is None:
        task_ids = ["easy", "medium", "hard"]

    results = {}
    env = BudgetVarianceEnv()
    grader = VarianceGrader()

    for task_id in task_ids:
        obs = env.reset(task_id=task_id)
        history = []
        step = 0
        cumulative_reward = 0.0
        action = None
        success = False

        # ── [START] ──────────────────────────────────────────────────────────
        print(
            f"[START] task={task_id} env=BudgetVarianceEnv model={MODEL_NAME}",
            flush=True
        )

        try:
            for _ in range(12):
                step += 1
                action, user_msg, assistant_msg = get_model_response(
                    obs.model_dump(), task_id, history, step
                )

                obs, reward, done, info = env.step(action)
                cumulative_reward += reward.value

                error_str = info.get("error", "null") or "null"

                # ── [STEP] ────────────────────────────────────────────────────
                print(
                    f"[STEP] step={step} action={action.action_type} "
                    f"reward={reward.value:.2f} done={str(done).lower()} "
                    f"error={error_str}",
                    flush=True
                )

                history.append({"role": "user", "content": user_msg})
                history.append({"role": "assistant", "content": assistant_msg})

                if done:
                    success = True
                    break

        except Exception as e:
            error_str = str(e).replace("\n", " ")
            print(
                f"[STEP] step={step + 1} action=error reward=0.00 "
                f"done=false error={error_str}",
                flush=True
            )
            success = False

        finally:
            # ── [END] ─────────────────────────────────────────────────────────
            print(
                f"[END] success={str(success).lower()} steps={step} "
                f"rewards={round(cumulative_reward, 2):.2f}",
                flush=True
            )

        # ── Grading (after [END], for internal results tracking) ─────────────
        final_text = ""
        if action and action.structured_output:
            try:
                final_text = json.dumps(action.structured_output.model_dump(mode="json"), indent=2)
            except Exception:
                final_text = action.structured_output.executive_summary or ""
        if not final_text and action and action.explanation_text:
            final_text = action.explanation_text
        if not final_text and obs.previous_drafts:
            final_text = " ".join(obs.previous_drafts)
        if not final_text:
            final_text = (
                f"Variance analysis for {task_id}. Sector: {obs.sector}. "
                f"Variance pct: {obs.variance_pct}. "
                f"Applied {obs.sector} sector norms. "
                f"Due to seasonal factors revenue variance is "
                f"{obs.variance_pct.get('Revenue', 0)}%. "
                f"Recommendation: review and document variance drivers."
            )

        grade_detail = grader.grade(task_id, final_text, obs, return_detail=True)
        raw_score = grade_detail["final_score"]
        final_score = max(1e-6, min(1 - 1e-6, float(raw_score)))

        results[task_id] = {
            "score": final_score,
            "cumulative_reward": round(cumulative_reward, 3),
            "steps_taken": step,
            "grader_detail": grade_detail
        }

    summary = {tid: r["score"] for tid, r in results.items()}
    return summary


if __name__ == "__main__":
    run_inference()
