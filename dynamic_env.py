from typing import Tuple, Dict, Any, List, Optional
import uuid
import os
import json
import re

from models import Observation, Action, Reward, clamp_openenv_score


_NORMS_CACHE: Dict[str, Dict] = {
    "Retail FMCG": {
        "revenue_tolerance_pct": (-8, 15),
        "cogs_tolerance_pct": (-5, 7),
        "gross_margin_floor_pct": 28.0,
        "red_flag_threshold_pct": 10.0,
        "common_drivers": ["seasonal_volume", "promo_effectiveness", "mix_shift", "price_realization"],
        "red_flags": ["shrinkage > 2%", "COGS variance > 10%", "promo spend > 8% with no volume lift"],
        "context": "Retail FMCG revenue swings are driven by promotions and seasonality.",
        "source": "Deloitte Retail Benchmark 2024"
    },
    "SaaS": {
        "revenue_tolerance_pct": (-5, 20),
        "cogs_tolerance_pct": (-10, 10),
        "gross_margin_floor_pct": 65.0,
        "red_flag_threshold_pct": 20.0,
        "common_drivers": ["new_logo_attainment", "expansion_arr", "churn_rate", "deal_slip"],
        "red_flags": ["Gross margin below 65%", "ARR churn > 5%", "OpEx growing faster than revenue"],
        "context": "SaaS revenue is lumpy due to deal timing and renewal cycles.",
        "source": "OpenView SaaS Benchmarks 2024"
    },
    "Pharma Manufacturing": {
        "revenue_tolerance_pct": (-5, 5),
        "cogs_tolerance_pct": (-3, 3),
        "gross_margin_floor_pct": 55.0,
        "red_flag_threshold_pct": 5.0,
        "common_drivers": ["api_price_spike", "yield_loss", "batch_rejection", "regulatory_hold"],
        "red_flags": ["COGS variance > 5% — immediate escalation", "Revenue miss > 5%"],
        "context": "Pharma manufacturing operates under strict cost controls.",
        "source": "KPMG Pharma Operations 2024"
    },
    "EdTech SaaS": {
        "revenue_tolerance_pct": (-10, 25),
        "cogs_tolerance_pct": (-8, 12),
        "gross_margin_floor_pct": 55.0,
        "red_flag_threshold_pct": 15.0,
        "common_drivers": ["enrollment_volume", "b2b_contract_timing", "content_cost", "refund_rate"],
        "red_flags": ["Refund rate > 8%", "Content cost > 20% of revenue"],
        "context": "EdTech revenue spikes around academic calendars and exam seasons.",
        "source": "Internal benchmark 2024"
    }
}

FORMAT_TEMPLATES: Dict[str, Dict] = {
    "one-pager": {
        "max_words": 300,
        "audience": "CFO / Finance Manager",
        "tone": "concise, factual",
        "required_sections": ["executive_summary", "top_3_variances", "root_cause", "recommended_action"],
        "style_guide": "No jargon. Numbers first, then explanation."
    },
    "board-slide": {
        "max_words": 200,
        "audience": "Board of Directors",
        "tone": "strategic, headline-driven",
        "required_sections": ["headline_number", "bridge_summary", "strategic_implication", "ask_or_decision"],
        "style_guide": "Lead with the so-what, not the numbers."
    },
    "exception-report": {
        "max_words": 500,
        "audience": "Operations + Finance + Legal",
        "tone": "urgent, evidence-based",
        "required_sections": ["alert_summary", "impact_quantification", "root_cause_chain", "corrective_action", "owner_and_timeline"],
        "style_guide": "Root cause chain must show A → B → C logic."
    },
    "memo": {
        "max_words": 400,
        "audience": "Internal Finance Team",
        "tone": "detailed, analytical",
        "required_sections": ["summary", "detailed_variance_table", "driver_analysis", "assumptions_and_risks"],
        "style_guide": "Include full variance table."
    }
}


def _clamp_reward(value: float) -> float:
    return clamp_openenv_score(value)


def _generate_norms_via_llm(sector: str, metrics: List[str]) -> Dict:
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "no-key")
        )
        model = os.getenv("MODEL_NAME", "gpt-4o-mini")

        prompt = f"""You are a senior FP&A expert. Generate industry benchmark norms for:

Sector: {sector}
Financial Metrics: {metrics}

Return ONLY valid JSON (no markdown) with this exact structure:
{{
  "revenue_tolerance_pct": [-X, +Y],
  "cogs_tolerance_pct": [-X, +Y],
  "gross_margin_floor_pct": 0.0,
  "red_flag_threshold_pct": 10.0,
  "common_drivers": ["driver1", "driver2", "driver3"],
  "red_flags": ["red flag 1", "red flag 2"],
  "context": "2-3 sentence industry context explaining variance patterns",
  "source": "Industry benchmark or your knowledge base"
}}

Use realistic industry benchmarks. Be specific to {sector}."""

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        norms = json.loads(raw)
        norms["source"] = f"LLM-generated for {sector}"
        norms["auto_generated"] = True
        return norms

    except Exception as e:
        return {
            "revenue_tolerance_pct": (-10, 10),
            "cogs_tolerance_pct": (-8, 8),
            "gross_margin_floor_pct": 30.0,
            "red_flag_threshold_pct": 15.0,
            "common_drivers": ["volume_change", "price_change", "cost_inflation", "mix_shift"],
            "red_flags": [f"Any metric variance > 15% needs investigation in {sector}"],
            "context": f"Generic norms applied for {sector}.",
            "source": "Generic fallback",
            "auto_generated": True,
            "error": str(e)
        }


def get_or_generate_norms(sector: str, metrics: List[str] = None) -> Dict:
    if sector in _NORMS_CACHE:
        return _NORMS_CACHE[sector]
    print(f"  Unknown sector '{sector}' — generating norms via LLM...")
    norms = _generate_norms_via_llm(sector, metrics or [])
    _NORMS_CACHE[sector] = norms
    return norms


class DynamicBudgetVarianceEnv:
    def __init__(self):
        self.episode_id: str = ""
        self.current_task: str = ""
        self.current_task_config: Dict = {}
        self.sector: str = ""
        self.budget: Dict[str, float] = {}
        self.actual: Dict[str, float] = {}
        self.previous_drafts: list = []
        self.step_count: int = 0
        self.done: bool = False
        self.requested_format: str = "one-pager"
        self.sector_norms: Dict = {}
        self.format_template: Dict = {}
        self._hallucination_count: int = 0
        self._is_custom: bool = False

    def reset(self, task_id: str = "easy") -> Observation:
        from env import TASK_LIBRARY
        if task_id not in TASK_LIBRARY:
            raise ValueError(f"Unknown task '{task_id}'. Use reset_custom() for custom data.")
        cfg = TASK_LIBRARY[task_id]
        return self._setup_episode(
            task_id=task_id,
            sector=cfg["sector"],
            budget=cfg["budget"],
            actual=cfg["actual"],
            requested_format=cfg["requested_format"],
            hint=cfg["hint"],
            task_config=cfg
        )

    def reset_custom(
        self,
        sector: str,
        budget: Dict[str, float],
        actual: Dict[str, float],
        requested_format: str = "one-pager",
        company_name: str = "",
        period: str = "",
        additional_context: str = ""
    ) -> Observation:
        self._is_custom = True
        metrics = list(budget.keys())
        norms = get_or_generate_norms(sector, metrics)

        hint_parts = [f"Custom task: {sector} analysis."]
        if company_name:
            hint_parts.append(f"Company: {company_name}.")
        if period:
            hint_parts.append(f"Period: {period}.")
        if additional_context:
            hint_parts.append(additional_context)
        hint_parts.append(f"Red-flag threshold: {norms.get('red_flag_threshold_pct', 10)}%.")
        if norms.get("auto_generated"):
            hint_parts.append("Note: Sector norms were auto-generated — verify with domain expert.")

        return self._setup_episode(
            task_id=f"custom_{sector.lower().replace(' ', '_')}",
            sector=sector,
            budget=budget,
            actual=actual,
            requested_format=requested_format,
            hint=" ".join(hint_parts),
            task_config={
                "name": f"{sector} — Custom Analysis",
                "difficulty": "custom",
                "sector": sector,
                "budget": budget,
                "actual": actual,
                "requested_format": requested_format,
                "hint": " ".join(hint_parts),
                "expected_drivers": norms.get("common_drivers", []),
                "trap": None,
                "company_name": company_name,
                "period": period
            }
        )

    def _setup_episode(self, task_id, sector, budget, actual,
                       requested_format, hint, task_config) -> Observation:
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.previous_drafts = []
        self.done = False
        self._hallucination_count = 0
        self.current_task = task_id
        self.current_task_config = task_config
        self.sector = sector
        self.budget = dict(budget)
        self.actual = dict(actual)
        self.requested_format = requested_format
        self.sector_norms = get_or_generate_norms(sector, list(budget.keys()))
        self.format_template = FORMAT_TEMPLATES.get(requested_format, FORMAT_TEMPLATES["one-pager"])

        variances, variance_pct = self._calc_variances()

        return Observation(
            task_id=task_id,
            sector=sector,
            budget=self.budget,
            actual=self.actual,
            variances=variances,
            variance_pct=variance_pct,
            requested_format=requested_format,
            format_requirements=self.format_template,
            previous_drafts=[],
            step_count=0,
            hint=hint
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        reward_value = 0.0
        breakdown: Dict[str, float] = {}

        if action.action_type == "analyze":
            reward_value = 0.10
            breakdown["analysis_base"] = 0.10

        elif action.action_type == "calculate":
            if action.calculations:
                correct = sum(
                    1 for c in action.calculations
                    if abs(c.absolute_variance - round(
                        self.actual.get(c.metric, 0) - self.budget.get(c.metric, 0), 2
                    )) < 0.05
                )
                reward_value = max(0.05, round(0.20 * correct / max(len(action.calculations), 1), 3))
                breakdown["calc_score"] = reward_value
            else:
                reward_value = 0.05

        elif action.action_type == "query_norms":
            if action.norm_query:
                reward_value = 0.20
                breakdown["norm_query"] = 0.20
            else:
                reward_value = 0.10

        elif action.action_type == "draft":
            reward_value = 0.15
            breakdown["draft_base"] = 0.15
            if action.structured_output:
                reward_value += 0.10
                breakdown["structured_bonus"] = 0.10
            if action.explanation_text:
                self.previous_drafts.append(action.explanation_text)

        elif action.action_type == "revise":
            reward_value = 0.10 if self.previous_drafts else 0.05
            if action.explanation_text:
                self.previous_drafts.append(action.explanation_text)

        elif action.action_type == "submit":
            self.done = True
            reward_value = 0.20
            breakdown["submit_base"] = 0.20
            if self.step_count <= 4:
                reward_value += 0.15
                breakdown["efficiency_bonus"] = 0.15
            elif self.step_count <= 7:
                reward_value += 0.05
                breakdown["efficiency_bonus"] = 0.05
            if action.structured_output:
                reward_value += 0.15
                breakdown["format_compliance"] = 0.15

            _, variance_pct = self._calc_variances()
            threshold = self.sector_norms.get("red_flag_threshold_pct", 10.0)
            any_breach = any(abs(v) > threshold for v in variance_pct.values())
            if action.structured_output:
                if any_breach and action.structured_output.risk_flag:
                    reward_value += 0.10
                    breakdown["risk_flag_correct"] = 0.10
                elif any_breach and not action.structured_output.risk_flag:
                    breakdown["missed_risk_flag"] = 0.05

        # CLAMP reward strictly between 0.05 and 0.95
        reward_value = _clamp_reward(reward_value)

        # Clamp breakdown values too
        safe_breakdown = {k: _clamp_reward(v) for k, v in breakdown.items()}

        variances, variance_pct = self._calc_variances()

        obs = Observation(
            task_id=self.current_task,
            sector=self.sector,
            budget=self.budget,
            actual=self.actual,
            variances=variances,
            variance_pct=variance_pct,
            requested_format=self.requested_format,
            format_requirements=self.format_template,
            previous_drafts=self.previous_drafts.copy(),
            step_count=self.step_count,
            hint=(
                f"Sector: {self.sector}. "
                f"Red-flag threshold: {self.sector_norms.get('red_flag_threshold_pct', '?')}%. "
                f"Common drivers: {', '.join(self.sector_norms.get('common_drivers', [])[:3])}."
            )
        )

        reward = Reward(value=reward_value, reason=f"{action.action_type} executed", breakdown=safe_breakdown)
        return obs, reward, self.done, {
            "episode_id": self.episode_id,
            "step": self.step_count,
            "hallucinations_so_far": self._hallucination_count,
            "sector_norms_auto_generated": self.sector_norms.get("auto_generated", False)
        }

    def state(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "current_task": self.current_task,
            "sector": self.sector,
            "step_count": self.step_count,
            "done": self.done,
            "is_custom": self._is_custom,
            "norms_auto_generated": self.sector_norms.get("auto_generated", False),
            "hallucination_count": self._hallucination_count
        }

    def get_sector_norms(self, sector: str = None) -> Dict:
        return get_or_generate_norms(sector or self.sector)

    def _calc_variances(self):
        variances = {
            k: round(self.actual.get(k, 0) - self.budget.get(k, 0), 2)
            for k in self.budget
        }
        variance_pct = {
            k: round(
                ((self.actual.get(k, 0) - self.budget.get(k, 0)) / self.budget.get(k, 1)) * 100, 2
            )
            for k in self.budget
        }
        return variances, variance_pct
