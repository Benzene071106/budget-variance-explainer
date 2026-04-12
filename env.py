from typing import Tuple, Dict, Any, List
import uuid
from models import Observation, Action, Reward, clamp_openenv_score


SECTOR_NORMS: Dict[str, Dict] = {
    "Retail FMCG": {
        "revenue_tolerance_pct": (-8, +15),
        "cogs_tolerance_pct": (-5, +7),
        "gross_margin_floor_pct": 28.0,
        "red_flag_threshold_pct": 10.0,
        "common_drivers": [
            "seasonal_volume", "promo_effectiveness",
            "mix_shift", "price_realization", "shrinkage"
        ],
        "red_flags": [
            "shrinkage > 2% of revenue",
            "promo spend > 8% with no volume lift",
            "COGS variance > 10%"
        ],
        "context": (
            "Retail FMCG revenue swings are driven by promotions and seasonality. "
            "Holiday quarters routinely run 8-15% above budget. "
            "COGS should track volume — if volume is up but COGS% is also up, investigate mix."
        ),
        "source": "Deloitte Retail Benchmark 2024"
    },
    "SaaS": {
        "revenue_tolerance_pct": (-5, +20),
        "cogs_tolerance_pct": (-10, +10),
        "gross_margin_floor_pct": 65.0,
        "arr_churn_threshold_pct": 5.0,
        "rule_of_40": True,
        "red_flag_threshold_pct": 20.0,
        "common_drivers": [
            "new_logo_attainment", "expansion_arr",
            "churn_rate", "deal_slip", "pricing_change", "seat_expansion"
        ],
        "red_flags": [
            "Gross margin below 65%",
            "ARR churn > 5% in a quarter",
            "OpEx growing faster than revenue 2 quarters in a row"
        ],
        "context": (
            "SaaS revenue is lumpy due to deal timing and renewal cycles. "
            "Q4 over-attainment and Q1 miss are both common. "
            "Gross margin is the key health metric — should stay above 65%. "
            "OpEx variance in sales is acceptable if pipeline coverage is healthy."
        ),
        "source": "OpenView SaaS Benchmarks 2024"
    },
    "Pharma Manufacturing": {
        "revenue_tolerance_pct": (-5, +5),
        "cogs_tolerance_pct": (-3, +3),
        "gross_margin_floor_pct": 55.0,
        "red_flag_threshold_pct": 5.0,
        "common_drivers": [
            "api_price_spike", "yield_loss",
            "batch_rejection", "fx_impact",
            "regulatory_hold", "production_downtime"
        ],
        "red_flags": [
            "COGS variance > 5% — immediate escalation required",
            "Revenue miss > 5% — check regulatory or supply chain hold",
            "R&D overrun > 10% — milestone review needed"
        ],
        "context": (
            "Pharma manufacturing operates under strict cost controls due to regulatory requirements. "
            "Any COGS variance above 5% is treated as catastrophic — it signals batch failures, "
            "API price spikes, or yield losses that must be investigated immediately. "
            "Revenue misses usually indicate regulatory holds or supply disruptions, not demand issues."
        ),
        "source": "KPMG Pharma Operations Benchmark 2024"
    },
    "EdTech SaaS": {
        "revenue_tolerance_pct": (-10, +25),
        "cogs_tolerance_pct": (-8, +12),
        "gross_margin_floor_pct": 55.0,
        "red_flag_threshold_pct": 15.0,
        "common_drivers": [
            "enrollment_volume", "course_completion_rate",
            "b2b_contract_timing", "refund_rate", "content_cost"
        ],
        "red_flags": [
            "Refund rate > 8%",
            "Content cost > 20% of revenue",
            "B2B renewal rate below 70%"
        ],
        "context": (
            "EdTech revenue spikes around academic calendars and exam seasons. "
            "B2B (school/university) contracts are lumpy; B2C is seasonal. "
            "Content production cost is the biggest COGS driver."
        ),
        "source": "Internal benchmark — EdTech India 2024"
    }
}


FORMAT_TEMPLATES: Dict[str, Dict] = {
    "one-pager": {
        "max_words": 300,
        "audience": "CFO / Finance Manager",
        "tone": "concise, factual",
        "required_sections": [
            "executive_summary",
            "top_3_variances",
            "root_cause",
            "recommended_action"
        ],
        "style_guide": "No jargon. Numbers first, then explanation. Max 3 bullet points per section."
    },
    "board-slide": {
        "max_words": 200,
        "audience": "Board of Directors",
        "tone": "strategic, headline-driven",
        "required_sections": [
            "headline_number",
            "bridge_summary",
            "strategic_implication",
            "ask_or_decision"
        ],
        "style_guide": "Max 3 bullets per section. Lead with the 'so what', not the numbers."
    },
    "exception-report": {
        "max_words": 500,
        "audience": "Operations + Finance + Legal",
        "tone": "urgent, evidence-based",
        "required_sections": [
            "alert_summary",
            "impact_quantification",
            "root_cause_chain",
            "corrective_action",
            "owner_and_timeline"
        ],
        "style_guide": (
            "Root cause chain must show A → B → C logic. "
            "Every claim must cite a data point. "
            "Flag regulatory or compliance risk explicitly."
        )
    },
    "memo": {
        "max_words": 400,
        "audience": "Internal Finance Team",
        "tone": "detailed, analytical",
        "required_sections": [
            "summary",
            "detailed_variance_table",
            "driver_analysis",
            "assumptions_and_risks"
        ],
        "style_guide": "Include full variance table. Distinguish between controllable and non-controllable variances."
    }
}


TASK_LIBRARY: Dict[str, Dict] = {
    "easy": {
        "name": "Retail FMCG — Holiday Surge",
        "difficulty": "easy",
        "sector": "Retail FMCG",
        "budget": {"Revenue": 1_000_000, "COGS": 600_000, "Gross_Margin": 400_000},
        "actual": {"Revenue": 1_120_000, "COGS": 630_000, "Gross_Margin": 490_000},
        "requested_format": "one-pager",
        "hint": "Single driver — holiday promotion drove volume. Use Retail FMCG sector norms.",
        "expected_drivers": ["seasonal_volume", "promo_effectiveness"],
        "trap": None
    },
    "easy_saas": {
        "name": "SaaS — Clean Quarter",
        "difficulty": "easy",
        "sector": "SaaS",
        "budget": {"Revenue": 1_000_000, "COGS": 300_000, "Gross_Margin": 700_000},
        "actual": {"Revenue": 1_090_000, "COGS": 310_000, "Gross_Margin": 780_000},
        "requested_format": "one-pager",
        "hint": "Simple new logo attainment above plan. SaaS norms apply.",
        "expected_drivers": ["new_logo_attainment"],
        "trap": None
    },
    "medium": {
        "name": "SaaS — Mixed Volume + Deal-Size",
        "difficulty": "medium",
        "sector": "SaaS",
        "budget": {"Revenue": 2_000_000, "COGS": 800_000, "Operating_Expense": 700_000},
        "actual": {"Revenue": 2_180_000, "COGS": 850_000, "Operating_Expense": 720_000},
        "requested_format": "board-slide",
        "hint": "Mixed volume and deal-size variance. Renewals and new logos both beat plan.",
        "expected_drivers": ["new_logo_attainment", "expansion_arr", "deal_slip"],
        "trap": None
    },
    "medium_retail_margin": {
        "name": "Retail FMCG — Revenue Up but Margin Squeezed",
        "difficulty": "medium",
        "sector": "Retail FMCG",
        "budget": {
            "Revenue": 1_500_000, "COGS": 900_000,
            "Gross_Margin": 600_000, "Promo_Spend": 75_000
        },
        "actual": {
            "Revenue": 1_650_000, "COGS": 1_040_000,
            "Gross_Margin": 610_000, "Promo_Spend": 130_000
        },
        "requested_format": "board-slide",
        "hint": (
            "TRAP: Revenue is up 10% but gross margin only grew 1.7%. "
            "Promo spend blew out 73%. AI must identify that the volume lift came at too high a cost."
        ),
        "expected_drivers": ["promo_effectiveness", "mix_shift"],
        "trap": "favorable_revenue_hides_margin_erosion"
    },
    "hard": {
        "name": "Pharma Manufacturing — Multi-Driver Crisis",
        "difficulty": "hard",
        "sector": "Pharma Manufacturing",
        "budget": {
            "Revenue": 1_500_000, "COGS": 900_000,
            "Gross_Margin": 600_000, "R&D": 250_000
        },
        "actual": {
            "Revenue": 1_420_000, "COGS": 980_000,
            "Gross_Margin": 440_000, "R&D": 280_000
        },
        "requested_format": "exception-report",
        "hint": (
            "Complex multi-driver variance. Revenue miss likely regulatory hold or supply issue. "
            "COGS overrun of 8.9% is catastrophic by pharma norms — investigate API price spike or batch failure."
        ),
        "expected_drivers": ["api_price_spike", "batch_rejection", "regulatory_hold"],
        "trap": None
    },
    "hard_saas_churn": {
        "name": "SaaS — Churn Quarter (Offsetting Variances)",
        "difficulty": "hard",
        "sector": "SaaS",
        "budget": {
            "Revenue": 3_000_000, "COGS": 900_000,
            "Gross_Margin": 2_100_000, "Sales_OpEx": 800_000
        },
        "actual": {
            "Revenue": 2_820_000, "COGS": 820_000,
            "Gross_Margin": 2_000_000, "Sales_OpEx": 650_000
        },
        "requested_format": "exception-report",
        "hint": (
            "TRAP: Net revenue miss is -6% but cost saves make EBITDA look OK. "
            "AI must flag that revenue miss + cost save is NOT healthy — "
            "it signals lost customers AND hiring freeze."
        ),
        "expected_drivers": ["churn_rate", "deal_slip", "headcount_freeze"],
        "trap": "offsetting_variances_mask_health_problem"
    },
    "hard_edtech_seasonal": {
        "name": "EdTech SaaS — Exam Season + B2B Slip",
        "difficulty": "hard",
        "sector": "EdTech SaaS",
        "budget": {
            "Revenue": 2_000_000, "COGS": 800_000,
            "Gross_Margin": 1_200_000, "Content_Cost": 200_000
        },
        "actual": {
            "Revenue": 2_310_000, "COGS": 970_000,
            "Gross_Margin": 1_340_000, "Content_Cost": 290_000
        },
        "requested_format": "memo",
        "hint": (
            "B2C beat by 25% due to exam season but B2B contracts slipped a quarter. "
            "Content cost overran 45% — investigate if new course launches were planned. "
            "Apply EdTech SaaS sector norms."
        ),
        "expected_drivers": ["enrollment_volume", "b2b_contract_timing", "content_cost"],
        "trap": None
    },
    "hard_conglomerate": {
        "name": "Conglomerate Rollup — Cross-Sector",
        "difficulty": "hard",
        "sector": "Retail FMCG",
        "budget": {
            "Retail_Revenue": 1_000_000, "Pharma_Revenue": 500_000,
            "Total_Revenue": 1_500_000, "Total_COGS": 900_000,
            "Consolidated_GM": 600_000
        },
        "actual": {
            "Retail_Revenue": 1_150_000, "Pharma_Revenue": 430_000,
            "Total_Revenue": 1_580_000, "Total_COGS": 1_010_000,
            "Consolidated_GM": 570_000
        },
        "requested_format": "board-slide",
        "hint": (
            "Retail beat plan; Pharma missed badly. Consolidated GM is down despite revenue beat. "
            "AI must apply different sector norms per division and explain why the net number is misleading."
        ),
        "expected_drivers": ["seasonal_volume", "api_price_spike", "mix_shift"],
        "trap": "consolidated_number_hides_pharma_crisis"
    }
}


def _clamp_reward(value: float) -> float:
    """Same rules as task scores: never exactly 0.0 or 1.0 after rounding."""
    return clamp_openenv_score(value)


class BudgetVarianceEnv:
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

    def reset(self, task_id: str = "easy") -> Observation:
        if task_id not in TASK_LIBRARY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Available: {list(TASK_LIBRARY.keys())}"
            )
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.previous_drafts = []
        self.done = False
        self._hallucination_count = 0
        self.current_task = task_id

        cfg = TASK_LIBRARY[task_id]
        self.current_task_config = cfg
        self.sector = cfg["sector"]
        self.budget = dict(cfg["budget"])
        self.actual = dict(cfg["actual"])
        self.requested_format = cfg["requested_format"]
        self.sector_norms = SECTOR_NORMS.get(self.sector, {})
        self.format_template = FORMAT_TEMPLATES.get(self.requested_format, {})

        variances, variance_pct = self._calc_variances()

        return Observation(
            task_id=task_id,
            sector=self.sector,
            budget=self.budget,
            actual=self.actual,
            variances=variances,
            variance_pct=variance_pct,
            requested_format=self.requested_format,
            format_requirements=self.format_template,
            previous_drafts=[],
            step_count=0,
            hint=cfg["hint"]
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        reward_value = 0.0
        breakdown: Dict[str, float] = {}
        reason = "Action taken"

        if action.action_type == "analyze":
            reward_value = 0.10
            breakdown["analysis_base"] = 0.10
            reason = "Analysis step — good start"

        elif action.action_type == "calculate":
            if action.calculations:
                correct = 0
                for calc in action.calculations:
                    expected_abs = round(
                        self.actual.get(calc.metric, 0) - self.budget.get(calc.metric, 0), 2
                    )
                    if abs(calc.absolute_variance - expected_abs) < 0.05:
                        correct += 1
                    else:
                        self._hallucination_count += 1

                base = round(0.20 * correct / max(len(action.calculations), 1), 3)
                reward_value = max(0.05, base)
                breakdown["correct_calc"] = reward_value
                reason = f"Calculations: {correct}/{len(action.calculations)} correct"
            else:
                reward_value = 0.05
                reason = "Calculate without calculations field — partial credit"

        elif action.action_type == "query_norms":
            if action.norm_query and action.norm_query.sector in SECTOR_NORMS:
                reward_value = 0.20
                breakdown["norm_query"] = 0.20
                reason = f"Queried norms for {action.norm_query.sector}"
            else:
                reward_value = 0.10
                reason = "query_norms without valid norm_query"

        elif action.action_type == "draft":
            reward_value = 0.15
            breakdown["draft_base"] = 0.15
            if action.structured_output:
                reward_value += 0.10
                breakdown["structured_bonus"] = 0.10
                if action.structured_output.risk_flag and not action.structured_output.risk_reason:
                    reward_value -= 0.05
                    breakdown["missing_risk_reason"] = 0.05
            if action.explanation_text:
                self.previous_drafts.append(action.explanation_text)
            reason = "Draft created"

        elif action.action_type == "revise":
            if len(self.previous_drafts) == 0:
                reward_value = 0.05
                reason = "Revise called with no prior draft"
            else:
                reward_value = 0.10
                reason = "Draft revised"
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

            norms = self.sector_norms
            threshold = norms.get("red_flag_threshold_pct", 10.0)
            _, variance_pct = self._calc_variances()
            any_breach = any(abs(v) > threshold for v in variance_pct.values())
            if action.structured_output:
                if any_breach and action.structured_output.risk_flag:
                    reward_value += 0.10
                    breakdown["risk_flag_correct"] = 0.10
                elif any_breach and not action.structured_output.risk_flag:
                    breakdown["missed_risk_flag"] = 0.05

            reason = "Final report submitted"

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
                f"Apply correct sector norms."
            )
        )

        reward = Reward(value=reward_value, reason=reason, breakdown=safe_breakdown)
        return obs, reward, self.done, {
            "episode_id": self.episode_id,
            "step": self.step_count,
            "hallucinations_so_far": self._hallucination_count,
            "task_trap": self.current_task_config.get("trap")
        }

    def state(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "current_task": self.current_task,
            "task_name": self.current_task_config.get("name", ""),
            "difficulty": self.current_task_config.get("difficulty", ""),
            "sector": self.sector,
            "step_count": self.step_count,
            "done": self.done,
            "drafts_count": len(self.previous_drafts),
            "hallucination_count": self._hallucination_count
        }

    def get_sector_norms(self, sector: str = None) -> Dict:
        s = sector or self.sector
        return SECTOR_NORMS.get(s, {})

    def list_tasks(self) -> List[Dict]:
        return [
            {
                "id": tid,
                "name": cfg["name"],
                "difficulty": cfg["difficulty"],
                "sector": cfg["sector"],
                "format": cfg["requested_format"],
                "has_trap": cfg["trap"] is not None
            }
            for tid, cfg in TASK_LIBRARY.items()
        ]

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
