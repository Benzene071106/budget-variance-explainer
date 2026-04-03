import os
import json
import re
from typing import Dict, Optional, Tuple
from models import Observation


try:
    from openai import OpenAI
    _client = OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "no-key")
    )
    _MODEL = os.getenv("GRADER_MODEL", os.getenv("MODEL_NAME", "gpt-4o-mini"))
    _LLM_AVAILABLE = True
except Exception:
    _LLM_AVAILABLE = False



_SECTOR_THRESHOLDS = {
    "Retail FMCG":         {"red_flag_pct": 10.0, "cogs_tight": False},
    "SaaS":                {"red_flag_pct": 20.0, "cogs_tight": False},
    "Pharma Manufacturing": {"red_flag_pct": 5.0,  "cogs_tight": True},
    "EdTech SaaS":         {"red_flag_pct": 15.0, "cogs_tight": False},
}

_EXPECTED_DRIVERS = {
    "easy":              ["seasonal", "holiday", "promotion", "volume"],
    "easy_saas":         ["logo", "new customer", "attainment"],
    "medium":            ["renewal", "deal", "expansion", "volume"],
    "medium_retail_margin": ["promo", "margin", "cost", "mix"],
    "hard":              ["raw material", "batch", "api", "yield", "regulatory"],
    "hard_saas_churn":   ["churn", "retention", "headcount", "hiring"],
    "hard_edtech_seasonal": ["exam", "enrollment", "content", "b2b"],
    "hard_conglomerate": ["retail", "pharma", "division", "mix"],
}

_FORMAT_KEYWORDS = {
    "one-pager":        ["summary", "recommendation", "root cause", "variance"],
    "board-slide":      ["headline", "strategic", "implication", "ask"],
    "exception-report": ["alert", "corrective", "impact", "timeline", "chain"],
    "memo":             ["detailed", "assumption", "controllable", "table"],
}


class VarianceGrader:
    """
    Two-layer grader:
      Layer 1 — Rule-based (always runs, instant, free)
      Layer 2 — LLM-as-judge (runs if API key available, adds nuance)

    Final score = weighted blend of both layers.
    """

   
    def grade(
        self,
        task_id: str,
        final_draft: str,
        observation: Observation,
        return_detail: bool = False
    ) -> float | Dict:

        # Flatten JSON structured output to plain text for rule-based grading
        flat_draft = self._flatten_draft(final_draft)
        rule_score, rule_detail = self._rule_based_grade(task_id, flat_draft, observation)

        if _LLM_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            llm_score, llm_feedback = self._llm_grade(task_id, final_draft, observation)
            # Weighted blend: rule 40%, LLM 60%
            final = round(0.40 * rule_score + 0.60 * llm_score, 3)
            source = "hybrid"
        else:
            final = rule_score
            llm_feedback = "LLM grader not available — rule-based only"
            source = "rule_based"

        detail = {
            "final_score": min(1.0, final),
            "rule_score": rule_score,
            "rule_detail": rule_detail,
            "llm_score": llm_score if _LLM_AVAILABLE and os.getenv("OPENAI_API_KEY") else None,
            "llm_feedback": llm_feedback,
            "grader_source": source
        }

        if return_detail:
            return detail
        return round(min(1.0, final), 3)

    def _flatten_draft(self, draft: str) -> str:
        """Convert structured JSON output to flat text for keyword matching"""
        try:
            data = json.loads(draft)
            parts = []
            def extract(obj):
                if isinstance(obj, str):
                    parts.append(obj)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        extract(v)
                elif isinstance(obj, list):
                    for item in obj:
                        extract(item)
            extract(data)
            return " ".join(parts)
        except Exception:
            return draft  # already plain text


    def _rule_based_grade(
        self, task_id: str, draft: str, obs: Observation
    ) -> Tuple[float, Dict]:
        score = 0.0
        detail: Dict[str, float] = {}
        draft_lower = draft.lower()

       
        num_score = self._check_numbers(draft, obs)
        score += 0.20 * num_score
        detail["numerical_accuracy"] = round(0.20 * num_score, 3)

        driver_score = self._check_drivers(task_id, draft_lower, obs)
        score += 0.25 * driver_score
        detail["driver_identification"] = round(0.25 * driver_score, 3)

     
        norm_score = self._check_sector_norm(obs.sector, draft, obs)
        score += 0.20 * norm_score
        detail["sector_norm_applied"] = round(0.20 * norm_score, 3)

      
        fmt_score = self._check_format(obs.requested_format, draft_lower)
        score += 0.20 * fmt_score
        detail["format_compliance"] = round(0.20 * fmt_score, 3)

        
        evidence_score = self._check_evidence(draft_lower)
        score += 0.15 * evidence_score
        detail["evidence_quality"] = round(0.15 * evidence_score, 3)

      
        halluc_penalty = self._hallucination_penalty(draft, obs)
        score -= halluc_penalty
        detail["hallucination_penalty"] = -round(halluc_penalty, 3)

        
        seasonality_score = self._check_seasonality_reasoning(draft_lower, obs)
        score += 0.05 * seasonality_score
        detail["seasonality_reasoning"] = round(0.05 * seasonality_score, 3)

       
        trap_score = self._check_offsetting_trap(draft_lower, obs)
        score += 0.05 * trap_score
        detail["trap_detection"] = round(0.05 * trap_score, 3)

        final = round(max(0.0, min(1.0, score)), 3)
        detail["rule_total"] = final
        return final, detail

    def _check_numbers(self, draft: str, obs: Observation) -> float:
        hits = 0
        for val in obs.variance_pct.values():
            abs_val = abs(val)
            variants = set()
            for v in [val, abs_val]:
                variants.add(str(v))
                variants.add(str(round(v, 1)))
                if v == int(v):
                    variants.add(str(int(v)))
            if any(v in draft for v in variants):
                hits += 1
        return hits / max(len(obs.variance_pct), 1)

    def _check_drivers(self, task_id: str, draft_lower: str, obs=None) -> float:
        keywords = _EXPECTED_DRIVERS.get(task_id, [])

        
        if not keywords and obs:
            try:
                from dynamic_env import get_or_generate_norms
                norms = get_or_generate_norms(obs.sector)
                keywords = norms.get("common_drivers", [])
                keywords = [k.split()[0].lower() for k in keywords if k]
            except Exception:
                pass

        universal = ["revenue", "cost", "margin", "volume", "price",
                     "material", "labor", "seasonal", "commodity",
                     "demand", "supply", "inflation", "variance"]

        if not keywords or task_id.startswith("custom_"):
           
            all_keywords = universal
            hits = sum(1 for kw in all_keywords if kw in draft_lower)
            return min(1.0, hits / 4)  

        hits = sum(1 for kw in keywords if kw in draft_lower)
        return min(1.0, hits / max(len(keywords) * 0.5, 1))

    def _check_sector_norm(self, sector: str, draft: str, obs=None) -> float:
        thresh = _SECTOR_THRESHOLDS.get(sector, {})
        draft_lower = draft.lower()
        score = 0.0

        
        if thresh:
            pct = thresh.get("red_flag_pct", 10)
        else:
            
            try:
                from dynamic_env import get_or_generate_norms
                norms = get_or_generate_norms(sector)
                pct = norms.get("red_flag_threshold_pct", 10)
            except Exception:
                pct = 10

    
        if str(pct) + "%" in draft or str(int(pct)) + "%" in draft:
            score += 0.5

 
        sector_keywords = {
            "Retail FMCG": ["seasonal", "fmcg", "promotional", "volume lift"],
            "SaaS": ["arr", "churn", "renewal", "gross margin"],
            "Pharma Manufacturing": ["catastrophic", "batch", "regulatory", "api", "yield"],
            "EdTech SaaS": ["enrollment", "exam", "content cost", "b2b", "refund"],
        }
        known_kws = sector_keywords.get(sector)
        if known_kws:
            if any(w in draft_lower for w in known_kws):
                score += 0.5
        else:
         
            universal_sector_words = [
                "commodity", "material", "labor", "workforce", "operational",
                "escalation", "investigation", "supplier", "procurement",
                "seasonal", "demand", "supply", "inflation", "cost overrun",
                "margin compression", "root cause", "corrective"
            ]
            hits = sum(1 for w in universal_sector_words if w in draft_lower)
            if hits >= 3:
                score += 0.5
            elif hits >= 1:
                score += 0.25
         
            try:
                from dynamic_env import get_or_generate_norms
                norms = get_or_generate_norms(sector)
                drivers = norms.get("common_drivers", [])
                driver_words = [d.split()[0].lower() for d in drivers]
                if any(w in draft_lower for w in driver_words):
                    score = min(1.0, score + 0.25)
                red_flags = norms.get("red_flags", [])
                flag_words = [f.split()[0].lower() for f in red_flags]
                if any(w in draft_lower for w in flag_words):
                    score = min(1.0, score + 0.25)
            except Exception:
                pass

        return min(1.0, score)

    def _check_format(self, fmt: str, draft_lower: str) -> float:
        keywords = _FORMAT_KEYWORDS.get(fmt, [])
        if not keywords:
            return 0.5
        hits = sum(1 for kw in keywords if kw in draft_lower)
        return hits / max(len(keywords), 1)

    def _check_evidence(self, draft_lower: str) -> float:
        causal_words = ["because", "due to", "caused by", "result of", "driven by", "attributed to"]
        hedging = ["suggests", "indicates", "appears to"]
        score = 0.0
        if any(w in draft_lower for w in causal_words):
            score += 0.6
        if any(w in draft_lower for w in hedging):
            score += 0.2
  
        if re.search(r"\d+\.?\d*\s*%", draft_lower):
            score += 0.2
        return min(1.0, score)

    def _check_seasonality_reasoning(self, draft_lower: str, obs: Observation) -> float:
        """
        Smart Rule 1 — Seasonality & Proportionality Reasoning.

        Checks if agent:
        a) Mentions seasonality when revenue variance is favorable
        b) Correctly flags margin squeeze when COGS grew faster than revenue
        c) Distinguishes one-time vs structural variance
        """
        score = 0.0
        vp = obs.variance_pct

        revenue_var = vp.get("Revenue", vp.get("revenue", 0))
        seasonal_words = ["seasonal", "holiday", "quarter", "cycle",
                         "temporary", "one-time", "period", "exam", "festival"]
        if revenue_var > 0 and any(w in draft_lower for w in seasonal_words):
            score += 0.4

        
        cogs_var = abs(vp.get("COGS", vp.get("Raw_Material", vp.get("cogs", 0))))
        rev_var = abs(revenue_var)
        if cogs_var > 0 and rev_var > 0:
            if cogs_var > rev_var:
            
                squeeze_words = ["margin", "squeeze", "compress", "erode",
                                "profitab", "cost overrun", "cost pressure"]
                if any(w in draft_lower for w in squeeze_words):
                    score += 0.4
            else:
                
                healthy_words = ["acceptable", "proportional", "tracking",
                                "in line", "within", "normal"]
                if any(w in draft_lower for w in healthy_words):
                    score += 0.3


        structural_words = ["sustainable", "structural", "recurring",
                           "investigate", "root cause", "systemic"]
        if any(w in draft_lower for w in structural_words):
            score += 0.2

        return min(1.0, score)

    def _check_offsetting_trap(self, draft_lower: str, obs: Observation) -> float:
        """
        Smart Rule 2 — Offsetting Variance Trap Detection.

        Real-world FP&A trap: revenue miss + cost save = net EBITDA ok.
        But this masks health problems (lost customers + hiring freeze).

        Also checks: favorable revenue but unfavorable margin = another trap.
        """
        score = 0.0
        vp = obs.variance_pct

        variances = list(vp.values())
        if len(variances) < 2:
            return 0.5  

        favorable = [v for v in variances if v > 0]
        unfavorable = [v for v in variances if v < 0]

     
        has_trap = len(favorable) > 0 and len(unfavorable) > 0

        if has_trap:
           
            offset_words = ["offset", "mask", "mislead", "despite", "however",
                           "although", "net", "overall", "underlying",
                           "concern", "warning", "flag", "investigate"]
            if any(w in draft_lower for w in offset_words):
                score += 0.6

            explain_words = ["because", "due to", "caused by", "driven by",
                            "result of", "attributed to"]
            if any(w in draft_lower for w in explain_words):
                score += 0.4
        else:
           
            score = 1.0

        return min(1.0, score)

    def _hallucination_penalty(self, draft: str, obs: Observation) -> float:
        """Penalise numbers in the draft that do not match the observation"""
        penalty = 0.0
        found_pcts = set(re.findall(r"(\d+\.?\d*)\s*%", draft))

   
        valid_pcts = set()
        for v in obs.variance_pct.values():
            abs_v = abs(v)
            valid_pcts.add(str(v))
            valid_pcts.add(str(abs_v))
            valid_pcts.add(str(round(abs_v, 1)))
            try:
                if abs_v == int(abs_v):
                    valid_pcts.add(str(int(abs_v)))
            except Exception:
                pass


        try:
            from dynamic_env import get_or_generate_norms
            norms = get_or_generate_norms(obs.sector)
            thresh = norms.get("red_flag_threshold_pct", 10)
            valid_pcts.add(str(thresh))
            valid_pcts.add(str(int(thresh)))
            valid_pcts.add(str(round(thresh, 1)))
        except Exception:
            pass

       
        range_nums = set(re.findall(r"(\d+)", " ".join(
            str(v) for v in obs.variance_pct.values()
        )))
        valid_pcts |= range_nums

        for pct in found_pcts:
            if pct not in valid_pcts and float(pct) > 0.5:
                penalty += 0.05

        return min(0.20, penalty)

    def _llm_grade(
        self, task_id: str, draft: str, obs: Observation
    ) -> Tuple[float, str]:
        prompt = f"""You are a senior FP&A quality reviewer. Grade this variance analysis report.

== GROUND TRUTH ==
Sector: {obs.sector}
Budget: {obs.budget}
Actual: {obs.actual}
Variance %: {obs.variance_pct}
Requested Format: {obs.requested_format}
Task: {task_id}

== AGENT'S REPORT ==
{draft}

== GRADING RUBRIC ==
Score from 0.0 to 1.0 on ALL of these:

1. NUMERICAL ACCURACY (0–1): Are all percentages and absolutes correct? No hallucinated numbers?
2. DRIVER QUALITY (0–1): Are the root causes correct and specific for this sector & task?
3. SECTOR NORM USAGE (0–1): Did the agent correctly apply {obs.sector} norms and thresholds?
4. FORMAT COMPLIANCE (0–1): Does the report match the {obs.requested_format} structure?
5. REASONING CHAIN (0–1): Is there a clear logical chain from data → driver → conclusion?
6. TRAP DETECTION (0–1): If there is a misleading signal in the data, did the agent catch it?

Return ONLY valid JSON (no markdown, no extra text):
{{
  "numerical_accuracy": 0.X,
  "driver_quality": 0.X,
  "sector_norm_usage": 0.X,
  "format_compliance": 0.X,
  "reasoning_chain": 0.X,
  "trap_detection": 0.X,
  "weighted_score": 0.X,
  "key_strength": "one sentence",
  "key_weakness": "one sentence"
}}

weighted_score = (numerical_accuracy*0.20 + driver_quality*0.25 + sector_norm_usage*0.20 + format_compliance*0.15 + reasoning_chain*0.15 + trap_detection*0.05)
"""
        try:
            resp = _client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400
            )
            raw = resp.choices[0].message.content.strip()
  
            raw = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)
            score = float(data.get("weighted_score", 0.5))
            feedback = (
                f"Strength: {data.get('key_strength', '')} | "
                f"Weakness: {data.get('key_weakness', '')}"
            )
            return round(min(1.0, max(0.0, score)), 3), feedback
        except Exception as e:
            return 0.5, f"LLM grader error: {e}"


if __name__ == "__main__":
    from env import BudgetVarianceEnv
    env = BudgetVarianceEnv()
    obs = env.reset("easy")
    grader = VarianceGrader()

    sample_draft = (
        "The Revenue variance of 12.0% is driven by holiday promotion volume lift, "
        "which is within the normal ±5-8% Retail FMCG seasonal range — actually above it, "
        "suggesting the promotion was highly effective. "
        "COGS increased by 5.0% due to higher volumes, which is acceptable. "
        "Gross Margin improved by 22.5% because revenue grew faster than costs. "
        "Recommendation: document promo ROI for future planning."
    )
    result = grader.grade("easy", sample_draft, obs, return_detail=True)
    import json as _j
    print(_j.dumps(result, indent=2))
