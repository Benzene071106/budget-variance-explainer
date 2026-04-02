# Budget Variance Explainer - OpenEnv

## Description & Motivation
This environment simulates the real-world task of **Budget Variance Analysis & Root-Cause Explanation**. Finance teams typically spend 45+ minutes manually calculating variances, linking numbers to business drivers, applying sector-specific norms, and formatting reports for executives. 

A well-trained agent can reduce this to ~2 minutes while maintaining high accuracy, evidence trails, and sector calibration — turning generic LLM output into trusted FP&A work.

## Key Features
- 3 tasks with clear easy → medium → hard progression
- Sector norms (Retail FMCG, SaaS, Pharma Manufacturing)
- Dense reward with partial progress signals
- Structured output requirement (one-pager, board-slide, exception-report)
- Zero-hallucination focus with evidence citation

## Action & Observation Spaces

**Observation** includes:
- Budget and Actual data
- Pre-computed variances (% and absolute)
- Sector name + sector norms hint
- Requested output format
- Draft history

**Actions**:
- `analyze`, `calculate`, `query_norms`, `draft`, `revise`, `submit`

## Tasks

1. **Easy (Retail FMCG)**: Single driver variance (holiday promotion). Output: one-pager.
2. **Medium (SaaS)**: Mixed volume + deal size. Output: board-slide style.
3. **Hard (Pharma Manufacturing)**: Complex multi-driver with strict norms. Output: exception report.

## Setup Instructions

### Local Development
```bash
pip install -r requirements.txt
uvicorn main:app --reload