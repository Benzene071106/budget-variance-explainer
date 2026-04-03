---
title: Budget Variance Explainer

colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Budget Variance Explainer — OpenEnv v2

FP&A Budget Variance Analysis with LLM-as-judge grader. Supports any sector dynamically.

## Endpoints

- `GET /` — Health check
- `GET /reset?task_id=easy` — Start episode
- `POST /step` — Take action
- `GET /grader?detail=true` — Get score
- `GET /baseline` — Run inference
- `GET /report` — Visual dashboard

## Environment Variables

Set these in Space Settings → Secrets:

- `HF_TOKEN` — Your Groq API key
- `API_BASE_URL` — https://api.groq.com/openai/v1
- `MODEL_NAME` — llama-3.3-70b-versatile
- `GRADER_MODEL` — llama-3.3-70b-versatile
