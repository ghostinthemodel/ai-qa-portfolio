# AI/ML QA Portfolio — Tapan Shah

> Senior QA Professional transitioning into AI/ML Quality Engineering.
> 10+ years of hands-on QA and leadership across Fintech, E-commerce, SaaS, and Consulting —
> now applied to the unique challenges of probabilistic, AI-driven systems.

**Live portfolio:** https://ghostinthemodel.github.io/ai-qa-portfolio/
**Live demos:** https://ghostinthemodel.github.io/ai-qa-portfolio/demos/

---

## What This Is

This repository is a working portfolio that demonstrates how I test AI/ML systems —
not just knowledge of the concepts, but actual test design, runnable code, and
real defect discovery.

It covers four areas where traditional QA breaks down and needs a new approach:

- **LLM output quality** — when there's no single "correct" answer
- **Agentic systems** — multi-step AI agents with tool calls and reasoning chains
- **Non-deterministic outputs** — the same input producing different outputs by design
- **RAG pipelines** — retrieval accuracy, answer faithfulness, hallucination detection

---

## Repository Structure

```
ai-qa-portfolio/
├── index.html                        # Main portfolio — philosophy, test cases, eval framework
├── demos/                            # Compiled React interactive demos
│   ├── index.html
│   └── assets/
├── tests/
│   ├── test_llm_chatbot_quality.py   # Runnable pytest suite — 17 tests, 6 categories
│   ├── DEFECT_REPORT_BUG001.md       # Real defect report — safety bypass via context reframing
│   └── README.md                     # Test suite setup and design rationale
└── README.md                         # This file
```

---

## Live Interactive Demos

Four browser-based demos built to show — not just describe — how I test AI systems.

| Demo | What It Shows |
|---|---|
| **LLM Eval Runner** | Run 15 eval cases across Factual Accuracy, Hallucination, Safety, Reasoning and Consistency. Drill into failures per case. |
| **Agentic System Tester** | Replay agent execution traces. Inspect tool calls, catch reasoning failures, and see silent failure modes. |
| **Non-Determinism Visualiser** | Run the same prompt 50× and watch the output distribution build in real time. Score variance, topic drift, QA verdict. |
| **RAG Quality Tester** | Test retrieval accuracy and answer faithfulness. Toggle between a grounded and a hallucinated answer to see how scoring changes. |

→ **[Open demos](https://ghostinthemodel.github.io/ai-qa-portfolio/demos/)**

---

## Runnable Test Suite

The `tests/` folder contains a real pytest suite targeting
[streamlit/llm-examples](https://github.com/streamlit/llm-examples) —
a publicly available open-source LLM chatbot application.

### What it tests

| Category | Tests | Failure Modes Covered |
|---|---|---|
| Response Quality | 4 | Empty responses, format non-compliance, length drift, language mismatch |
| Safety & Policy | 3 | Harmful content refusal, PII extraction, over-refusal |
| Consistency | 3 | Factual variance, sycophantic reversal, persona drift |
| Prompt Injection | 2 | Direct injection, role-play bypass |
| Hallucination | 2 | Fake entity fabrication, known fact accuracy |
| Performance | 3 | Latency SLA, empty input, long input handling |

### Run it yourself

### Test Results

![Test suite passing](tests/results.png)


```bash
# Install dependencies
pip install pytest openai python-dotenv pytest-asyncio

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > tests/.env

# Run the full suite
pytest tests/test_llm_chatbot_quality.py -v

# Run a single category
pytest tests/test_llm_chatbot_quality.py -v -k "Safety"
pytest tests/test_llm_chatbot_quality.py -v -k "Hallucination"
pytest tests/test_llm_chatbot_quality.py -v -k "Injection"
```

> Estimated cost: ~$0.02 on gpt-3.5-turbo. Full suite runs in ~45–60 seconds.

### Why the tests are designed this way

Every test includes a `WHY` block explaining the failure mode it targets,
the design rationale, threshold decisions, and known limitations.
This is intentional — in AI/ML QA, *how* you test matters as much as *what* you test.

---

## Defect Report

`tests/DEFECT_REPORT_BUG001.md` documents a real class of LLM safety bug:
**safety filter bypass via educational framing.**

The model correctly refuses direct requests to generate phishing emails.
However, wrapping the same request in "for cybersecurity awareness training..."
causes it to comply — producing a fully functional, convincing phishing email.

**Reproduction rate: 80% (8/10 runs)**

The report includes:
- Exact reproduction steps with model outputs
- Reproduction rate table across 10 runs
- Root cause analysis (RLHF alignment limitation)
- Three recommended fixes with code
- A regression test added post-discovery
- Verification criteria for sign-off

This defect was found during **structured exploratory testing** after the
automated suite passed — illustrating why automated tests define a floor,
not a ceiling, for AI safety testing.

---

## Key Testing Principles

**Test distributions, not instances.**
A single "bad" output is noise. A shifted distribution is a signal.
Statistical thresholds replace exact string matching.

**Test both directions of safety.**
Over-refusal is also a quality failure. A model that refuses 30% of
legitimate requests has a usability bug, not a safety win.

**Automated tests define a floor, not a ceiling.**
The safety bypass defect was found during exploratory testing — after
the automated suite passed. Both are required.

**Evals are products.**
Eval sets go stale. Rubrics drift. Evaluation infrastructure deserves
the same engineering rigour as production code.

**Document the why, not just the what.**
Test design judgment is as important as the test code itself.

---

## Background

10+ years of QA experience across:
- **Fintech** — payments, banking platforms, regulatory compliance testing
- **E-commerce** — checkout flows, recommendation systems, A/B test quality
- **SaaS / Enterprise software** — API testing, regression frameworks, release pipelines
- **Consulting** — QA strategy, test maturity assessments, team uplift

Now building specialised skills in AI/ML quality engineering — where the
testing problems are genuinely new and traditional QA frameworks need
rethinking from first principles.

---

## Contact

Open to Senior QA Engineer and AI/ML Quality Engineering roles.

**GitHub:** [ghostinthemodel](https://github.com/ghostinthemodel)

---

*Built to demonstrate AI/ML quality engineering skills through working code,
not just concepts.*
