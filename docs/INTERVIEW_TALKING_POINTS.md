# Interview Talking Points — User Retention & Churn Prediction

---

## Elevator Pitch

I built an end-to-end retention intelligence framework simulating how a company
like Apple could proactively identify and intervene on subscription churn risk.

The solution combines predictive modeling, SHAP explainability, and a business
decisioning layer that transforms raw churn predictions into prioritized,
actionable retention recommendations.

---

## Why This Project Matters

Many churn models stop at prediction.

I wanted to demonstrate not only predictive modeling capability, but also how
to operationalize model outputs into business decisions and intervention logic.

---

## Technical Highlights

- Built modular production-style architecture.
- Benchmarked multiple candidate models.
- Applied SHAP for explainability.
- Developed multi-factor decision engine.

---

## Key Insight Highlights

- Behavioral engagement significantly outperformed demographics.
- Engagement decay is the strongest leading churn signal.
- Critical/high-value cohorts drive disproportionate revenue risk.

---

## Business Impact Framing

The framework enables retention teams to prioritize interventions based on:
- predicted churn risk,
- economic value,
- behavioral deterioration drivers.

---

## Lessons / Design Decisions

- Prioritized explainability over black-box complexity.
- Focused on business operationalization, not raw prediction only.
- Designed modular pipeline for production extensibility.

---