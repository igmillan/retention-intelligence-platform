# Technical Report — User Retention & Churn Prediction

---

## Objective

Develop an end-to-end machine learning pipeline to predict subscription churn,
explain model behavior, and operationalize retention prioritization decisions.

---

## Architecture

The solution was implemented using a modular architecture:

Simulation → Features → Modeling → Explainability → Decisioning

---

## Data Generation

Synthetic user-level behavioral data was simulated to emulate streaming
platform engagement patterns.

Population generated:
- 10,000 users.

Included variables:
- engagement metrics,
- subscription metadata,
- behavioral activity,
- churn target labels.

---

## Feature Engineering

Feature engineering included:

- ratio metrics,
- temporal recency variables,
- behavioral aggregation features,
- engagement intensity indicators.

---

## Modeling Framework

Benchmark models evaluated:

- Logistic Regression
- Random Forest
- XGBoost

Champion selected using:
- ROC-AUC,
- PR-AUC,
- Lift.

---

## Explainability Methodology

SHAP explainability was applied for:

- global driver ranking,
- local prediction explanation,
- user-level attribution.

---

## Decision Engine Logic

Decisioning combines:

- churn probability,
- driver categorization,
- revenue-at-risk,
- playbook recommendation logic.

---

## Output Artifacts

Generated outputs include:

- prediction tables,
- SHAP reports,
- decisioning outputs,
- prioritization summaries.

---