# Case 01 — User Retention & Churn Prediction: Retention Intelligence Framework

**Author:** Israel Gómez Millán · igm4487@gmail.com · [linkedin.com/in/igm4487](https://linkedin.com/in/igm4487)  
**Stack:** Python · XGBoost · SHAP · pandas · scikit-learn · Streamlit · Power BI  
**Seed:** 42 · **Population:** 10,000 Synthetic Streaming Users  

---

## Executive Summary

Subscription-based digital platforms such as Apple Music and Apple TV+ face
constant retention pressure as user engagement fluctuates over time.

While predicting which users may churn is valuable, modern Decision Intelligence
organizations require more than raw prediction—they need explainable, actionable,
and economically prioritized intervention strategies.

This project develops a production-style **Retention Intelligence Framework**
that predicts user churn within a 30-day horizon, explains the behavioral drivers
behind churn risk, and operationalizes model outputs into actionable retention
recommendations through a multi-factor decisioning engine.

---

## Business Problem

Digital subscription ecosystems frequently struggle to detect churn risk early
enough to intervene before subscription cancellation occurs.

Without predictive retention systems, organizations face:

- Reactive churn management after revenue is already lost,
- Generic retention campaigns lacking personalization,
- Poor prioritization of limited retention resources,
- Minimal visibility into behavioral drivers behind churn.

---

## Strategic Objective

Build an end-to-end retention intelligence system capable of:

- Predicting churn likelihood within a 30-day forecast horizon,
- Explaining the behavioral drivers behind churn risk,
- Prioritizing users by both risk and economic value,
- Recommending retention playbooks tailored to user deterioration patterns.

---

## Solution Overview

A modular Decision Intelligence pipeline was architected to simulate a
production-grade retention analytics workflow:

```text
Simulation Layer
    ↓
Feature Engineering Layer
    ↓
Benchmark Modeling Layer
    ↓
Explainability Layer
    ↓
Decision Intelligence Layer
```

---

## Pipeline Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│               STAGE 1: SIMULATION LAYER                    │
│ Synthetic user generation · behavioral activity simulation │
│ churn target assignment                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            STAGE 2: FEATURE ENGINEERING LAYER              │
│ Behavioral metrics · engagement ratios · derived features  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            STAGE 3: BENCHMARK MODELING LAYER               │
│ Logistic Regression · Random Forest · XGBoost Benchmarking │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            STAGE 4: EXPLAINABILITY LAYER                   │
│ SHAP global importance · local driver attribution          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         STAGE 5: DECISION INTELLIGENCE LAYER               │
│ Risk tiering · revenue scoring · playbook recommendation   │
└─────────────────────────────────────────────────────────────┘
```

---

## Methodology

### Synthetic Behavioral Simulation
Generated realistic streaming-platform behavioral data including:

- subscription plan allocation,
- session activity,
- watch/listen time,
- skip/completion behavior,
- discovery metrics,
- churn target generation.

### Feature Engineering
Developed behavioral and derived engagement features including:

- session frequency,
- watch efficiency ratios,
- platform adoption scores,
- discovery breadth indicators.

### Benchmark Modeling
Evaluated multiple candidate models:

- Logistic Regression,
- Random Forest,
- XGBoost.

Champion/challenger methodology selected the highest-performing model.

### Explainability Framework
Applied SHAP analysis for:

- global feature importance ranking,
- local user-level churn explanation,
- model interpretability enhancement.

### Decision Intelligence Engine
Operationalized predictions into:

- risk segmentation tiers,
- revenue-at-risk prioritization,
- intervention playbook assignment.

---

## Modeling Performance

| Metric | Champion Result |
|---|---|
| Champion Model | XGBoost |
| ROC-AUC | ~0.90+ |
| PR-AUC | ~0.65+ |
| Lift @ Top Decile | ~3.8x Baseline |
| Users Scored | 2,000 Test Users |

The selected XGBoost model delivered superior predictive performance while
maintaining explainability and operational interpretability.

---

## Behavioral Insights Generated

SHAP explainability analysis revealed that churn risk is primarily driven by:

| Rank | Driver | Strategic Interpretation |
|---|---|---|
| 1 | Watch Time Last 30D | Declining consumption intensity strongly predicts churn |
| 2 | Completion Rate | Lower engagement quality increases churn propensity |
| 3 | Days Since Last Session | Inactivity sharply elevates churn likelihood |
| 4 | Sessions Last 30D | Reduced platform frequency predicts disengagement |
| 5 | Content Diversity | Narrow exploration breadth correlates with churn |

Behavioral engagement signals significantly outperformed demographic/contextual
attributes as churn predictors.

---

## Decision Intelligence Framework

Predictions were operationalized into a retention prioritization engine that:

### Segments Users into Risk Tiers

- Low Risk
- Medium Risk
- High Risk
- Critical Risk

### Estimates Revenue-at-Risk

Combining:

- churn probability,
- subscription value proxy,
- ecosystem depth.

### Assigns Retention Playbooks

Examples include:

- Re-Engagement Nudges
- Content Discovery Boosts
- Engagement Quality Recovery
- Feature Education Campaigns
- Commercial Retention Offers

---

## Key Results

| KPI | Result |
|---|---|
| Total Users Simulated | 10,000 |
| Test Users Scored | 2,000 |
| Critical Risk Users | ~8.5% |
| Revenue-at-Risk Identified | \$5,600+ |
| Retention Playbooks Generated | 5 |
| SHAP Drivers Tracked | Top 10 Global / Top 3 Local |

---

## Strategic Recommendations

### 1. Prioritize Critical High-Value Users
Retention resources should focus first on users exhibiting both:

- critical churn probability,
- elevated revenue-at-risk.

### 2. Personalize Retention Strategy by Driver Category
Retention actions should align with the dominant behavioral deterioration pattern
rather than applying uniform campaigns.

### 3. Monitor Engagement Decay as Primary Early Warning Signal
Declining watch/listen time and session inactivity represent the strongest
leading indicators of upcoming churn.

---

## Repository Structure

```text
case_01_user_retention_churn/
│
├── config/
├── data/
│   ├── raw/
│   ├── features/
│   └── scored/
│
├── src/
│   ├── simulation/
│   ├── features/
│   ├── modeling/
│   ├── explainability/
│   ├── decisioning/
│   └── pipeline/
│
├── reports/
├── assets/
├── README.md
└── requirements.txt
```

---

## How to Run

Execute the modular pipeline sequentially:

```bash
python -m src.pipeline.run_simulation
python -m src.pipeline.run_features
python -m src.pipeline.run_modeling
python -m src.pipeline.run_explainability
python -m src.pipeline.run_decisioning
```

---

## Governance

| Standard | Convention |
|---|---|
| Reproducibility | `SEED = 42` across all scripts |
| Naming | `snake_case` + semantic suffixing |
| Logging | Structured timestamped logs |
| Modeling | Champion/Challenger Benchmarking |
| Explainability | SHAP Global + Local Attribution |
| Documentation | Enterprise-style README + modular architecture |

---

## Future Enhancements

Potential future roadmap includes:

- uplift modeling for treatment optimization,
- campaign ROI simulation,
- causal inference for intervention analysis,
- reinforcement learning for adaptive retention policies.

---