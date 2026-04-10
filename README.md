Case 01 — Retention Intelligence Platform

Overview

An end-to-end retention intelligence framework designed to predict subscription churn, explain behavioral churn drivers, and operationalize retention prioritization using explainable machine learning and business-aware decisioning.

This project simulates a real-world subscription business environment and demonstrates the complete lifecycle of a production-grade data science solution—from synthetic data generation to predictive modeling, explainability, and executive-facing dashboard deployment.

⸻

Business Problem

Subscription businesses face revenue leakage due to preventable customer churn. Traditional churn analysis often stops at prediction and fails to translate insights into actionable retention strategies.

This project solves that gap by creating an intelligence layer that:
	•	Predicts 30-day churn probability.
	•	Identifies the behavioral drivers behind churn risk.
	•	Prioritizes customers by economic exposure.
	•	Assigns recommended intervention playbooks.

⸻

Solution Architecture

1. Simulation Engine

Synthetic subscription ecosystem generation for 10,000 users including:
	•	Demographics
	•	Plan structure
	•	Behavioral engagement metrics
	•	Platform usage metrics
	•	Simulated churn labels

2. Feature Engineering Pipeline

Derived retention intelligence variables including:
	•	Engagement ratios
	•	Recency metrics
	•	Behavioral efficiency KPIs
	•	Consumption quality indicators

3. Modeling Benchmark

Compared multiple candidate models:
	•	Logistic Regression
	•	Random Forest
	•	XGBoost

Champion model selected via business-aware evaluation framework.

4. Explainability Layer

SHAP-based model explainability for:
	•	Global churn driver importance
	•	Behavioral insight extraction
	•	User-level driver decomposition

5. Decision Engine

Operational scoring layer producing:
	•	Risk tier assignment
	•	Revenue at risk estimation
	•	Priority scoring
	•	Recommended retention playbooks

6. Dashboard Layer

Interactive Streamlit application presenting:
	•	Executive KPIs
	•	Benchmark comparison
	•	Churn driver analysis
	•	Retention prioritization dashboard

⸻

Key Results

Metric	Value
Champion Model	XGBoost
ROC-AUC	0.901
PR-AUC	0.658
Critical Users Identified	169
Revenue Exposure Detected	$5,646


⸻

Business Insights
	•	Churn risk is primarily driven by declining watch time, lower completion rate, and increased inactivity recency.
	•	Behavioral variables materially outperform demographic variables in predictive importance.
	•	A concentrated subset of users drives disproportionate revenue exposure, enabling focused intervention strategies.

⸻

Tech Stack
	•	Python
	•	Pandas / NumPy
	•	Scikit-learn
	•	XGBoost
	•	SHAP
	•	Plotly
	•	Streamlit

⸻

Repository Structure

case_01_user_retention_churn/
│
├── app/
│   └── dashboard_app.py
├── config/
├── data/
│   ├── raw/
│   ├── features/
│   └── scored/
├── docs/
├── reports/
├── src/
│   ├── simulation/
│   ├── features/
│   ├── modeling/
│   ├── explainability/
│   ├── decisioning/
│   └── pipeline/
└── README.md


⸻

How to Run

Execute Full Pipeline

python -m src.pipeline.run_simulation
python -m src.pipeline.run_features
python -m src.pipeline.run_modeling
python -m src.pipeline.run_explainability
python -m src.pipeline.run_decisioning

Launch Dashboard

streamlit run app/dashboard_app.py


⸻

Portfolio Positioning

This case demonstrates:
	•	End-to-end ML system design
	•	Product-oriented analytics thinking
	•	Explainable AI implementation
	•	Decision intelligence / prioritization frameworks
	•	Executive storytelling and dashboard design

Designed as part of a FAANG-grade Data Science portfolio targeting Apple Decision Intelligence and similar strategic analytics organizations.