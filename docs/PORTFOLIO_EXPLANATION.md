# Portfolio Explanation — Synthetic Retention Intervention Learning Loop

## One-line summary

This project demonstrates a clean-room, synthetic machine learning workflow for prioritising voluntary attrition interventions in a financial-services context.

## What the project shows

The project shows how an analytics workflow can move from a business problem to a reproducible technical delivery:

1. Define a retention-risk problem.
2. Generate a fully synthetic customer-month dataset.
3. Engineer reusable behavioural, engagement, relationship, and risk features.
4. Build a transparent rules-based baseline.
5. Train a first supervised machine learning model.
6. Create a capacity-constrained intervention-prioritisation list.
7. Simulate treatment/control assignment and intervention outcomes.
8. Train a treatment-aware second model.
9. Produce governance checks, model-comparison outputs, and visual reporting.

## Why synthetic data was used

The project uses only synthetic data to demonstrate the workflow safely.

It does not include, reproduce, reference, or depend on any real customer, member, employer, financial, operational, confidential, or proprietary organisation data.

## Business interpretation

The project is not simply a prediction model. Its purpose is to show how predictive analytics can be connected to operational action.

The learning loop demonstrates that risk scoring should be connected to:

- intervention capacity;
- recommended action type;
- treatment/control design;
- treatment logging;
- outcome measurement;
- governance checks;
- iterative model learning.

## Responsible analytics framing

The project includes documentation and safeguards to make clear that it is educational and demonstrative.

It is not intended for real credit decisions, real eligibility decisions, real customer treatment decisions, or automated decision-making affecting real individuals.

Any production use would require real data governance, privacy review, legal review, fairness assessment, model risk management, monitoring, and human oversight.

## Explanation

I built this as a clean-room portfolio project to demonstrate how I think about analytics delivery beyond dashboards or isolated models.

The project simulates a financial-services retention problem and implements an end-to-end learning loop: synthetic data generation, feature engineering, baseline modelling, machine learning prioritisation, intervention allocation, synthetic treatment logging, second-model learning, governance checks, and visual reporting.

The most important idea is that a model is only useful if it can be connected responsibly to a business process. That is why the project includes intervention capacity, treatment/control groups, outcome logging, documentation, and governance checks.

## What I would improve next

Future improvements could include:

- clearer uplift-modelling logic;
- stronger treatment-effect estimation;
- model calibration reporting;
- fairness diagnostics across synthetic customer segments;
- dashboard-ready summary outputs;
- a lighter notebook version for non-technical reviewers.
