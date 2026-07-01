# Model Card

## Model status

No model has been trained yet.

## Planned model

The project will use a supervised classification model to estimate voluntary attrition risk over a future 90-day window.

The planned main model is a histogram-based gradient boosting classifier.

## Planned comparison

The project will compare:

1. Rules-based baseline.
2. Machine learning model v1.
3. Treatment-aware machine learning model v2.

## Planned evaluation

The models will be evaluated using:

- ROC-AUC.
- PR-AUC.
- Precision at K.
- Lift at K.
- Brier score.
- Calibration by risk decile.
- Expected value at risk.
- Treatment/control outcome comparison.

## Intended use

Educational and portfolio demonstration of governed retention analytics, intervention prioritisation, and synthetic treatment-loop design.

## Not intended for

This project is not intended for real-world customer decisioning without appropriate data governance, legal review, validation, fairness assessment, monitoring, and human oversight.