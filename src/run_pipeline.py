"""Main executable pipeline for the synthetic intervention learning-loop project.

Run from the project root with:

    python -m src.run_pipeline

The pipeline intentionally uses only synthetic data. It does not require,
reference, or depend on any proprietary organisation data.
"""

from pathlib import Path

from src.config import get_profile
from src.feature_engineering import (
    build_feature_table,
    feature_table_summary_as_text,
    save_feature_table_outputs,
    summarise_feature_table,
)
from src.governance_reporting import (
    run_governance_reporting,
    save_governance_reporting_outputs,
)
from src.intervention_policy import (
    create_intervention_list,
    intervention_summary_as_text,
    save_intervention_outputs,
    summarise_intervention_list,
)
from src.model_training import (
    model_metrics_as_text,
    run_model_v1,
    save_model_v1_outputs,
)
from src.model_v2_learning import (
    model_v2_metrics_as_text,
    run_model_v2,
    save_model_v2_outputs,
)
from src.rules_baseline import (
    rules_metrics_as_text,
    run_rules_baseline,
    save_rules_baseline_outputs,
)
from src.synthetic_data import (
    generate_customer_month_panel,
    save_synthetic_panel_outputs,
    summarise_synthetic_panel,
    synthetic_panel_summary_as_text,
)
from src.treatment_log import (
    generate_treatment_log,
    save_treatment_log_outputs,
    summarise_treatment_log,
    treatment_log_summary_as_text,
)
from src.visual_reporting import run_visual_reporting_from_outputs

# Safe display/config fallbacks for compatibility with SyntheticProfile.
DEFAULT_PROFILE = "deep"
TARGET_COLUMN = "voluntary_attrition_next_90d"
RANDOM_SEED = 42
RULES_BASELINE_VERSION = "rules_baseline_v1"
ML_MODEL_V1_VERSION = "ml_model_v1"
ML_MODEL_V2_VERSION = "treatment_aware_ml_model_v2"
INTERVENTION_CAPACITY = 1000
TREATMENT_SHARE = 0.80
CONTROL_SHARE = 0.20


def main() -> None:
    """Run the full synthetic retention-intervention learning-loop pipeline."""
    print("Synthetic Voluntary Attrition Intervention Learning Loop")

    profile = get_profile()
    profile_name = getattr(profile, "profile_name", DEFAULT_PROFILE)
    target_column = getattr(profile, "target_column", TARGET_COLUMN)
    random_seed = getattr(profile, "random_seed", RANDOM_SEED)
    rules_baseline_version = getattr(
        profile, "rules_baseline_version", RULES_BASELINE_VERSION
    )
    ml_model_v1_version = getattr(profile, "ml_model_v1_version", ML_MODEL_V1_VERSION)
    ml_model_v2_version = getattr(profile, "ml_model_v2_version", ML_MODEL_V2_VERSION)
    intervention_capacity = getattr(
        profile, "intervention_capacity", INTERVENTION_CAPACITY
    )
    treatment_share = getattr(profile, "treatment_share", TREATMENT_SHARE)
    control_share = getattr(profile, "control_share", CONTROL_SHARE)

    print("Project configuration is ready.")
    print(f"Project root: {Path.cwd()}")
    print(f"Default profile: {profile_name}")
    print(f"Synthetic customers: {profile.n_customers:,}")
    print(f"Monthly snapshots: {profile.n_months}")
    print(f"Target column: {target_column}")
    print(f"Random seed: {random_seed}")
    print(f"Rules baseline version: {rules_baseline_version}")
    print(f"ML model v1 version: {ml_model_v1_version}")
    print(f"ML model v2 version: {ml_model_v2_version}")
    print(f"Intervention capacity: {intervention_capacity:,}")
    print(f"Treatment share: {treatment_share:.0%}")
    print(f"Control share: {control_share:.0%}")

    print()
    print("Generating synthetic customer-month panel...")

    synthetic_panel = generate_customer_month_panel(profile=profile)
    synthetic_summary = summarise_synthetic_panel(synthetic_panel)
    synthetic_output_paths = save_synthetic_panel_outputs(synthetic_panel)

    print("Synthetic data generated successfully.")
    print(synthetic_panel_summary_as_text(synthetic_summary))
    print(f"Output file: {synthetic_output_paths['synthetic_panel']}")

    print()
    print("Building engineered feature table...")

    feature_result = build_feature_table(synthetic_panel)
    feature_summary = summarise_feature_table(feature_result.feature_table)
    feature_output_paths = save_feature_table_outputs(feature_result)

    print("Feature engineering completed successfully.")
    print(feature_table_summary_as_text(feature_summary))
    print(f"Output file: {feature_output_paths['feature_table']}")

    print()
    print("Running rules-based baseline...")

    rules_result = run_rules_baseline(feature_result.feature_table)
    rules_output_paths = save_rules_baseline_outputs(rules_result)

    print("Rules-based baseline completed successfully.")
    print("Rules baseline metrics")
    print(rules_metrics_as_text(rules_result.metrics))
    print(f"Rules metrics output file: {rules_output_paths['metrics']}")
    print(f"Rules scored output file: {rules_output_paths['scored_table']}")

    print()
    print("Training ML model v1...")

    model_v1_result = run_model_v1(feature_result.feature_table)
    model_v1_output_paths = save_model_v1_outputs(model_v1_result)

    print("ML model v1 completed successfully.")
    print("Model metrics")
    print(model_metrics_as_text(model_v1_result.metrics))
    print(f"ML v1 metrics output file: {model_v1_output_paths['metrics']}")
    print(f"ML v1 scored output file: {model_v1_output_paths['scored_table']}")

    print()
    print("Creating intervention policy list...")

    intervention_result = create_intervention_list(
        scored_table=model_v1_result.scored_table,
        feature_table=feature_result.feature_table,
        profile_name=profile_name,
    )
    intervention_summary = summarise_intervention_list(
        intervention_result.intervention_list
    )
    intervention_output_paths = save_intervention_outputs(intervention_result)

    print("Intervention policy completed successfully.")
    print(intervention_summary_as_text(intervention_summary))
    print(
        "Intervention output file: "
        f"{intervention_output_paths['intervention_list']}"
    )

    print()
    print("Generating synthetic treatment log...")

    treatment_result = generate_treatment_log(
        intervention_result.intervention_list
    )
    treatment_summary = summarise_treatment_log(treatment_result.treatment_log)
    treatment_output_paths = save_treatment_log_outputs(treatment_result)

    print("Treatment log generated successfully.")
    print(treatment_log_summary_as_text(treatment_summary))
    print(
        "Treatment log output file: "
        f"{treatment_output_paths['treatment_log']}"
    )

    print()
    print("Training treatment-aware ML model v2...")

    model_v2_result = run_model_v2(
        feature_table=feature_result.feature_table,
        treatment_log=treatment_result.treatment_log,
    )
    model_v2_output_paths = save_model_v2_outputs(model_v2_result)

    print("Treatment-aware ML model v2 completed successfully.")
    print("Model v2 metrics")
    print(model_v2_metrics_as_text(model_v2_result.metrics))
    print(
        "ML v2 learning table output file: "
        f"{model_v2_output_paths['learning_table']}"
    )
    print(f"ML v2 metrics output file: {model_v2_output_paths['metrics']}")
    print(f"ML v2 scored output file: {model_v2_output_paths['scored_table']}")

    print()
    print("Creating governance and model-comparison reports...")

    governance_result = run_governance_reporting(
        synthetic_panel=synthetic_panel,
        feature_table=feature_result.feature_table,
        rules_metrics=rules_result.metrics,
        rules_scored_table=rules_result.scored_table,
        model_v1_metrics=model_v1_result.metrics,
        model_v1_scored_table=model_v1_result.scored_table,
        intervention_list=intervention_result.intervention_list,
        treatment_log=treatment_result.treatment_log,
        model_v2_metrics=model_v2_result.metrics,
        model_v2_learning_table=model_v2_result.learning_table,
        model_v2_scored_table=model_v2_result.scored_table,
    )
    governance_output_paths = save_governance_reporting_outputs(
        governance_result
    )

    print("Governance and model-comparison reporting completed successfully.")
    print("Model comparison")
    print(governance_result.model_comparison.to_string(index=False))
    print()
    print("Governance checks summary")
    print(
        governance_result.governance_checks.groupby(
            ["severity", "status"]
        ).size().reset_index(name="checks").to_string(index=False)
    )
    print(
        "Model comparison output file: "
        f"{governance_output_paths['model_comparison']}"
    )
    print(
        "Governance checks output file: "
        f"{governance_output_paths['governance_checks']}"
    )

    print()
    print("Generating portfolio visual reports...")

    visual_result = run_visual_reporting_from_outputs()

    print("Visual reporting completed successfully.")
    print(visual_result.as_text())

    print()
    print("Learning-loop pipeline completed successfully.")


if __name__ == "__main__":
    main()