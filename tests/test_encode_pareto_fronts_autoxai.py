from pathlib import Path

import encode_pareto_fronts_autoxai as enc


def test_autoxai_schema_parses_randint_and_categorical() -> None:
    cfg = enc.load_hyperparameter_config(Path("src/configs/explainer_hyperparameters.yml"))
    schema = enc.build_hyperparameter_schema(cfg)

    assert "autoxai_lime" in schema.numeric_specs
    assert "lime_num_samples" in schema.numeric_specs["autoxai_lime"]

    assert "autoxai_shap" in schema.categorical_specs
    assert "shap_explainer_type" in schema.categorical_specs["autoxai_shap"]
    assert set(schema.categorical_params["shap_explainer_type"]) >= {"kernel", "sampling"}


def test_autoxai_shap_conditional_applicability() -> None:
    cfg = enc.load_hyperparameter_config(Path("src/configs/explainer_hyperparameters.yml"))
    schema = enc.build_hyperparameter_schema(cfg)

    # Sampling SHAP: shap_l1_reg and shap_l1_reg_k should be treated as not applicable.
    features, applicable = enc.encode_hyperparameters(
        "autoxai_shap",
        {"shap_explainer_type": "sampling", "shap_nsamples": "128", "shap_l1_reg": "aic", "shap_l1_reg_k": "5"},
        schema=schema,
        dataset_n_features=20,
    )
    assert applicable["is_applicable_shap_l1_reg"] == 0
    assert applicable["is_applicable_shap_l1_reg_k"] == 0
    assert features["hp_shap_l1_reg_oh_aic"] == 0.0

    # Kernel SHAP with num_features: shap_l1_reg_k should become applicable.
    features, applicable = enc.encode_hyperparameters(
        "autoxai_shap",
        {"shap_explainer_type": "kernel", "shap_nsamples": "128", "shap_l1_reg": "num_features", "shap_l1_reg_k": "5"},
        schema=schema,
        dataset_n_features=20,
    )
    assert applicable["is_applicable_shap_l1_reg"] == 1
    assert applicable["is_applicable_shap_l1_reg_k"] == 1
    assert "hp_shap_l1_reg_k" in features

