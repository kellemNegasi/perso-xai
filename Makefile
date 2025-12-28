PYTHON ?= python3

.PHONY: test tests test-evaluators test-models test-datasets run-experiments sanity run-openml-adult-income run-openml-bank-marketing run-openml-german-credit test-all tune-train

test test-evaluators:
	$(PYTHON) -m pytest tests/evaluators

test-models:
	$(PYTHON) -m pytest tests/models

test-datasets:
	$(PYTHON) -m pytest tests/test_dataset_adapters.py

test-all:
	$(PYTHON) -m pytest

# Example: make run-experiments RUN_EXPERIMENTS="tabular_demo_suite" OUTPUT_DIR=results
RUN_EXPERIMENTS ?= tabular_demo_suite
MAX_INSTANCES ?=
	OUTPUT_DIR ?=
	EXPERIMENT_RESULTS_SUBDIR ?=
	MODEL ?=
	LOG_LEVEL ?=
	PRINT_SUMMARY ?=
OPENML_MAX_INSTANCES ?=
OPENML_OUTPUT_DIR ?= openml_results

	run-experiments:
		$(PYTHON) -m src.cli.main $(RUN_EXPERIMENTS) $(if $(MAX_INSTANCES),--max-instances $(MAX_INSTANCES),) $(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),) $(if $(EXPERIMENT_RESULTS_SUBDIR),--experiment-results-subdir $(EXPERIMENT_RESULTS_SUBDIR),) $(if $(MODEL),--model $(MODEL),) $(if $(LOG_LEVEL),--log-level $(LOG_LEVEL),) $(if $(PRINT_SUMMARY),--print-summary,)

# Example: make tune-train TUNE_EXPERIMENT=openml_adult_suite TUNE_MODEL=random_forest
TUNE_EXPERIMENT ?= openml_adult_suite
TUNE_MODEL ?=
TUNE_MAX_INSTANCES ?=
TUNING_OUTPUT_DIR ?= saved_models/tuning_results
MODEL_STORE_DIR ?= saved_models
STOP_AFTER_TRAINING ?= 1

tune-train:
	$(PYTHON) -m src.cli.main $(TUNE_EXPERIMENT) $(if $(TUNE_MAX_INSTANCES),--max-instances $(TUNE_MAX_INSTANCES),) $(if $(TUNE_MODEL),--model $(TUNE_MODEL),) --tune-models --use-tuned-params --reuse-trained-models --tuning-output-dir $(TUNING_OUTPUT_DIR) --model-store-dir $(MODEL_STORE_DIR) $(if $(filter 1 yes true TRUE,$(STOP_AFTER_TRAINING)),--stop-after-training,)

sanity:
	$(PYTHON) -m src.cli.main breast_cancer_rf_suite --max-instances 10 --output-dir demo_results

run-openml-adult-income:
	$(PYTHON) -m src.cli.main openml_adult_suite $(if $(OPENML_MAX_INSTANCES),--max-instances $(OPENML_MAX_INSTANCES),) --output-dir $(OPENML_OUTPUT_DIR)/adult_income

run-openml-bank-marketing:
	$(PYTHON) -m src.cli.main openml_bank_suite $(if $(OPENML_MAX_INSTANCES),--max-instances $(OPENML_MAX_INSTANCES),) --output-dir $(OPENML_OUTPUT_DIR)/bank_marketing

run-openml-german-credit:
	$(PYTHON) -m src.cli.main openml_german_suite $(if $(OPENML_MAX_INSTANCES),--max-instances $(OPENML_MAX_INSTANCES),) --output-dir $(OPENML_OUTPUT_DIR)/german_credit
