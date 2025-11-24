PYTHON ?= python3

.PHONY: test tests test-evaluators

test tests test-evaluators:
	$(PYTHON) -m pytest tests/evaluators
