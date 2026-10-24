PYTHON ?= python
CLI := $(PYTHON) -m fraud_detection

.PHONY: install data eda train all test lint clean

install:  ## install the package with dev tools
	$(PYTHON) -m pip install -e ".[dev]"

data:  ## generate the synthetic dataset
	$(CLI) generate

eda: data  ## exploratory figure and summary
	$(CLI) eda

train: data  ## select, tune, evaluate, save model and figures
	$(CLI) train

all: eda train  ## reproduce every artefact in reports/

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -rf data models .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
