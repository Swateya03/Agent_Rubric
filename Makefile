# AgentRubric — common commands
# Run: make help  to see all available commands
# Requires: GNU make (pre-installed on Mac/Linux, install via chocolatey on Windows)

.PHONY: help run verbose quiet export train verify test test-integration clean

help:
	@echo ""
	@echo "  AgentRubric — available commands"
	@echo ""
	@echo "  Pipeline:"
	@echo "    make run        Run pipeline on sample data (INFO logging)"
	@echo "    make verbose    Run with full DEBUG output"
	@echo "    make quiet      Run silently (WARNING and ERROR only)"
	@echo "    make export     Run pipeline and export preference pairs"
	@echo ""
	@echo "  Training:"
	@echo "    make train      Quick-test reward model trainer (10 steps)"
	@echo ""
	@echo "  Development:"
	@echo "    make test              Run tests (excludes API calls)"
	@echo "    make test-integration  Run tests that require real API calls"
	@echo "    make verify            Run all code quality checks"
	@echo "    make clean             Remove caches and generated result files"
	@echo ""

run:
	python -m agentrubric.run_pipeline \
		--data data/sample_responses.json

verbose:
	python -m agentrubric.run_pipeline \
		--data data/sample_responses.json \
		--verbose

quiet:
	python -m agentrubric.run_pipeline \
		--data data/sample_responses.json \
		--quiet

export:
	python -m agentrubric.run_pipeline \
		--data data/sample_responses.json \
		--export-pairs \
		--verbose

train:
	python -m agentrubric.training.reward_trainer --quick-test

verify:
	python scripts/verify.py

test:
	python -m pytest agentrubric/tests/ -v --tb=short -m "not integration"

test-integration:
	python -m pytest agentrubric/tests/ -v --tb=short -m "integration"

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	@find . -name "*.pyc" -delete 2>/dev/null; true
	@rm -f data/results.json data/phase3_results.json
	@echo "Cleaned."
