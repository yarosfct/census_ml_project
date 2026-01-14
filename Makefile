.PHONY: help venv install test lint format quickcheck clean

PYTHON := python3
VENV := venv
BIN := $(VENV)/bin

help:
	@echo "Available targets:"
	@echo "  venv        - Create a virtual environment"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linter (ruff)"
	@echo "  format      - Format code (ruff format)"
	@echo "  quickcheck  - Run quick check script"
	@echo "  clean       - Remove generated files"

venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
		echo "Virtual environment created. Activate with: source $(VENV)/bin/activate"; \
	else \
		echo "Virtual environment already exists."; \
	fi

install: venv
	@echo "Installing dependencies..."
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
	@echo "Dependencies installed."

test:
	@echo "Running tests..."
	$(BIN)/pytest

lint:
	@echo "Running linter..."
	$(BIN)/ruff check .

format:
	@echo "Formatting code..."
	$(BIN)/ruff format .

quickcheck:
	@echo "Running quick check..."
	$(BIN)/python src/scripts/quick_check.py

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleanup complete."
