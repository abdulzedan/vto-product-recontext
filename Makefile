.PHONY: help install install-dev format lint type-check test test-cov clean run

help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  format       Format code with black and isort"
	@echo "  lint         Run linting with ruff"
	@echo "  type-check   Run type checking with mypy"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  clean        Clean up generated files"
	@echo "  run          Run the bulk processor"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

format:
	black src tests
	isort src tests
	ruff check --fix src tests

lint:
	black --check src tests
	isort --check src tests
	ruff check src tests

type-check:
	mypy src

test:
	pytest

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov .pytest_cache .mypy_cache .ruff_cache
	rm -rf output/* logs/*

run:
	python -m bulk_image_processor