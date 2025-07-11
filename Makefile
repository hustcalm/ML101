# Makefile for ML101 project
.PHONY: help install install-dev clean test coverage lint format build publish docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package"
	@echo "  install-dev  - Install package in development mode"
	@echo "  clean        - Clean build artifacts"
	@echo "  test         - Run tests"
	@echo "  coverage     - Run tests with coverage"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  build        - Build package"
	@echo "  publish      - Publish to PyPI"
	@echo "  docs         - Build documentation"

install:
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	pytest tests/ -v

coverage:
	pytest tests/ --cov=ml101 --cov-report=html --cov-report=term

lint:
	flake8 ml101/ tests/ examples/
	mypy ml101/

format:
	black ml101/ tests/ examples/
	isort ml101/ tests/ examples/

build: clean
	python -m build

publish: build
	twine upload dist/*

docs:
	cd docs && make html

# Development workflow
check: lint test coverage

# Full development setup
setup: install-dev
	@echo "Development environment ready!"
	@echo "Run 'make check' to run all checks"

# Release workflow
release: clean test lint build
	@echo "Package ready for release!"
	@echo "Run 'make publish' to upload to PyPI"
