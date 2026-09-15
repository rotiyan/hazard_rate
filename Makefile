.PHONY: install test clean lint format docs

# Install package and dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Run tests
test:
	pytest safe_fraud_detection/tests/ -v

# Run tests with coverage
test-cov:
	pytest safe_fraud_detection/tests/ --cov=safe_fraud_detection --cov-report=html --cov-report=term

# Run unit tests only
test-unit:
	pytest safe_fraud_detection/tests/unit/ -v

# Run integration tests only
test-integration:
	pytest safe_fraud_detection/tests/integration/ -v

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Lint code
lint:
	flake8 safe_fraud_detection/ --max-line-length=100 --ignore=E203,W503

# Format code
format:
	black safe_fraud_detection/ --line-length=100

# Type checking
typecheck:
	mypy safe_fraud_detection/ --ignore-missing-imports

# Build documentation
docs:
	cd docs && make html

# Run all quality checks
quality: lint typecheck test

# Build package
build:
	python setup.py sdist bdist_wheel

# Help
help:
	@echo "Available commands:"
	@echo "  make install          - Install package and dependencies"
	@echo "  make install-dev      - Install with development dependencies"
	@echo "  make test             - Run all tests"
	@echo "  make test-cov         - Run tests with coverage"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make clean            - Clean build artifacts"
	@echo "  make lint             - Lint code"
	@echo "  make format           - Format code with black"
	@echo "  make typecheck        - Run type checking"
	@echo "  make quality          - Run all quality checks"
	@echo "  make build            - Build package"
	@echo "  make docs             - Build documentation"
