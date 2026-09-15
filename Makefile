.PHONY: install install-dev test test-cov test-unit test-integration clean lint format typecheck quality build help

# Install package and runtime dependencies
install:
	pip install -e .

# Install with development tools and plotting (for the examples)
install-dev:
	pip install -e ".[dev,viz]"

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

# Run all quality checks
quality: lint typecheck test

# Build sdist and wheel into dist/ (needs the dev extras)
build:
	python -m build

# Help
help:
	@echo "Available commands:"
	@echo "  make install          - Install package and runtime dependencies"
	@echo "  make install-dev      - Install with development tools and plotting"
	@echo "  make test             - Run all tests"
	@echo "  make test-cov         - Run tests with coverage"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make clean            - Clean build artifacts"
	@echo "  make lint             - Lint code"
	@echo "  make format           - Format code with black"
	@echo "  make typecheck        - Run type checking"
	@echo "  make quality          - Run all quality checks"
	@echo "  make build            - Build sdist and wheel"
