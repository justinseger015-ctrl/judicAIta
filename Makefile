# Makefile for Judicaita

.PHONY: help install install-dev test lint format type-check clean docs serve docker-build docker-up

help:
	@echo "Judicaita Development Commands"
	@echo "==============================="
	@echo ""
	@echo "Installation:"
	@echo "  make install         Install production dependencies"
	@echo "  make install-dev     Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make format          Format code with black"
	@echo "  make lint            Lint code with ruff"
	@echo "  make type-check      Run type checking with mypy"
	@echo "  make test            Run tests"
	@echo "  make test-cov        Run tests with coverage"
	@echo "  make clean           Clean build artifacts"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs            Build documentation"
	@echo "  make docs-serve      Serve documentation locally"
	@echo ""
	@echo "Running:"
	@echo "  make serve           Start API server"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-up       Start Docker compose services"
	@echo "  make docker-down     Stop Docker compose services"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test,docs]"
	pre-commit install

format:
	black src/ tests/ examples/
	ruff check --fix src/ tests/ examples/

lint:
	ruff check src/ tests/ examples/

type-check:
	mypy src/

test:
	pytest

test-cov:
	pytest --cov=judicaita --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && mkdocs build

docs-serve:
	cd docs && mkdocs serve

serve:
	judicaita serve --reload

docker-build:
	docker build -t judicaita:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Development workflow
dev: install-dev format lint type-check test

# CI workflow
ci: lint type-check test

# Release workflow
release: clean test
	python -m build
	@echo "Ready to release! Run: twine upload dist/*"
