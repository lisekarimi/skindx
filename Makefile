# =====================================
# 🌱 Project & Environment Configuration
# =====================================

# Read from pyproject.toml using grep (works on all platforms)
PROJECT_NAME = $(shell python3 -c "import re; print(re.search('name = \"(.*)\"', open('pyproject.toml').read()).group(1))")
VERSION = $(shell python3 -c "import re; print(re.search('version = \"(.*)\"', open('pyproject.toml').read()).group(1))")

include .env
export DOCKER_USERNAME PROJECT_NAME VERSION

DOCKER_IMAGE_NB = $(DOCKER_USERNAME)/$(PROJECT_NAME)-nb
DOCKER_IMAGE_APP = $(DOCKER_USERNAME)/$(PROJECT_NAME)-app
CONTAINER_NAME_NB = $(PROJECT_NAME)-notebooks
CONTAINER_NAME_APP = $(PROJECT_NAME)-app

# =======================
# 🐳 Docker Compose Commands
# =======================
build: ## Build all services
	docker-compose build

up: ## Run all services
	docker-compose up

notebooks: ## Build and run notebooks service
	docker-compose up notebooks

app: ## Run app in dev mode
	docker-compose up app

down: ## Stop all services
	docker-compose down

# =======================
# 🐋 Docker Commands (HF Spaces)
# =======================
DOCKER_IMAGE_HF = $(DOCKER_USERNAME)/$(PROJECT_NAME)-hf:$(VERSION)
CONTAINER_NAME_HF = $(PROJECT_NAME)-container

hf-build: ## Build Docker image for HF Spaces
	docker build -t $(DOCKER_IMAGE_HF) .

hf-run: ## Run Docker container locally (test HF deployment) with hot reload
	docker run -it --rm --name $(CONTAINER_NAME_HF) \
	  --env-file .env \
	  -v $(CURDIR):/app \
	  -w /app \
	  -p 7860:7860 -p 8000:8000 \
	  --user root \
	  $(DOCKER_IMAGE_HF)

hf-stop: ## Stop running Docker container
	docker stop $(CONTAINER_NAME_HF)

# =======================
# 🪝 Hooks
# =======================

hooks:	## Install pre-commit on local machine
	pip install pre-commit && pre-commit install && pre-commit install --hook-type commit-msg

# Pre-commit ensures code quality before commits.
# Installing globally lets you use it across all projects.
# Check if pre-commit command exists : pre-commit --version

# =====================================
# ✨ Code Quality
# =====================================

lint:	## Run code linting and formatting
	uvx ruff check .
	uvx ruff format .

fix:	## Fix code issues and format
	uvx ruff check --fix .
	uvx ruff format .

# =======================
# 🔍 Security Scanning
# =======================

# check-secrets:		## Check for secrets/API keys
# 	gitleaks detect --source . --verbose

# bandit-scan:		## Check Python code for security issues
# 	uvx bandit -r src/

# audit:	## Audit dependencies for vulnerabilities
# 	uv run --with pip-audit pip-audit

security-scan:		## Run all security checks
	gitleaks detect --source . --verbose && uv run --with pip-audit pip-audit && uvx bandit -r src/

# =======================
# 🧪 Testing Commands
# =======================

test: 	## Run all tests in the tests/ directory
	uv run --isolated --with pytest --with torch --with plotly --with pillow --with huggingface_hub pytest

test-file: 	## Run specific test file
	uv run --isolated --with pytest --with torch --with plotly --with pillow --with huggingface_hub pytest tests/test_models.py

test-func: 	## Run specific test function by name
	uv run --isolated --with pytest --with torch --with plotly --with pillow --with huggingface_hub pytest -k test_extract_code

test-cov: 	## Run tests with coverage
	uv run --isolated --with pytest --with pytest-cov --with torch --with plotly --with pillow --with huggingface_hub pytest --cov=src

test-cov-html: 	## Run tests with coverage and generate HTML report
	uv run --isolated --with pytest --with pytest-cov --with torch --with plotly --with pillow --with huggingface_hub pytest --cov=src --cov-report html

open-cov: 	## Open HTML coverage report in browser
	@echo "To open the HTML coverage report, run:"
	@echo "  start htmlcov\\index.html        (Windows)"
	@echo "  open htmlcov/index.html          (macOS)"
	@echo "  xdg-open htmlcov/index.html      (Linux)"


# =====================================
# 📚 Documentation & Help
# =====================================

help: ## Show this help message
	@echo "Available commands:"
	@echo ""
	@python3 -c "import re; lines=open('Makefile', encoding='utf-8').readlines(); targets=[re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$',l) for l in lines]; [print(f'  make {m.group(1):<20} {m.group(2)}') for m in targets if m]"


# =======================
# 🎯 PHONY Targets
# =======================

# Auto-generate PHONY targets (cross-platform)
.PHONY: $(shell python3 -c "import re; print(' '.join(re.findall(r'^([a-zA-Z_-]+):\s*.*?##', open('Makefile', encoding='utf-8').read(), re.MULTILINE)))")

# Test the PHONY generation
# test-phony:
# 	@echo "$(shell python3 -c "import re; print(' '.join(sorted(set(re.findall(r'^([a-zA-Z0-9_-]+):', open('Makefile', encoding='utf-8').read(), re.MULTILINE)))))")"
