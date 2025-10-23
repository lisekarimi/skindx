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
build-all: ## Build all services
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
# 🐋 Docker Commands
# =======================
DOCKER_IMAGE = $(DOCKER_USERNAME)/$(PROJECT_NAME):$(VERSION)
CONTAINER_NAME = $(PROJECT_NAME)-container

build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE) .

run: ## Run container locally
	docker run -d --rm --name $(CONTAINER_NAME) \
	  --env-file .env \
	  -p 80:80 \
	  -v $(CURDIR)/assets:/app/assets \
	  -v $(CURDIR)/src:/app/src \
	  -v $(CURDIR)/docs:/app/docs \
	  $(DOCKER_IMAGE)
	@echo "🚀 Running app at http://localhost:80"

dev: build run ## Build and run container in development mode

list : ## List files inside the container
	docker run --rm $(DOCKER_IMAGE) ls -la /app

logs: ## View container logs
	docker logs -f $(CONTAINER_NAME)

stop: ## Stop the Docker container
	docker stop $(CONTAINER_NAME) || true

clean: stop ## Stop and remove the Docker container and image
	docker rm $(CONTAINER_NAME) || true
	docker rmi $(DOCKER_IMAGE) || true

restart: clean run ## Restart Docker container (clean + run)

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
