# 🧪 SKINDX Unit Tests

Simple unit tests for the core components of the SKINDX skin cancer detection application, focusing on business logic rather than UI components.

## 🩺 Medical AI Testing Considerations

Medical AI validation (model accuracy, false positives/negatives, clinical validation) has been conducted in separate Jupyter notebooks during model development. These unit tests focus on the application components - ensuring the web app correctly loads models, processes files, and handles user interactions reliably.

## 🎯 Current Scope

This focused approach tests the critical ML inference pipeline while avoiding the complexity of testing UI components. The 32% coverage reflects intentional focus on core business logic rather than comprehensive application testing.


## 📊 Test Coverage

```
Name                  Stmts   Miss  Cover
-----------------------------------------
src\__init__.py           0      0   100%
src\constants.py          7      7     0%
src\model_loader.py      46      7    85%
src\ui\__init__.py        0      0   100%
src\ui\app.py            80     80     0%
src\ui\utils.py          89     63    29%
src\utils.py              8      0   100%
-----------------------------------------
TOTAL                   230    157    32%
```

## ✅ What We Test

**Model Inference** (`test_model_loader.py`)
- Model loading and initialization
- Basic prediction functionality
- Device selection (CPU/GPU)

**Image Processing** (`test_image_preprocessing.py`)
- Image preprocessing pipeline
- Format conversion (RGB, grayscale, RGBA)
- Shape consistency (224x224x3 output)

**File Validation** (`test_file_validation.py`)
- File upload validation from UI layer
- File size limits (10MB max)
- MIME type checking (JPEG, PNG only)

## 🚫 What We Don't Test

- **UI Components**: Streamlit app components (0% coverage)
- **API Functions**: HTTP request functions (partially covered at 29%)
- **Constants**: Static configuration values (0% coverage)
- **Visualization**: Plotly chart generation functions

## 🏃‍♂️ Running Tests

```bash
make test
```

All test commands and options are detailed in the Makefile.

## 🛡️ Code Quality & CI

All tests follow Ruff formatting and linting standards. The CI pipeline runs both linting and tests to maintain code quality. Check `.github/workflows/code_quality.yml` for implementation details.

## 🤝 Contributing

When contributing to the project:

1. Write simple tests for new core functionality
2. Ensure all existing tests pass
3. Follow Ruff formatting standards
4. Test business logic only, not UI components
5. Use simple assertions and basic mocks
6. Focus on happy path scenarios
7. Use uv's isolated environment for dependencies
