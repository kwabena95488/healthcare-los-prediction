# Contributing to Healthcare Length of Stay Prediction

Welcome to the Healthcare Length of Stay Prediction project! We appreciate your interest in contributing. This guide will help you get started with development and contribution workflows.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Release Process](#release-process)

---

## Development Setup

### Prerequisites

- **Python**: 3.8 or higher
- **Git**: Latest version
- **Make**: For using Makefile commands (optional but recommended)

### Initial Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Healthcare-Analytics-main
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Using pip
   pip install -r requirements.txt
   pip install -e .

   # Using Makefile
   make install
   ```

4. **Verify setup**
   ```bash
   python validate_setup.py
   python -m pytest tests/ -v
   ```

### Development Environment

- **IDE Recommendations**: VS Code, PyCharm, or Jupyter Lab
- **Extensions**: Python, Pylint, Black formatter, GitLens
- **Configuration**: Use the provided `.gitignore` and `pytest.ini`

---

## Project Architecture

### Directory Structure

```
Healthcare-Analytics-main/
├── src/                    # Source code modules
│   ├── data/              # Data processing and loading
│   ├── models/            # ML model implementations  
│   ├── evaluation/        # Metrics and model evaluation
│   ├── visualization/     # Plotting and visualization
│   └── utils/             # Utilities and configuration
├── data/                  # Data storage
│   ├── raw/              # Original, immutable datasets
│   ├── processed/        # Cleaned, feature-engineered data
│   └── interim/          # Intermediate processing files
├── models/               # Trained model artifacts
├── results/              # Analysis outputs
├── config/               # Configuration files
├── scripts/              # CLI executable scripts
├── tests/                # Test suite
├── docs/                 # Documentation
└── experiments/          # MLflow experiment tracking
```

### Design Principles

1. **Modularity**: Each module has a single responsibility
2. **Configuration-Driven**: All parameters externalized to YAML files
3. **Reproducibility**: Fixed random seeds and version-controlled dependencies
4. **Testability**: Comprehensive test coverage with fixtures
5. **Documentation**: JSDoc-style documentation for all functions
6. **CLI-First**: All major workflows accessible via command line

### Key Components

- **Base Models**: Abstract base class ensuring consistent interface
- **Data Pipeline**: Modular preprocessing with configurable steps
- **Configuration System**: YAML-based with environment variable support
- **CLI Scripts**: Professional command-line tools for all workflows
- **Testing Framework**: pytest with comprehensive fixtures and coverage

---

## Development Workflow

### Feature Development

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes following code standards**
   - Write tests first (TDD approach recommended)
   - Follow the coding standards below
   - Update documentation as needed

3. **Run tests locally**
   ```bash
   make test
   # Or manually:
   python -m pytest tests/ --cov=src --cov-report=html
   ```

4. **Validate your changes**
   ```bash
   python validate_setup.py
   ```

5. **Commit with descriptive messages**
   ```bash
   git add .
   git commit -m "feat: add new data validation feature"
   ```

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

### Commit Message Format

Follow conventional commits:
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

---

## Code Standards

### Python Style Guide

- **PEP 8**: Follow Python style guide
- **Line Length**: Maximum 88 characters (Black formatter default)
- **Imports**: Use absolute imports, group by standard/third-party/local
- **Type Hints**: Use type hints for all function signatures

### Naming Conventions

- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Files**: `snake_case.py`
- **Directories**: `lowercase`

### Documentation Standards

All functions must include JSDoc-style documentation:

```python
def process_data(data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Process raw data according to configuration settings.
    
    @param {pd.DataFrame} data - Raw input data
    @param {Dict[str, Any]} config - Processing configuration
    @returns {pd.DataFrame} Processed data
    @throws {ValueError} When data validation fails
    @version 1.0.0
    @public
    
    @example
    # Process training data
    processed_data = process_data(raw_data, config)
    """
    # Implementation...
```

### Error Handling

- Use specific exception types
- Include helpful error messages
- Log errors appropriately
- Handle edge cases gracefully

### Configuration Management

- All parameters in YAML configuration files
- Use `src.utils.config` for loading configurations
- Support environment variable overrides
- Validate configuration on load

---

## Testing Guidelines

### Test Structure

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test module interactions
- **End-to-End Tests**: Test complete workflows

### Test Organization

```
tests/
├── conftest.py           # Shared fixtures
├── test_data_processing.py   # Data module tests
├── test_models.py        # Model tests
├── test_evaluation.py    # Evaluation tests
└── test_integration.py   # Integration tests
```

### Writing Tests

1. **Use descriptive test names**
   ```python
   def test_load_raw_data_success_with_valid_files():
   def test_preprocess_features_handles_missing_values():
   ```

2. **Use fixtures for test data**
   ```python
   def test_model_training(sample_data, test_config):
       X, y = sample_data
       # Test implementation
   ```

3. **Test edge cases and error conditions**
   ```python
   def test_load_data_raises_error_for_missing_file():
       with pytest.raises(FileNotFoundError):
           load_raw_data({'train_path': 'nonexistent.csv'})
   ```

### Test Commands

```bash
# Run all tests
make test

# Run specific test categories
python -m pytest tests/ -m unit
python -m pytest tests/ -m integration
python -m pytest tests/ -m slow

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_data_processing.py -v
```

### Test Requirements

- **Minimum Coverage**: 80% code coverage
- **All Tests Pass**: No failing tests in pull requests
- **Test Isolation**: Tests should not depend on each other
- **Mock External Dependencies**: Use mocks for file I/O, APIs, etc.

---

## Documentation Standards

### Code Documentation

- **JSDoc Style**: Use JSDoc-style comments for all functions
- **Type Information**: Include parameter and return types
- **Examples**: Provide usage examples
- **Version Tags**: Include version information

### File Documentation

- **Module Docstrings**: Every module should have a descriptive docstring
- **Class Documentation**: Document class purpose and key methods
- **Function Documentation**: Complete parameter and return documentation

### Documentation Updates

- Update documentation when changing APIs
- Include examples for new features
- Update README for significant changes
- Maintain API documentation accuracy

---

## Pull Request Process

### Before Submitting

1. **Run full test suite**
   ```bash
   make test
   ```

2. **Validate setup**
   ```bash
   python validate_setup.py
   ```

3. **Update documentation**
   - Update docstrings for changed functions
   - Update README if needed
   - Add examples for new features

4. **Check code quality**
   ```bash
   # Format code
   black src/ tests/ scripts/
   
   # Check linting
   pylint src/
   ```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
```

### Review Process

1. **Automated Checks**: All tests and validation must pass
2. **Code Review**: At least one reviewer approval required
3. **Documentation Review**: Ensure documentation is complete
4. **Testing Verification**: Verify test coverage and quality

---

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce the behavior

**Expected behavior**
What you expected to happen

**Environment:**
- OS: [e.g. macOS 12.0]
- Python version: [e.g. 3.9.7]
- Package version: [e.g. 1.0.0]

**Additional context**
Any other context about the problem
```

### Feature Requests

```markdown
**Feature Description**
Clear description of the proposed feature

**Motivation**
Why is this feature needed?

**Proposed Implementation**
How should this feature work?

**Alternatives Considered**
What alternatives have you considered?
```

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

---

## Release Process

### Version Numbering

Follow semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

### Release Checklist

1. **Update version numbers**
   - `setup.py`
   - Configuration files
   - Documentation

2. **Update CHANGELOG.md**
   - Add new version section
   - List all changes since last release

3. **Run comprehensive tests**
   ```bash
   make test
   python validate_setup.py
   ```

4. **Create release tag**
   ```bash
   git tag -a v1.2.0 -m "Release version 1.2.0"
   git push origin v1.2.0
   ```

5. **Update documentation**
   - API documentation
   - README badges
   - Installation instructions

---

## Getting Help

### Resources

- **Documentation**: Check `docs/` directory
- **API Reference**: See `docs/API.md`
- **Examples**: Check `scripts/` for usage examples
- **Tests**: Review `tests/` for implementation examples

### Communication

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Code Review**: Participate in pull request reviews

### Development Tips

1. **Start Small**: Begin with small, focused contributions
2. **Ask Questions**: Don't hesitate to ask for clarification
3. **Read the Code**: Understand existing patterns before contributing
4. **Test Thoroughly**: Write comprehensive tests for your changes
5. **Document Well**: Good documentation helps everyone

---

## Code of Conduct

We expect all contributors to:

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming environment
- Follow the technical guidelines outlined here

Thank you for contributing to the Healthcare Length of Stay Prediction project! Your contributions help improve healthcare analytics and patient outcomes. 