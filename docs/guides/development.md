# Development Setup Guide

This guide will help you set up your development environment for contributing to Judicaita.

## Prerequisites

- Python 3.10 or higher
- Git
- A code editor (VS Code, PyCharm, etc.)
- Google Cloud account (for API access)

## Initial Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/judicAIta.git
cd judicAIta

# Add upstream remote
git remote add upstream https://github.com/clduab11/judicAIta.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### 4. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# Add your Google API key and other configuration
```

### 5. Verify Setup

```bash
# Run tests
make test

# Check formatting
make format

# Run linting
make lint

# Check types
make type-check
```

## Development Workflow

### Creating a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### Making Changes

1. Write code following our style guide
2. Add tests for new functionality
3. Update documentation as needed
4. Run tests frequently

```bash
# Run tests
make test

# Run specific test
pytest tests/unit/test_your_module.py -v

# Run with coverage
make test-cov
```

### Code Quality

Before committing, ensure code quality:

```bash
# Format code
make format

# Lint code
make lint

# Type check
make type-check

# Run all checks
make dev
```

### Committing Changes

```bash
# Stage changes
git add .

# Commit with conventional commit message
git commit -m "feat(module): add new feature"

# Push to your fork
git push origin feature/your-feature-name
```

### Creating Pull Request

1. Go to GitHub and create a Pull Request
2. Fill in the PR template
3. Link related issues
4. Wait for review

## IDE Setup

### VS Code

Recommended extensions:
- Python
- Pylance
- Black Formatter
- Ruff
- GitLens

Settings (`.vscode/settings.json`):
```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false
}
```

### PyCharm

1. Configure Python interpreter to use venv
2. Enable Black formatter in settings
3. Enable pytest as test runner
4. Install Ruff plugin

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=judicaita

# Specific test file
pytest tests/unit/test_citation_parser.py

# Specific test
pytest tests/unit/test_citation_parser.py::TestCitationParser::test_extract_citation

# With output
pytest -v -s
```

### Writing Tests

```python
# tests/unit/test_your_module.py
import pytest
from judicaita.your_module import YourClass

class TestYourClass:
    """Test suite for YourClass."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        instance = YourClass()
        result = instance.method()
        assert result == expected_value
    
    @pytest.mark.asyncio
    async def test_async_method(self):
        """Test async method."""
        instance = YourClass()
        result = await instance.async_method()
        assert result is not None
```

### Test Coverage

```bash
# Generate coverage report
make test-cov

# View HTML report
open htmlcov/index.html  # On Mac
xdg-open htmlcov/index.html  # On Linux
start htmlcov/index.html  # On Windows
```

## Documentation

### Building Documentation

```bash
# Build documentation
make docs

# Serve documentation locally
make docs-serve

# View at http://localhost:8000
```

### Writing Documentation

- Use Markdown for documentation files
- Follow Google-style docstrings
- Include code examples
- Keep documentation up-to-date with code

Example docstring:
```python
def process_document(file_path: Path, max_size: int) -> DocumentContent:
    """
    Process a document and extract its content.

    This function processes legal documents from various formats and
    extracts structured content including text, metadata, and citations.

    Args:
        file_path: Path to the document file
        max_size: Maximum file size in bytes

    Returns:
        DocumentContent: Extracted document content with metadata

    Raises:
        DocumentProcessingError: If processing fails
        DocumentTooLargeError: If file exceeds max_size

    Example:
        >>> service = DocumentInputService()
        >>> content = await service.process_document(Path("case.pdf"))
        >>> print(content.text[:100])

    Note:
        This function is async and must be awaited.
    """
```

## Debugging

### Using Python Debugger

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use built-in breakpoint()
breakpoint()
```

### Using VS Code Debugger

Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        },
        {
            "name": "Python: Pytest",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["-v"],
            "console": "integratedTerminal"
        }
    ]
}
```

### Debug Logging

```python
from loguru import logger

logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
```

## Docker Development

### Building Image

```bash
# Build Docker image
make docker-build

# Or directly
docker build -t judicaita:dev .
```

### Running with Docker Compose

```bash
# Start services
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

### Development with Docker

```bash
# Run with volume mount for live reload
docker-compose -f docker-compose.dev.yml up
```

## Common Tasks

### Adding New Dependencies

```bash
# Add to requirements.txt or pyproject.toml
# Then install
pip install -e ".[dev]"

# Update requirements
pip freeze > requirements-freeze.txt
```

### Updating Pre-commit Hooks

```bash
# Update hooks
pre-commit autoupdate

# Run on all files
pre-commit run --all-files
```

### Database Migrations

```bash
# Create migration (when DB schema changes)
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```

## Troubleshooting

### Import Errors

```bash
# Reinstall in editable mode
pip install -e .

# Check PYTHONPATH
echo $PYTHONPATH
```

### Test Failures

```bash
# Clear pytest cache
pytest --cache-clear

# Run with verbose output
pytest -vv

# Run failed tests only
pytest --lf
```

### Type Checking Issues

```bash
# Install type stubs
pip install types-requests types-redis

# Run mypy with more output
mypy --show-error-codes src/
```

## Performance Profiling

### Using cProfile

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Run with profiler
python -m memory_profiler your_script.py
```

## Resources

- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Type Hints (PEP 484)](https://www.python.org/dev/peps/pep-0484/)
- [pytest Documentation](https://docs.pytest.org/)
- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://beta.ruff.rs/docs/)

## Getting Help

- Check documentation in `docs/`
- Ask in GitHub Discussions
- Review existing issues and PRs
- Contact maintainers

---

Happy coding! 🚀
