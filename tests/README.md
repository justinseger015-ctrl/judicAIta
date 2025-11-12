# Judicaita Tests

This directory contains the test suite for Judicaita.

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures and configuration
├── unit/                 # Unit tests (fast, isolated)
│   ├── test_config.py
│   ├── test_exceptions.py
│   └── test_citation_parser.py
├── integration/          # Integration tests (slower, with dependencies)
└── e2e/                  # End-to-end tests (full workflows)
```

## Running Tests

### All Tests

```bash
pytest
```

### With Coverage

```bash
pytest --cov=judicaita --cov-report=html
```

### Specific Test Types

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# End-to-end tests only
pytest tests/e2e/ -v
```

### Specific Test File

```bash
pytest tests/unit/test_citation_parser.py -v
```

### Specific Test

```bash
pytest tests/unit/test_citation_parser.py::TestCitationParser::test_extract_citation -v
```

### With Output

```bash
pytest -v -s
```

## Test Markers

Tests are marked with pytest markers for selective running:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Slow tests

Run tests by marker:

```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Run integration tests only
pytest -m integration
```

## Writing Tests

### Test File Naming

- Unit tests: `test_<module_name>.py`
- Place in appropriate directory (`unit/`, `integration/`, `e2e/`)

### Test Function Naming

```python
def test_<what_is_being_tested>():
    """Clear description of what the test does."""
    # Arrange
    # Act
    # Assert
```

### Using Fixtures

```python
def test_with_fixture(sample_legal_text):
    """Test using a fixture."""
    # sample_legal_text is provided by conftest.py
    assert len(sample_legal_text) > 0
```

### Async Tests

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await some_async_function()
    assert result is not None
```

### Test Example

```python
"""Test suite for citation parser."""

import pytest
from judicaita.citation_mapping.parser import CitationParser

class TestCitationParser:
    """Test citation parser functionality."""
    
    def test_extract_citation(self):
        """Test basic citation extraction."""
        # Arrange
        parser = CitationParser()
        text = "Brown v. Board of Education, 347 U.S. 483"
        
        # Act
        citations = parser.extract_citations(text)
        
        # Assert
        assert len(citations) == 1
        assert citations[0].citation.case_name == "Brown v. Board of Education"
    
    def test_extract_no_citations(self):
        """Test with text containing no citations."""
        parser = CitationParser()
        text = "Plain text with no citations"
        
        citations = parser.extract_citations(text)
        
        assert len(citations) == 0
```

## Test Coverage

### Viewing Coverage

```bash
# Run tests with coverage
pytest --cov=judicaita --cov-report=html

# Open coverage report
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Goals

- Overall: 80%+
- Core modules: 90%+
- Utilities: 85%+

## Fixtures

Common fixtures are defined in `conftest.py`:

- `temp_dir` - Temporary directory for test files
- `sample_legal_text` - Sample legal text for testing
- `sample_citation_text` - Text with citations
- `test_settings` - Test configuration
- `sample_pdf_path` - Path to test PDF
- `sample_docx_path` - Path to test Word document

### Using Fixtures

```python
def test_with_temp_dir(temp_dir):
    """Test that uses temporary directory."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()
```

## Mocking

### Mock External Services

```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Test with mocked dependency."""
    with patch('judicaita.module.ExternalService') as mock_service:
        mock_service.return_value.method.return_value = "mocked"
        result = function_using_service()
        assert result == "expected"
```

### Mock Async Functions

```python
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_async_with_mock():
    """Test async function with mock."""
    mock = AsyncMock(return_value="result")
    result = await mock()
    assert result == "result"
```

## Test Data

Test data should be:

- Small and focused
- Reproducible
- Version controlled (when appropriate)
- Located in `tests/data/` or created in fixtures

## Debugging Tests

### Run with Debug Output

```bash
pytest -vv -s
```

### Stop at First Failure

```bash
pytest -x
```

### Run Last Failed Tests

```bash
pytest --lf
```

### Use Python Debugger

```python
def test_with_debugger():
    """Test with debugger."""
    import pdb; pdb.set_trace()
    # Test code here
```

## Continuous Integration

Tests run automatically on:

- Every push to main/develop
- Every pull request
- Scheduled daily runs

See `.github/workflows/ci.yml` for CI configuration.

## Best Practices

1. **One assertion per test** (when possible)
2. **Clear test names** that describe what's being tested
3. **Use fixtures** for common setup
4. **Mock external dependencies** (APIs, databases, file system)
5. **Test edge cases** and error conditions
6. **Keep tests fast** - unit tests should run in milliseconds
7. **Test behavior, not implementation** - test what, not how
8. **Use appropriate markers** - mark slow/integration tests
9. **Write docstrings** for test classes and functions
10. **Follow AAA pattern** - Arrange, Act, Assert

## Contributing Tests

When adding new features:

1. Write tests first (TDD) or alongside code
2. Aim for high coverage of new code
3. Include unit tests for all functions
4. Add integration tests for workflows
5. Update fixtures as needed
6. Document any special test requirements

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

## Getting Help

- Check existing tests for examples
- Review [pytest documentation](https://docs.pytest.org/)
- Ask in GitHub Discussions
- Contact maintainers

---

Happy testing! 🧪
