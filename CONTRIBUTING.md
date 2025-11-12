# Contributing to Judicaita

Thank you for your interest in contributing to Judicaita! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/judicAIta.git`
3. Add upstream remote: `git remote add upstream https://github.com/clduab11/judicAIta.git`
4. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv or conda)

### Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

4. Set up environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-citation-parser`
- `fix/document-processing-bug`
- `docs/update-readme`
- `refactor/simplify-audit-logger`

### Commit Messages

Follow conventional commit format:
```
type(scope): brief description

Longer description if needed

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(citation): add support for EU case citations
fix(pdf): handle corrupted PDF files gracefully
docs(api): add examples for summary generator
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=judicaita --cov-report=html

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m e2e

# Run specific test file
pytest tests/unit/test_citation_parser.py
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Place end-to-end tests in `tests/e2e/`
- Name test files with `test_` prefix
- Use descriptive test names: `test_pdf_processor_handles_large_files`
- Include docstrings explaining what is being tested

Example test:
```python
def test_citation_parser_extracts_us_case():
    """Test that CitationParser correctly extracts US case citations."""
    parser = CitationParser()
    text = "Brown v. Board of Education, 347 U.S. 483"
    citations = parser.extract_citations(text)
    
    assert len(citations) == 1
    assert citations[0].citation.case_name == "Brown v. Board of Education"
```

## Code Style

We follow strict code quality standards:

### Python Style

- Follow PEP 8
- Use type hints for all functions
- Maximum line length: 100 characters
- Use docstrings for all public functions/classes

### Formatting

We use automated tools:
- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking

Run formatters:
```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

Pre-commit hooks run these automatically.

### Documentation

- Use Google-style docstrings
- Include type hints
- Add examples for complex functions
- Update relevant documentation in `docs/`

Example docstring:
```python
def process_document(file_path: Path, max_size: int = 1024) -> DocumentContent:
    """
    Process a document and extract its content.

    Args:
        file_path: Path to the document file
        max_size: Maximum file size in KB

    Returns:
        DocumentContent: Extracted document content with metadata

    Raises:
        DocumentProcessingError: If processing fails
        DocumentTooLargeError: If file exceeds max_size

    Example:
        >>> service = DocumentInputService()
        >>> content = await service.process_document(Path("case.pdf"))
        >>> print(content.text[:100])
    """
```

## Submitting Changes

### Pull Request Process

1. Update your fork:
```bash
git fetch upstream
git rebase upstream/main
```

2. Push your changes:
```bash
git push origin feature/your-feature-name
```

3. Create Pull Request:
   - Go to GitHub and create a PR from your fork
   - Fill in the PR template
   - Link related issues
   - Request review from maintainers

### PR Requirements

- [ ] Tests pass (`pytest`)
- [ ] Code is formatted (`black`, `ruff`)
- [ ] Type checks pass (`mypy`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (for significant changes)
- [ ] Commits follow conventional commit format
- [ ] PR description explains the changes

### Review Process

- Maintainers will review your PR
- Address any requested changes
- Once approved, maintainers will merge

## Reporting Issues

### Bug Reports

Include:
- Clear, descriptive title
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Error messages/logs
- Code samples if applicable

### Feature Requests

Include:
- Clear description of the feature
- Use cases and motivation
- Proposed implementation (if any)
- Alternatives considered

### Security Issues

**Do not open public issues for security vulnerabilities.**

Email security issues to: security@example.com (or see SECURITY.md)

## Development Guidelines

### Adding New Features

1. Check existing issues/PRs for similar work
2. Open an issue to discuss the feature
3. Wait for maintainer approval before starting work
4. Follow the development process above
5. Add comprehensive tests
6. Update documentation

### Working with Models

When adding model integration:
- Use configuration for model parameters
- Add proper error handling
- Include logging for debugging
- Test with mock models first
- Document model requirements

### Performance Considerations

- Profile code for bottlenecks
- Use async/await for I/O operations
- Cache expensive operations when appropriate
- Consider memory usage for large documents

## Getting Help

- Check [documentation](docs/)
- Search existing [issues](https://github.com/clduab11/judicAIta/issues)
- Ask in [discussions](https://github.com/clduab11/judicAIta/discussions)
- Reach out to maintainers

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in commit history

Thank you for contributing to Judicaita! 🙏
