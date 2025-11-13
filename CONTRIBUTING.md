# Contributing to JudicAIta

Thank you for your interest in contributing to JudicAIta! This document provides guidelines and instructions for contributing to the project.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Process](#development-process)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Submitting Changes](#submitting-changes)
7. [Issue Guidelines](#issue-guidelines)
8. [Pull Request Process](#pull-request-process)
9. [Community](#community)

---

## Code of Conduct

### Our Pledge

We pledge to make participation in JudicAIta a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include:**
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by opening an issue or contacting the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

---

## Getting Started

### Prerequisites

- **Python**: 3.10 or higher
- **Git**: For version control
- **Kaggle Account**: Optional, for TPU training

### Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/judicAIta.git
cd judicAIta

# Add upstream remote
git remote add upstream https://github.com/judicaita/judicAIta.git
```

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (editable mode)
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify setup
python -c "import src; print('Setup successful!')"
```

---

## Development Process

### Branching Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `fix/*`: Bug fixes
- `refactor/*`: Code refactoring
- `docs/*`: Documentation updates

### Workflow

1. **Sync with upstream**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make changes and commit**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

5. **Open Pull Request** on GitHub

---

## Coding Standards

### Python

#### Style Guide
- Follow **PEP 8** conventions
- Use **black** for formatting (line length: 100)
- Use **ruff** for linting
- Use **mypy** for type checking

#### Formatting
```bash
# Format code
black src/ tests/

# Check with ruff
ruff check src/ tests/

# Type check
mypy src/
```

#### Naming Conventions
- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

#### Docstrings
Use **Google-style docstrings**:

```python
def extract_clauses(document: str, clause_types: list[str]) -> dict[str, list[str]]:
    """Extract specific clauses from a legal document.

    Args:
        document: The full text of the legal document.
        clause_types: List of clause types to extract (e.g., ['indemnity', 'termination']).

    Returns:
        Dictionary mapping clause types to extracted text segments.

    Raises:
        ValueError: If document is empty or clause_types is invalid.

    Example:
        >>> doc = "This agreement... termination clause..."
        >>> extract_clauses(doc, ["termination"])
        {'termination': ['...termination clause text...']}
    """
    pass
```

#### Type Hints
Always use type hints for function signatures:

```python
from typing import Optional, List, Dict, Any

def process_document(
    file_path: str,
    extract_entities: bool = True,
    max_length: Optional[int] = None
) -> Dict[str, Any]:
    """Process a legal document."""
    pass
```

### TypeScript

#### Style Guide
- Follow **Airbnb TypeScript** style
- Use **Prettier** for formatting
- Use **ESLint** for linting

#### Naming Conventions
- **Functions/variables**: `camelCase`
- **Components/classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `#privateField` (native private)

---

## Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests (individual functions/classes)
├── integration/       # Integration tests (multiple components)
└── e2e/              # End-to-end tests (full workflows)
```

### Writing Tests

#### Unit Tests

```python
# tests/unit/test_citation.py
import pytest
from src.legal_utils.citation import validate_bluebook_citation

def test_validate_bluebook_citation_valid():
    """Test validation of valid Bluebook citation."""
    citation = "Brown v. Board of Educ., 347 U.S. 483 (1954)"
    assert validate_bluebook_citation(citation) is True

def test_validate_bluebook_citation_invalid():
    """Test validation of invalid citation."""
    citation = "Brown v Board of Education"
    assert validate_bluebook_citation(citation) is False

@pytest.mark.parametrize("citation,expected", [
    ("Roe v. Wade, 410 U.S. 113 (1973)", True),
    ("Invalid citation", False),
])
def test_validate_bluebook_citation_parametrized(citation, expected):
    """Parametrized test for citation validation."""
    assert validate_bluebook_citation(citation) == expected
```

#### Integration Tests

```python
# tests/integration/test_contract_workflow.py
import pytest
from src.agents.orchestration import ContractReviewWorkflow

@pytest.mark.asyncio
async def test_contract_review_workflow():
    """Test complete contract review workflow."""
    workflow = ContractReviewWorkflow(model="google/gemma-3-1b")

    result = await workflow.execute({
        "file_path": "tests/fixtures/sample_contract.pdf"
    })

    # Assertions
    assert "parties" in result
    assert len(result["parties"]) >= 2
    assert "clauses" in result
    assert "summary" in result
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test file
pytest tests/unit/test_citation.py

# Run specific test
pytest tests/unit/test_citation.py::test_validate_bluebook_citation_valid

# Run tests matching pattern
pytest -k "citation"
```

### Coverage Requirements

- **Minimum coverage**: 80%
- **Critical paths**: 100% coverage required
- **New code**: Must maintain or improve coverage

```bash
# Check coverage
pytest --cov=src --cov-report=term-missing tests/

# Generate HTML report
pytest --cov=src --cov-report=html tests/
open htmlcov/index.html
```

---

## Submitting Changes

### Commit Messages

Use **Conventional Commits** format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(agents): add contract parsing agent with clause extraction

- Implement ContractParsingAgent class
- Add clause type detection using spaCy NER
- Support PDF and DOCX input formats

Closes #12
```

```
fix(citation): correct Bluebook format validation regex

The previous regex failed to match citations with multiple reporters.
Updated to handle cases like "Roe v. Wade, 410 U.S. 113, 93 S. Ct. 705 (1973)".

Fixes #45
```

### Pre-commit Checks

Before committing, ensure:

1. **Code is formatted**
   ```bash
   black src/ tests/
   ```

2. **Linting passes**
   ```bash
   ruff check src/ tests/
   ```

3. **Type checking passes**
   ```bash
   mypy src/
   ```

4. **Tests pass**
   ```bash
   pytest tests/
   ```

5. **Coverage is maintained**
   ```bash
   pytest --cov=src tests/
   ```

### Pre-commit Hooks

We use pre-commit hooks to automate checks:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Issue Guidelines

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check documentation** for answers
3. **Try latest version** to see if issue is fixed

### Creating an Issue

Use issue templates when available:

#### Bug Report

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.8]
- JudicAIta version: [e.g., 0.1.0]

**Additional context**
Add any other context about the problem.
```

#### Feature Request

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Additional context**
Any other context or screenshots.
```

---

## Pull Request Process

### Before Submitting

1. **Update documentation** if needed
2. **Add tests** for new features
3. **Update CHANGELOG.md** (if applicable)
4. **Ensure CI passes** (tests, linting, type checking)

### PR Template

```markdown
## Description

Brief description of changes.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?

Describe the tests you ran.

## Checklist

- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally
- [ ] Any dependent changes have been merged and published

## Related Issues

Closes #<issue_number>
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by at least one maintainer
3. **Address feedback** and update PR
4. **Approval** from maintainer
5. **Merge** (squash and merge preferred)

### After Merge

- Delete your feature branch
- Close related issues
- Update local repository

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Kaggle Forum**: Competition-related discussions

### Getting Help

- Check [README.md](README.md) for setup and usage
- Read [CLAUDE.md](CLAUDE.md) for project conventions
- Search existing issues and discussions
- Ask in GitHub Discussions

### Recognition

Contributors are recognized in:
- GitHub contributors page
- CHANGELOG.md
- Annual contributor acknowledgments

---

## Legal

### License

By contributing to JudicAIta, you agree that your contributions will be licensed under the **AGPL-3.0 License**.

### Copyright

- You retain copyright to your contributions
- You grant JudicAIta a perpetual, worldwide, non-exclusive, royalty-free license to use your contributions

### Attribution

- Contributors are credited in commits and release notes
- Significant contributions are acknowledged in documentation

---

## Thank You!

Thank you for contributing to JudicAIta and helping build better tools for the legal community!

---

**Questions?** Open an issue or discussion on [GitHub](https://github.com/judicaita/judicAIta).

**Last Updated**: 2025-11-13
