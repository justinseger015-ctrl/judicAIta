"""
Test configuration and fixtures for Judicaita tests.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from judicaita.core.config import Settings


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_legal_text() -> str:
    """Sample legal text for testing."""
    return """
    In the matter of Brown v. Board of Education, 347 U.S. 483 (1954),
    the Supreme Court held that state laws establishing separate public schools
    for black and white students were unconstitutional. This decision overturned
    Plessy v. Ferguson, 163 U.S. 537 (1896), which had established the
    "separate but equal" doctrine.

    The Court's decision was based on the Equal Protection Clause of the
    Fourteenth Amendment. Chief Justice Warren wrote: "We conclude that in the
    field of public education the doctrine of 'separate but equal' has no place.
    Separate educational facilities are inherently unequal."

    This case is cited in numerous subsequent decisions, including 42 U.S.C. § 2000d,
    which prohibits discrimination in federally funded programs.
    """


@pytest.fixture
def sample_citation_text() -> str:
    """Sample text with various citation formats."""
    return """
    The plaintiff cites Brown v. Board of Education, 347 U.S. 483 (1954) in support
    of their argument. Additionally, 42 U.S.C. § 1983 provides a remedy for
    constitutional violations. See also Miranda v. Arizona, 384 U.S. 436 (1966)
    and 18 U.S.C. 1001 regarding false statements.
    """


@pytest.fixture
def test_settings() -> Settings:
    """Test settings with safe defaults."""
    return Settings(
        google_api_key="test-key",
        debug=True,
        audit_log_enabled=False,
        cache_enabled=False,
    )


@pytest.fixture
def sample_pdf_path(temp_dir: Path) -> Path:
    """Create a sample PDF file for testing."""
    pdf_path = temp_dir / "test.pdf"
    # Create a minimal PDF (this won't be valid, just for path testing)
    pdf_path.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    return pdf_path


@pytest.fixture
def sample_docx_path(temp_dir: Path) -> Path:
    """Create a sample Word document for testing."""
    docx_path = temp_dir / "test.docx"
    # Just create an empty file for path testing
    docx_path.touch()
    return docx_path
