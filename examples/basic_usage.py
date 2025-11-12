"""
Example: Basic usage of Judicaita document processing.

This script demonstrates how to process a legal document and extract
its content, citations, and metadata.
"""

import asyncio

from judicaita.audit_logs import AuditEventType, AuditLogger
from judicaita.citation_mapping import CitationMappingService
from judicaita.document_input import DocumentInputService


async def main():
    """Main example function."""
    print("=" * 60)
    print("Judicaita Document Processing Example")
    print("=" * 60)

    # Initialize services
    doc_service = DocumentInputService()
    citation_service = CitationMappingService()
    audit_logger = AuditLogger()

    # For this example, we'll create a sample text
    sample_text = """
    In Brown v. Board of Education, 347 U.S. 483 (1954), the Supreme Court
    held that state laws establishing separate public schools for black and
    white students were unconstitutional. This landmark decision overturned
    Plessy v. Ferguson, 163 U.S. 537 (1896).

    The decision was based on the Equal Protection Clause of the Fourteenth
    Amendment, as codified in 42 U.S.C. § 2000d. The Court concluded that
    "separate educational facilities are inherently unequal."
    """

    print("\n1. Processing sample legal text...")
    print("-" * 60)
    print(sample_text[:200] + "...")

    # Extract citations
    print("\n2. Extracting citations...")
    citations = await citation_service.extract_and_map_citations(sample_text)

    print(f"\nFound {len(citations)} citations:")
    for i, match in enumerate(citations, 1):
        citation = match.citation
        print(f"\n  Citation {i}:")
        print(f"    Text: {citation.raw_citation}")
        print(f"    Type: {citation.citation_type.value}")
        print(f"    Jurisdiction: {citation.jurisdiction.value}")
        if citation.case_name:
            print(f"    Case: {citation.case_name}")
        if citation.statute_section:
            print(f"    Section: {citation.statute_section}")

    # Log the operation
    print("\n3. Logging audit entry...")
    await audit_logger.log_event(
        event_type=AuditEventType.DOCUMENT_PROCESS,
        action="Processed sample legal text",
        status="success",
        details={
            "text_length": len(sample_text),
            "citations_found": len(citations),
        },
    )
    print("   Audit entry created successfully")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
