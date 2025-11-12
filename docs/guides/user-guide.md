# Judicaita User Guide

Welcome to Judicaita, your AI-powered legal assistant! This guide will help you get started with using Judicaita effectively.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Processing Documents](#processing-documents)
5. [Analyzing Legal Questions](#analyzing-legal-questions)
6. [Working with Citations](#working-with-citations)
7. [Generating Summaries](#generating-summaries)
8. [Compliance and Auditing](#compliance-and-auditing)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Introduction

Judicaita is an explainable legal AI assistant designed to help lawyers and paralegals with:

- **Document Processing**: Extract and structure information from legal documents
- **Legal Reasoning**: Generate transparent, step-by-step reasoning traces
- **Citation Management**: Extract, validate, and map legal citations
- **Plain-English Summaries**: Make complex legal text accessible
- **Compliance**: Maintain audit trails for transparency

### Key Benefits

✅ **Explainable AI**: Every decision includes transparent reasoning  
✅ **Time-Saving**: Automate document processing and analysis  
✅ **Accuracy**: Automated citation validation and verification  
✅ **Accessibility**: Plain-English summaries for any audience  
✅ **Compliance**: Built-in audit logging for regulatory requirements

## Installation

### System Requirements

- Python 3.10 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection for API access

### Step-by-Step Installation

1. **Install Python** (if not already installed):
   ```bash
   # Check Python version
   python --version
   
   # Should be 3.10 or higher
   ```

2. **Install Judicaita**:
   ```bash
   pip install judicaita
   ```

3. **Set up configuration**:
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env and add your Google API key
   nano .env
   ```

4. **Verify installation**:
   ```bash
   judicaita version
   ```

## Quick Start

### Command Line Interface

Process your first document:

```bash
# Process a legal document
judicaita process-document /path/to/document.pdf

# Analyze a legal query
judicaita analyze-query "What is the precedent for contract breach?"

# Generate audit report
judicaita audit-report --days 7
```

### Python API

```python
import asyncio
from judicaita.document_input import DocumentInputService

async def main():
    service = DocumentInputService()
    document = await service.process_document("case.pdf")
    print(f"Processed {len(document.text)} characters")

asyncio.run(main())
```

## Processing Documents

### Supported Formats

- **PDF** (.pdf)
- **Word** (.docx, .doc)
- **Plain Text** (.txt) - planned

### Basic Document Processing

```python
from pathlib import Path
from judicaita.document_input import DocumentInputService

# Initialize service
service = DocumentInputService(max_size_bytes=50*1024*1024)  # 50MB limit

# Process document
document = await service.process_document(Path("legal_brief.pdf"))

# Access extracted content
print(document.text)
print(document.metadata.title)
print(f"Found {len(document.citations)} citations")
```

### Working with Document Content

```python
# Access metadata
metadata = document.metadata
print(f"Author: {metadata.author}")
print(f"Pages: {metadata.page_count}")
print(f"Created: {metadata.created_date}")

# Access sections
for section in document.sections:
    print(f"Page {section['page']}: {section['text'][:100]}...")

# Access tables
for table in document.tables:
    print(f"Table on page {table['page']}")
    # Process table data
    for row in table['data']:
        print(row)
```

### Handling Large Documents

For documents over 10MB:

```python
# Use larger size limit
service = DocumentInputService(max_size_bytes=100*1024*1024)

# Process in chunks if needed
from judicaita.utils import chunk_text

chunks = chunk_text(document.text, chunk_size=5000, overlap=500)
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}")
    # Process each chunk
```

## Analyzing Legal Questions

### Creating Reasoning Traces

```python
from judicaita.reasoning_trace import ReasoningTraceGenerator

# Initialize generator
generator = ReasoningTraceGenerator()
await generator.initialize()

# Generate trace
trace = await generator.generate_trace(
    query="Does the plaintiff have standing to sue?",
    context=document.text,
    citations=document.citations
)

# Review reasoning steps
for step in trace.steps:
    print(f"{step.step_type}: {step.description}")
    print(f"Confidence: {step.confidence_score:.2%}")

# Get conclusion
print(f"Conclusion: {trace.final_conclusion}")
```

### Exporting Traces

```python
# Export to markdown
markdown = trace.to_markdown()
with open("reasoning_trace.md", "w") as f:
    f.write(markdown)

# Access individual steps
analysis_steps = trace.get_steps_by_type("analysis")
for step in analysis_steps:
    print(step.description)
```

## Working with Citations

### Extracting Citations

```python
from judicaita.citation_mapping import CitationMappingService

service = CitationMappingService()

# Extract from text
citations = await service.extract_and_map_citations(
    text=document.text,
    validate=True  # Validate against databases
)

# Review citations
for match in citations:
    citation = match.citation
    print(f"Citation: {citation.raw_citation}")
    print(f"Type: {citation.citation_type.value}")
    print(f"Valid: {citation.is_valid}")
```

### Validating Individual Citations

```python
# Validate a single citation
citation = await service.validate_citation("347 U.S. 483")

if citation:
    print(f"Valid: {citation.raw_citation}")
    print(f"Type: {citation.citation_type.value}")
    print(f"URL: {citation.url}")
```

### Building Citation Graphs

```python
# Build relationship graph
citations_list = [match.citation for match in citations]
graph = await service.build_citation_graph(citations_list)

# Explore relationships
for citation in graph.nodes:
    related = graph.get_related_citations(citation)
    print(f"{citation.raw_citation} is related to {len(related)} citations")
```

## Generating Summaries

### Basic Summaries

```python
from judicaita.summary_generator import (
    SummaryGenerator,
    SummaryLevel,
    ReadingLevel
)

generator = SummaryGenerator()
await generator.initialize()

# Generate summary
summary = await generator.generate_summary(
    text=document.text,
    summary_level=SummaryLevel.MEDIUM,
    reading_level=ReadingLevel.HIGH_SCHOOL
)

print(summary.summary)
```

### Advanced Summary Options

```python
# Detailed summary with sections
summary = await generator.generate_summary(
    text=document.text,
    summary_level=SummaryLevel.DETAILED,
    reading_level=ReadingLevel.COLLEGE,
    include_sections=True
)

# Access sections
for section in summary.sections:
    print(f"\n{section.title}")
    print(section.content)
    for point in section.key_points:
        print(f"  - {point}")

# Access key terms
for term, definition in summary.key_terms.items():
    print(f"{term}: {definition}")
```

### Different Reading Levels

```python
# For general public
simple = await generator.generate_summary(
    text=legal_text,
    reading_level=ReadingLevel.MIDDLE_SCHOOL
)

# For legal professionals
detailed = await generator.generate_summary(
    text=legal_text,
    reading_level=ReadingLevel.PROFESSIONAL
)
```

## Compliance and Auditing

### Logging Events

```python
from judicaita.audit_logs import (
    AuditLogger,
    AuditEventType,
    AuditSeverity,
    ComplianceStatus
)

logger = AuditLogger()

# Log an operation
entry = await logger.log_event(
    event_type=AuditEventType.DOCUMENT_PROCESS,
    action="Processed legal brief",
    user_id="user123",
    severity=AuditSeverity.INFO,
    status="success",
    details={"document": "brief.pdf"}
)
```

### Querying Logs

```python
from datetime import datetime, timedelta

# Query recent logs
logs = await logger.query_logs(
    start_date=datetime.now() - timedelta(days=7),
    event_types=[AuditEventType.DOCUMENT_PROCESS]
)

for entry in logs:
    print(f"{entry.timestamp}: {entry.action}")
```

### Generating Reports

```python
# Generate compliance report
report = await logger.generate_report(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)

print(f"Total events: {report.total_events}")
print(f"Violations: {len(report.compliance_violations)}")

# Export report
markdown = report.to_markdown()
with open("compliance_report.md", "w") as f:
    f.write(markdown)
```

## Best Practices

### Document Processing

1. **Validate before processing**: Check file size and format
2. **Handle errors gracefully**: Use try-except blocks
3. **Log all operations**: Enable audit logging
4. **Cache results**: Store processed documents for reuse
5. **Clean up**: Delete temporary files after processing

### Working with AI Models

1. **Provide context**: More context = better results
2. **Review outputs**: Always review AI-generated content
3. **Use confidence scores**: Consider confidence when making decisions
4. **Iterate**: Refine queries based on results
5. **Combine methods**: Use multiple analysis approaches

### Security

1. **Protect API keys**: Never commit keys to version control
2. **Validate inputs**: Always validate user inputs
3. **Enable audit logs**: Track all operations
4. **Regular backups**: Backup important data
5. **Keep updated**: Update Judicaita regularly

## Troubleshooting

### Common Issues

#### Installation Problems

**Problem**: `pip install judicaita` fails

**Solution**:
```bash
# Update pip
pip install --upgrade pip

# Install with verbose output
pip install -v judicaita
```

#### API Key Issues

**Problem**: "Invalid API key" error

**Solution**:
1. Check `.env` file exists and contains key
2. Verify key is valid on Google Cloud Console
3. Restart application after updating key

#### Document Processing Fails

**Problem**: PDF processing throws error

**Solution**:
1. Check file is valid PDF
2. Verify file size is within limits
3. Try with different PDF if corrupted

#### Model Not Responding

**Problem**: Reasoning trace generation hangs

**Solution**:
1. Check internet connection
2. Verify API quota/limits
3. Check model service status

### Getting Help

- **Documentation**: Check docs/ directory
- **Issues**: https://github.com/clduab11/judicAIta/issues
- **Discussions**: https://github.com/clduab11/judicAIta/discussions

### Debug Mode

Enable debug logging:

```python
from judicaita.core import Settings

settings = Settings(
    debug=True,
    log_level="DEBUG"
)
```

Or via environment:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
```

## Next Steps

1. **Explore Examples**: Check `examples/` directory
2. **Read API Docs**: See `docs/api/reference.md`
3. **Join Community**: Participate in discussions
4. **Contribute**: See `CONTRIBUTING.md`

---

For more information, visit the [documentation](https://github.com/clduab11/judicAIta/docs).
