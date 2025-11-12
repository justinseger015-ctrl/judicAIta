# Judicaita 🏛️⚖️

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**An Explainable Legal AI Assistant for Lawyers and Paralegals**

Judicaita is an AI companion built with **Google Tunix** and **Gemma 3n** for the Kaggle hackathon. It generates explainable legal reasoning, stepwise traces, citation mapping, plain-English summaries, and compliance audit logs from case files, ensuring transparent and efficient legal workflows.

## 🌟 Features

### Core Capabilities

- **📄 Document Processing**: Intelligent extraction from PDF, Word, and other legal document formats
- **🧠 Reasoning Trace Generation**: Step-by-step explainable AI reasoning for legal analysis
- **📚 Legal Citation Mapping**: Automatic citation extraction, validation, and relationship mapping
- **💬 Plain-English Summaries**: Convert complex legal text into accessible summaries at various reading levels
- **📊 Compliance Audit Logs**: Comprehensive audit trails for transparency and accountability

### Key Advantages

- ✅ **Explainable AI**: Every decision includes transparent reasoning traces
- ✅ **Citation Accuracy**: Automated citation validation and mapping
- ✅ **Accessibility**: Plain-English summaries make legal content accessible to all
- ✅ **Compliance-First**: Built-in audit logging for regulatory compliance
- ✅ **Production-Ready**: Modern architecture following 2025 best practices

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Google API key for Tunix and Gemma access

### Installation

1. Clone the repository:
```bash
git clone https://github.com/clduab11/judicAIta.git
cd judicAIta
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
# Or for development:
pip install -e ".[dev]"
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your Google API key
```

### Usage

#### Command Line Interface

Process a legal document:
```bash
judicaita process-document /path/to/document.pdf --output ./results
```

Analyze a legal query:
```bash
judicaita analyze-query "What is the precedent for contract breach in California?"
```

Generate audit report:
```bash
judicaita audit-report --days 30 --output report.md
```

Start API server:
```bash
judicaita serve --host 0.0.0.0 --port 8000
```

#### Python API

```python
from judicaita.document_input import DocumentInputService
from judicaita.reasoning_trace import ReasoningTraceGenerator
from judicaita.citation_mapping import CitationMappingService
from judicaita.summary_generator import SummaryGenerator
from judicaita.audit_logs import AuditLogger

# Process a document
doc_service = DocumentInputService()
document = await doc_service.process_document("case.pdf")

# Generate reasoning trace
trace_gen = ReasoningTraceGenerator()
await trace_gen.initialize()
trace = await trace_gen.generate_trace(
    query="Analyze this case",
    context=document.text
)

# Extract and map citations
citation_service = CitationMappingService()
citations = await citation_service.extract_and_map_citations(document.text)

# Generate plain-English summary
summary_gen = SummaryGenerator()
await summary_gen.initialize()
summary = await summary_gen.generate_summary(
    document.text,
    summary_level="medium",
    reading_level="high_school"
)

# Log for audit
audit_logger = AuditLogger()
await audit_logger.log_event(
    event_type="document_process",
    action="Processed legal document",
    status="success"
)
```

## 📁 Project Structure

```
judicAIta/
├── src/judicaita/           # Main package
│   ├── document_input/      # Document processing (PDF, Word)
│   ├── reasoning_trace/     # Explainable reasoning generation
│   ├── citation_mapping/    # Citation extraction and validation
│   ├── summary_generator/   # Plain-English summaries
│   ├── audit_logs/          # Compliance audit logging
│   ├── core/                # Core configuration and exceptions
│   ├── utils/               # Utility functions
│   └── cli.py               # Command-line interface
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-end tests
├── docs/                    # Documentation
│   ├── api/                # API documentation
│   ├── guides/             # User guides
│   └── architecture/       # Architecture docs
├── examples/               # Example scripts and notebooks
│   ├── notebooks/          # Jupyter notebooks
│   └── sample_data/        # Sample legal documents
├── config/                 # Configuration files
├── pyproject.toml          # Project configuration
├── requirements.txt        # Production dependencies
└── README.md              # This file
```

## 🔧 Configuration

Judicaita uses environment variables for configuration. Key settings:

- `GOOGLE_API_KEY`: Your Google API key for Tunix/Gemma
- `GEMMA_MODEL_NAME`: Model name (default: gemma-3n)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `AUDIT_LOG_ENABLED`: Enable compliance audit logging
- `CACHE_ENABLED`: Enable caching for performance

See `.env.example` for all configuration options.

## 📚 Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [API Reference](docs/api/reference.md)
- [User Guide](docs/guides/user-guide.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Development Setup](docs/guides/development.md)

## 🧪 Testing

Run tests:
```bash
# All tests
pytest

# With coverage
pytest --cov=judicaita

# Specific test type
pytest -m unit
pytest -m integration
```

## 🛡️ Security & Compliance

Judicaita takes security and compliance seriously:

- **Audit Logging**: All operations are logged for compliance
- **Data Retention**: Configurable retention policies
- **Access Control**: Role-based access controls (planned)
- **Encryption**: Data encryption at rest and in transit (planned)

See [SECURITY.md](docs/SECURITY.md) for security policy and reporting vulnerabilities.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Google Tunix](https://tunix.google.com) and [Gemma 3n](https://ai.google.dev/gemma)
- Developed for the Kaggle Hackathon
- Inspired by the legal tech community's commitment to access to justice

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/clduab11/judicAIta/issues)
- **Discussions**: [GitHub Discussions](https://github.com/clduab11/judicAIta/discussions)

## 🗺️ Roadmap

- [x] Core document processing
- [x] Reasoning trace generation
- [x] Citation mapping
- [x] Plain-English summaries
- [x] Audit logging
- [ ] API server implementation
- [ ] Web UI dashboard
- [ ] Multi-language support
- [ ] Advanced citation databases
- [ ] Real-time collaboration features

---

Made with ❤️ for the legal community
