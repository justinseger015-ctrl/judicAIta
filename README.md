# Judicaita 🏛️⚖️

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-AGPL%203.0-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**An Explainable Legal AI Assistant for Lawyers and Paralegals**

Judicaita is an AI companion built with **Google Tunix** and **Gemma3-1B-IT** for the Kaggle hackathon. It generates explainable legal reasoning, stepwise traces, citation mapping, plain-English summaries, and compliance audit logs from case files, ensuring transparent and efficient legal workflows...fitting all this into a tiny form factor that can go anywhere!

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

## ⚠️ Important Setup Notes

### TPU Training Dependencies (Critical)

> **🔴 ATTENTION KAGGLE HACKATHON PARTICIPANTS:** The training notebook was updated in **December 2025** to fix critical dependency issues. If you encounter `ModuleNotFoundError: No module named 'tunix'`, ensure you're using the latest notebook version.

| Package | Required Version | Notes |
|---------|------------------|-------|
| `google-tunix` | `>=0.1.0,<=0.1.5` | ⚠️ **NOT** 0.5.0+ (does not exist on PyPI) |
| `jax[tpu]` | TPU-compatible | Use official libtpu releases, **NOT** `jax==0.4.35` |
| `flax` | `0.10.2` | Compatible with JAX TPU builds |

**Common Pitfalls:**
- ❌ `pip install google-tunix>=0.5.0` → Package doesn't exist, causes `ModuleNotFoundError`
- ❌ `pip install jax==0.4.35 jaxlib==0.4.35` → Incompatible with Colab TPU runtime
- ✅ Use: `pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`

**Expected Warnings (Harmless):**
- `jax_cuda12_plugin` warnings are **normal** on Colab TPU and can be safely ignored
- These appear because Colab has GPU packages pre-installed alongside TPU runtime

For complete version compatibility details, see the [Version Compatibility Summary](examples/notebooks/train_tunix_reasoning.ipynb) in the training notebook.

## 🔧 Training & Fine-tuning

### Tunix/TPU Training (Kaggle Hackathon)

For training Gemma models with GRPO on Google Cloud TPU using the Tunix framework:

**Notebook:** [`examples/notebooks/train_tunix_reasoning.ipynb`](examples/notebooks/train_tunix_reasoning.ipynb)

> 📢 **Updated December 2025:** This notebook was recently updated to fix critical dependency installation errors. See [PR #13](https://github.com/clduab11/judicAIta/pull/13) for details.

This specialized training approach uses:
- **Framework:** JAX/Flax with Google Tunix (different from main PyTorch codebase)
- **Hardware:** TPU v2-8+ on Google Colab
- **Model:** Gemma 3-1B-IT with LoRA adapters
- **Format:** XML-tagged reasoning (`<reasoning>`/`<answer>`)
- **Method:** GRPO (Group Relative Policy Optimization)

**Dependency Requirements:**
```bash
# ✅ Correct installation (from notebook Step 1)
!pip install -q "google-tunix[tpu]>=0.1.0,<=0.1.5"
!pip install -q "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
!pip install -q flax==0.10.2

# ❌ Do NOT use these (outdated/incorrect)
# !pip install "google-tunix>=0.5.0"  # Version doesn't exist!
# !pip install jax==0.4.35 jaxlib==0.4.35  # Incompatible with Colab TPU
```

**Prerequisites:**
- Google Colab account with TPU access
- Hugging Face account for model downloads
- Kaggle account for submissions

**Quick Start:**
1. Open [`train_tunix_reasoning.ipynb`](examples/notebooks/train_tunix_reasoning.ipynb) in Colab
2. Set runtime to TPU (Runtime → Change runtime type → TPU)
3. Run Step 1 (dependencies) - expect `jax_cuda12_plugin` warnings (harmless)
4. **Restart runtime** when prompted
5. Continue with Step 2 onwards

**Quick Troubleshooting:**
| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'tunix'` | Wrong Tunix version | Use `>=0.1.0,<=0.1.5` |
| JAX TPU initialization fails | Wrong JAX version | Use `jax[tpu]` with libtpu releases |
| `jax_cuda12_plugin` warnings | Normal for Colab | Ignore - harmless for TPU |

See [examples/notebooks/README.md](examples/notebooks/README.md) for more training options including PyTorch-based GRPO training.

## 📚 Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [API Reference](docs/api/reference.md)
- [User Guide](docs/guides/user-guide.md)
- [GRPO Training Guide](docs/GRPO_TRAINING.md)
- [Tunix/TPU Training Notebook](examples/notebooks/train_tunix_reasoning.ipynb)
- [Contributing Guide](CONTRIBUTING.md)
- [Development Setup](docs/guides/development.md)

## 🐛 Known Issues & Troubleshooting

### TPU Training Notebook Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'tunix'` | Incorrect Tunix version | Install `google-tunix[tpu]>=0.1.0,<=0.1.5` (max is 0.1.5, **NOT** 0.5.0) |
| JAX TPU initialization fails | Incompatible JAX version | Use `pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html` |
| `RuntimeError: TPU not found` | Wrong Colab runtime | Set runtime to TPU: Runtime → Change runtime type → TPU |
| Imports fail after install | Runtime not restarted | Restart runtime after Step 1, then continue from Step 2 |

### Expected Warnings (Safe to Ignore)

- **`jax_cuda12_plugin` warnings**: Normal on Google Colab TPU runtime. These appear because Colab has GPU packages pre-installed. They do not affect TPU training.

### Version Constraints

**Google Tunix:**
- Maximum available version: **0.1.5** (as of December 2025)
- Do NOT use `>=0.5.0` - this version does not exist on PyPI
- Optimized for Kaggle's Google Tunix hackathon requirements

**JAX for TPU:**
- Use `jax[tpu]` with official libtpu releases
- JAX 0.4+ requires TPU VMs (not available on Colab)
- Colab TPU runtime requires TPU-specific builds

For comprehensive troubleshooting, see the **Troubleshooting Guide** section in [`train_tunix_reasoning.ipynb`](examples/notebooks/train_tunix_reasoning.ipynb).

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

- Built with [Google Tunix](https://tunix.google.com) (0.1.x series) and [Gemma 3n](https://ai.google.dev/gemma)
- Optimized for the **Kaggle Google Tunix Hackathon** requirements
- TPU training tested on Google Colab TPU runtime (note: JAX 0.4+ requires TPU VMs not available on Colab)
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
