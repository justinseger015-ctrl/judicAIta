# Judicaita 🏛️⚖️

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-AGPL%203.0-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**An Explainable Legal AI Assistant for Lawyers and Paralegals**

Judicaita is an AI companion built with **Google Tunix** and **Gemma3-1B-IT** for the Kaggle hackathon. It generates explainable legal reasoning, stepwise traces, citation mapping, plain-English summaries, and compliance audit logs from case files, ensuring transparent and efficient legal workflows...fitting all this into a tiny form factor that can go anywhere!

<img width="2560" height="1440" alt="VeniceAI_sNE72d5_@2x" src="https://github.com/user-attachments/assets/0eab368e-69b0-4b35-8715-7252a04d2301" />

## 🏆 Kaggle Google TuniX Hackathon

This project is developed for the [Google TuniX Hackathon on Kaggle](https://www.kaggle.com/competitions/google-tunix-hackathon). JudicAIta demonstrates GRPO (Group Relative Policy Optimization) training for legal AI, producing explainable reasoning traces in structured XML format. The system fine-tunes Google Gemma models on legal reasoning tasks to generate transparent, auditable legal analysis.

**Submission Deadline**: January 12, 2026

### Competition Submission

- **Primary Training Notebook**: [`examples/notebooks/train_tunix_reasoning.ipynb`](examples/notebooks/train_tunix_reasoning.ipynb)
- **Validation Guide**: [`docs/COLAB_VALIDATION_GUIDE.md`](docs/COLAB_VALIDATION_GUIDE.md) - 4-phase validation process
- **Submission Checklist**: [`docs/HACKATHON_SUBMISSION_CHECKLIST.md`](docs/HACKATHON_SUBMISSION_CHECKLIST.md)
- **Submission Record**: [`docs/SUBMISSION_RECORD.md`](docs/SUBMISSION_RECORD.md)
- **Technical Writeup**: [`docs/hackathon_writeup.md`](docs/hackathon_writeup.md)

### Reward Function

JudicAIta uses a multi-objective reward function with the following weights:

| Component | Weight | Description |
|-----------|--------|-------------|
| Correctness | 40% | Accuracy of final legal conclusion |
| Reasoning Quality | 30% | Structured, logical step-by-step reasoning |
| Citation Accuracy | 20% | Proper citation format and relevance |
| Clarity | 10% | Readability and accessibility |

The reward function uses GRPO (Group Relative Policy Optimization) for memory-efficient training on TPU. See [`src/judicaita/training/rewards.py`](src/judicaita/training/rewards.py) and [`docs/hackathon_writeup.md`](docs/hackathon_writeup.md) for implementation details.

## 🌟 Features

### Document Processing
- **PDF & Word Processing**: Extract text from legal documents in PDF and DOCX formats
- **Citation Extraction**: Identify and parse legal citations (U.S. Code, case law, regulations)

### Legal Analysis
- **Reasoning Trace Generation**: Generate XML-structured reasoning with `<reasoning>` and `<answer>` tags
- **Citation Validation**: Validate citation format and structure
- **Query Analysis**: Analyze legal queries with context-aware responses

### Training & Evaluation
- **GRPO Training**: Train models using Group Relative Policy Optimization on TPU
- **Multi-objective Rewards**: Composite reward function with correctness, reasoning, citation, and clarity components
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning of Gemma models
- **Model Evaluation**: Evaluate trained checkpoints on legal reasoning tasks

### Infrastructure
- **CLI Interface**: Seven commands: `process_document`, `analyze_query`, `audit_report`, `validate_citation`, `serve`, `train_grpo`, `evaluate_model`
- **Docker Support**: Three-service architecture with PostgreSQL and Redis
- **Configuration**: Environment-based configuration via `.env` files

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

## 🐳 Docker Setup

Judicaita provides a three-service Docker architecture for development:

| Service | Image | Purpose |
|---------|-------|---------|
| Judicaita App | `judicaita:latest` | Main application container |
| PostgreSQL 16 | `postgres:16-alpine` | Database storage |
| Redis 7 | `redis:7-alpine` | Caching layer |

### Docker Commands

```bash
# Build Docker image
make docker-build

# Start all services
make docker-up

# Stop all services
make docker-down

# View logs
make docker-logs
```

**⚠️ Security Warning**: The `docker-compose.yml` contains hardcoded credentials (`POSTGRES_PASSWORD=password`) for development convenience only. **Do not use in production** - use environment variables or a secret management tool like Docker secrets, HashiCorp Vault, or AWS Secrets Manager.

The Docker configuration includes volume mounts for `./data`, `./logs`, and `./uploads` directories for development.

## 📖 Usage

### Command Line Interface

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

### Python API

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

> **🔴 ATTENTION KAGGLE HACKATHON PARTICIPANTS:** The training notebook was updated in **December 2025** with comprehensive validation cells for end-to-end submission readiness. If you encounter any issues, refer to the detailed validation guide.

| Package | Required Version | Notes |
|---------|------------------|-------|
| `google-tunix` | `0.1.0 - 0.1.6` | Max version: 0.1.5 (Dec 2025), **NOT** 0.5.0+ |
| `jax` | TPU-compatible (0.8.x) | Use `jax[tpu]` with libtpu releases |
| `flax` | `0.10.2` or `0.12.x` | Compatible with JAX TPU builds |
| `transformers` | `>=4.40.0,<=4.57.1` | For Gemma model support |

**Common Pitfalls:**
- ❌ `pip install google-tunix>=0.5.0` → Version doesn't exist, causes `ModuleNotFoundError`
- ❌ `pip install jax==0.4.35 jaxlib==0.4.35` → Incompatible with Colab TPU runtime
- ✅ Use: `pip install git+https://github.com/google/tunix` (recommended)
- ✅ Use: `pip install git+https://github.com/jax-ml/jax` for latest TPU support

**Expected Warnings (Harmless):**
- `jax_cuda12_plugin` warnings are **normal** on Colab TPU and can be safely ignored
- These appear because Colab has GPU packages pre-installed alongside TPU runtime

### 📋 Complete Validation Guide (4 Phases)

**NEW:** Comprehensive validation guide covering all phases from environment setup to submission:

**[📖 Complete Validation Guide](docs/COLAB_VALIDATION_GUIDE.md)** - Detailed validation procedures including:

1. **Phase 1: Environment & Dependency Validation**
   - ✅ TPU runtime configuration
   - ✅ Package version verification
   - ✅ Import validation (Tunix, JAX, Flax)
   - ✅ HBM memory check
   - ✅ LoRA adapter configuration

2. **Phase 2: Training Pipeline Verification**
   - ✅ Model download & initialization
   - ✅ Dataset preparation validation
   - ✅ GRPO configuration review
   - ✅ Reward function testing
   - ✅ Training setup validation

3. **Phase 3: Inference & Output Quality**
   - ✅ XML format validation
   - ✅ Reasoning quality assessment
   - ✅ Citation extraction testing
   - ✅ Output quality metrics

4. **Phase 4: Submission Preparation**
   - ✅ Package structure validation
   - ✅ JSON file validation
   - ✅ Final submission checklist

**Quick Start Validation:**

Before running full training, execute the validation cells in the notebook:

1. Run **Phase 1 validation cells** after Step 1 installation
2. Verify all checks pass (8 TPU cores, imports successful)
3. Proceed to **Phase 2** training after validation
4. Use **Phase 3** cells to validate inference quality
5. Complete **Phase 4** checklist before submission

See the [Validation Guide](docs/COLAB_VALIDATION_GUIDE.md) for detailed procedures and troubleshooting.

## 🔧 Training on Google Cloud TPU

### Primary Hackathon Deliverable

The training notebook is the **primary deliverable** for the Kaggle Google TuniX Hackathon submission.

**Notebook:** [`examples/notebooks/train_tunix_reasoning.ipynb`](examples/notebooks/train_tunix_reasoning.ipynb)

> 📢 **Updated December 2025:** This notebook now includes **comprehensive validation cells** covering all 4 phases:
> - **Phase 1**: Environment & dependency validation (TPU, imports, packages)
> - **Phase 2**: Training pipeline verification (setup, config, reward functions)
> - **Phase 3**: Inference & output quality (XML format, reasoning, citations)
> - **Phase 4**: Submission preparation (package validation, checklist)
> 
> **Total validation cells**: 11 new cells added for end-to-end validation
> 
> See [Complete Validation Guide](docs/COLAB_VALIDATION_GUIDE.md) for detailed procedures.

This specialized training approach uses:
- **Framework:** JAX/Flax with Google Tunix (different from main PyTorch codebase)
- **Hardware:** TPU v2-8+ on Google Colab
- **Model:** Gemma 3-1B-IT with LoRA adapters
- **Format:** XML-tagged reasoning (`<reasoning>`/`<answer>`)
- **Method:** GRPO (Group Relative Policy Optimization)

**Dependency Requirements:**
```bash
# ✅ Recommended installation (from notebook Step 1 - December 2025)
!pip install -q dotenv kagglehub ipywidgets tensorflow tensorflow_datasets tensorboardX
!pip install -q transformers>=4.40.0 grain huggingface_hub>=0.20.0 datasets>=2.14.0
!pip install -q 'numpy>2' sentencepiece>=0.1.99 safetensors>=0.4.0

# Install JAX, Tunix, Qwix, and Flax from GitHub (latest versions)
!pip install -q git+https://github.com/jax-ml/jax
!pip install git+https://github.com/google/tunix
!pip install git+https://github.com/google/qwix
!pip uninstall -q flax -y
!pip install git+https://github.com/google/flax

# ❌ Do NOT use these (outdated/incorrect)
# !pip install "google-tunix>=0.5.0"  # Version doesn't exist!
# !pip install jax==0.4.35 jaxlib==0.4.35  # Incompatible with Colab TPU
```

**Note:** After installation, you **MUST** restart the Colab runtime before proceeding.

**Prerequisites:**
- Google Colab account with TPU access
- Hugging Face account for model downloads
- Kaggle account for submissions

**Quick Start with Validation:**
1. Open [`train_tunix_reasoning.ipynb`](examples/notebooks/train_tunix_reasoning.ipynb) in Colab
2. Set runtime to TPU (Runtime → Change runtime type → TPU)
3. Run **Step 1** (dependencies) - expect `jax_cuda12_plugin` warnings (harmless)
4. **Restart runtime** when installation completes
5. Run **Step 2** (TPU initialization)
6. **Execute Phase 1 validation cells** - verify all checks pass:
   - ✅ 8 TPU cores detected
   - ✅ Tunix/Flax/JAX imports successful
   - ✅ Package versions correct
7. Continue with Steps 3-5 (model download, dataset prep, reward function)
8. **Execute Phase 2 validation cells** - verify training setup
9. Run training execution
10. **Execute Phase 3 validation cells** - verify output quality
11. Export adapters and run **Phase 4 validation** - prepare submission

**Complete Validation Flow:**
- **Phase 1** (pre-training): Environment and dependencies
- **Phase 2** (pre-training): Training pipeline setup
- **Phase 3** (post-training): Inference and output quality
- **Phase 4** (pre-submission): Package validation and checklist

**Quick Troubleshooting:**
| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'tunix'` | Wrong install method | Use `git+https://github.com/google/tunix` |
| JAX TPU initialization fails | Wrong JAX version | Use `git+https://github.com/jax-ml/jax` |
| `jax_cuda12_plugin` warnings | Normal for Colab | Ignore - harmless for TPU |
| Runtime not restarted | Skipped Step 4 | Runtime → Restart runtime after Step 1 |

**Complete Troubleshooting:** See [Validation Guide](docs/COLAB_VALIDATION_GUIDE.md#troubleshooting-reference) for detailed solutions to all common issues.

See [examples/notebooks/README.md](examples/notebooks/README.md) for more training options including PyTorch-based GRPO training.

## 📚 Documentation

- **[Complete Colab Validation Guide](docs/COLAB_VALIDATION_GUIDE.md)** - NEW! Comprehensive 4-phase validation
- [Architecture Overview](docs/architecture/overview.md)
- [API Reference](docs/api/reference.md)
- [User Guide](docs/guides/user-guide.md)
- [GRPO Training Guide](docs/GRPO_TRAINING.md)
- [Tunix/TPU Training Notebook](examples/notebooks/train_tunix_reasoning.ipynb)
- [Notebook README](examples/notebooks/README.md) - Training options and Phase 1 guide
- [Contributing Guide](CONTRIBUTING.md)
- [Development Setup](docs/guides/development.md)

## 🐛 Known Issues & Troubleshooting

### Quick Reference

For comprehensive troubleshooting covering all phases, see the **[Complete Troubleshooting Guide](docs/COLAB_VALIDATION_GUIDE.md#troubleshooting-reference)**.

### Most Common Issues

| Issue | Solution | Guide Section |
|-------|----------|---------------|
| `ImportError: cannot import name 'GenerationMixin'` | Transformers version mismatch - see [Issue #35 Solution](#transformers-version-fix) | [Phase 1](docs/COLAB_VALIDATION_GUIDE.md#phase-1-environment--dependency-validation) |
| `ModuleNotFoundError: No module named 'tunix'` | Install from GitHub: `git+https://github.com/google/tunix` | [Phase 1](docs/COLAB_VALIDATION_GUIDE.md#phase-1-environment--dependency-validation) |
| JAX TPU initialization fails | Install from GitHub: `git+https://github.com/jax-ml/jax` | [Phase 1](docs/COLAB_VALIDATION_GUIDE.md#14-tpu-detection-validation) |
| `RuntimeError: TPU not found` | Set runtime to TPU: Runtime → Change runtime type | [Phase 1](docs/COLAB_VALIDATION_GUIDE.md#11-colab-runtime-configuration) |
| Imports fail after install | Restart runtime after Step 1 | [Phase 1](docs/COLAB_VALIDATION_GUIDE.md#13-runtime-restart-checkpoint) |
| Out of Memory during training | Reduce batch_size, num_generations | [Phase 2](docs/COLAB_VALIDATION_GUIDE.md#phase-2-training-pipeline-verification) |
| All rewards are 0.0 | Check XML format validation | [Phase 3](docs/COLAB_VALIDATION_GUIDE.md#phase-3-inference--output-quality) |

### Expected Warnings (Safe to Ignore)

- **`jax_cuda12_plugin` warnings**: Normal on Google Colab TPU runtime. These appear because Colab has GPU packages pre-installed. They do not affect TPU training.

### Transformers Version Fix

#### Issue: `ImportError: cannot import name 'GenerationMixin'`

**Root Cause:** Version mismatch in the `transformers` library. The `GenerationMixin` class location changed between versions, causing import failures.

**Solution:** The project requires `transformers>=4.40.0,<4.57.1`. This is automatically handled in the `train_tunix_reasoning.ipynb` notebook Cell 7, which force-reinstalls the correct version.

**If you encounter this error:**

1. **In Colab Notebook**: The `train_tunix_reasoning.ipynb` Cell 7 includes automatic fix via force-reinstall:
   ```python
   # Note: flax and datasets are co-dependencies required for Gemma model training
   subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "transformers>=4.40.0,<4.57.1", "flax>=0.10.2,<0.13.0", "datasets"])
   ```

2. **After reinstall**: Restart the runtime (Runtime → Restart runtime)

3. **For local development**: Use the corrected `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

**Reference:** [GitHub Issue #35](https://github.com/clduab11/judicAIta/issues/35)

### Complete Troubleshooting

The validation guide includes detailed troubleshooting for:
- Environment setup issues
- Dependency conflicts
- TPU initialization problems
- Training pipeline errors
- Memory management
- Output quality issues
- Submission package problems

**[→ View Complete Troubleshooting Guide](docs/COLAB_VALIDATION_GUIDE.md#troubleshooting-reference)**

## 🧪 Testing

Unit tests exist in `tests/unit/` with 8 test files covering configuration, exceptions, citation parsing, and training components. Integration and end-to-end test directories exist but are currently empty.

Run tests:
```bash
# All tests
pytest

# With coverage
pytest --cov=judicaita

# Unit tests only
pytest tests/unit/ -v
```

See [tests/README.md](tests/README.md) for detailed testing documentation.

## 🛡️ Security & Compliance

Judicaita implements security measures appropriate for legal AI applications:

**Implemented:**
- **Input Validation**: Strict validation of all inputs using Pydantic
- **File Size Limits**: Configurable limits on document upload sizes
- **Type Safety**: Comprehensive type hints and static type checking
- **Dependency Management**: Regular updates and security scanning of dependencies

**Configured (pending full implementation):**
- **Audit Logging**: Audit logging settings are configured but not actively logging in all operations
- **Compliance Modes**: Data retention settings are defined but not enforced

**Planned:**
- **Encryption**: Data encryption at rest and in transit
- **Authentication & Authorization**: JWT-based authentication with role-based access control
- **Rate Limiting**: API rate limiting to prevent abuse

See [SECURITY.md](docs/SECURITY.md) for security policy and reporting vulnerabilities.

## 🤖 GitHub Copilot Integration

Judicaita includes GitHub Copilot configuration for enhanced GRPO development assistance.

### Copilot Configuration

The repository includes [`.github/copilot-instructions.md`](.github/copilot-instructions.md) which provides Copilot with:

- Project context (legal AI, Kaggle hackathon, TPU training)
- GRPO-specific patterns and best practices
- Notebook development guidance
- Debugging and troubleshooting tips

### GRPO Reference Patterns

Advanced GRPO patterns from AllenAI's `grpo_fast.py` are documented for optimization and debugging:

- **[GRPO Fast Patterns](docs/references/grpo_fast_patterns.md)**: Advantage computation, loss variants, memory optimization
- **[Quick Reference](docs/references/grpo_quick_reference.md)**: Common scenarios with code examples

### For Contributors

When working on GRPO-related code:

1. Reference the pattern documentation for optimization ideas
2. Use Copilot prompts that mention "grpo_fast.py patterns" for targeted suggestions
3. Check [`docs/GRPO_TRAINING.md`](docs/GRPO_TRAINING.md) for the Advanced Patterns section

See [`docs/references/`](docs/references/) for complete reference documentation.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Google Tunix](https://tunix.google.com) (0.1.x series) and [Gemma 3n](https://ai.google.dev/gemma)
- Optimized for the **Kaggle Google Tunix Hackathon** requirements
- TPU training tested on Google Colab TPU runtime (note: JAX 0.4+ requires TPU VMs not available on Colab)
- Inspired by the legal tech community's commitment to access to justice

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/clduab11/judicAIta/issues)
- **Discussions**: [GitHub Discussions](https://github.com/clduab11/judicAIta/discussions)

---

Made with ❤️ for the legal community
