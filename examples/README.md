# Judicaita Examples

This directory contains example scripts and notebooks demonstrating how to use Judicaita.

## Available Examples

### Python Scripts

#### `basic_usage.py`
Basic example showing document processing and citation extraction.

**Run:**
```bash
python examples/basic_usage.py
```

**What it demonstrates:**
- Document input service initialization
- Citation extraction and mapping
- Audit logging

#### `reasoning_trace_example.py`
Example of generating explainable reasoning traces for legal analysis.

**Run:**
```bash
python examples/reasoning_trace_example.py
```

**What it demonstrates:**
- Reasoning trace generation
- Step-by-step legal analysis
- Confidence scoring
- Markdown export

## Jupyter Notebooks

Coming soon! We're working on interactive Jupyter notebooks that will include:

- **Getting Started**: Introduction to Judicaita
- **Document Processing**: Deep dive into document handling
- **Citation Analysis**: Working with legal citations
- **Summary Generation**: Creating plain-English summaries
- **Compliance Workflows**: Audit logging and reporting

## Sample Data

The `sample_data/` directory will contain sample legal documents for testing:

- Sample briefs
- Sample contracts
- Sample court opinions
- Sample statutes

**Note**: These are example documents for demonstration purposes only.

## Running the Examples

### Prerequisites

1. Install Judicaita:
```bash
pip install -e .
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

```bash
# Navigate to examples directory
cd examples

# Run an example
python basic_usage.py
```

### With Custom Configuration

```bash
# Set environment variables
export GOOGLE_API_KEY=your-key-here
export LOG_LEVEL=DEBUG

# Run example
python reasoning_trace_example.py
```

## Creating Your Own Examples

Feel free to create your own examples based on these templates!

### Example Template

```python
"""
Example: Your example description

This script demonstrates...
"""

import asyncio
from judicaita import ...


async def main():
    """Main example function."""
    print("=" * 60)
    print("Your Example Title")
    print("=" * 60)
    
    # Your code here
    
    print("\nExample completed!")


if __name__ == "__main__":
    asyncio.run(main())
```

## Contributing Examples

We welcome example contributions! If you have a useful example:

1. Create a new file in `examples/`
2. Follow the template above
3. Add documentation
4. Test thoroughly
5. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.

## Getting Help

- Check the [User Guide](../docs/guides/user-guide.md)
- Review [API Documentation](../docs/api/reference.md)
- Ask in [GitHub Discussions](https://github.com/clduab11/judicAIta/discussions)

## License

All examples are licensed under the same Apache 2.0 license as Judicaita.
