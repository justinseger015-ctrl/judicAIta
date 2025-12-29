# GRPO Training Demo for Judicaita

This directory contains example notebooks and scripts for training legal reasoning models using GRPO (Group Relative Policy Optimization).

## Available Notebooks

### `train_tunix_reasoning.ipynb` - **NEW! Tunix/TPU Training** 🚀

Complete Google Colab notebook for training Gemma 3-1B-IT with GRPO on TPU using Google Tunix framework.

**Features:**
- TPU v2-8+ setup and initialization
- Gemma 3-1B-IT model loading and fine-tuning
- XML-formatted reasoning (`<reasoning>`/`<answer>` tags)
- Custom reward function (format + length + correctness)
- LoRA parameter-efficient training
- Kaggle submission package export

**Requirements:**
- Google Colab with TPU runtime
- Hugging Face account (for model access)
- Kaggle account (for submission)

**Usage:**
1. Open in Google Colab
2. Set runtime to TPU (Runtime → Change runtime type → TPU)
3. Run cells sequentially
4. Follow inline instructions for authentication

**Note:** This uses JAX/Flax/Tunix stack (different from PyTorch main codebase).

## 📋 Phase 1: TPU Smoke Test & Validation

Before running full training, the notebook includes a **Phase 1 validation flow** to verify environment setup:

### Validation Checklist

| Step | Cell | What it Validates |
|------|------|-------------------|
| 1 | Install Dependencies | Package installation completes |
| 2 | Package Versions | Tunix 0.1.x, JAX TPU, Flax 0.10.2 |
| 3 | Runtime Restart | Fresh Python interpreter |
| 4 | TPU Detection | 8 TPU cores visible |
| 5 | Import Verification | JAX, Tunix, Flax imports |
| 6 | HBM Check | Memory visibility (optional) |
| 7 | LoRA Test | Adapter modules accessible |
| 8 | Validation Summary | All checks passed |

### Phase 1 Success Criteria

- ✅ All 8 TPU cores detected
- ✅ Tunix/Flax/JAX imports successful
- ✅ HBM memory stats visible (optional)
- ✅ LoRA adapter modules accessible

**→ Phase 2 training can proceed when all critical checks pass**

### Common Issues & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'tunix'` | Wrong Tunix version | Use `>=0.1.0,<=0.1.5` |
| JAX TPU initialization fails | Wrong JAX version | Use `jax[tpu]` with libtpu releases |
| `RuntimeError: TPU not found` | Wrong Colab runtime | Set runtime to TPU |
| Imports fail after install | Runtime not restarted | Restart runtime after Step 1 |
| `jax_cuda12_plugin` warnings | Normal for Colab | Ignore - harmless for TPU |

### Related PRs

- **[PR #13](https://github.com/clduab11/judicAIta/pull/13)** - Fixed `ModuleNotFoundError` dependency version constraints
- **[PR #7](https://github.com/clduab11/judicAIta/issues/7)** - ground_truth metadata bug (blocks Phase 2)

## Quick Start

### Training a Model

```python
from judicaita.training import GRPOTrainer, TrainingConfig, create_training_dataset

# Load datasets
legalbench_data, pile_of_law_data = create_training_dataset(
    generate_cot=False,  # Set True for synthetic CoT
    samples_per_subset=1000,
)

# Configure training
config = TrainingConfig(
    base_model="google/gemma-3-1b-it",
    use_lora=True,
    num_epochs=3,
    batch_size=4,
    learning_rate=1e-5,
)

# Train
trainer = GRPOTrainer(config=config, train_dataset=legalbench_data)
trainer.initialize()
metrics = trainer.train()
```

### Evaluating a Model

```python
from judicaita.training.evaluation import evaluate_checkpoint

results = evaluate_checkpoint(
    checkpoint_path="./checkpoints/grpo/final",
    eval_dataset=eval_data,
    max_samples=100,
)
```

### Using in Reasoning Trace Generator

```python
from judicaita.reasoning_trace import ReasoningTraceGenerator

# Load with GRPO checkpoint
generator = ReasoningTraceGenerator(
    checkpoint_path="./checkpoints/grpo/final"
)
await generator.initialize()

trace = await generator.generate_trace(
    query="Legal question here",
    context="Relevant context"
)
```

## CLI Commands

Train a model:
```bash
judicaita train-grpo --base-model google/gemma-3-1b-it --epochs 3 --batch-size 4
```

Evaluate a checkpoint:
```bash
judicaita evaluate-model ./checkpoints/grpo/final --max-samples 100
```

## Notebooks

- **`train_tunix_reasoning.ipynb`** - Tunix/TPU training with XML-formatted reasoning (Kaggle hackathon)
- `grpo_training_demo.ipynb` - Complete training demonstration (PyTorch)
- `evaluation_demo.ipynb` - Model evaluation examples
- `inference_demo.ipynb` - Inference with trained models

## Key Features

1. **Data Curation**: Automatic loading of LegalBench and Pile-of-Law datasets
2. **Synthetic CoT**: Optional generation of Chain-of-Thought reasoning traces
3. **GRPO Training**: Group Relative Policy Optimization for reasoning improvement
4. **LoRA/PEFT**: Parameter-efficient training for large models
5. **Reward Functions**: Multi-component rewards for format, outcome, and verbosity
6. **Evaluation**: Comprehensive metrics including ROUGE, BLEU, and task accuracy

## Configuration

See `TrainingConfig` in `judicaita/training/grpo_trainer.py` for all configuration options.

Key hyperparameters:
- `learning_rate`: 1e-5 (default)
- `grpo_tau`: 0.1 (temperature for advantage normalization)
- `grpo_gamma`: 0.99 (discount factor)
- `num_rollouts`: 4 (number of rollouts per prompt)
- `lora_r`: 16 (LoRA rank)
- `lora_alpha`: 32 (LoRA alpha)

## Resources

- [LegalBench Dataset](https://huggingface.co/datasets/nguha/legalbench)
- [Pile-of-Law Dataset](https://huggingface.co/datasets/pile-of-law/pile-of-law)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Gemma Models](https://ai.google.dev/gemma)
