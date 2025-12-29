# GRPO Training Demo for Judicaita

This directory contains example notebooks and scripts for training legal reasoning models using GRPO (Group Relative Policy Optimization).

## Available Notebooks

### `train_tunix_reasoning.ipynb` - **NEW! Tunix/TPU Training** 🚀

Complete Google Colab notebook for training Gemma 3-1B-IT with GRPO on TPU using Google Tunix framework.

**Features:**
- **Phase 1 Validation**: Comprehensive smoke test for TPU setup and dependencies
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

#### Phase 1: Environment Validation (Required First)
1. Open in Google Colab
2. Set runtime to TPU (Runtime → Change runtime type → TPU)
3. Run **Step 1** (dependency installation)
4. Run validation cell to verify package versions
5. **Restart runtime** (critical!)
6. Run **Step 2** (TPU initialization)
7. Run all Phase 1 validation cells (imports, HBM, LoRA, summary)
8. Verify "🎉 ALL CHECKS PASSED" message before proceeding

#### Phase 2: Training Execution
9. Complete Steps 3-5 (model download, dataset preparation)
10. Execute GRPO training
11. Export LoRA adapters for submission

**Validation Criteria:**
- ✅ 8 TPU cores detected (or 4 for TPU v2-4)
- ✅ JAX version: 0.8.x (TPU-compatible)
- ✅ Tunix imported successfully
- ✅ Flax version: 0.12.x or 0.10.2
- ✅ HBM memory stats visible (after model load)
- ✅ LoRA adapter configuration validated

**⚠️ Important Notes:**
- `jax_cuda12_plugin` warnings are **normal and harmless** on Colab TPU
- Phase 2 blocked by [PR #7](https://github.com/clduab11/judicAIta/pull/7) - `ground_truth` metadata bug
- Uses JAX/Flax/Tunix stack (different from PyTorch main codebase)

**Troubleshooting:**
See inline troubleshooting reference in the notebook for common issues.

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
