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

## Using Copilot for GRPO Development

GitHub Copilot is configured to assist with GRPO development in this repository.

### Getting Started with Copilot

1. Ensure GitHub Copilot is enabled in your IDE
2. Reference [`.github/copilot-instructions.md`](../../.github/copilot-instructions.md) for project context
3. Use specific prompts to trigger GRPO-aware suggestions

### GRPO Reference Documentation

- **[GRPO Fast Patterns](../../docs/references/grpo_fast_patterns.md)**: Advanced patterns from AllenAI's grpo_fast.py
- **[Quick Reference](../../docs/references/grpo_quick_reference.md)**: Common scenarios with code examples
- **[Training Guide](../../docs/GRPO_TRAINING.md)**: Complete training documentation

### Troubleshooting "GRPO Gremlins"

Common issues and their solutions using grpo_fast.py patterns:

#### TPU Issues

| Problem | Solution |
|---------|----------|
| TPU not detected | `Runtime → Change runtime type → TPU`, then restart |
| Wrong device count | Verify with `len(jax.devices())`, expect 8 for TPU v2-8 |
| `jax_cuda12_plugin` warnings | **Safe to ignore** - normal on Colab TPU |

#### Dependency Resolution

```bash
# Clean install order (if imports fail)
pip uninstall jax jaxlib flax -y
pip install git+https://github.com/jax-ml/jax
pip install git+https://github.com/google/tunix
pip install git+https://github.com/google/flax
# RESTART RUNTIME after installation
```

#### Training Stability

| Issue | Pattern from grpo_fast.py |
|-------|---------------------------|
| Extreme advantage values | Add clipping: `torch.clamp(advantages, -10, 10)` |
| Gradient explosions | Reduce `max_grad_norm` from 1.0 to 0.5 |
| Loss becomes NaN | Check gradient monitor, reduce learning rate |

#### Memory Optimization

```python
# If OOM errors occur:
config = TrainingConfig(
    batch_size=2,              # Reduce
    gradient_accumulation_steps=8,  # Increase
    num_rollouts=2,            # Reduce
    bf16=True,                 # Enable
)
```

#### Advantage Computation Validation

```python
# Validate advantage computation is working
# Note: Access training metrics through the train() return value
# or add a public validation method to GRPOTrainer
training_metrics = trainer.train()
if 'advantages' in training_metrics:
    print(f"Advantage stats from training metrics")
```

### Example Copilot Prompts

For notebook development:

```
"Add TPU initialization validation cell"
"Create HBM memory monitoring"
"Add dependency version checking"
```

For GRPO improvements:

```
"Add advantage clipping to _compute_advantages following grpo_fast.py"
"Implement gradient checkpointing for memory efficiency"
"Add KL penalty to the loss function"
```

For debugging:

```
"Add gradient norm logging"
"Create reward breakdown visualization"
"Add checkpoint recovery logic"
```

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
