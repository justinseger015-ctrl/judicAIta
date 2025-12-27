# GRPO Training Demo for Judicaita

This directory contains example notebooks and scripts for training legal reasoning models using GRPO (Group Relative Policy Optimization).

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
    base_model="google/gemma-2-2b-it",
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
judicaita train-grpo --base-model google/gemma-2-2b-it --epochs 3 --batch-size 4
```

Evaluate a checkpoint:
```bash
judicaita evaluate-model ./checkpoints/grpo/final --max-samples 100
```

## Notebooks

- `grpo_training_demo.ipynb` - Complete training demonstration
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
