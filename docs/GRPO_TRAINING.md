# GRPO Training Pipeline Documentation

## Overview

This document describes the GRPO (Group Relative Policy Optimization) training pipeline implementation for the Judicaita legal AI assistant.

## Architecture

### Module Structure

```
src/judicaita/training/
├── __init__.py              # Module exports
├── data_curation.py         # Dataset loading and preprocessing
├── grpo_trainer.py          # GRPO training implementation
├── rewards.py               # Reward function components
└── evaluation.py            # Evaluation metrics and harness
```

### Key Components

#### 1. Data Curation (`data_curation.py`)

**LegalBenchDataset**: Loads reasoning-heavy tasks from nguha/legalbench
- Task types: rule_qa, contract_qa, issue_spotting
- Configurable number of samples per task
- Automatic formatting for GRPO training

**PileOfLawDataset**: Loads legal corpus for domain adaptation
- Subsets: courtlistener_opinions, uscode, contracts
- Configurable subsampling for efficiency
- Supports mixed-batch pretraining

**SyntheticCoTGenerator**: Generates Chain-of-Thought reasoning traces
- Uses frontier model (e.g., gemma-2-27b-it) for annotation
- Step-by-step format: "Step 1: ... Step 2: ... Conclusion: ..."
- Batch processing support

#### 2. GRPO Trainer (`grpo_trainer.py`)

**TrainingConfig**: Comprehensive training configuration
- Model settings (base model, max length, device)
- LoRA/PEFT settings (rank, alpha, dropout, target modules)
- GRPO hyperparameters (tau, gamma, num_rollouts)
- Optimization settings (learning rate, scheduler, weight decay)
- Checkpointing and logging

**GRPOTrainer**: Main training class
- Initializes model with optional LoRA adaptation
- Implements GRPO algorithm with group relative advantages
- Supports multiple rollouts per prompt
- Integrated evaluation during training
- Checkpoint management

Key GRPO features:
- Group normalization for advantages
- Multiple rollouts per prompt for variance reduction
- Temperature-controlled advantage computation
- Gradient accumulation for large effective batch sizes

#### 3. Reward Functions (`rewards.py`)

**FormatReward**: Validates step-by-step reasoning structure
- Checks for "Step N:" markers
- Validates conclusion presence
- Configurable minimum steps requirement

**OutcomeReward**: Evaluates conclusion correctness
- Semantic similarity to reference answer
- Configurable similarity threshold
- Supports exact match or token overlap

**VerbosityReward**: Balances response length
- Target length with tolerance
- Penalties for too short or too long responses
- Encourages optimal reasoning depth

**CompositeReward**: Combines multiple reward signals
- Weighted aggregation (default: 30% format, 50% outcome, 20% verbosity)
- Batch processing support
- Returns detailed breakdown of component scores

#### 4. Evaluation (`evaluation.py`)

**EvaluationHarness**: Comprehensive model evaluation
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- BLEU score
- Task accuracy based on token overlap
- Format compliance checking
- Detailed per-sample metrics

**evaluate_checkpoint()**: Utility for checkpoint evaluation
- Loads model and tokenizer
- Runs evaluation on dataset
- Saves results to JSON

## Usage

### Training

#### Via CLI

```bash
# Basic training
judicaita train-grpo --base-model google/gemma-2-2b-it --epochs 3

# With custom settings
judicaita train-grpo \
    --base-model google/gemma-2-2b-it \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 5e-6 \
    --output-dir ./my_checkpoints \
    --generate-cot  # Enable synthetic CoT generation
```

#### Via Python API

```python
from judicaita.training import (
    GRPOTrainer,
    TrainingConfig,
    create_training_dataset,
    CompositeReward,
)

# Load datasets
legalbench_data, pile_of_law_data = create_training_dataset(
    generate_cot=False,  # Set True for synthetic CoT
    samples_per_subset=5000,
)

# Configure training
config = TrainingConfig(
    base_model="google/gemma-2-2b-it",
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    num_epochs=3,
    batch_size=4,
    learning_rate=1e-5,
    grpo_tau=0.1,
    grpo_gamma=0.99,
    num_rollouts=4,
)

# Create custom reward function
reward_fn = CompositeReward(
    format_weight=0.3,
    outcome_weight=0.5,
    verbosity_weight=0.2,
)

# Initialize and train
trainer = GRPOTrainer(
    config=config,
    train_dataset=legalbench_data,
    reward_fn=reward_fn,
)
trainer.initialize()
metrics = trainer.train()
```

### Evaluation

```bash
# Evaluate checkpoint
judicaita evaluate-model ./checkpoints/grpo/final \
    --max-samples 100 \
    --output results.json
```

### Integration with Reasoning Trace Generator

```python
from judicaita.reasoning_trace import ReasoningTraceGenerator

# Load GRPO-tuned model
generator = ReasoningTraceGenerator(
    checkpoint_path="./checkpoints/grpo/final"
)
await generator.initialize()

# Generate reasoning trace
trace = await generator.generate_trace(
    query="Is this non-compete clause enforceable?",
    context="Employee worked for 2 years, clause was 5 years..."
)
```

## Configuration

### Environment Variables

Add to `.env`:

```
GRPO_CHECKPOINT_PATH=./checkpoints/grpo/final
GRPO_BASE_MODEL=google/gemma-2-2b-it
GRPO_LEARNING_RATE=1e-5
GRPO_NUM_EPOCHS=3
GRPO_BATCH_SIZE=4
GRPO_USE_LORA=True
```

### Hyperparameter Tuning

Key hyperparameters to tune:

1. **Learning Rate** (`learning_rate`): Start with 1e-5, adjust based on loss curves
2. **GRPO Tau** (`grpo_tau`): Controls advantage normalization (default: 0.1)
3. **Num Rollouts** (`num_rollouts`): More rollouts = better estimates but slower (default: 4)
4. **LoRA Rank** (`lora_r`): Higher rank = more capacity but more memory (default: 16)
5. **Reward Weights**: Adjust based on desired behavior emphasis

## Performance Considerations

### Memory Optimization

- Use LoRA (default enabled) for 60-90% memory reduction
- Gradient accumulation for larger effective batch sizes
- BF16 precision (default enabled on supported hardware)
- Gradient checkpointing for very large models

### Training Time

Approximate training times (on single A100 GPU):
- 1000 samples, 3 epochs, 4 rollouts: ~2-3 hours
- 10000 samples, 3 epochs, 4 rollouts: ~20-30 hours

Speed up training:
- Reduce `num_rollouts` (2 for fast iteration)
- Use smaller base model (gemma-2-2b-it vs gemma-2-9b-it)
- Increase `batch_size` and `gradient_accumulation_steps`
- Pre-generate CoT traces offline

### TPU Support

The pipeline supports TPU training:
- Set `device="tpu"` in TrainingConfig
- Use `torch_xla` for TPU acceleration
- Batch size can be larger on TPU

## Best Practices

1. **Start Small**: Test with 100-1000 samples before full training
2. **Monitor Metrics**: Watch loss, reward, and evaluation metrics
3. **Checkpointing**: Save frequently (every 500 steps recommended)
4. **Evaluation**: Run eval every 100 steps to catch overfitting
5. **CoT Quality**: Validate synthetic CoT traces before training
6. **Reward Balancing**: Tune reward weights based on desired behavior

## Troubleshooting

### Common Issues

**Out of Memory**:
- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Use smaller base model
- Enable gradient checkpointing

**Slow Training**:
- Reduce `num_rollouts`
- Increase `batch_size`
- Use smaller dataset for testing

**Poor Performance**:
- Check reward function behavior
- Validate data quality
- Increase training epochs
- Tune GRPO hyperparameters (tau, gamma)

**Model Not Improving**:
- Check learning rate (try 5e-6 or 2e-5)
- Verify reward signals are meaningful
- Ensure sufficient training data

## Future Enhancements

Planned improvements:
1. Distributed training support (multi-GPU/TPU)
2. Advanced reward models (trained on LegalBench)
3. Curriculum learning for progressive difficulty
4. Integration with RAG for citation grounding
5. Fine-grained task-specific rewards
6. Real-time training monitoring dashboard

## Advanced Patterns

For advanced optimization and debugging, we maintain reference documentation based on AllenAI's `grpo_fast.py` implementation patterns.

### Reference Documentation

- **[GRPO Fast Patterns](references/grpo_fast_patterns.md)**: Comprehensive documentation of advanced patterns including:
  - Alternative advantage computation approaches (clipping, centering, MAD normalization)
  - Loss computation variants (DAPO, CISPO, KL penalties)
  - Memory optimization techniques
  - Gradient stability patterns
  
- **[Quick Reference Guide](references/grpo_quick_reference.md)**: Practical code examples for common scenarios

### When to Consult Reference Patterns

| Scenario | Recommended Patterns |
|----------|---------------------|
| Training instability | Advantage clipping, gradient monitoring |
| OOM errors | Memory optimization, gradient checkpointing |
| Poor convergence | Alternative normalization, KL penalties |
| Debugging | Gradient tracking, loss component logging |

### Important Note

> **The existing judicAIta implementation is sufficient for the Kaggle hackathon requirements.** The advanced patterns are provided for optimization, debugging, and future development. Consult them when:
> - Training becomes unstable
> - Memory constraints require optimization
> - You need to debug specific issues
> - Extending the implementation beyond hackathon scope

## Using Copilot for GRPO Development

GitHub Copilot is configured with judicAIta-specific context to assist with GRPO development.

### Copilot Configuration

See [`.github/copilot-instructions.md`](../.github/copilot-instructions.md) for the full Copilot configuration.

### How to Reference Patterns in Prompts

When using Copilot, include specific references to trigger pattern-aware suggestions:

```
"Add advantage clipping following grpo_fast.py patterns"
"Implement gradient checkpointing like in AllenAI open-instruct"
"Add KL penalty integration to the loss function"
```

### Expected Copilot Assistance

Copilot can help with:

1. **Advantage Computation**: Clipping strategies, normalization variants
2. **Memory Optimization**: Gradient checkpointing, batch processing
3. **Debugging**: Gradient monitoring, loss validation
4. **Notebook Development**: TPU initialization, dependency management

### Example Prompts

```
# For stability issues
"Add symmetric clipping to _compute_advantages with configurable range"

# For memory issues  
"Enable gradient checkpointing in GRPOTrainer"

# For debugging
"Add gradient norm logging before and after clipping"
```

## References

- [LegalBench Dataset](https://huggingface.co/datasets/nguha/legalbench)
- [Pile-of-Law Dataset](https://huggingface.co/datasets/pile-of-law/pile-of-law)
- [PEFT/LoRA](https://huggingface.co/docs/peft)
- [Gemma Models](https://ai.google.dev/gemma)
- GRPO: Group Relative Policy Optimization (based on Tunix demos)

## Contributing

When contributing to the training pipeline:
1. Follow existing code style (black, ruff)
2. Add tests for new reward functions
3. Document hyperparameters and their effects
4. Update this documentation
5. Benchmark changes on small dataset first

## License

Same as main Judicaita project (Apache 2.0)
