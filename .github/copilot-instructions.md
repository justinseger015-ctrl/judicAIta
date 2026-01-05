# GitHub Copilot Instructions for Judicaita

## Project Context

Judicaita is a **legal AI reasoning system** designed for the Kaggle Google Tunix Hackathon. The project focuses on generating explainable legal reasoning traces using GRPO (Group Relative Policy Optimization).

### Technical Stack

- **Framework**: PyTorch with Hugging Face Transformers
- **Training**: JAX/Flax with Google Tunix for TPU training
- **Hardware**: TPU v2-8 (Google Colab) for training
- **Model**: Gemma 3-1B-IT with LoRA adapters
- **Format**: XML-tagged reasoning (`<reasoning>`/`<answer>`)

### Existing Implementation

The repository has a working GRPO implementation in `src/judicaita/training/grpo_trainer.py` that includes:

- `GRPOTrainer`: Main training class with LoRA support
- `TrainingConfig`: Comprehensive configuration dataclass
- `_compute_advantages()`: Group-relative advantage computation
- `_grpo_step()`: GRPO training step with rollouts
- Memory profiling and gradient monitoring

## When to Reference AllenAI grpo_fast.py Patterns

Reference patterns from AllenAI's `grpo_fast.py` when working on:

### Advantage Computation Optimization

- Improving stability of `_compute_advantages()` method
- Implementing clipping strategies for extreme values
- Adding centered vs. standard normalization options
- Reducing variance in advantage estimates

### Memory Efficiency

- Optimizing batch processing in `_grpo_step()`
- Reducing memory footprint during rollout generation
- Implementing gradient checkpointing
- Streaming advantage computation for large batches

### Gradient Stability

- Debugging gradient explosions or vanishing gradients
- Implementing gradient monitoring improvements
- Adding loss scaling for mixed precision training
- Checkpoint recovery after gradient instability

### Loss Computation

- Implementing alternative loss formulations (DAPO, CISPO)
- Adding KL penalty integration
- Optimizing reference policy updates
- Tuning clipping bounds

See `docs/references/grpo_fast_patterns.md` for detailed pattern documentation.

## Notebook Development Guidance

### TPU Initialization

When working on `examples/notebooks/train_tunix_reasoning.ipynb`:

```python
# Correct TPU initialization pattern
import jax
devices = jax.devices()
assert len(devices) == 8, f"Expected 8 TPU cores, got {len(devices)}"
```

### Dependency Management

```bash
# Recommended installation order (from grpo_fast.py patterns)
pip install git+https://github.com/jax-ml/jax
pip install git+https://github.com/google/tunix
pip install transformers>=4.40.0,<4.57.1
```

### Validation Suggestions

- Always run Phase 1 validation before training
- Check TPU device count before starting
- Verify package versions match requirements
- Monitor HBM memory during training

## Advantage Computation Patterns

### Clipping Strategies

When implementing advantage clipping:

```python
# Pattern: Clip extreme advantages for stability
advantages = (rewards - mean_rewards) / (std_rewards + eps)
advantages = torch.clamp(advantages, -clip_range, clip_range)
```

### Normalization Approaches

- **Standard normalization**: `(x - mean) / std`
- **Centered normalization**: Subtract mean only, useful for sparse rewards
- **Per-group normalization**: Current judicaita approach, normalize within rollout groups

### Variance Reduction

```python
# Pattern: Multiple rollouts reduce variance
num_rollouts = 4  # Default in TrainingConfig
# More rollouts = lower variance but slower training
```

## Memory Optimization Contexts

### Batch Processing

```python
# Pattern: Gradient accumulation for memory efficiency
config = TrainingConfig(
    batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
)
```

### Tensor Operations

- Use `torch.no_grad()` for generation
- Enable bfloat16 for reduced memory (`bf16=True` in config)
- Consider gradient checkpointing for large models

### Gradient Accumulation

- Current implementation uses configurable gradient accumulation
- Increase `gradient_accumulation_steps` when memory constrained
- Balance between memory usage and training stability

## Debugging Guidance

### Gradient Monitoring

```python
# Pattern: Check for gradient issues
if self._profiling_enabled and self.gradient_monitor is not None:
    grad_norm, is_stable = self.gradient_monitor.check_gradients(
        self.model, global_step
    )
    if not is_stable:
        # Emergency checkpoint and early stopping
        self.save_checkpoint(f"emergency-step-{global_step}")
```

### Loss Validation

- Monitor `avg_reward` in training logs
- Check for reward collapse (all rewards becoming 0.0)
- Verify XML format in generated responses

### Checkpoint Recovery

- Use emergency checkpoints on gradient instability
- Load checkpoints with `trainer.load_checkpoint(path)`
- Verify checkpoint integrity before resuming

## Notebook Troubleshooting

### TPU Issues

| Error | Solution |
|-------|----------|
| `RuntimeError: TPU not found` | Set runtime to TPU in Colab |
| JAX TPU initialization fails | Use `git+https://github.com/jax-ml/jax` |
| `jax_cuda12_plugin` warnings | Safe to ignore on TPU |

### JAX/Tunix Compatibility

```bash
# Known working combination
pip install git+https://github.com/google/tunix
pip install git+https://github.com/jax-ml/jax
pip install flax>=0.10.2,<0.13.0
```

### Dependency Resolution

- Always restart runtime after installing JAX packages
- Install transformers with version bounds: `>=4.40.0,<4.57.1`
- Use `pip install -q` for cleaner output

## Code Style

### Follow Existing Patterns

- Extend `GRPOTrainer` rather than replacing it
- Use dataclasses for configuration (like `TrainingConfig`)
- Follow loguru logging patterns

### Transformers Library Consistency

- Use `AutoModelForCausalLM` and `AutoTokenizer`
- Follow PEFT/LoRA integration patterns
- Maintain compatibility with Hugging Face ecosystem

### Documentation

- Add docstrings following existing style
- Update `docs/GRPO_TRAINING.md` for training changes
- Include validation mode support in new features

## Reference Documentation

- **Detailed Patterns**: `docs/references/grpo_fast_patterns.md`
- **Quick Reference**: `docs/references/grpo_quick_reference.md`
- **Training Guide**: `docs/GRPO_TRAINING.md`
- **Notebook Guide**: `examples/notebooks/README.md`
