# GRPO Quick Reference

Quick reference for common GRPO scenarios with specific patterns and code examples. Use this guide when working with GitHub Copilot for GRPO development.

---

## Common Development Scenarios

### 1. Improving Advantage Computation Stability

**Symptom**: Training loss fluctuates wildly, gradients spike

**Pattern**: Add advantage clipping

```python
# In _compute_advantages()
advantages = (rewards - mean_rewards) / (std_rewards * self.config.grpo_tau)

# Add clipping for stability
clip_range = 10.0
advantages = torch.clamp(advantages, -clip_range, clip_range)
```

**Copilot Prompt**:
> "Add symmetric clipping to advantage computation following grpo_fast.py patterns with configurable clip_range"

---

### 2. Reducing Memory Usage

**Symptom**: OOM errors, training crashes on large batches

**Pattern A**: Enable gradient checkpointing

```python
# In initialize()
if self.config.gradient_checkpointing:
    self.model.gradient_checkpointing_enable()
```

**Pattern B**: Reduce batch, increase accumulation

```python
config = TrainingConfig(
    batch_size=2,  # Reduced from 4
    gradient_accumulation_steps=8,  # Increased from 4
    # Effective batch size unchanged: 16
)
```

**Pattern C**: Stream advantage computation

```python
# Process in chunks
for chunk in torch.split(rewards, chunk_size):
    chunk_adv = compute_advantages(chunk)
    # Process immediately to avoid storing all
```

**Copilot Prompt**:
> "Enable gradient checkpointing in GRPOTrainer following transformers library patterns"

---

### 3. Debugging Gradient Issues

**Symptom**: NaN loss, training stops early, emergency checkpoints triggered

**Pattern**: Enhanced gradient monitoring

```python
# Monitor before and after clipping
pre_clip_norm = torch.nn.utils.get_grad_norm_(model.parameters())
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
post_clip_norm = torch.nn.utils.get_grad_norm_(model.parameters())

if pre_clip_norm > 10 * max_norm:
    logger.warning(f"Large gradient norm: {pre_clip_norm:.2f}")
```

**Pattern**: Reduce learning rate on instability

```python
if not gradient_is_stable:
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5
    logger.info("Reduced learning rate due to gradient instability")
```

**Copilot Prompt**:
> "Add gradient norm logging before and after clipping in the training loop"

---

### 4. Optimizing Reward Calculation

**Symptom**: Slow training, reward computation bottleneck

**Pattern**: Batch reward computation

```python
# Instead of:
for response in responses:
    reward = reward_fn.compute(prompt, response, reference)

# Use batched:
def batch_compute_rewards(prompts, responses, references):
    # Vectorize format checking
    format_scores = [self._check_format(r) for r in responses]
    
    # Parallel semantic similarity
    similarity_scores = batch_semantic_similarity(responses, references)
    
    return combine_scores(format_scores, similarity_scores)
```

**Copilot Prompt**:
> "Refactor reward computation to process responses in batches for efficiency"

---

## Notebook-Specific Scenarios

### 5. Fixing TPU Initialization

**Symptom**: `RuntimeError: TPU not found`, wrong device count

**Solution**:

```python
# Step 1: Verify runtime is TPU
import jax
devices = jax.devices()
print(f"Available devices: {len(devices)}")

# Expected: 8 TPU cores
assert len(devices) == 8, "Set runtime to TPU: Runtime → Change runtime type → TPU"
```

**Troubleshooting**:

| Issue | Solution |
|-------|----------|
| 0 devices | Restart runtime |
| GPU detected | Change runtime type to TPU |
| 4 cores instead of 8 | Using TPU v2-4, adjust batch size |

**Copilot Prompt**:
> "Add TPU initialization validation following judicaita notebook patterns"

---

### 6. Resolving JAX Conflicts

**Symptom**: `ModuleNotFoundError`, version conflicts, `jax_cuda12_plugin` errors

**Solution**: Correct installation order

```bash
# Clear existing installations
pip uninstall jax jaxlib flax -y

# Install from GitHub (latest TPU-compatible)
pip install git+https://github.com/jax-ml/jax
pip install git+https://github.com/google/tunix
pip install git+https://github.com/google/flax

# ALWAYS restart runtime after installation
```

**Note**: `jax_cuda12_plugin` warnings are **harmless** on TPU

**Copilot Prompt**:
> "Check and fix JAX installation for TPU compatibility"

---

### 7. Handling Memory Errors in Notebook

**Symptom**: `OutOfMemoryError`, HBM exhausted, training crashes

**Solution A**: Reduce generation parameters

```python
# In training config
config = TrainingConfig(
    batch_size=2,
    num_rollouts=2,  # Reduced from 4
    generation_max_length=256,  # Reduced from 512
)
```

**Solution B**: Clear memory between operations

```python
import gc
import torch

# After each epoch or major operation
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**Solution C**: Monitor HBM usage

```python
# Add to notebook
import jax
for device in jax.devices():
    stats = device.memory_stats()
    print(f"HBM: {stats['bytes_in_use'] / 1e9:.2f} GB")
```

**Copilot Prompt**:
> "Add HBM memory monitoring following judicaita validation patterns"

---

## Copilot Prompt Suggestions

### For Advantage Computation

```
"Implement centered advantage normalization from grpo_fast.py patterns"
"Add winsorization for advantage outliers following AllenAI patterns"
"Implement robust normalization using MAD for _compute_advantages"
```

### For Loss Functions

```
"Add KL penalty to GRPO loss following grpo_fast.py patterns"
"Implement PPO-style ratio clipping in _grpo_step"
"Add adaptive KL coefficient adjustment"
```

### For Memory Optimization

```
"Enable gradient checkpointing in GRPOTrainer"
"Implement streaming advantage computation for large batches"
"Add memory profiling to training loop"
```

### For Debugging

```
"Add gradient norm tracking to GRPOTrainer"
"Implement emergency checkpoint on gradient instability"
"Add loss component logging for debugging rewards"
```

### For Notebook Development

```
"Add Phase 1 TPU validation cells to notebook"
"Implement HBM memory monitoring for TPU training"
"Add dependency version checking cell"
```

---

## Quick Fixes

### All Rewards are 0.0

```python
# Check XML format in responses
response = generate_response(prompt)
print(f"Response: {response}")
assert "<reasoning>" in response, "Missing reasoning tags"
assert "<answer>" in response, "Missing answer tags"
```

### Training Too Slow

```python
# Quick optimizations
config = TrainingConfig(
    num_rollouts=2,  # Reduce from 4
    eval_steps=500,  # Reduce eval frequency
    logging_steps=50,  # Reduce logging frequency
)
```

### Gradients Exploding

```python
# Reduce learning rate and add clipping
config = TrainingConfig(
    learning_rate=5e-6,  # Reduced from 1e-5
    max_grad_norm=0.5,  # Reduced from 1.0
)
```

### Memory Issues

```python
# Quick memory reduction
config = TrainingConfig(
    batch_size=1,
    gradient_accumulation_steps=16,
    num_rollouts=2,
    bf16=True,  # Use bfloat16
)
```

---

## Reference Links

- [Full Pattern Documentation](grpo_fast_patterns.md)
- [GRPO Training Guide](../GRPO_TRAINING.md)
- [Notebook README](../../examples/notebooks/README.md)
- [Copilot Instructions](../../.github/copilot-instructions.md)
