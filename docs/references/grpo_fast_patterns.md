# GRPO Fast Patterns Reference

This document extracts key patterns from AllenAI's `grpo_fast.py` implementation for reference during judicAIta development. These patterns can enhance advantage computation, loss calculation, memory efficiency, and training stability.

**Source**: [AllenAI open-instruct grpo_fast.py](https://github.com/allenai/open-instruct/blob/main/open_instruct/grpo_fast.py)

---

## Table of Contents

1. [Advantage Computation](#advantage-computation)
2. [Loss Computation Variants](#loss-computation-variants)
3. [Memory Optimization Patterns](#memory-optimization-patterns)
4. [Gradient Stability Patterns](#gradient-stability-patterns)
5. [Reward Computation Patterns](#reward-computation-patterns)
6. [Comparison to Judicaita Implementation](#comparison-to-judicaita-implementation)
7. [When to Apply](#when-to-apply)

---

## Advantage Computation

### Group-Relative Normalization

The core GRPO approach normalizes advantages within groups of rollouts for each prompt.

**Pattern: Standard Group Normalization**

```python
# Normalize within each group (rollouts for same prompt)
mean_rewards = rewards.mean(dim=1, keepdim=True)
std_rewards = rewards.std(dim=1, keepdim=True) + eps
advantages = (rewards - mean_rewards) / std_rewards
```

**Pattern: Temperature-Scaled Normalization**

```python
# Add temperature parameter for controlling distribution spread
advantages = (rewards - mean_rewards) / (std_rewards * tau)
```

### Clipping Strategies

Clipping prevents extreme advantages from destabilizing training.

**Pattern: Symmetric Clipping**

```python
# Clip advantages to fixed range
clip_range = 10.0  # or configurable
advantages = torch.clamp(advantages, -clip_range, clip_range)
```

**Pattern: Asymmetric Clipping**

```python
# Different bounds for positive/negative advantages
advantages = torch.clamp(advantages, min=-5.0, max=10.0)
```

**Pattern: Soft Clipping (Tanh)**

```python
# Smooth clipping using tanh scaling
advantages = clip_range * torch.tanh(advantages / clip_range)
```

### Outlier Handling

**Pattern: Winsorization**

```python
# Cap extreme values at percentiles
lower = torch.quantile(advantages, 0.01)
upper = torch.quantile(advantages, 0.99)
advantages = torch.clamp(advantages, lower, upper)
```

**Pattern: Robust Normalization (MAD)**

```python
# Use median absolute deviation instead of std
median = rewards.median(dim=1, keepdim=True).values
mad = (rewards - median).abs().median(dim=1, keepdim=True).values + eps
advantages = (rewards - median) / (mad * 1.4826)  # Scale factor for normal dist
```

### Centered vs. Standard Normalization

**Pattern: Centered Normalization (Mean Only)**

```python
# Useful for sparse rewards where variance is uninformative
advantages = rewards - mean_rewards
```

**Pattern: Full Normalization**

```python
# Standard z-score normalization
advantages = (rewards - mean_rewards) / (std_rewards + eps)
```

---

## Loss Computation Variants

### Standard GRPO Loss

```python
# Policy gradient with advantages
loss = -(advantages * log_probs).mean()
```

### DAPO (Decoupled Alignment via Policy Optimization)

**Pattern: Decoupled Positive/Negative Updates**

```python
# Separate treatment for positive and negative advantages
positive_mask = advantages > 0
negative_mask = advantages <= 0

positive_loss = -(advantages[positive_mask] * log_probs[positive_mask]).mean()
negative_loss = -(advantages[negative_mask] * log_probs[negative_mask]).mean()

loss = alpha * positive_loss + beta * negative_loss
```

### CISPO (Constrained Importance Sampling Policy Optimization)

**Pattern: Importance Sampling with Constraints**

```python
# Compute importance ratios
ratio = torch.exp(log_probs - old_log_probs)

# Clip ratios
ratio_clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

# Take minimum (pessimistic bound)
loss = -torch.min(
    ratio * advantages,
    ratio_clipped * advantages
).mean()
```

### KL Penalty Integration

**Pattern: Add KL Divergence Penalty**

```python
# Encourage staying close to reference policy
kl_penalty = kl_coef * (log_probs - ref_log_probs).mean()
loss = policy_loss + kl_penalty
```

**Pattern: Adaptive KL Coefficient**

```python
# Adjust KL coefficient based on actual KL
if actual_kl > target_kl * 1.5:
    kl_coef *= 2.0
elif actual_kl < target_kl / 1.5:
    kl_coef /= 2.0
```

### Reference Policy Updates

**Pattern: Exponential Moving Average**

```python
# Update reference policy with EMA
for ref_param, param in zip(ref_model.parameters(), model.parameters()):
    ref_param.data = tau * param.data + (1 - tau) * ref_param.data
```

### Clipping Bounds

**Pattern: Configurable Ratio Clipping**

```python
# Clipping bounds for policy ratio
clip_epsilon = 0.2  # Standard PPO value
# Lower for more conservative updates: 0.1
# Higher for more aggressive: 0.3
```

---

## Memory Optimization Patterns

### Streaming Advantage Computation

**Pattern: Compute Advantages in Chunks**

```python
# Process in chunks to reduce peak memory
chunk_size = 32
all_advantages = []

for i in range(0, len(rewards), chunk_size):
    chunk_rewards = rewards[i:i+chunk_size]
    chunk_advantages = compute_advantages(chunk_rewards)
    all_advantages.append(chunk_advantages)

advantages = torch.cat(all_advantages)
```

### Tensor Operation Batching

**Pattern: Fused Operations**

```python
# Avoid intermediate tensors
# Instead of:
# centered = rewards - mean
# scaled = centered / std
# advantages = scaled / tau

# Use:
advantages = torch.div(
    torch.sub(rewards, mean), 
    std * tau
)
```

**Pattern: In-Place Operations**

```python
# Reduce memory allocations
advantages = rewards.clone()
advantages.sub_(mean)
advantages.div_(std)
```

### Gradient Checkpointing

**Pattern: Selective Checkpointing**

```python
# Checkpoint specific transformer layers
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(model, hidden_states):
    # Checkpoint expensive transformer blocks
    for layer in model.transformer.layers:
        hidden_states = checkpoint(layer, hidden_states)
    return model.lm_head(hidden_states)
```

**Pattern: Enable in Model**

```python
# Enable gradient checkpointing on transformer
model.gradient_checkpointing_enable()
```

---

## Gradient Stability Patterns

### Gradient Monitoring

**Pattern: Norm Tracking**

```python
# Track gradient norms across training
grad_norms = []
for p in model.parameters():
    if p.grad is not None:
        grad_norms.append(p.grad.norm().item())

total_norm = torch.tensor(grad_norms).norm()
```

**Pattern: Anomaly Detection**

```python
# Detect gradient anomalies
if torch.isnan(total_norm) or torch.isinf(total_norm):
    logger.warning(f"Gradient anomaly at step {step}")
    # Skip update or reduce learning rate
```

### Gradient Clipping

**Pattern: Global Norm Clipping**

```python
# Standard approach in judicaita
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Pattern: Value Clipping**

```python
# Clip individual gradient values
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### Loss Scaling

**Pattern: Dynamic Loss Scaling (AMP)**

```python
# For mixed precision training
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    loss = compute_loss()

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Reward Computation Patterns

### Multi-Mode Rewards

**Pattern: Composable Reward Functions**

```python
# Similar to judicaita's CompositeReward
class MultiModeReward:
    def __init__(self, components, weights):
        self.components = components
        self.weights = weights
    
    def compute(self, prompt, response, reference):
        total = 0.0
        details = {}
        for comp, weight in zip(self.components, self.weights):
            score = comp.compute(prompt, response, reference)
            total += weight * score
            details[comp.name] = score
        return total, details
```

### Verifiable Rewards

**Pattern: Format Verification**

```python
# Check for required structure
def format_reward(response):
    has_reasoning = "<reasoning>" in response and "</reasoning>" in response
    has_answer = "<answer>" in response and "</answer>" in response
    return 1.0 if (has_reasoning and has_answer) else 0.0
```

**Pattern: Correctness Verification**

```python
# For tasks with verifiable answers
def correctness_reward(response, ground_truth):
    extracted = extract_answer(response)
    return 1.0 if extracted == ground_truth else 0.0
```

### Batched Calculation

**Pattern: Vectorized Reward Computation**

```python
# Compute rewards for batch in parallel
def batch_rewards(prompts, responses, references):
    # Vectorize where possible
    format_scores = torch.tensor([format_check(r) for r in responses])
    length_scores = compute_length_scores(responses)  # Vectorized
    
    return format_weight * format_scores + length_weight * length_scores
```

---

## Comparison to Judicaita Implementation

### Advantage Computation

| Aspect | Judicaita (`_compute_advantages`) | grpo_fast.py Patterns |
|--------|-----------------------------------|----------------------|
| Normalization | Group-relative with tau scaling | Multiple options (standard, centered, MAD) |
| Clipping | Not implemented | Symmetric, asymmetric, soft clipping |
| Outlier Handling | epsilon (1e-8) for division | Winsorization, quantile capping |
| When to Enhance | If experiencing extreme advantages | Apply clipping for stability |

**Enhancement Opportunity**: Add configurable clipping to `_compute_advantages()`:

```python
def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True) + 1e-8
    
    advantages = (rewards - mean_rewards) / (std_rewards * self.config.grpo_tau)
    
    # Optional clipping (from grpo_fast.py)
    if self.config.advantage_clip_range is not None:
        advantages = torch.clamp(
            advantages, 
            -self.config.advantage_clip_range, 
            self.config.advantage_clip_range
        )
    
    return advantages
```

### Loss Computation

| Aspect | Judicaita (`_grpo_step`) | grpo_fast.py Patterns |
|--------|--------------------------|----------------------|
| Loss Function | Standard policy gradient | DAPO, CISPO, KL penalty options |
| Clipping | No ratio clipping | PPO-style ratio clipping |
| Reference Policy | Not used | EMA updates, KL penalties |
| When to Enhance | For more stable updates | Add ratio clipping for safety |

### Memory Patterns

| Aspect | Judicaita | grpo_fast.py Patterns |
|--------|-----------|----------------------|
| Gradient Accumulation | ✅ Implemented | Similar |
| Checkpointing | Config option | Selective layer checkpointing |
| Batch Processing | Sequential rollouts | Streaming chunks |
| When to Enhance | OOM errors | Apply streaming patterns |

---

## When to Apply

### Apply Advantage Clipping When:

- Training loss becomes unstable
- Advantage values show extreme outliers
- Gradient norms spike unexpectedly
- **Hackathon Relevance**: High - stability is critical for limited compute time

### Apply KL Penalty When:

- Model drifts too far from base model
- Generated outputs lose coherence
- Need to maintain base model capabilities
- **Hackathon Relevance**: Medium - helps preserve Gemma capabilities

### Apply Memory Optimization When:

- Encountering OOM errors during training
- Need to increase effective batch size
- Training on TPU v2-8 with limited HBM
- **Hackathon Relevance**: High - TPU memory is constrained

### Apply Gradient Stability Patterns When:

- Training crashes unexpectedly
- Loss becomes NaN
- Need emergency recovery
- **Hackathon Relevance**: High - can save training runs

### Keep Current Implementation When:

- Training is stable and converging
- Memory usage is acceptable
- Reward signals are behaving as expected
- **Note**: The existing judicaita implementation is sufficient for hackathon requirements

---

## References

- [AllenAI Open Instruct](https://github.com/allenai/open-instruct)
- [GRPO Paper/Concept](https://arxiv.org/abs/2402.03300)
- [Judicaita GRPO Training Guide](../GRPO_TRAINING.md)
- [Judicaita GRPOTrainer](../../src/judicaita/training/grpo_trainer.py)
