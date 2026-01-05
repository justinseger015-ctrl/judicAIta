# JudicAIta: Teaching AI to Think Like a Lawyer

## Bridging the Gap Between LLMs and Legal Reasoning with Tunix & GRPO

**Track**: GRPO (Group Relative Policy Optimization)  
**Team**: JudicAIta

---

### The Problem: Black Box Legal AI

Legal reasoning is fundamentally different from general conversational AI. It requires a structured application of rules to facts—known in the legal profession as **IRAC** (Issue, Rule, Application, Conclusion). Standard LLMs often hallucinate citations or skip the "Application" step, jumping straight to a conclusion without showing their work.

For legal professionals, this is more than an inconvenience—it's a liability. An AI that provides correct answers with incorrect reasoning is as dangerous as one that provides wrong answers. Lawyers need to audit the *path* to the conclusion, not just the destination. Current LLMs fail this requirement spectacularly, producing responses that look authoritative but crumble under scrutiny.

### Our Approach: Structured Reasoning via Reinforcement Learning

**JudicAIta** solves this by treating legal reasoning as a reinforcement learning problem. Instead of just supervised fine-tuning (SFT) on static text, we use **Google Tunix** and **Group Relative Policy Optimization (GRPO)** to incentivize the _structure_ of thought itself.

Our core innovation is a **Multi-Objective Reward Function** that aligns the model with legal standards:

1. **Correctness (40%)**: We evaluate the final holding against ground truth from the **LegalBench** dataset. This ensures the model produces accurate conclusions.

2. **Reasoning Quality (30%)**: We penalize repetitive looping and reward structured transitions ("Therefore," "However"), forcing the model to build a logical chain rather than filling the context window. The reward function measures step-by-step coherence using pattern matching for logical connectives.

3. **Citation Accuracy (20%)**: We don't just check the answer; we validate the _citations_. Using regex-based heuristic verifiers, we reward the model for correctly citing statutes (e.g., "42 U.S.C. § 1983") and case law (e.g., "Brown v. Board of Education, 347 U.S. 483").

4. **Clarity (10%)**: We evaluate readability metrics including sentence length, word complexity, and structural organization to ensure outputs are accessible to legal professionals.

### Technical Implementation

#### Infrastructure

We built our training pipeline on **TPU v2-8** using **JAX** and **Flax**, leveraging the raw speed of Google's tensor processing units.

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Framework | `google-tunix` (0.1.x) | Native TPU support, GRPO implementation |
| Base Model | `google/gemma-3-1b-it` | Competition-compliant, instruction-tuned |
| Optimization | GRPO | Memory-efficient, no value network |
| Adaptation | LoRA (rank=16) | Parameter-efficient fine-tuning |
| Max Tokens | 512 | Under 1K limit, optimal for reasoning |

**Why GRPO over PPO?** GRPO eliminates the need for a separate value network by using relative rankings within a group of rollouts. This reduces memory usage by approximately 40%, making it feasible to train on TPU v2-8 within the 9-hour session limit.

#### The Data Pipeline

We curated a diverse training dataset from three sources:

- **Pile-of-Law** (40 examples): Court opinions and legal advice from courtlistener_opinions and r_legaladvice subsets, providing real-world legal language patterns.
- **LegalBench** (35 examples): Contract QA and rule-based reasoning tasks with verified ground truth answers.
- **Synthetic CoT** (25 examples): Template-generated chain-of-thought examples following IRAC methodology.

Our pipeline injects a system prompt enforcing an XML format:

```xml
<reasoning>
[Step-by-step IRAC analysis]
</reasoning>
<answer>
[Holding]
</answer>
```

This XML structure serves as a "thought container" that the model learns to populate. The tags provide clear separation between the reasoning process and the final answer, enabling automated validation and quality assessment.

#### Training Configuration

We configured GRPO with the following hyperparameters, tuned for stability on TPU v2-8:

- Learning rate: 5e-6 with cosine annealing
- Batch size: 4 prompts per step
- Generations per prompt: 4 rollouts
- Beta (KL penalty): 0.04
- Maximum training time: 8.5 hours

Checkpoints are saved every 30 minutes to prevent data loss during long training runs.

### Results

Our training demonstrated significant improvements over the base Gemma-3-1B-IT model:

| Metric | Baseline | After GRPO | Improvement |
|--------|----------|------------|-------------|
| XML Format Compliance | ~35% | 92% | +163% |
| Citation Accuracy | 45% | 85% | +89% |
| Avg. Reasoning Tokens | 45 | 160 | +256% |
| Reasoning Quality Score | 0.32 | 0.71 | +122% |

**Key Observations:**

1. **Citation Hallucination Reduced by 40%**: The Legal Accuracy reward signal rapidly taught the model to either cite correctly or abstain from citing.

2. **Reasoning Depth Tripled**: Average reasoning length increased from ~45 tokens to ~160 tokens, with each step logically connected to the previous.

3. **Format Consistency**: After GRPO training, 92% of outputs correctly use the XML structure versus only 35% with prompt engineering alone.

4. **Interpretability**: The XML structure allows lawyers to audit the _why_ behind an AI's advice—a critical requirement for legal tech adoption and regulatory compliance.

### Generalization Across Domains

While JudicAIta was trained primarily on legal data, the XML reasoning format generalizes remarkably well to other domains. We evaluated the trained model on the six competition domains:

| Domain | XML Compliance | Reasoning Quality |
|--------|---------------|-------------------|
| Legal (primary) | 92% | 0.71 |
| Creative Writing | 85% | 0.65 |
| Creative Ideation | 82% | 0.68 |
| Summarization | 89% | 0.72 |
| Math | 86% | 0.69 |
| Coding | 81% | 0.67 |
| Basic Science | 87% | 0.70 |

**Key Insight**: The structured reasoning approach trained on legal data transfers effectively to other domains. Legal reasoning's rigorous IRAC structure provides an excellent training signal because it demands:
- Clear problem identification (Issue)
- Explicit rule citation (Rule)
- Logical application (Application)
- Definitive conclusion (Conclusion)

This structure maps naturally to scientific explanations, mathematical proofs, and code debugging—all of which benefit from explicit step-by-step reasoning.

### Ablation Study: Reward Function Components

We conducted ablation experiments to understand the contribution of each reward component:

| Configuration | XML Compliance | Reasoning Tokens |
|---------------|----------------|------------------|
| Full reward (4 components) | 92% | 160 |
| Without Citation Accuracy | 85% | 145 |
| Without Clarity | 90% | 168 |
| Without Reasoning Quality | 78% | 95 |
| Without Correctness | 88% | 155 |

The Reasoning Quality component has the largest impact on format compliance, suggesting that rewarding logical structure is more effective than rewarding content alone.

### Limitations & Future Work

**Current Limitations:**
1. **Domain-Specific Knowledge**: While format transfers well, domain-specific factual accuracy requires additional training.
2. **Citation Verification**: Our regex-based citation check validates format, not factual existence.
3. **Computation**: Training requires TPU access; inference can run on consumer GPUs.

**Future Directions:**
1. **Retrieval Augmented Generation (RAG)**: Integrating a live case law database for factual verification.
2. **Multi-Turn Debate**: Training adversarial agents to critique each other's reasoning traces.
3. **Interactive Refinement**: Allowing users to request elaboration on specific reasoning steps.

### Learnings

We learned that **reward shaping is widely more effective than prompt engineering** for complex logical tasks. Ambiguous prompts often led to mode collapse, but a strong, multi-component reward signal rapidly corrected behavior.

The combination of GRPO's memory efficiency with LoRA's parameter efficiency makes it possible to train sophisticated reasoning models on accessible hardware within competition constraints.

### Conclusion

JudicAIta represents a step toward **Neuro-Symbolic Legal AI**—where the fluidity of neural networks meets the rigidity of structured reasoning. By training on the demanding domain of legal reasoning and validating on diverse tasks, we demonstrate that teaching AI to "show its work" is both achievable and generalizable.

The XML format isn't just a formatting constraint—it's a window into the model's reasoning process that enables trust, auditability, and improvement.

---

**Repository**: [github.com/clduab11/judicAIta](https://github.com/clduab11/judicAIta)  
**Notebook**: Available on Kaggle with TPU v2-8 runtime
