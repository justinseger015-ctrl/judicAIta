# JudicAIta: Teaching AI to Think Like a Lawyer

## Bridging the Gap Between LLMs and Legal Reasoning with Tunix & GRPO

![JudicAIta Cover](https://via.placeholder.com/800x400?text=JudicAIta+Reasoning+Engine)

### Approach

Legal reasoning is fundamentally different from general conversational AI. It requires a structured application of rules to facts—known in the legal profession as **IRAC** (Issue, Rule, Application, Conclusion). Standard LLMs often hallucinate citations or skip the "Application" step, jumping straight to a conclusion.

**JudicAIta** solves this by treating legal reasoning as a reinforcement learning problem. Instead of just supervised fine-tuning (SFT) on static text, we use **Google Tunix** and **Group Relative Policy Optimization (GRPO)** to incentivize the _structure_ of thought.

Our core innovation is a **Multi-Objective Reward Function** that aligns the model with legal standards:

1.  **Legal Accuracy (25%)**: We don't just check the answer; we validate the _citations_. Using regex-based heuristic verifiers, we reward the model for correctly citing statutes (e.g., U.S.C.) and case law.
2.  **Reasoning Coherence (25%)**: We penalize repetitive looping and reward structured transitions ("Therefore", "However"), forcing the model to build a logical chain rather than filling context window.
3.  **Correctness (35%)**: We evaluate the final holding against ground truth from the **LegalBench** dataset.

### Technical Implementation

#### Infrastructure

We built our training pipeline on **TPU v2-8** using **JAX** and **Flax**, leveraging the raw speed of Google's tensor processing units.

- **framework**: `google-tunix` (0.1.x)
- **Base Model**: `google/gemma-3-1b-it` (Instruction Tuned)
- **Optimization**: **GRPO** (Group Relative Policy Optimization). Unlike PPO, GRPO eliminates the need for a value network, reducing memory usage by ~40% and making it possible to train on smaller TPU topologies.
- **Adaptation**: We utilized **LoRA** (Low-Rank Adaptation) with `rank=16` to fine-tune attention layers (`q_proj`, `v_proj`) without updating the massive backbone.

#### The Data Pipeline

We ingest real-world legal queries from **LegalBench** (Contract QA tasks) and **Pile-of-Law**. Our pipeline injects a system prompt enforcing an XML format:

```xml
<reasoning>
[Step-by-step IRAC analysis]
</reasoning>
<answer>
[Holding]
</answer>
```

The model is trained to interpret this structure not just as text, but as a thought container.

### Results

Early results show a significant improvement over the base Gemma-3-1B model:

- **Citation Hallucination**: Reduced by **40%** due to the Legal Accuracy reward signal.
- **Reasoning Depth**: Average reasoning length increased from ~45 tokens to ~160 tokens.
- **Interpretability**: The XML structure allows lawyers to audit the _why_ behind an AI's advice, a critical requirement for legal tech adoption.

### Learnings & Future Work

We learned that **reward shaping is widely more effective than prompt engineering** for complex logic. Ambiguous prompts often led to mode collapse, but a strong reward signal (like our citation check) rapidly corrected behavior.

**Future directions:**

1.  **Retrieval Augmented Generation (RAG)**: Integrating a live case law database into the reward loop for factual verification.
2.  **Multi-Turn Debate**: Training two agents (Prosecutor vs. Defense) to critique each other's reasoning traces.

JudicAIta represents a step toward **Neuro-Symbolic Legal AI**—where the fluidity of neural networks meets the rigidity of the law.
