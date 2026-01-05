# JudicAIta: Google Tunix Hackathon Video Script

**Target Duration:** 3 Minutes (≤180 seconds)  
**Word Count Target:** ~400 words (roughly 130 words/minute)  
**Speaker:** Voiceover (or Presenter)

---

## 0:00 - 0:15 | The Hook

**Visual:** Split screen comparing generic AI output vs. JudicAIta's structured XML output.

**Audio:** "What if your AI didn't just guess the answer, but actually _showed its work_? In law, a correct answer with wrong reasoning is malpractice. Today, we're fixing that."

## 0:15 - 0:40 | The Problem

**Visual:** Animation of a "black box" neural network with flashing warnings: "Hallucination Risk," "No Citations," "Unverifiable."

**Audio:** "Standard LLMs operate like black boxes—conclusions without explanations. For legal AI, this is a liability. You can't trust what you can't audit."

## 0:40 - 1:20 | The Solution (Tunix & GRPO)

**Visual:** Diagram of JudicAIta pipeline with icons for Tunix, TPU, and Reward Function. Highlight `<reasoning>` and `<answer>` XML tags appearing.

**Audio:** "JudicAIta uses Google Tunix and GRPO on TPUs. Instead of teaching the model _what_ to say, we incentivize _how to think_. Our multi-objective reward function penalizes lazy reasoning and rewards accurate citations, logical structure, and clarity. The model learns to wrap its thinking in XML tags—making every step auditable."

## 1:20 - 2:10 | The Demo

**Visual:** Screen recording of notebook running on TPU.

1. **Input:** User asks: "Is a verbal contract for selling a house enforceable?"
2. **Process:** Show `<reasoning>` tag being generated with Statute of Frauds citation.
3. **Output:** Final `<answer>` tag with clear conclusion.

**Audio:** "Let's see it work. We ask about verbal real estate contracts. Watch how JudicAIta opens a reasoning block first—it identifies the issue, applies the Statute of Frauds, then delivers the verdict. This isn't retrieval; it's generated reasoning you can audit."

## 2:10 - 2:40 | Results

**Visual:** Bar charts showing before/after metrics: XML Compliance (35%→92%), Reasoning Tokens (45→160).

**Audio:** "Results speak clearly. GRPO training increased XML format compliance from 35% to 92% and tripled reasoning depth. The model doesn't just answer—it explains."

## 2:40 - 3:00 | Call to Action

**Visual:** GitHub and Kaggle links. Text: "Try it on Kaggle TPU."

**Audio:** "JudicAIta is open source. Run it on a single TPU session, explore the code, and help build the future of explainable AI. Thanks for watching."

---

**Estimated Word Count:** ~385 words
