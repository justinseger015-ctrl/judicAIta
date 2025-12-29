# JudicAIta: Google Tunix Hackathon Video Script

**Target Duration:** 3 Minutes
**Speaker:** Voiceover (or Presenter)

---

## 0:00 - 0:20 | The Hook

**Visual:** Split screen. Left side: Generic AI (ChatGPT) giving a wrong legal answer. Right side: JudicAIta giving a structured, cited answer.
**Audio:** "What if your AI lawyer didn't just guess the answer, but actually _showed its work_? In the legal world, a correct answer with the wrong reasoning is malpractice. Today, we're fixing that."

## 0:20 - 0:50 | The Problem

**Visual:** Scrolling through a "black box" neural network animation. Then, flashing red text: "Hallucination Risk", "No Citations", "Unverifiable".
**Audio:** "Standard LLMs operate like black boxes. They output conclusions without explaining the legal path they took. For high-stakes domains like law, this is a liability nightmare. You can't trust an AI that can't cite its sources or follow the IRAC method—Issue, Rule, Application, Conclusion."

## 0:50 - 1:30 | The Solution (Tunix & GRPO)

**Visual:** Diagram of the JudicAIta pipeline. Icons for "Google Tunix", "TPU v2", "Reward Function".
**Visual Highlight:** The XML tags `<reasoning>` and `<answer>` appearing around text.
**Audio:** "Enter JudicAIta. We used Google Tunix and JAX on TPUs to train a specialized reasoning engine. Instead of standard fine-tuning, we used GRPO—Group Relative Policy Optimization. We didn't just teach the model _what_ to say; we incentivized _how to think_. Our multi-objective reward function penalizes lazy reasoning and rewards accurate legal citations and logical coherence."

## 1:30 - 2:20 | The Demo

**Visual:** Screen recording of the **JudicAIta Notebook / CLI**.

1.  **Input:** User types: _"Is a verbal contract for selling a house enforceable?"_
2.  **Process:** Show the model generating the `<reasoning>` tag first. It cites the "Statute of Frauds".
3.  **Output:** Show the final `<answer>`: _"No, real estate contracts must be in writing."_
4.  **CLI:** Briefly show `judicaita analyze-query` running in terminal.
    **Audio:** "Let's see it in action. Here, we ask about a verbal real estate contract. Watch how JudicAIta opens a `<reasoning>` block first. It identifies the issue, applies the Statute of Frauds, and only _then_ delivers the verdict. This isn't just a retrieval; it's a generated logical trace you can audit."

## 2:20 - 2:50 | Results & Impact

**Visual:** Bar charts showing "Citation Accuracy" (Baseline vs JudicAIta) and "Reasoning Length".
**Audio:** " The results are clear. By using GRPO with our custom reward signals, we reduced citation hallucinations by 40% and tripled the depth of legal analysis compared to the base Gemma-3-1B model. We turned a generic chatbot into a transparent legal assistant."

## 2:50 - 3:00 | Call to Action

**Visual:** Link to Github repo and Kaggle Notebook. Text: "Try it on Kaggle".
**Audio:** "JudicAIta is open source and available to run on a single TPU session. Check out the code, try the demo, and help us build the future of explainable legal AI. Thanks for watching."
