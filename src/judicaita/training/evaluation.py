"""
Evaluation harness for GRPO-trained models.

This module provides evaluation utilities including ROUGE/BLEU metrics
for synthetic traces and accuracy metrics on LegalBench tasks.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets import Dataset
from loguru import logger


@dataclass
class EvaluationResult:
    """Results from model evaluation."""

    rouge_scores: dict[str, float]
    bleu_score: float
    task_accuracy: float
    format_score: float
    avg_length: float
    num_samples: int
    per_sample_scores: list[dict[str, Any]]


class EvaluationHarness:
    """
    Evaluation harness for GRPO-trained legal reasoning models.

    Provides comprehensive evaluation including:
    - ROUGE/BLEU for trace quality
    - Task accuracy on LegalBench
    - Format compliance scores
    """

    def __init__(self) -> None:
        """Initialize evaluation harness."""
        self._rouge = None
        self._bleu = None
        self._load_metrics()

    def _load_metrics(self) -> None:
        """Load evaluation metrics from HuggingFace datasets."""
        try:
            from datasets import load_metric

            self._rouge = load_metric("rouge")
            self._bleu = load_metric("bleu")
            logger.info("Evaluation metrics loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load HuggingFace metrics: {e}")
            logger.info("Will use fallback metric implementations")

    def evaluate_model(
        self,
        model: Any,
        tokenizer: Any,
        eval_dataset: Dataset,
        max_samples: int | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate model on dataset.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            eval_dataset: Evaluation dataset
            max_samples: Maximum number of samples to evaluate
            generation_config: Configuration for text generation

        Returns:
            EvaluationResult with comprehensive metrics
        """
        logger.info("Starting model evaluation...")

        if max_samples is not None:
            eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

        logger.info(f"Evaluating on {len(eval_dataset)} samples")

        # Default generation config
        if generation_config is None:
            generation_config = {
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
            }

        # Generate responses
        predictions = []
        references = []
        per_sample_scores = []

        for i, sample in enumerate(eval_dataset):
            if i % 10 == 0:
                logger.info(f"Evaluating sample {i}/{len(eval_dataset)}")

            prompt = sample["prompt"]
            reference = sample.get("response", "") or sample.get("cot_reasoning", "")

            # Generate prediction
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

            with model.no_grad():
                outputs = model.generate(**inputs, **generation_config)

            prediction = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            predictions.append(prediction)
            references.append(reference)

            # Compute per-sample metrics
            sample_scores = self._compute_sample_metrics(prediction, reference)
            per_sample_scores.append(sample_scores)

        # Compute aggregate metrics
        rouge_scores = self._compute_rouge(predictions, references)
        bleu_score = self._compute_bleu(predictions, references)
        task_accuracy = self._compute_task_accuracy(per_sample_scores)
        format_score = self._compute_format_score(per_sample_scores)
        avg_length = np.mean([len(p.split()) for p in predictions])

        result = EvaluationResult(
            rouge_scores=rouge_scores,
            bleu_score=bleu_score,
            task_accuracy=task_accuracy,
            format_score=format_score,
            avg_length=avg_length,
            num_samples=len(eval_dataset),
            per_sample_scores=per_sample_scores,
        )

        logger.info("Evaluation completed!")
        self._log_results(result)

        return result

    def _compute_sample_metrics(self, prediction: str, reference: str) -> dict[str, Any]:
        """Compute metrics for a single sample."""
        import re

        # Check format compliance
        has_steps = bool(re.search(r"Step\s+\d+:", prediction, re.IGNORECASE))
        has_conclusion = bool(re.search(r"Conclusion:", prediction, re.IGNORECASE))
        num_steps = len(re.findall(r"Step\s+\d+:", prediction, re.IGNORECASE))

        # Check task correctness (simple token overlap)
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        overlap = len(pred_tokens & ref_tokens) / len(ref_tokens) if ref_tokens else 0.0

        return {
            "has_steps": has_steps,
            "has_conclusion": has_conclusion,
            "num_steps": num_steps,
            "format_compliant": has_steps and has_conclusion and num_steps >= 2,
            "token_overlap": overlap,
            "length": len(prediction.split()),
        }

    def _compute_rouge(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        """Compute ROUGE scores."""
        if self._rouge is not None:
            try:
                results = self._rouge.compute(
                    predictions=predictions, references=references, use_stemmer=True
                )

                return {
                    "rouge1": results["rouge1"].mid.fmeasure,
                    "rouge2": results["rouge2"].mid.fmeasure,
                    "rougeL": results["rougeL"].mid.fmeasure,
                }
            except Exception as e:
                logger.warning(f"ROUGE computation failed: {e}")

        # Fallback: simple n-gram overlap
        return self._compute_rouge_fallback(predictions, references)

    def _compute_rouge_fallback(
        self, predictions: list[str], references: list[str]
    ) -> dict[str, float]:
        """Fallback ROUGE implementation using simple n-gram overlap."""
        rouge1_scores = []
        rouge2_scores = []

        for pred, ref in zip(predictions, references, strict=False):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            # ROUGE-1: unigram overlap
            pred_unigrams = set(pred_tokens)
            ref_unigrams = set(ref_tokens)
            if ref_unigrams:
                rouge1 = len(pred_unigrams & ref_unigrams) / len(ref_unigrams)
            else:
                rouge1 = 0.0
            rouge1_scores.append(rouge1)

            # ROUGE-2: bigram overlap
            pred_bigrams = set(zip(pred_tokens[:-1], pred_tokens[1:], strict=False))
            ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:], strict=False))
            if ref_bigrams:
                rouge2 = len(pred_bigrams & ref_bigrams) / len(ref_bigrams)
            else:
                rouge2 = 0.0
            rouge2_scores.append(rouge2)

        return {
            "rouge1": np.mean(rouge1_scores),
            "rouge2": np.mean(rouge2_scores),
            "rougeL": np.mean(rouge1_scores),  # Use ROUGE-1 as approximation
        }

    def _compute_bleu(self, predictions: list[str], references: list[str]) -> float:
        """Compute BLEU score."""
        if self._bleu is not None:
            try:
                # Format references for BLEU metric
                references_formatted = [[ref] for ref in references]

                results = self._bleu.compute(
                    predictions=predictions, references=references_formatted
                )

                return results["bleu"]
            except Exception as e:
                logger.warning(f"BLEU computation failed: {e}")

        # Fallback: simple BLEU approximation
        return self._compute_bleu_fallback(predictions, references)

    def _compute_bleu_fallback(self, predictions: list[str], references: list[str]) -> float:
        """Fallback BLEU implementation."""
        scores = []

        for pred, ref in zip(predictions, references, strict=False):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            # Simple unigram precision
            if pred_tokens:
                matches = sum(1 for token in pred_tokens if token in ref_tokens)
                precision = matches / len(pred_tokens)
            else:
                precision = 0.0

            # Length penalty
            bp = (
                1.0
                if len(pred_tokens) >= len(ref_tokens)
                else np.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))
            )

            scores.append(bp * precision)

        return np.mean(scores)

    def _compute_task_accuracy(self, per_sample_scores: list[dict[str, Any]]) -> float:
        """Compute task accuracy based on token overlap."""
        if not per_sample_scores:
            return 0.0

        # Consider task correct if token overlap > 0.5
        correct = sum(1 for s in per_sample_scores if s["token_overlap"] > 0.5)
        return correct / len(per_sample_scores)

    def _compute_format_score(self, per_sample_scores: list[dict[str, Any]]) -> float:
        """Compute format compliance score."""
        if not per_sample_scores:
            return 0.0

        compliant = sum(1 for s in per_sample_scores if s["format_compliant"])
        return compliant / len(per_sample_scores)

    def _log_results(self, result: EvaluationResult) -> None:
        """Log evaluation results."""
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Number of samples: {result.num_samples}")
        logger.info(f"Average length: {result.avg_length:.1f} tokens")
        logger.info("")
        logger.info("ROUGE Scores:")
        logger.info(f"  ROUGE-1: {result.rouge_scores['rouge1']:.4f}")
        logger.info(f"  ROUGE-2: {result.rouge_scores['rouge2']:.4f}")
        logger.info(f"  ROUGE-L: {result.rouge_scores['rougeL']:.4f}")
        logger.info("")
        logger.info(f"BLEU Score: {result.bleu_score:.4f}")
        logger.info(f"Task Accuracy: {result.task_accuracy:.4f}")
        logger.info(f"Format Compliance: {result.format_score:.4f}")
        logger.info("=" * 60)


def evaluate_checkpoint(
    checkpoint_path: str,
    eval_dataset: Dataset,
    max_samples: int | None = None,
    output_file: str | None = None,
) -> EvaluationResult:
    """
    Evaluate a model checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        eval_dataset: Evaluation dataset
        max_samples: Maximum number of samples to evaluate
        output_file: Optional file to save results

    Returns:
        EvaluationResult
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model from {checkpoint_path}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, revision="main", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, revision="main")

    # Create evaluation harness
    harness = EvaluationHarness()

    # Run evaluation
    results = harness.evaluate_model(model, tokenizer, eval_dataset, max_samples)

    # Save results if requested
    if output_file:
        import json

        with open(output_file, "w") as f:
            json.dump(
                {
                    "rouge_scores": results.rouge_scores,
                    "bleu_score": results.bleu_score,
                    "task_accuracy": results.task_accuracy,
                    "format_score": results.format_score,
                    "avg_length": results.avg_length,
                    "num_samples": results.num_samples,
                },
                f,
                indent=2,
            )
        logger.info(f"Results saved to {output_file}")

    return results
