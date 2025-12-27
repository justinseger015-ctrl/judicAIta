"""
Data curation module for GRPO training pipeline.

This module handles loading and preprocessing datasets from LegalBench and Pile-of-Law,
along with synthetic Chain-of-Thought (CoT) generation for reasoning traces.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LegalBenchTask:
    """Configuration for a LegalBench reasoning task."""

    name: str
    description: str
    reasoning_type: str  # e.g., "rule_qa", "contract_qa", "issue_spotting"
    num_samples: int = 1000


# Reasoning-heavy tasks from LegalBench
REASONING_TASKS = [
    LegalBenchTask(
        name="rule_qa",
        description="Rule-based question answering requiring step-by-step legal reasoning",
        reasoning_type="rule_qa",
        num_samples=1000,
    ),
    LegalBenchTask(
        name="contract_qa",
        description="Contract analysis with multi-step reasoning",
        reasoning_type="contract_qa",
        num_samples=1000,
    ),
    LegalBenchTask(
        name="issue_spotting",
        description="Legal issue identification requiring analytical reasoning",
        reasoning_type="issue_spotting",
        num_samples=1000,
    ),
]


class LegalBenchDataset:
    """
    Loader for LegalBench dataset with focus on reasoning-heavy tasks.

    This class handles loading specific reasoning tasks from the LegalBench dataset
    and preparing them for GRPO training.
    """

    def __init__(self, tasks: list[LegalBenchTask] | None = None) -> None:
        """
        Initialize LegalBench dataset loader.

        Args:
            tasks: List of reasoning tasks to load. If None, uses default reasoning tasks.
        """
        self.tasks = tasks or REASONING_TASKS
        self.dataset: Dataset | None = None

    def load(self, cache_dir: Path | None = None) -> Dataset:
        """
        Load LegalBench dataset for specified reasoning tasks.

        Args:
            cache_dir: Optional cache directory for downloaded datasets

        Returns:
            Dataset: Loaded and filtered LegalBench dataset
        """
        logger.info(f"Loading LegalBench dataset with {len(self.tasks)} reasoning tasks...")

        all_samples = []

        try:
            # Load the nguha/legalbench dataset
            dataset = load_dataset("nguha/legalbench", cache_dir=cache_dir)

            # Filter for reasoning-heavy tasks
            for task in self.tasks:
                logger.info(f"Processing task: {task.name}")

                # Extract relevant samples based on task type
                # Note: Actual task filtering depends on LegalBench structure
                # This is a placeholder that should be adapted to actual dataset schema
                if isinstance(dataset, DatasetDict):
                    task_samples = self._filter_task_samples(
                        dataset, task.name, task.num_samples
                    )
                    all_samples.extend(task_samples)

            # Create unified dataset
            self.dataset = Dataset.from_pandas(pd.DataFrame(all_samples))
            logger.info(f"Loaded {len(self.dataset)} samples from LegalBench")

            return self.dataset

        except Exception as e:
            logger.error(f"Failed to load LegalBench dataset: {e}")
            # Return empty dataset as fallback
            self.dataset = Dataset.from_dict({"prompt": [], "response": [], "task": []})
            return self.dataset

    def _filter_task_samples(
        self, dataset: DatasetDict, task_name: str, num_samples: int
    ) -> list[dict[str, Any]]:
        """
        Filter dataset samples for a specific task.

        Args:
            dataset: Full LegalBench dataset
            task_name: Name of the task to filter
            num_samples: Maximum number of samples to extract

        Returns:
            List of sample dictionaries
        """
        samples = []

        # Placeholder implementation - adapt based on actual dataset structure
        # This assumes dataset has 'train' split with 'task' and 'question' fields
        try:
            if "train" in dataset:
                train_data = dataset["train"]
                for i, item in enumerate(train_data):
                    if i >= num_samples:
                        break
                    # Extract prompt and create sample
                    samples.append(
                        {
                            "prompt": item.get("question", ""),
                            "response": item.get("answer", ""),
                            "task": task_name,
                            "context": item.get("context", ""),
                        }
                    )
        except Exception as e:
            logger.warning(f"Could not filter samples for task {task_name}: {e}")

        return samples


class PileOfLawDataset:
    """
    Loader for Pile-of-Law dataset subsets for domain adaptation.

    This class handles loading and subsampling specific legal document types
    from the Pile-of-Law dataset.
    """

    # Relevant subsets from Pile-of-Law
    SUBSETS = ["courtlistener_opinions", "uscode", "contracts"]

    def __init__(
        self, subsets: list[str] | None = None, samples_per_subset: int = 5000
    ) -> None:
        """
        Initialize Pile-of-Law dataset loader.

        Args:
            subsets: List of subsets to load. If None, uses default subsets.
            samples_per_subset: Number of samples to load from each subset
        """
        self.subsets = subsets or self.SUBSETS
        self.samples_per_subset = samples_per_subset
        self.dataset: Dataset | None = None

    def load(self, cache_dir: Path | None = None) -> Dataset:
        """
        Load and subsample Pile-of-Law dataset.

        Args:
            cache_dir: Optional cache directory for downloaded datasets

        Returns:
            Dataset: Loaded and subsampled Pile-of-Law dataset
        """
        logger.info(
            f"Loading Pile-of-Law dataset with {len(self.subsets)} subsets, "
            f"{self.samples_per_subset} samples each..."
        )

        all_samples = []

        try:
            # Load the pile-of-law dataset
            for subset in self.subsets:
                logger.info(f"Loading subset: {subset}")

                try:
                    # Load specific subset
                    dataset = load_dataset(
                        "pile-of-law/pile-of-law",
                        subset,
                        split=f"train[:{self.samples_per_subset}]",
                        cache_dir=cache_dir,
                    )

                    # Extract text content
                    for item in dataset:
                        all_samples.append(
                            {"text": item.get("text", ""), "source": subset, "type": "pretraining"}
                        )

                except Exception as e:
                    logger.warning(f"Could not load subset {subset}: {e}")
                    continue

            # Create unified dataset
            self.dataset = Dataset.from_pandas(pd.DataFrame(all_samples))
            logger.info(f"Loaded {len(self.dataset)} samples from Pile-of-Law")

            return self.dataset

        except Exception as e:
            logger.error(f"Failed to load Pile-of-Law dataset: {e}")
            # Return empty dataset as fallback
            self.dataset = Dataset.from_dict({"text": [], "source": [], "type": []})
            return self.dataset


class SyntheticCoTGenerator:
    """
    Generator for synthetic Chain-of-Thought (CoT) completions.

    Uses a strong frontier model to generate step-by-step reasoning traces
    for legal prompts from LegalBench.
    """

    def __init__(
        self, model_name: str = "google/gemma-2-27b-it", device: str = "cuda"
    ) -> None:
        """
        Initialize synthetic CoT generator.

        Args:
            model_name: Name of the model to use for CoT generation
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device
        self.model: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None

    def initialize(self) -> None:
        """Initialize the model and tokenizer for CoT generation."""
        logger.info(f"Initializing CoT generator with model: {self.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=self.device,
            )
            logger.info("CoT generator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CoT generator: {e}")
            raise

    def generate_cot(self, prompt: str, context: str = "") -> str:
        """
        Generate synthetic Chain-of-Thought reasoning for a legal prompt.

        Args:
            prompt: The legal question or prompt
            context: Optional context information

        Returns:
            Generated CoT reasoning trace
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("CoT generator not initialized. Call initialize() first.")

        # Create CoT prompt
        cot_prompt = self._create_cot_prompt(prompt, context)

        # Generate CoT reasoning
        inputs = self.tokenizer(cot_prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (remove prompt)
        cot_reasoning = generated_text[len(cot_prompt) :].strip()

        return cot_reasoning

    def _create_cot_prompt(self, prompt: str, context: str = "") -> str:
        """
        Create a prompt that encourages step-by-step reasoning.

        Args:
            prompt: The legal question
            context: Optional context

        Returns:
            Formatted prompt for CoT generation
        """
        cot_template = """You are a legal reasoning assistant. Analyze the following legal question with detailed step-by-step reasoning.

Question: {prompt}
{context_section}

Provide a detailed analysis with the following structure:
Step 1: Identify the applicable legal rule or principle
Step 2: Map the facts to the legal elements
Step 3: Apply the rule to the facts
Conclusion: State the final answer with reasoning

Analysis:"""

        context_section = f"\nContext: {context}" if context else ""

        return cot_template.format(prompt=prompt, context_section=context_section)

    def generate_dataset_cot(self, dataset: Dataset, num_samples: int | None = None) -> Dataset:
        """
        Generate CoT traces for an entire dataset.

        Args:
            dataset: Input dataset with prompts
            num_samples: Number of samples to process (None for all)

        Returns:
            Dataset with added CoT reasoning traces
        """
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        logger.info(f"Generating CoT traces for {len(dataset)} samples...")

        cot_traces = []

        for i, sample in enumerate(dataset):
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(dataset)} samples")

            try:
                prompt = sample.get("prompt", "")
                context = sample.get("context", "")
                cot_trace = self.generate_cot(prompt, context)
                cot_traces.append(cot_trace)

            except Exception as e:
                logger.warning(f"Failed to generate CoT for sample {i}: {e}")
                cot_traces.append("")

        # Add CoT traces to dataset
        dataset = dataset.add_column("cot_reasoning", cot_traces)

        logger.info("CoT generation completed")
        return dataset


def create_training_dataset(
    legalbench_tasks: list[LegalBenchTask] | None = None,
    pile_of_law_subsets: list[str] | None = None,
    samples_per_subset: int = 5000,
    generate_cot: bool = True,
    cot_model: str = "google/gemma-2-27b-it",
    cache_dir: Path | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Create complete training dataset with LegalBench + Pile-of-Law.

    Args:
        legalbench_tasks: LegalBench reasoning tasks to include
        pile_of_law_subsets: Pile-of-Law subsets to include
        samples_per_subset: Number of samples per Pile-of-Law subset
        generate_cot: Whether to generate synthetic CoT traces
        cot_model: Model to use for CoT generation
        cache_dir: Cache directory for datasets

    Returns:
        Tuple of (legalbench_dataset, pile_of_law_dataset)
    """
    logger.info("Creating training dataset...")

    # Load LegalBench dataset
    legalbench_loader = LegalBenchDataset(tasks=legalbench_tasks)
    legalbench_data = legalbench_loader.load(cache_dir=cache_dir)

    # Generate synthetic CoT traces if requested
    if generate_cot and len(legalbench_data) > 0:
        logger.info("Generating synthetic CoT traces...")
        cot_generator = SyntheticCoTGenerator(model_name=cot_model)
        cot_generator.initialize()
        legalbench_data = cot_generator.generate_dataset_cot(legalbench_data)

    # Load Pile-of-Law dataset
    pile_of_law_loader = PileOfLawDataset(
        subsets=pile_of_law_subsets, samples_per_subset=samples_per_subset
    )
    pile_of_law_data = pile_of_law_loader.load(cache_dir=cache_dir)

    logger.info(
        f"Training dataset created: {len(legalbench_data)} LegalBench samples, "
        f"{len(pile_of_law_data)} Pile-of-Law samples"
    )

    return legalbench_data, pile_of_law_data
