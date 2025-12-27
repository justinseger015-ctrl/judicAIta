"""
GRPO (Group Relative Policy Optimization) trainer for legal domain adaptation.

This module implements the GRPO training algorithm adapted for legal reasoning
with support for LoRA/PEFT for parameter-efficient training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)

from judicaita.core.config import get_settings
from judicaita.training.rewards import CompositeReward


@dataclass
class TrainingConfig:
    """Configuration for GRPO training."""

    # Model settings
    base_model: str = "google/gemma-2-2b-it"
    model_max_length: int = 2048
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # LoRA/PEFT settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # GRPO settings
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # GRPO-specific hyperparameters
    grpo_tau: float = 0.1  # Temperature for advantage normalization
    grpo_gamma: float = 0.99  # Discount factor
    num_rollouts: int = 4  # Number of rollouts per prompt

    # Optimization
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"

    # Checkpointing
    checkpoint_dir: Path = Path("./checkpoints/grpo")
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10

    # Evaluation
    eval_batch_size: int = 8
    generation_max_length: int = 1024
    generation_temperature: float = 0.7
    generation_top_p: float = 0.95

    # Mixed precision
    fp16: bool = False
    bf16: bool = True

    # Seed for reproducibility
    seed: int = 42


class GRPOTrainer:
    """
    GRPO trainer for legal domain adaptation.

    Implements Group Relative Policy Optimization for training language models
    on legal reasoning tasks with step-by-step trace generation.
    """

    def __init__(
        self,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        reward_fn: CompositeReward | None = None,
    ) -> None:
        """
        Initialize GRPO trainer.

        Args:
            config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            reward_fn: Reward function for evaluating generations
        """
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.reward_fn = reward_fn or CompositeReward()

        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizer | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: Any | None = None

        # Set random seed
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def initialize(self) -> None:
        """Initialize model, tokenizer, and training components."""
        logger.info(f"Initializing GRPO trainer with base model: {self.config.base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        logger.info("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            device_map=self.config.device,
        )

        # Apply LoRA if enabled
        if self.config.use_lora:
            logger.info("Applying LoRA adaptation...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Initialize optimizer
        self._initialize_optimizer()

        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("GRPO trainer initialized successfully")

    def _initialize_optimizer(self) -> None:
        """Initialize optimizer and learning rate scheduler."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Create optimizer
        if self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        # Create learning rate scheduler
        if self.config.lr_scheduler == "cosine":
            total_steps = (
                len(self.train_dataset)
                // self.config.batch_size
                // self.config.gradient_accumulation_steps
                * self.config.num_epochs
            )

            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
        elif self.config.lr_scheduler == "linear":
            from transformers import get_linear_schedule_with_warmup

            total_steps = (
                len(self.train_dataset)
                // self.config.batch_size
                // self.config.gradient_accumulation_steps
                * self.config.num_epochs
            )

            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps,
            )

    def train(self) -> dict[str, Any]:
        """
        Run GRPO training loop.

        Returns:
            Training metrics and statistics
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Trainer not initialized. Call initialize() first.")

        logger.info("Starting GRPO training...")
        logger.info(f"Training for {self.config.num_epochs} epochs")
        logger.info(f"Training dataset size: {len(self.train_dataset)}")

        self.model.train()

        global_step = 0
        total_loss = 0.0
        training_metrics = {"losses": [], "rewards": [], "advantages": []}

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            epoch_loss = 0.0
            num_batches = 0

            # Create dataloader
            dataloader = self._create_dataloader(self.train_dataset, self.config.batch_size)

            for batch_idx, batch in enumerate(dataloader):
                # Generate rollouts for each prompt
                prompts = batch["prompt"]
                references = batch.get("response", [""] * len(prompts))

                # Perform GRPO update
                loss, metrics = self._grpo_step(prompts, references)

                # Backward pass with gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                    # Optimizer step
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    self.optimizer.zero_grad()
                    global_step += 1

                # Track metrics
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1

                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / self.config.logging_steps
                    logger.info(
                        f"Step {global_step}, Loss: {avg_loss:.4f}, "
                        f"Avg Reward: {metrics.get('avg_reward', 0.0):.4f}"
                    )
                    training_metrics["losses"].append(avg_loss)
                    training_metrics["rewards"].append(metrics.get("avg_reward", 0.0))
                    total_loss = 0.0

                # Evaluation
                if (
                    self.eval_dataset is not None
                    and global_step % self.config.eval_steps == 0
                ):
                    eval_metrics = self.evaluate()
                    logger.info(f"Evaluation metrics: {eval_metrics}")
                    self.model.train()

                # Checkpointing
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(global_step)

            # End of epoch logging
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"Epoch {epoch + 1} completed, Average Loss: {avg_epoch_loss:.4f}")

        logger.info("Training completed!")

        # Save final checkpoint
        self.save_checkpoint("final")

        return training_metrics

    def _grpo_step(
        self, prompts: list[str], references: list[str]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Perform a single GRPO training step.

        Args:
            prompts: Batch of input prompts
            references: Reference responses for reward computation

        Returns:
            Tuple of (loss, metrics)
        """
        batch_size = len(prompts)

        # Generate multiple rollouts per prompt
        all_responses = []
        all_log_probs = []

        for prompt in prompts:
            rollout_responses = []
            rollout_log_probs = []

            for _ in range(self.config.num_rollouts):
                response, log_prob = self._generate_with_log_probs(prompt)
                rollout_responses.append(response)
                rollout_log_probs.append(log_prob)

            all_responses.append(rollout_responses)
            all_log_probs.append(rollout_log_probs)

        # Compute rewards for all rollouts
        rewards = []
        for i, (prompt, rollout_responses) in enumerate(zip(prompts, all_responses)):
            reference = references[i] if i < len(references) else ""
            rollout_rewards = []

            for response in rollout_responses:
                reward_result = self.reward_fn.compute(prompt, response, reference)
                rollout_rewards.append(reward_result.score)

            rewards.append(rollout_rewards)

        # Convert to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.config.device)
        log_probs_tensor = torch.stack(
            [torch.stack(lp) for lp in all_log_probs]
        )  # [batch_size, num_rollouts]

        # Compute advantages (group relative)
        advantages = self._compute_advantages(rewards_tensor)

        # Compute GRPO loss
        loss = -(advantages * log_probs_tensor).mean()

        # Metrics
        metrics = {
            "avg_reward": rewards_tensor.mean().item(),
            "avg_advantage": advantages.mean().item(),
            "loss": loss.item(),
        }

        return loss, metrics

    def _generate_with_log_probs(self, prompt: str) -> tuple[str, torch.Tensor]:
        """
        Generate response and compute log probabilities.

        Args:
            prompt: Input prompt

        Returns:
            Tuple of (generated_text, log_prob)
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(
            self.config.device
        )

        # Generate with model
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.generation_max_length,
                temperature=self.config.generation_temperature,
                top_p=self.config.generation_top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Extract generated text
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1] :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute log probability (simplified - use scores from generation)
        # In production, compute proper log probs from logits
        log_prob = torch.tensor(0.0, device=self.config.device)  # Placeholder

        return generated_text, log_prob

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute group relative advantages.

        Args:
            rewards: Tensor of rewards [batch_size, num_rollouts]

        Returns:
            Advantages tensor
        """
        # Group normalization: normalize within each group (prompt)
        mean_rewards = rewards.mean(dim=1, keepdim=True)
        std_rewards = rewards.std(dim=1, keepdim=True) + 1e-8

        advantages = (rewards - mean_rewards) / (std_rewards * self.config.grpo_tau)

        return advantages

    def _create_dataloader(self, dataset: Dataset, batch_size: int) -> Any:
        """Create a dataloader from dataset."""
        from torch.utils.data import DataLoader

        def collate_fn(examples: list[dict[str, Any]]) -> dict[str, list[Any]]:
            """Collate examples into batch."""
            batch = {}
            for key in examples[0].keys():
                batch[key] = [ex[key] for ex in examples]
            return batch

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

    def evaluate(self) -> dict[str, float]:
        """
        Evaluate model on evaluation dataset.

        Returns:
            Evaluation metrics
        """
        if self.eval_dataset is None:
            logger.warning("No evaluation dataset provided")
            return {}

        logger.info("Running evaluation...")

        self.model.eval()

        total_reward = 0.0
        num_samples = 0

        with torch.no_grad():
            for sample in self.eval_dataset:
                prompt = sample["prompt"]
                reference = sample.get("response", "")

                # Generate response
                response, _ = self._generate_with_log_probs(prompt)

                # Compute reward
                reward_result = self.reward_fn.compute(prompt, response, reference)
                total_reward += reward_result.score
                num_samples += 1

                if num_samples >= 100:  # Limit evaluation samples
                    break

        avg_reward = total_reward / num_samples if num_samples > 0 else 0.0

        metrics = {"eval_avg_reward": avg_reward, "eval_samples": num_samples}

        return metrics

    def save_checkpoint(self, step: int | str) -> None:
        """
        Save model checkpoint.

        Args:
            step: Training step or identifier
        """
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_path}")

        # Save model
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training config
        import json

        config_path = checkpoint_path / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)

        logger.info("Checkpoint saved successfully")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        logger.info("Checkpoint loaded successfully")
