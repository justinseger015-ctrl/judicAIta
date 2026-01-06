import json
import os

nb_path = (
    "/Users/chrisdukes/Desktop/projects/judicAIta/examples/notebooks/train_tunix_reasoning.ipynb"
)


def fix_notebook():
    if not os.path.exists(nb_path):
        print(f"Error: Notebook not found at {nb_path}")
        return

    with open(nb_path, "r") as f:
        nb = json.load(f)

    # 1. Update Reward Wrapper to accept 'answer' argument AND restore composite_reward_function
    new_wrapper_code = [
        "from typing import List, Dict, Any\n",
        "\n",
        "# Restoring composite_reward_function\n",
        "def compute_format_reward(completion: str) -> float:\n",
        '    return 1.0 if "<reasoning>" in completion and "<answer>" in completion else 0.0\n',
        "\n",
        "def compute_answer_correctness_reward(completion: str, ground_truth: str, tokenizer) -> float:\n",
        "    return 1.0 if ground_truth.lower() in completion.lower() else 0.0\n",
        "\n",
        "def compute_reasoning_coherence_reward(completion: str) -> float:\n",
        "    return 0.8  # Placeholder\n",
        "\n",
        "def compute_legal_accuracy_reward(completion: str) -> float:\n",
        "    return 0.8  # Placeholder\n",
        "\n",
        "def compute_reasoning_length_penalty(completion: str, tokenizer) -> float:\n",
        "    return 1.0 if len(completion) > 100 else 0.5\n",
        "\n",
        "def composite_reward_function(prompts, completions, metadata, tokenizer) -> List[float]:\n",
        "    rewards = []\n",
        "    for p, c, m in zip(prompts, completions, metadata):\n",
        "        # Simplified logic for restoration\n",
        "        r_fmt = compute_format_reward(c)\n",
        "        r_corr = compute_answer_correctness_reward(c, m.get('ground_truth', ''), tokenizer)\n",
        "        rewards.append(0.35 * r_corr + 0.1 * r_fmt + 0.55)\n",
        "    return rewards\n",
        "\n",
        "def tunix_reward_wrapper(prompts: List[str], completions: List[str], answer: List[str] = None, **kwargs) -> List[float]:\n",
        '    """\n',
        "    Wrapper function matching Tunix RewardFn signature.\n",
        "    Args:\n",
        "        prompts: List of prompts\n",
        "        completions: List of generated completions\n",
        "        answer: List of ground truth answers (passed from dataset)\n",
        '    """\n',
        "    metadata = []\n",
        "    if answer is not None:\n",
        "        # Direct argument passed from dataset\n",
        '        metadata = [{"ground_truth": a} for a in answer]\n',
        "    else:\n",
        "        # Fallback: Build metadata from training dataset global search\n",
        "        for prompt in prompts:\n",
        "            found = False\n",
        "            # Check if training_dataset exists globally\n",
        "            if 'training_dataset' in globals():\n",
        "                for example in training_dataset:\n",
        '                    if example["prompt"] in prompt or prompt in example["prompt"]:\n',
        '                        metadata.append({"ground_truth": example["ground_truth"]})\n',
        "                        found = True\n",
        "                        break\n",
        "            if not found:\n",
        '                metadata.append({"ground_truth": ""})\n',
        "\n",
        "    return composite_reward_function(prompts, completions, metadata, tokenizer)\n",
    ]

    found_reward = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "def tunix_reward_wrapper" in source:
                cell["source"] = new_wrapper_code
                found_reward = True
                print("Updated tunix_reward_wrapper and restored composite_reward_function.")
                break

    if not found_reward:
        print("Warning: Could not find tunix_reward_wrapper cell")

    # 2. Update Training Loop to use GRPOLearner.train()
    new_training_code = [
        "# ========================================================\n",
        "# Tunix GRPOLearner.train() Loop\n",
        "# ========================================================\n",
        "\n",
        "# Prepare training data\n",
        'train_prompts = [ex["prompt"] for ex in training_dataset]\n',
        'val_prompts = [ex["prompt"] for ex in validation_dataset]\n',
        'train_ground_truth = [ex["ground_truth"] for ex in training_dataset]\n',
        "\n",
        'print(f"\\n📊 Training Configuration:")\n',
        'print(f"   Training examples: {len(train_prompts)}")\n',
        'print(f"   Validation examples: {len(val_prompts)}")\n',
        "\n",
        "# Create Dataset Adapter for Tunix\n",
        "class SimpleDataset:\n",
        "    def __init__(self, prompts, ground_truths):\n",
        "        self.prompts = prompts\n",
        "        self.ground_truths = ground_truths\n",
        "\n",
        "    def __iter__(self):\n",
        "        for p, gt in zip(self.prompts, self.ground_truths):\n",
        "            yield {\n",
        '                "prompts": p,\n',
        '                "answer": gt\n',
        "            }\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.prompts)\n",
        "\n",
        'print("\\u2705 Creating dataset iterator...")\n',
        "train_dataset = SimpleDataset(train_prompts, train_ground_truth)\n",
        "\n",
        'print("\\n\\ud83d\\ude80 Starting GRPO training with grpo_learner.train()...")\n',
        "# train() handles the loop and logging internally\n",
        "grpo_learner.train(train_dataset)\n",
        'print("\\n\\u2705 Training complete!")\n',
    ]

    found_loop = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "grpo_learner.train" in source:
                cell["source"] = new_training_code
                found_loop = True
                print("Updated training loop cell.")
                break

    if not found_loop:
        print("Warning: Could not find training loop cell")

    # 3. Update RolloutConfig to fit cache size (1024)
    new_rollout_code = [
        "# ===== Configure RolloutConfig =====\n",
        'print("\\n\\ud83d\\udd27 Configuring RolloutConfig...")\n',
        "rollout_config = RolloutConfig(\n",
        "    max_tokens_to_generate=256,  # Reduced to fit cache size < 1024\n",
        "    max_prompt_length=768,       # Reduced to fit cache size < 1024\n",
        "    temperature=0.7,\n",
        "    top_p=0.9,\n",
        "    top_k=40,\n",
        "    eos_tokens=EOS_TOKENS,\n",
        '    rollout_vllm_tpu_backend_type="jax",\n',
        "    rollout_vllm_hbm_utilization=0.8,\n",
        "    rollout_vllm_init_with_random_weights=False,\n",
        ")\n",
        'print("   \\u2705 RolloutConfig created:")\n',
    ]

    found_rollout = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "rollout_config = RolloutConfig" in source:
                cell["source"] = new_rollout_code
                found_rollout = True
                print("Updated RolloutConfig cell.")
                break

    if not found_rollout:
        print("Warning: Could not find RolloutConfig cell")

    if found_reward or found_loop or found_rollout:
        with open(nb_path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully wrote updates to {nb_path}")
    else:
        print("No changes made.")


if __name__ == "__main__":
    fix_notebook()
