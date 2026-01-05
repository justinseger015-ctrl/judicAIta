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

    # 1. Update Reward Wrapper to accept 'answer' argument
    # We keep the fallback logic just in case, but prefer the direct argument
    new_wrapper_code = [
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
                print("Updated tunix_reward_wrapper cell.")
                break

    if not found_reward:
        print("Warning: Could not find tunix_reward_wrapper cell")

    # 2. Update Training Loop to use GRPOLearner.train()
    new_training_code = [
        "# ========================================================\n",
        "# Tunix GRPOLearner.train() Loop\n",
        "# ========================================================\n",
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
        "grpo_learner.train(\n",
        "    dataset=train_dataset,\n",
        ")\n",
        'print("\\n\\u2705 Training complete!")\n',
    ]

    found_loop = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "grpo_learner.train_step" in source:
                cell["source"] = new_training_code
                found_loop = True
                print("Updated training loop cell.")
                break

    if not found_loop:
        print("Warning: Could not find training loop cell containing 'grpo_learner.train_step'")

    if found_reward or found_loop:
        with open(nb_path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully wrote updates to {nb_path}")
    else:
        print("No changes made.")


if __name__ == "__main__":
    fix_notebook()
