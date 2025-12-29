import json

NOTEBOOK_PATH = "examples/notebooks/train_tunix_reasoning.ipynb"

NEW_DATA_CODE = [
    "import json\n",
    "import re\n",
    "from typing import List, Dict, Any\n",
    "from datasets import load_dataset\n",
    "\n",
    "def prepare_dataset_for_tunix(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:\n",
    '    """\n',
    "    Prepare dataset in Tunix-compatible JSONL format.\n",
    "\n",
    "    Args:\n",
    "        examples: List of dicts with 'question' and 'answer' fields\n",
    "\n",
    "    Returns:\n",
    "        List of dicts with 'prompt', 'ground_truth', and 'metadata'\n",
    '    """\n',
    "    prepared = []\n",
    "\n",
    "    for idx, ex in enumerate(examples):\n",
    "        prepared.append({\n",
    '            "prompt": ex.get("question", ex.get("prompt", "")),\n',
    '            "ground_truth": ex.get("answer", ex.get("ground_truth", "")),\n',
    '            "metadata": {\n',
    '                "example_id": idx,\n',
    '                "original_question": ex.get("question", ""),\n',
    '                "task_type": ex.get("task_type", "general_reasoning")\n',
    "            }\n",
    "        })\n",
    "\n",
    "    return prepared\n",
    "\n",
    'print("📥 Loading real legal data from HuggingFace (nguha/legalbench)...")\n',
    "\n",
    "# Load subset: 'contract_qa' (Contract Law questions)\n",
    "try:\n",
    '    dataset = load_dataset("nguha/legalbench", "contract_qa", split="train")\n',
    '    print(f"   Loaded {len(dataset)} examples from LegalBench (contract_qa)")\n',
    "    \n",
    "    # Take first 100 examples for this demo/hackathon training\n",
    "    # (In full training, use more)\n",
    "    real_examples = []\n",
    "    for item in dataset.select(range(min(len(dataset), 100))):\n",
    "        real_examples.append({\n",
    '            "question": item["question"],\n',
    '            "answer": item["answer"], # LegalBench uses \'answer\' usually yes/no or text\n',
    '            "task_type": "contract_qa"\n',
    "        })\n",
    "        \n",
    "except Exception as e:\n",
    '    print(f"⚠️ Failed to load LegalBench: {e}")\n',
    '    print("   Falling back to synthetic examples for demonstration.")\n',
    "    real_examples = [\n",
    "        {\n",
    '            "question": "Can an employer in California enforce a non-compete clause against a former employee?",\n',
    '            "answer": "No, non-compete clauses are generally unenforceable in California except in limited circumstances involving sale of business or dissolution of partnership.",\n',
    '            "task_type": "legal_qa"\n',
    "        },\n",
    "        {\n",
    '            "question": "What is the statute of limitations for filing a breach of contract claim?",\n',
    '            "answer": "The statute of limitations varies by jurisdiction. In many states, it is 4-6 years for written contracts and 2-3 years for oral contracts.",\n',
    '            "task_type": "legal_qa"\n',
    "        },\n",
    "        {\n",
    '            "question": "Under what circumstances can a contract be voided for duress?",\n',
    '            "answer": "A contract can be voided for duress when one party was forced to enter the agreement through threats, violence, or other improper pressure that overcame their free will.",\n',
    '            "task_type": "legal_qa"\n',
    "        },\n",
    "        {\n",
    '            "question": "What is required to establish an attorney-client privilege?",\n',
    '            "answer": "Attorney-client privilege requires: (1) an attorney-client relationship, (2) confidential communication, (3) made for the purpose of seeking or providing legal advice.",\n',
    '            "task_type": "legal_qa"\n',
    "        },\n",
    "    ]\n",
    "\n",
    "# Prepare dataset\n",
    "prepared_dataset = prepare_dataset_for_tunix(real_examples)\n",
    "\n",
    'print(f"✅ Prepared {len(prepared_dataset)} training examples")\n',
    'print(f"\\n📝 Sample example:")\n',
    "print(json.dumps(prepared_dataset[0], indent=2))",
]


def update_notebook():
    print(f"Reading {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, "r") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    updated = False

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue

        source_lines = cell.get("source", [])
        source_text = "".join(source_lines)

        # Identification Logic: Look for the function def and the dummy data
        if "def prepare_dataset_for_tunix" in source_text and "sample_examples = [" in source_text:
            print("Found Data Preparation cell. Updating...")
            cell["source"] = NEW_DATA_CODE
            updated = True
            break

    if updated:
        print(f"Saving updated notebook to {NOTEBOOK_PATH}...")
        with open(NOTEBOOK_PATH, "w") as f:
            json.dump(nb_data, f, indent=2)
        print("✅ Notebook data loading updated successfully!")
    else:
        print("❌ Could not find Data Preparation cell.")


if __name__ == "__main__":
    update_notebook()
