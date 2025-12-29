import json

NOTEBOOK_PATH = "examples/notebooks/train_tunix_reasoning.ipynb"


def update_notebook():
    print(f"Reading {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, "r") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    updated_count = 0

    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue

        source_lines = cell.get("source", [])
        source_text = "".join(source_lines)

        # Update 1: "What This Notebook Does"
        if (
            "**Reward Function**: Score outputs based on format, reasoning length, and correctness"
            in source_text
        ):
            print("Found 'What This Notebook Does' cell. Updating...")

            new_source = []
            for line in source_lines:
                if "**Reward Function**:" in line:
                    new_source.append(
                        "4. **Reward Function**: Multi-objective scoring including **Legal Accuracy**, **Reasoning Coherence**, **Answer Correctness** (35%), Format, and Length.\n"
                    )
                else:
                    new_source.append(line)
            cell["source"] = new_source
            updated_count += 1

        # Update 2: "Task 3: Implement Custom Reward Function"
        elif "## 🎯 Task 3: Implement Custom Reward Function" in source_text:
            print("Found 'Task 3 Reward' cell. Updating...")
            cell["source"] = [
                "## 🎯 Task 3: Implement Custom Reward Function\n",
                "\n",
                "Create a competition-compliant reward function that scores:\n",
                "1. **Answer Correctness** (35%): Match with ground truth (exact or Jaccard)\n",
                "2. **Legal Accuracy** (25%): Valid legal citation patterns (e.g., U.S.C., v., §)\n",
                "3. **Reasoning Coherence** (25%): Structural integrity and lack of repetition\n",
                "4. **Format Compliance** (10%): Proper XML `<reasoning>` and `<answer>` tags\n",
                "5. **Reasoning Length** (5%): Encouraging detailed analysis (>150 tokens)\n",
            ]
            updated_count += 1

    if updated_count > 0:
        print(f"Saving updated notebook to {NOTEBOOK_PATH}...")
        with open(NOTEBOOK_PATH, "w") as f:
            json.dump(nb_data, f, indent=2)
        print(f"✅ Notebook markdown updated successfully ({updated_count} cells)!")
    else:
        print("❌ Could not find markdown cells to update.")


if __name__ == "__main__":
    update_notebook()
