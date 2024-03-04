import json
import os
from pathlib import Path

from datasets import load_dataset

gsm8k = load_dataset("gsm8k", "socratic")

output_dir = "data/processed/gsm8k"
output_path = os.path.join(output_dir, "gsm8k_data.jsonl")

Path(output_dir).mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as fout:
    gsm8k_prompt = "Please solve the following math problem with detailed steps."

    for idx, example in enumerate(gsm8k["train"]):
        messages = []
        messages.append({"role": "user", "content": f"""{gsm8k_prompt}\n\nProblem: {example["question"]}"""})
        messages.append({"role": "assistant", "content": f"Answer: {example['answer']}."})

        fout.write(
            json.dumps(
                {
                    "dataset": "gsm8k",
                    "id": f"gsm8k_{idx}",
                    "messages": messages,
                }
            )
            + "\n"
        )
