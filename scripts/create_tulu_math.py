# scripts/create_tulu_math.py
import json
import os
from datasets import load_dataset


def main():
    # Output directory and file
    output_dir = "data/tulu_math"
    os.makedirs(output_dir, exist_ok=True)

    # Stream the dataset
    ds_iterable = load_dataset(
        "allenai/tulu-3-sft-personas-math-filtered",
        streaming=True,
    )

    for split, itr in ds_iterable.items():  # likely only "train"
        out_path = os.path.join(output_dir, f"{split}.jsonl")
        n = 0
        with open(out_path, "w", encoding="utf-8") as fout:
            for row in itr:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        print(f"Wrote {n:,} records to {out_path}")

if __name__ == "__main__":
    main()
