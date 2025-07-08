# scripts/create_tulu_math_chunks.py
import json
import os
from datasets import load_dataset

MAX_BYTES = 100 * 1024 * 1024  # 100 MiB


def write_split(split_name: str, itr, base_dir: str) -> None:
    """Write one dataset split into ≤100 MiB chunks inside base_dir/split_name/."""
    split_dir = os.path.join(base_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    file_idx = 0
    bytes_in_file = 0
    n_records_in_split = 0

    def new_path(idx: int) -> str:
        return os.path.join(split_dir, f"{split_name}-{idx:05d}.jsonl")

    fout = open(new_path(file_idx), "w", encoding="utf-8")
    try:
        for row in itr:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            line_bytes = len(line.encode("utf-8"))

            if bytes_in_file and bytes_in_file + line_bytes > MAX_BYTES:
                fout.close()
                print(
                    f"  » Closed {split_name}-{file_idx:05d}.jsonl "
                    f"({bytes_in_file:,} bytes)"
                )
                file_idx += 1
                fout = open(new_path(file_idx), "w", encoding="utf-8")
                bytes_in_file = 0

            fout.write(line)
            bytes_in_file += line_bytes
            n_records_in_split += 1

        print(
            f"Wrote {n_records_in_split:,} records to {split_dir} "
            f"across {file_idx + 1} file(s)."
        )
    finally:
        fout.close()


def main() -> None:
    output_root = "data/dart_math"
    os.makedirs(output_root, exist_ok=True)

    # Stream the dataset so we never load everything into RAM
    ds_iterable = load_dataset(
        "hkust-nlp/dart-math-uniform",
        streaming=True,
    )

    for split, itr in ds_iterable.items():  # e.g. "train"
        print(f"Processing split '{split}' …")
        write_split(split, itr, output_root)


if __name__ == "__main__":
    main()
