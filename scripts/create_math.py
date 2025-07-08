#!/usr/bin/env python3
"""
Convert MATH dataset from individual JSON files to JSONL format.
Creates train.jsonl and validation.jsonl from the train and test directories respectively.
"""

import json
from pathlib import Path
from typing import Dict, Any


def load_math_problems(data_dir: Path, split: str) -> list[Dict[str, Any]]:
    """Load all MATH problems from a given split (train or test)."""
    split_dir = data_dir / split
    problems = []

    if not split_dir.exists():
        raise FileNotFoundError(f"Directory {split_dir} does not exist")

    # Get all subject directories
    subject_dirs = [d for d in split_dir.iterdir() if d.is_dir()]

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        print(f"Processing {subject}...")

        # Get all JSON files in this subject directory
        json_files = list(subject_dir.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    problem_data = json.load(f)

                # Add subject and file ID to the problem data
                problem_data["subject"] = subject
                problem_data["file_id"] = json_file.stem

                problems.append(problem_data)

            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue

    print(f"Loaded {len(problems)} problems from {split}")
    return problems


def save_jsonl(problems: list[Dict[str, Any]], output_path: Path) -> None:
    """Save problems to JSONL format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for problem in problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")

    print(f"Saved {len(problems)} problems to {output_path}")


def main():
    """Main function to convert MATH dataset to JSONL format."""
    # Define paths
    data_dir = Path("data/MATH")
    output_dir = data_dir  # Output in the MATH directory

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Process train split
    print("Processing train split...")
    train_problems = load_math_problems(data_dir, "train")
    train_output = output_dir / "train.jsonl"
    save_jsonl(train_problems, train_output)

    # Process test split (save as validation.jsonl)
    print("\nProcessing test split...")
    test_problems = load_math_problems(data_dir, "test")
    validation_output = output_dir / "validation.jsonl"
    save_jsonl(test_problems, validation_output)

    print("\nConversion complete!")
    print(f"Train: {len(train_problems)} problems -> {train_output}")
    print(f"Validation: {len(test_problems)} problems -> {validation_output}")


if __name__ == "__main__":
    main()
