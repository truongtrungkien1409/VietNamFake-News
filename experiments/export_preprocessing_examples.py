"""Export real before/after preprocessing examples and summary statistics."""

import argparse
import os
import re
from pathlib import Path
from typing import List

import pandas as pd

from src.preprocessing.preprocess import TextPreprocessor, _get_normalization_key


def _transforms(original: str, cleaned: str) -> str:
    transforms = []
    if re.search(r"https?://|www\.|<[^>]+>", original, flags=re.IGNORECASE):
        transforms.append("url_html_cleanup")
    if "#" in original or re.search(r"[!?.,;:]{2,}", original):
        transforms.append("hashtag_punctuation")
    if original != original.lower():
        transforms.append("lowercase")
    if any(token in original.lower().split() for token in ("ko", "k", "dc", "j", "ntn", "vs")):
        transforms.append("teencode")
    if any(ord(char) > 0xFFFF for char in original):
        transforms.append("emoji")
    if "_" in cleaned:
        transforms.append("word_segmentation")
    if not transforms and original != cleaned:
        transforms.append("normalization")
    return ",".join(transforms) or "none_observed"


def _choose_examples(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    candidates = []
    for label in (0, 1):
        subset = df[df["label"] == label]
        if not subset.empty:
            candidates.append(subset.sample(n=min(sample_size // 2, len(subset)), random_state=seed + label))
    selected = pd.concat(candidates, ignore_index=True) if candidates else df.head(0)
    if len(selected) < sample_size:
        remaining = df.drop(selected.index, errors="ignore")
        selected = pd.concat([selected, remaining.head(sample_size - len(selected))], ignore_index=True)
    return selected.head(sample_size).reset_index(drop=True)


def export_examples(input_paths: List[str], output_dir: str, sample_size: int, seed: int) -> None:
    frames = [pd.read_csv(input_path, encoding="utf-8") for input_path in input_paths]
    df = pd.concat(frames, ignore_index=True)
    required = {"text", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {', '.join(sorted(missing))}")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    selected = _choose_examples(df, sample_size, seed)
    preprocessor = TextPreprocessor()
    selected["cleaned_text"] = selected["text"].map(preprocessor.clean_text)
    examples = pd.DataFrame({
        "id": range(1, len(selected) + 1),
        "label": selected["label"].astype(int),
        "source": selected["source"] if "source" in selected else "",
        "original_text": selected["text"],
        "cleaned_text": selected["cleaned_text"],
    })
    examples["applied_transforms"] = [
        _transforms(original, cleaned)
        for original, cleaned in zip(examples["original_text"], examples["cleaned_text"])
    ]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    examples.to_csv(output / "preprocessing_examples.csv", index=False, encoding="utf-8")

    before_cleaning = len(df)
    cleaned = df["text"].map(preprocessor.clean_text)
    nonempty = cleaned.str.strip().ne("")
    at_least_three = cleaned.str.split().str.len().ge(3)
    eligible = df.loc[nonempty & at_least_three].copy()
    eligible["cleaned_text"] = cleaned[nonempty & at_least_three]
    eligible["norm_key"] = eligible["cleaned_text"].map(_get_normalization_key)
    label_counts = eligible.groupby("norm_key")["label"].nunique()
    conflicting_keys = set(label_counts[label_counts > 1].index)
    non_conflicting = eligible[~eligible["norm_key"].isin(conflicting_keys)]
    unique_after_filters = non_conflicting.drop_duplicates("norm_key")
    summary = pd.DataFrame([{
        "input_samples": before_cleaning,
        "nonempty_after_cleaning": int(nonempty.sum()),
        "removed_empty": int((~nonempty).sum()),
        "at_least_three_words": int(at_least_three.sum()),
        "removed_short": int((nonempty & ~at_least_three).sum()),
        "removed_conflicting_labels": int(eligible["norm_key"].isin(conflicting_keys).sum()),
        "unique_normalized_text": int(len(unique_after_filters)),
        "duplicate_candidates": int(len(non_conflicting) - len(unique_after_filters)),
    }])
    summary.to_csv(output / "preprocessing_summary.csv", index=False, encoding="utf-8")
    print(f"Wrote {output / 'preprocessing_examples.csv'}")
    print(f"Wrote {output / 'preprocessing_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        default=[
            "data/raw/vn_news_label_0_real.csv",
            "data/raw/vn_news_label_1_fake.csv",
        ],
        help="Một hoặc nhiều CSV có cột text và label."
    )
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    export_examples(args.input, args.output_dir, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
