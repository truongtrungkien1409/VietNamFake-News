"""Export representative FP/FN samples from the real error-analysis table."""

import argparse
from pathlib import Path

import pandas as pd


def _shorten(value: object, limit: int = 300) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def export_error_samples(input_path: str, output_path: str, limit: int = 10) -> None:
    errors = pd.read_csv(input_path, encoding="utf-8")
    required = {"original_text", "true_label", "predicted_label", "fake_probability", "error_type"}
    missing = required.difference(errors.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {', '.join(sorted(missing))}")

    fp = errors[errors["error_type"] == "FP"].sort_values(
        "fake_probability", ascending=False
    ).head(limit)
    fn = errors[errors["error_type"] == "FN"].sort_values(
        "fake_probability", ascending=True
    ).head(limit)

    lines = [
        "# Mẫu dự đoán sai trên tập Test",
        "",
        "Nguồn: `results/tables/error_analysis.csv`.",
        "Với FP, mẫu được sắp theo xác suất tin giả giảm dần. Với FN, mẫu được sắp theo xác suất tin giả tăng dần để đưa ra các trường hợp bị bỏ sót với độ tin cậy cao nhất của nhãn tin thật.",
        "",
        f"- Tổng FP trong bảng nguồn: {int((errors['error_type'] == 'FP').sum())}.",
        f"- Tổng FN trong bảng nguồn: {int((errors['error_type'] == 'FN').sum())}.",
        "",
        "## 1. False Positive (FP)",
        "",
        "Các mẫu có nhãn thật là 0 nhưng bị dự đoán là 1.",
        "",
    ]

    for index, (_, row) in enumerate(fp.iterrows(), start=1):
        lines.extend([
            f"### FP {index}",
            "",
            f"- Nhãn thật: `{int(row['true_label'])}`",
            f"- Nhãn dự đoán: `{int(row['predicted_label'])}`",
            f"- Xác suất tin giả: `{float(row['fake_probability']):.6f}`",
            f"- Văn bản (tối đa 300 ký tự): {_shorten(row['original_text'])}",
            "",
        ])

    lines.extend([
        "## 2. False Negative (FN)",
        "",
        "Các mẫu có nhãn thật là 1 nhưng bị dự đoán là 0.",
        "",
    ])

    for index, (_, row) in enumerate(fn.iterrows(), start=1):
        lines.extend([
            f"### FN {index}",
            "",
            f"- Nhãn thật: `{int(row['true_label'])}`",
            f"- Nhãn dự đoán: `{int(row['predicted_label'])}`",
            f"- Xác suất tin giả: `{float(row['fake_probability']):.6f}`",
            f"- Văn bản (tối đa 300 ký tự): {_shorten(row['original_text'])}",
            "",
        ])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/tables/error_analysis.csv")
    parser.add_argument("--output", default="results/error_samples.md")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    export_error_samples(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()
