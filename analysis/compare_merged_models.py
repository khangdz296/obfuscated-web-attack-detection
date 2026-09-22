"""Build a checked comparison table for the two 1024-token external tests."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "merged_external_evaluation"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_row(model_name: str, dataset_name: str, result: dict, summary: dict) -> dict:
    tn, fp, fn, tp = result["confusion_matrix_tn_fp_fn_tp"]
    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "Rows": result["rows"],
        "Max_len": summary["max_len"],
        "Threshold": summary["threshold"],
        "Accuracy": result["accuracy"],
        "AUC-ROC": result.get("roc_auc"),
        "PR-AUC": result.get("average_precision"),
        "Attack Precision": result["attack"]["precision"],
        "Attack Recall": result["attack"]["recall"],
        "Attack F1": result["attack"]["f1-score"],
        "FP": fp,
        "FN": fn,
        "Truncated": summary["truncated_rows"] if dataset_name == "merged_all" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-lstm-summary", type=Path, default=REPORT_DIR / "summary.json")
    parser.add_argument("--cnn-only-summary", type=Path, default=REPORT_DIR / "cnn_only" / "summary.json")
    parser.add_argument("--cnn-only-label", default="CNN-only (1024)")
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "comparison_matched_1024.csv")
    args = parser.parse_args()

    summaries = {
        "CNN-LSTM (WebApp)": json.loads(args.cnn_lstm_summary.read_text(encoding="utf-8")),
        args.cnn_only_label: json.loads(args.cnn_only_summary.read_text(encoding="utf-8")),
    }
    lstm, cnn = summaries.values()
    if lstm["partial_test"] or cnn["partial_test"]:
        raise ValueError("Both evaluations must cover the full merged dataset")
    if lstm["rows_tested"] != cnn["rows_tested"]:
        raise ValueError("The evaluations have different row counts")
    if lstm["max_len"] != cnn["max_len"]:
        raise ValueError("The models have different max_len values")
    if lstm["threshold"] != cnn["threshold"]:
        raise ValueError("The models use different decision thresholds")
    if file_sha256(Path(lstm["prepared_csv"])) != file_sha256(Path(cnn["prepared_csv"])):
        raise ValueError("The prepared model inputs differ between evaluations")

    rows = []
    for model_name, summary in summaries.items():
        rows.append(metric_row(model_name, "merged_all", summary["overall"], summary))
        for source, result in summary["by_source"].items():
            rows.append(metric_row(model_name, source, result, summary))
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Saved checked comparison: {args.output}")


if __name__ == "__main__":
    main()
