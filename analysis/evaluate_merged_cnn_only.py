"""Evaluate the saved CNN-only obfu_http model on the two new datasets.

The preprocessing, row policy and metrics match evaluate_merged_external.py.
Only the model, tokenizer and model input length change.
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from evaluate_merged_external import DEFAULT_TRAIN_PATH, PROJECT_ROOT, evaluate, metrics, prepare_merged


DEFAULT_MODEL_DIR = (
    PROJECT_ROOT / "cnn_only" / "artifacts_cnn_only_matched_1024"
    / "by_dataset" / "obfu_http"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "merged_external_evaluation" / "cnn_only"


def load_cnn_only_assets(model_dir: Path) -> tuple:
    model_path = model_dir / "best_cnn_only.keras"
    tokenizer_path = model_dir / "tokenizer.pkl"
    metadata_path = model_dir / "metadata_and_results.json"
    missing = [path for path in (model_path, tokenizer_path, metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing CNN-only artifact(s): " + ", ".join(map(str, missing)))
    model = tf.keras.models.load_model(model_path, compile=False)
    with tokenizer_path.open("rb") as file:
        tokenizer = pickle.load(file)
    max_len = int(model.input_shape[1])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    configured_len = metadata.get("model", {}).get("max_len")
    if configured_len is not None and int(configured_len) != max_len:
        raise ValueError(f"CNN-only max_len mismatch: model={max_len}, metadata={configured_len}")
    return model, tokenizer, pad_sequences, max_len


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obfuscated-grouped", type=Path, default=PROJECT_ROOT / "obfuscated_grouped.csv")
    parser.add_argument("--xss-payloads", type=Path, default=PROJECT_ROOT / "xss_payloads_with_obfuscated.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, help="Smoke test only: evaluate the first N prepared rows.")
    parser.add_argument("--threshold", type=float, help="Override the threshold saved with the CNN-only model.")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size and --limit must be positive")

    metadata_path = args.model_dir / "metadata_and_results.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"CNN-only metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_threshold = metadata.get("model", {}).get("threshold_selection_on_validation", {}).get("threshold", 0.5)
    threshold = float(args.threshold if args.threshold is not None else saved_threshold)
    if not 0 <= threshold <= 1:
        parser.error("Threshold must be between 0 and 1")

    frame, preparation = prepare_merged({
        "obfuscated_grouped": args.obfuscated_grouped,
        "xss_payloads_with_obfuscated": args.xss_payloads,
    })
    if args.limit:
        frame = frame.head(args.limit).copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = args.output_dir / "merged_preprocessed.csv"
    frame.to_csv(prepared_path, index=False, encoding="utf-8")

    assets = load_cnn_only_assets(args.model_dir)
    frame, max_len = evaluate(frame, args.batch_size, threshold, assets=assets)
    if args.training_csv.is_file():
        train_inputs = set(pd.read_csv(args.training_csv, usecols=["payload"], dtype=str)["payload"].dropna())
        frame["in_training_data"] = frame["payload"].isin(train_inputs)
    else:
        frame["in_training_data"] = False

    predictions_path = args.output_dir / "predictions.csv"
    frame.to_csv(predictions_path, index=False, encoding="utf-8")
    summary = {
        "model_type": "cnn_only",
        "model_path": str(args.model_dir / "best_cnn_only.keras"),
        "tokenizer_path": str(args.model_dir / "tokenizer.pkl"),
        "threshold": threshold,
        "max_len": max_len,
        "preprocessing": "preprocessing.preprocess_data.wrap_payload_as_request + preprocess_inference_input",
        "preparation": preparation,
        "rows_tested": len(frame),
        "partial_test": args.limit is not None,
        "prepared_csv": str(prepared_path),
        "predictions_csv": str(predictions_path),
        "truncated_rows": int(frame["truncated"].sum()),
        "zero_token_rows": int(frame["token_count"].eq(0).sum()),
        "training_overlap_rows": int(frame["in_training_data"].sum()),
        "training_overlap_checked": args.training_csv.is_file(),
        "overall": metrics(frame, threshold),
        "excluding_training_overlap": metrics(frame.loc[~frame["in_training_data"]], threshold),
        "by_source": {name: metrics(part, threshold) for name, part in frame.groupby("source")},
        "by_technique": {name: metrics(part, threshold) for name, part in frame.groupby("technique")},
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved merged data: {prepared_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {summary_path}")
    print("Overall:", json.dumps(summary["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
