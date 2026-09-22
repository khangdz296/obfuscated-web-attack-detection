"""Prepare two external payload datasets and evaluate the model used by webapp.

Run from any directory:
    python analysis/evaluate_merged_external.py

The output is an external evaluation. Neither the tokenizer nor the model is
refitted, and the saved training threshold is used unless overridden.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.preprocess_data import normalize_payload, preprocess_inference_input, wrap_payload_as_request
from webapp.app import METADATA_PATH, load_inference_assets


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "merged_external_evaluation"
DEFAULT_TRAIN_PATH = (
    PROJECT_ROOT
    / "cnn_lstm"
    / "artifacts_cnn_lstm_by_dataset"
    / "processed_data_by_dataset"
    / "obfu_http"
    / "train.csv"
)
SOURCES = (
    ("obfuscated_grouped", "query", "label"),
    ("xss_payloads_with_obfuscated", "Sentence", "Label"),
)


def load_source(path: Path, name: str, text_column: str, label_column: str) -> tuple[pd.DataFrame, int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    source = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {text_column, label_column}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    input_count = len(source)
    source["_source_row"] = np.arange(2, input_count + 2, dtype=np.int64)
    # Deduplicate only fully identical original records inside each file.
    # Different obfuscation variants may collapse after whitespace handling;
    # they remain separate test cases as requested.
    source = source.drop_duplicates(subset=[column for column in source if column != "_source_row"]).reset_index(drop=True)
    exact_duplicates = input_count - len(source)
    labels = source[label_column].str.strip()
    invalid = sorted(set(labels) - {"0", "1"})
    if invalid:
        raise ValueError(f"{path} has invalid binary labels: {invalid[:10]}")

    result = pd.DataFrame(
        {
            "source": name,
            "source_row": source["_source_row"],
            "raw_payload": source[text_column],
            "label": labels.astype(np.int8),
            "technique": source["technique"] if "technique" in source else "unspecified",
            "source_group": source["group"] if "group" in source else "",
        }
    )
    return result, input_count, exact_duplicates


def prepare_merged(paths: dict[str, Path]) -> tuple[pd.DataFrame, dict]:
    loaded = [load_source(paths[name], name, text_col, label_col)
              for name, text_col, label_col in SOURCES]
    frames = [item[0] for item in loaded]
    raw_counts = {name: item[1] for (name, _, _), item in zip(SOURCES, loaded)}
    exact_duplicate_counts = {name: item[2] for (name, _, _), item in zip(SOURCES, loaded)}
    merged = pd.concat(frames, ignore_index=True)
    merged["raw_payload"] = merged["raw_payload"].map(normalize_payload)
    empty_rows = int(merged["raw_payload"].eq("").sum())
    merged = merged.loc[merged["raw_payload"].ne("")].copy()

    # These are payload-only datasets, so use the same wrapper as the Kaggle
    # payload source in preprocessing/preprocess_data.py. Pass that envelope
    # through the webapp's inference preprocessor for the final whitespace
    # normalization also applied by clean() in the training pipeline.
    wrapped = merged["raw_payload"].map(wrap_payload_as_request)
    inference = wrapped.str[0].map(preprocess_inference_input)
    if not inference.str[1].eq("unified_http_envelope").all():
        raise ValueError("Webapp did not recognize a training-style HTTP envelope")
    # clean() in the training pipeline normalizes redundant whitespace after
    # wrapping; preprocess_inference_input applies the same final step.
    merged["payload"] = inference.str[0]
    merged["input_kind"] = inference.str[1]
    merged = merged.reset_index(drop=True)
    merged["technique"] = merged["technique"].replace("", "unspecified")

    stats = {
        "input_rows_by_source": raw_counts,
        "input_rows_total": sum(raw_counts.values()),
        "exact_duplicate_rows_removed_by_source": exact_duplicate_counts,
        "empty_rows_removed": empty_rows,
        "evaluated_rows": len(merged),
        "repeated_model_inputs_retained": int(merged.duplicated(subset=["payload"]).sum()),
        "model_inputs_with_conflicting_labels": int(merged.groupby("payload")["label"].nunique().gt(1).sum()),
        "label_counts": {str(k): int(v) for k, v in merged["label"].value_counts().items()},
        "evaluated_rows_by_source": {k: int(v) for k, v in merged["source"].value_counts().items()},
        "input_kinds": {k: int(v) for k, v in merged["input_kind"].value_counts().items()},
    }
    return merged, stats


def metrics(frame: pd.DataFrame, threshold: float) -> dict:
    if frame.empty:
        return {"rows": 0}
    true = frame["label"].to_numpy(dtype=np.int8)
    prob = frame["attack_probability"].to_numpy(dtype=np.float64)
    predicted = (prob >= threshold).astype(np.int8)
    report = classification_report(
        true, predicted, labels=[0, 1], target_names=["normal", "attack"],
        output_dict=True, zero_division=0,
    )
    result = {
        "rows": len(frame),
        "label_counts": {"normal": int((true == 0).sum()), "attack": int((true == 1).sum())},
        "accuracy": float(accuracy_score(true, predicted)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(true, predicted, labels=[0, 1]).ravel().astype(int).tolist(),
        "normal": report["normal"],
        "attack": report["attack"],
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
    }
    if len(np.unique(true)) == 2:
        result["roc_auc"] = float(roc_auc_score(true, prob))
        result["average_precision"] = float(average_precision_score(true, prob))
    return result


def evaluate(
    frame: pd.DataFrame,
    batch_size: int,
    threshold: float,
    assets: tuple | None = None,
) -> tuple[pd.DataFrame, int]:
    model, tokenizer, pad_sequences, max_len = assets or load_inference_assets()
    probabilities = np.empty(len(frame), dtype=np.float32)
    token_counts = np.empty(len(frame), dtype=np.int32)
    for start in range(0, len(frame), batch_size):
        end = min(start + batch_size, len(frame))
        sequences = tokenizer.texts_to_sequences(frame["payload"].iloc[start:end].tolist())
        token_counts[start:end] = [len(sequence) for sequence in sequences]
        vectors = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
        probabilities[start:end] = model.predict(vectors, batch_size=batch_size, verbose=0).reshape(-1)
        if end == len(frame) or end % (batch_size * 20) == 0:
            print(f"Evaluated {end:,}/{len(frame):,} rows", flush=True)
    frame = frame.copy()
    frame["attack_probability"] = probabilities
    frame["predicted_label"] = (probabilities >= threshold).astype(np.int8)
    frame["token_count"] = token_counts
    frame["truncated"] = token_counts > max_len
    return frame, max_len


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obfuscated-grouped", type=Path, default=PROJECT_ROOT / "obfuscated_grouped.csv")
    parser.add_argument("--xss-payloads", type=Path, default=PROJECT_ROOT / "xss_payloads_with_obfuscated.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, help="Smoke test only: evaluate the first N prepared rows.")
    parser.add_argument("--prepare-only", action="store_true", help="Merge and preprocess without loading the model.")
    parser.add_argument("--threshold", type=float, help="Override the threshold saved with the webapp model.")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size and --limit must be positive")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.is_file() else {}
    threshold = float(args.threshold if args.threshold is not None else metadata.get("threshold", 0.5))
    if not 0 <= threshold <= 1:
        parser.error("Threshold must be between 0 and 1")

    frame, preparation = prepare_merged({
        "obfuscated_grouped": args.obfuscated_grouped,
        "xss_payloads_with_obfuscated": args.xss_payloads,
    })
    if args.limit:
        frame = frame.head(args.limit).copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_dir / "merged_preprocessed.csv"
    frame.to_csv(merged_path, index=False, encoding="utf-8")
    summary = {
        "model_path": str(METADATA_PATH.parent / "best_tuned_hybrid_cnn_lstm.keras"),
        "tokenizer_path": str(METADATA_PATH.parent / "tokenizer.pkl"),
        "threshold": threshold,
        "preprocessing": "preprocessing.preprocess_data.wrap_payload_as_request + preprocess_inference_input",
        "preparation": preparation,
        "rows_tested": len(frame),
        "partial_test": args.limit is not None,
        "prepared_csv": str(merged_path),
    }
    if args.prepare_only:
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Prepared {len(frame):,} rows: {merged_path}")
        return

    frame, max_len = evaluate(frame, args.batch_size, threshold)
    if args.training_csv.is_file():
        train_inputs = set(pd.read_csv(args.training_csv, usecols=["payload"], dtype=str)["payload"].dropna())
        frame["in_training_data"] = frame["payload"].isin(train_inputs)
    else:
        frame["in_training_data"] = False

    predictions_path = args.output_dir / "predictions.csv"
    frame.to_csv(predictions_path, index=False, encoding="utf-8")
    summary["predictions_csv"] = str(predictions_path)
    summary["max_len"] = max_len
    summary["truncated_rows"] = int(frame["truncated"].sum())
    summary["zero_token_rows"] = int(frame["token_count"].eq(0).sum())
    summary["training_overlap_rows"] = int(frame["in_training_data"].sum())
    summary["training_overlap_checked"] = args.training_csv.is_file()
    summary["overall"] = metrics(frame, threshold)
    summary["excluding_training_overlap"] = metrics(frame.loc[~frame["in_training_data"]], threshold)
    summary["by_source"] = {name: metrics(part, threshold) for name, part in frame.groupby("source")}
    summary["by_technique"] = {name: metrics(part, threshold) for name, part in frame.groupby("technique")}
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved merged data: {merged_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {summary_path}")
    print("Overall:", json.dumps(summary["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
