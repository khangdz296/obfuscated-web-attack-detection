"""Evaluate trained models on a unified obfuscation robustness dataset.

This script does not train any model. It merges the group-generated
obfuscated HTTP dataset with two additional obfuscation datasets, then uses
existing artifacts for evaluation.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.preprocess_data import (
    load_obfu_http,
    normalize_payload,
    preprocess_inference_input,
)


DATASETS = [
    {
        "name": "external_sql_obfuscated",
        "path": PROJECT_ROOT / "dataset" / "obfuscated_grouped.csv",
        "text_col": "query",
        "label_col": "label",
    },
    {
        "name": "external_xss_obfuscated",
        "path": PROJECT_ROOT / "dataset" / "xss_payloads_with_obfuscated.csv",
        "text_col": "Sentence",
        "label_col": "Label",
    },
]

ADDITIONAL_OBFU_MERGED_NAME = "additional_obfu_merged"
GROUP_OBFU_HTTP_TEST_NAME = "group_obfuscated_http_test_splits"
COMBINED_DATASET_NAME = "combined_obfu_all_sources"
OBFU_HTTP_DATASET_PATH = PROJECT_ROOT / "dataset" / "obfuscated_http_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "obfu_eval_outputs"

MODELS = [
    {
        "name": "CNN-LSTM tuned",
        "model_path": PROJECT_ROOT
        / "cnn_lstm"
        / "artifacts_cnn_lstm_tuning"
        / "obfu_http"
        / "final"
        / "best_tuned_hybrid_cnn_lstm.keras",
        "tokenizer_path": PROJECT_ROOT
        / "cnn_lstm"
        / "artifacts_cnn_lstm_tuning"
        / "obfu_http"
        / "final"
        / "tokenizer.pkl",
        "metadata_path": PROJECT_ROOT
        / "cnn_lstm"
        / "artifacts_cnn_lstm_tuning"
        / "obfu_http"
        / "final"
        / "metadata_and_results.json",
    },
    {
        "name": "CNN-only obfu",
        "model_path": PROJECT_ROOT
        / "cnn_only"
        / "artifacts_cnn_only_by_dataset"
        / "by_dataset"
        / "obfu_http"
        / "best_cnn_only.keras",
        "tokenizer_path": PROJECT_ROOT
        / "cnn_only"
        / "artifacts_cnn_only_by_dataset"
        / "by_dataset"
        / "obfu_http"
        / "tokenizer.pkl",
        "metadata_path": PROJECT_ROOT
        / "cnn_only"
        / "artifacts_cnn_only_by_dataset"
        / "by_dataset"
        / "obfu_http"
        / "metadata_and_results.json",
    },
]


def load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def metadata_value(metadata: dict, *keys: str, default=None):
    current = metadata
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_external_dataset(spec: dict) -> tuple[list[str], np.ndarray, int]:
    df = pd.read_csv(spec["path"])
    text_col = spec["text_col"]
    label_col = spec["label_col"]
    before = len(df)
    df = df[df[text_col].map(lambda value: bool(normalize_payload(value)))].copy()
    texts = [preprocess_inference_input(value)[0] for value in df[text_col].astype(str)]
    labels = df[label_col].astype(int).to_numpy()
    return texts, labels, before - len(df)


def load_external_frame(spec: dict) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(spec["path"])
    text_col = spec["text_col"]
    label_col = spec["label_col"]
    before = len(df)
    df = df[df[text_col].map(lambda value: bool(normalize_payload(value)))].copy()

    out = pd.DataFrame()
    out["payload"] = df[text_col].astype(str)
    out["label"] = df[label_col].astype(int)
    out["source_dataset"] = spec["name"]
    out["source_split"] = "external"
    if "technique" in df.columns:
        out["technique"] = df["technique"].astype(str)
    else:
        out["technique"] = ""
    if "group" in df.columns:
        out["group"] = df["group"].astype(str)
    else:
        out["group"] = ""
    return out, before - len(df)


def load_group_obfu_http_frame(test_only: bool = True) -> pd.DataFrame:
    df = load_obfu_http(str(OBFU_HTTP_DATASET_PATH))
    if "split" not in df.columns:
        raise ValueError(f"{OBFU_HTTP_DATASET_PATH} must contain a split column.")

    if test_only:
        df = df[df["split"].astype(str).str.startswith("test")].copy()

    out = pd.DataFrame()
    out["payload"] = df["payload"].astype(str)
    out["label"] = df["label"].astype(int)
    out["source_dataset"] = "group_obfuscated_http"
    out["source_split"] = df["split"].astype(str)
    out["technique"] = (
        df["obfuscation_techniques"].astype(str)
        if "obfuscation_techniques" in df.columns
        else ""
    )
    out["group"] = df["seed_id"].astype(str) if "seed_id" in df.columns else ""
    return out


def evaluate_one(model_spec: dict, dataset_spec: dict, texts: list[str], labels: np.ndarray) -> dict:
    metadata = load_metadata(model_spec["metadata_path"])
    max_len = int(
        metadata_value(metadata, "model", "max_len", default=None)
        or metadata.get("max_len", 1024)
    )
    threshold = float(
        metadata.get("threshold")
        or metadata_value(metadata, "threshold_selection", "threshold", default=None)
        or metadata_value(metadata, "model", "decision_threshold", default=0.5)
    )

    with model_spec["tokenizer_path"].open("rb") as file:
        tokenizer = pickle.load(file)

    model = tf.keras.models.load_model(model_spec["model_path"], compile=False)
    sequences = tokenizer.texts_to_sequences(texts)
    features = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    started = time.perf_counter()
    probabilities = model.predict(features, batch_size=256, verbose=0).ravel()
    inference_seconds = time.perf_counter() - started
    predictions = (probabilities >= threshold).astype(int)

    attack_precision, attack_recall, attack_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    return {
        "model": model_spec["name"],
        "dataset": dataset_spec["name"],
        "rows": int(len(labels)),
        "max_len": max_len,
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "auc_roc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
        "attack_precision": attack_precision,
        "attack_recall": attack_recall,
        "attack_f1": attack_f1,
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
        "inference_ms_per_sample": inference_seconds / max(len(labels), 1) * 1000,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prepared = []
    merged_frames = []
    for dataset_spec in DATASETS:
        frame, _ = load_external_frame(dataset_spec)
        merged_frames.append(frame)
        print(
            f"Loaded {dataset_spec['name']}: {len(frame):,} rows"
        )

    additional_obfu = pd.concat(merged_frames, ignore_index=True)
    additional_obfu_path = OUTPUT_DIR / "additional_obfu_merged.csv"
    additional_obfu.to_csv(additional_obfu_path, index=False, encoding="utf-8")
    print(f"Saved {ADDITIONAL_OBFU_MERGED_NAME}: {len(additional_obfu):,} rows")

    group_obfu_test = load_group_obfu_http_frame(test_only=True)
    group_obfu_test_path = OUTPUT_DIR / "group_obfuscated_http_test_splits.csv"
    group_obfu_test.to_csv(group_obfu_test_path, index=False, encoding="utf-8")
    print(f"Saved {GROUP_OBFU_HTTP_TEST_NAME}: {len(group_obfu_test):,} rows")

    combined = pd.concat([group_obfu_test, additional_obfu], ignore_index=True)
    combined_path = OUTPUT_DIR / "combined_obfu_all_sources.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8")
    combined_texts = [
        preprocess_inference_input(value)[0]
        for value in combined["payload"].astype(str)
    ]
    combined_labels = combined["label"].to_numpy()
    prepared.append(
        (
            {"name": COMBINED_DATASET_NAME},
            combined_texts,
            combined_labels,
            0,
        )
    )
    print(f"Prepared {COMBINED_DATASET_NAME}: {len(combined_labels):,} rows")

    group_obfu_full = load_group_obfu_http_frame(test_only=False)
    full_combined_path = OUTPUT_DIR / "combined_obfu_all_sources_full.csv"
    pd.concat([group_obfu_full, additional_obfu], ignore_index=True).to_csv(
        full_combined_path,
        index=False,
        encoding="utf-8",
    )

    rows = []
    for model_spec in MODELS:
        for dataset_spec, texts, labels, _ in prepared:
            result = evaluate_one(model_spec, dataset_spec, texts, labels)
            rows.append(result)
            print(
                f"{result['model']} on {result['dataset']}: "
                f"accuracy={result['accuracy']:.6f}, "
                f"attack_f1={result['attack_f1']:.6f}, "
                f"FP={result['fp']}, FN={result['fn']}"
            )

    output_path = OUTPUT_DIR / "external_obfu_eval_results.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
