"""
All-in-one pipeline for char-level Hybrid 1D-CNN + LSTM web attack detection.

This file intentionally keeps preprocessing, vectorization, training, and
evaluation together so the research workflow is easy to run and explain.

Main idea:
- Preserve obfuscation evidence: no URL decode, no HTML unescape, no lowercase.
- Normalize only redundant whitespace.
- Split each dataset separately.
- Train one model per dataset and evaluate each model on every dataset test split.
- Fit each tokenizer on that model's own train split only to avoid data leakage.
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import preprocess_data as prep

KAGGLE_PATH = prep.KAGGLE_PATH
CSIC_PATH = prep.CSIC_PATH
OBFUSCATION_PATH = prep.OBFU_PATH
OBFU_PAYLOAD_PATH = prep.OBFU_PAYLOAD_PATH
OUTPUT_DIR = str(MODEL_DIR / "artifacts_cnn_lstm_by_dataset")
MAX_LEN = 768
EMBEDDING_DIM = 64
SEED = 42
DECISION_THRESHOLD = 0.5
EARLY_STOPPING_MIN_DELTA = 1e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def portable_artifact_path(path: Path) -> str:
    """Prefer project-relative POSIX paths so artifacts remain portable."""
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved_path)


# Legacy helper kept for cnn_only/cnn_bilstm/sequence_pool scripts that still
# train on the old combined Kaggle+CSIC split.
def build_datasets(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    base_df = pd.concat(
        [
            prep.load_kaggle(args.kaggle_path),
            prep.load_csic(args.csic_path),
        ],
        ignore_index=True,
    )
    base_df = prep.clean(base_df, deduplicate=True, drop_label_conflicts=True)
    base_df = base_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    if args.sample_size:
        base_df = base_df.sample(n=min(args.sample_size, len(base_df)), random_state=args.seed)
        base_df = base_df.reset_index(drop=True)

    train_val_df, test_df = prep.safe_train_test_split(base_df, args.test_size, args.seed)
    train_df, val_df = prep.safe_train_test_split(train_val_df, args.val_size, args.seed)

    obfuscated_df = prep.clean(
        prep.load_obfuscation(args.obfuscation_path),
        deduplicate=True,
        drop_label_conflicts=False,
    )
    if args.obfu_sample_size:
        obfuscated_df = obfuscated_df.sample(
            n=min(args.obfu_sample_size, len(obfuscated_df)),
            random_state=args.seed,
        ).reset_index(drop=True)

    metadata = {
        "preprocessing_policy": {
            "url_decode": False,
            "html_unescape": False,
            "lowercase": False,
            "whitespace_normalization_only": True,
            "input_representation": "Unified HTTP envelope for every source.",
            "csic_payload_policy": "Keep method, path, query, body, cookie and content type.",
            "drop_label_conflicts_base": True,
            "drop_label_conflicts_obfuscated_test": False,
            "tokenizer_rule": "Tokenizer is fit on train split only.",
        },
        "splits": {
            "base_clean": prep.summarize(base_df),
            "train": prep.summarize(train_df),
            "val": prep.summarize(val_df),
            "test": prep.summarize(test_df),
            "obfuscated_test": prep.summarize(obfuscated_df),
        },
    }
    return train_df, val_df, test_df, obfuscated_df, metadata


def build_tokenizer(train_payloads: pd.Series) -> Tokenizer:
    tokenizer = Tokenizer(
        char_level=True,
        lower=False,
        filters="",
        oov_token="<OOV>",
    )
    tokenizer.fit_on_texts(train_payloads)
    return tokenizer


def vectorize(tokenizer: Tokenizer, payloads: pd.Series, max_len: int) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(payloads)
    return pad_sequences(
        sequences,
        maxlen=max_len,
        padding="post",
        truncating="post",
    )


def build_model(
    vocab_size: int,
    max_len: int,
    embedding_dim: int,
    cnn_filters: int = 128,
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
) -> Sequential:
    """Build the sequential CNN-LSTM model.

    The optional arguments keep the original architecture as the default while
    allowing the notebook to tune model capacity, regularisation and learning
    rate without duplicating the topology.
    """
    model = Sequential(name="Hybrid_1D_CNN_LSTM_Sequence_Pooling_Web_Attack_Detector")
    model.add(Input(shape=(max_len,), name="payload_tokens"))
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="char_embedding"))

    model.add(Conv1D(filters=cnn_filters, kernel_size=3, padding="same", activation="relu", name="conv_k3"))
    model.add(MaxPooling1D(pool_size=4, name="pool_1"))

    model.add(Conv1D(filters=cnn_filters, kernel_size=5, padding="same", activation="relu", name="conv_k5"))
    model.add(MaxPooling1D(pool_size=4, name="pool_2"))

    model.add(LSTM(lstm_units, return_sequences=True, name="lstm_context_sequence"))
    model.add(GlobalMaxPooling1D(name="lstm_global_max_pool"))
    model.add(Dense(dense_units, activation="relu", name="dense_classifier"))
    model.add(Dropout(dropout, name="dropout"))
    model.add(Dense(1, activation="sigmoid", name="attack_probability"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def choose_validation_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Choose an F1-optimal threshold using validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if not len(thresholds):
        return {
            "threshold": DECISION_THRESHOLD,
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": 0.0,
        }
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1))
    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1[best_index]),
    }


def evaluate_model(
    model: Sequential,
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    batch_size: int,
    threshold: float,
) -> dict:
    y_prob = model.predict(X, batch_size=batch_size).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    result = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "auc_roc": float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, y_prob)) if len(np.unique(y)) > 1 else None,
        "decision_threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y,
            y_pred,
            labels=[0, 1],
            target_names=["Normal (0)", "Attack (1)"],
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
    }

    print(f"\n=== {name.upper()} RESULTS ===")
    print(f"Accuracy: {result['accuracy']:.4f}")
    if result["auc_roc"] is not None:
        print(f"AUC-ROC : {result['auc_roc']:.4f}")
        print(f"PR-AUC  : {result['pr_auc']:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print("Confusion matrix:")
    print(np.array(result["confusion_matrix"]))
    print("Classification report:")
    print(
        classification_report(
            y,
            y_pred,
            labels=[0, 1],
            target_names=["Normal (0)", "Attack (1)"],
            digits=4,
            zero_division=0,
        )
    )
    return result


def save_processed_csvs(
    output_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    obfuscated_df: pd.DataFrame,
) -> None:
    processed_dir = output_dir / "processed_data"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(processed_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(processed_dir / "test.csv", index=False, encoding="utf-8")
    obfuscated_df.to_csv(processed_dir / "obfuscated_test.csv", index=False, encoding="utf-8")


def build_source_datasets(args: argparse.Namespace) -> tuple[dict[str, dict[str, pd.DataFrame]], dict]:
    datasets = prep.load_clean_datasets(
        args.kaggle_path, args.csic_path, args.obfuscation_path,
        sources=getattr(args, "datasets", None),
        obfu_payload_path=getattr(args, "obfu_payload_path", OBFU_PAYLOAD_PATH),
    )
    sampled_datasets = {}
    for name, frame in datasets.items():
        sample_size = args.obfu_sample_size if name.startswith("obfu") else args.sample_size
        if sample_size and sample_size < len(frame):
            if "split" in frame.columns:
                # Sample inside each split so the shipped held-out design survives.
                frame = (
                    frame.groupby("split", sort=False)
                    .sample(frac=sample_size / len(frame), random_state=args.seed)
                    .reset_index(drop=True)
                )
            else:
                frame = frame.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)
        sampled_datasets[name] = frame

    dataset_splits = prep.split_all_datasets(
        sampled_datasets,
        args.test_size,
        args.val_size,
        args.seed,
        split_protocol=args.split_protocol,
    )
    metadata = {
        "preprocessing_policy": {
            "url_decode": False,
            "html_unescape": False,
            "lowercase": False,
            "whitespace_normalization_only": True,
            "input_representation": "Unified HTTP envelope for every source.",
            "csic_payload_policy": "Keep method, path, query, body, cookie and content type.",
            "split_protocol": args.split_protocol,
            "split_protocol_note": (
                "Stratified row split: request families may cross splits."
                if args.split_protocol == "random_stratified_row"
                else "Family-group split: canonical split_group never crosses splits."
            ),
            "tokenizer_rule": "Each model fits its tokenizer on its own dataset's train split only.",
        },
        "datasets": {},
    }
    for name, frame in sampled_datasets.items():
        metadata["datasets"][name] = {
            "clean": prep.summarize(frame),
            "splits": {
                split_name: prep.summarize(split_df)
                for split_name, split_df in dataset_splits[name].items()
            },
        }
    return dataset_splits, metadata


def save_source_processed_csvs(
    output_dir: Path,
    dataset_splits: dict[str, dict[str, pd.DataFrame]],
) -> None:
    processed_dir = output_dir / "processed_data_by_dataset"
    for dataset_name, splits in dataset_splits.items():
        for split_name, split_df in splits.items():
            split_dir = processed_dir / dataset_name
            split_dir.mkdir(parents=True, exist_ok=True)
            split_df.to_csv(split_dir / f"{split_name}.csv", index=False, encoding="utf-8")


def resolve_train_sources(raw_sources: list[str], available_sources: list[str]) -> list[str]:
    if not raw_sources or "all" in raw_sources:
        return available_sources
    invalid = sorted(set(raw_sources) - set(available_sources))
    if invalid:
        raise ValueError(f"Unknown train source(s): {invalid}. Available: {available_sources}")
    return raw_sources


def metric_from_report(result: dict, label: str, metric: str) -> float | None:
    value = result.get("classification_report", {}).get(label, {}).get(metric)
    return float(value) if value is not None else None


def evaluation_summary_row(train_source: str, test_source: str, result: dict) -> dict:
    return {
        "train_source": train_source,
        "test_source": test_source,
        "accuracy": result["accuracy"],
        "auc_roc": result["auc_roc"],
        "pr_auc": result["pr_auc"],
        "decision_threshold": result["decision_threshold"],
        "attack_precision": metric_from_report(result, "Attack (1)", "precision"),
        "attack_recall": metric_from_report(result, "Attack (1)", "recall"),
        "attack_f1": metric_from_report(result, "Attack (1)", "f1-score"),
        "normal_precision": metric_from_report(result, "Normal (0)", "precision"),
        "normal_recall": metric_from_report(result, "Normal (0)", "recall"),
        "normal_f1": metric_from_report(result, "Normal (0)", "f1-score"),
    }


def class_weights_for(y_train: np.ndarray) -> dict[int, float]:
    classes = np.unique(y_train)
    if len(classes) < 2:
        return {int(classes[0]): 1.0}
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return {int(label): float(weight) for label, weight in zip(classes, class_weights)}


def train_and_evaluate_source_model(
    train_source: str,
    dataset_splits: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    tf.keras.backend.clear_session()
    set_seed(args.seed)

    source_dir = output_dir / "by_dataset" / train_source
    source_dir.mkdir(parents=True, exist_ok=True)
    train_df = dataset_splits[train_source]["train"]
    val_df = dataset_splits[train_source]["val"]

    print(f"\n=== TRAINING MODEL FOR {train_source.upper()} ===")
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    tokenizer = build_tokenizer(train_df["payload"])
    vocab_size = len(tokenizer.word_index) + 1
    X_train = vectorize(tokenizer, train_df["payload"], args.max_len)
    X_val = vectorize(tokenizer, val_df["payload"], args.max_len)
    y_train = train_df["label"].to_numpy(dtype=np.int32)
    y_val = val_df["label"].to_numpy(dtype=np.int32)
    class_weight_dict = class_weights_for(y_train)

    print(f"Vocabulary size: {vocab_size}")
    print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")
    print(f"Class weights: {class_weight_dict}")

    model = build_model(vocab_size, args.max_len, args.embedding_dim)
    best_model_path = source_dir / "best_hybrid_cnn_lstm.keras"
    last_model_path = source_dir / "last_hybrid_cnn_lstm.keras"
    tokenizer_path = source_dir / "tokenizer.pkl"
    history_path = source_dir / "training_history.csv"
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=EARLY_STOPPING_MIN_DELTA,
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    model.save(last_model_path)
    with tokenizer_path.open("wb") as file:
        pickle.dump(tokenizer, file)
    pd.DataFrame(history.history).to_csv(history_path, index=False, encoding="utf-8")

    best_model = tf.keras.models.load_model(best_model_path)

    threshold_strategy = getattr(args, "threshold_strategy", "val_f1")
    if threshold_strategy == "val_f1":
        val_prob = best_model.predict(X_val, batch_size=args.batch_size).flatten()
        threshold_selection = choose_validation_threshold(y_val, val_prob)
    else:
        threshold_selection = {
            "threshold": DECISION_THRESHOLD,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    selected_threshold = float(threshold_selection["threshold"])
    print(
        f"Decision threshold ({threshold_strategy}): "
        f"{selected_threshold:.4f}"
    )

    evaluations = {}
    summary_rows = []
    for test_source, splits in dataset_splits.items():
        # Evaluate on every split whose name starts with "test". v2 ships three
        # generalisation splits alongside the in-distribution one, and the
        # comparison between them is the actual result.
        test_split_names = sorted(name for name in splits if name.startswith("test"))
        for split_name in test_split_names:
            if splits[split_name].empty:
                continue
            test_df = splits[split_name]
            label = test_source if split_name == "test" else f"{test_source}:{split_name}"
            X_test = vectorize(tokenizer, test_df["payload"], args.max_len)
            y_test = test_df["label"].to_numpy(dtype=np.int32)
            result = evaluate_model(
                best_model,
                X_test,
                y_test,
                f"{train_source} model on {label}",
                args.batch_size,
                selected_threshold,
            )
            evaluations[label] = result
            summary_rows.append(evaluation_summary_row(train_source, label, result))

    preprocessing_config = {
        "dataset": train_source,
        "seed": args.seed,
        "preprocessing_source": "preprocessing/preprocess_data.py",
        "split_target": {"train": 0.72, "validation": 0.08, "test": 0.20},
        "group_column": "split_group",
        "url_decode": False,
        "html_unescape": False,
        "lowercase": False,
        "whitespace_normalization_only": True,
        "input_representation": "unified_http_envelope",
        "tokenizer_fit_split": "train",
        "vocab_size": int(vocab_size),
        "max_len": int(args.max_len),
        "padding": "post",
        "truncating": "post",
        "decision_threshold": selected_threshold,
        "threshold_strategy": threshold_strategy,
        "threshold_selection_on_validation": threshold_selection,
        "class_weight": {str(key): float(value) for key, value in class_weight_dict.items()},
        "splits": {
            split_name: prep.summarize(split_df)
            for split_name, split_df in dataset_splits[train_source].items()
        },
    }
    model_metadata = {
        "train_source": train_source,
        "preprocessing": preprocessing_config,
        "model": {
            "max_len": args.max_len,
            "embedding_dim": args.embedding_dim,
            "vocab_size": vocab_size,
            "architecture": "Embedding -> Conv1D(k3) -> MaxPool(4) -> Conv1D(k5) -> MaxPool(4) -> LSTM(128, return_sequences=True) -> GlobalMaxPooling1D -> Dense(64) -> Dropout -> Sigmoid",
            "architecture_note": "One independent model is trained per dataset source, then evaluated on every dataset test split.",
            "cnn_filters": [128, 128],
            "cnn_kernel_sizes": [3, 5],
            "cnn_pool_sizes": [4, 4],
            "lstm_units": 128,
            "lstm_return_sequences": True,
            "lstm_pooling": "GlobalMaxPooling1D",
            "dense_units": 64,
            "dropout": 0.3,
            "learning_rate": 1e-3,
            "parameter_count": int(best_model.count_params()),
            "class_weight": class_weight_dict,
            "epochs_requested": int(args.epochs),
            "epochs_ran": int(len(history.history["loss"])),
            "early_stopping_monitor": "val_loss",
            "early_stopping_mode": "min",
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
            "early_stopping_patience": 3,
            "decision_threshold": selected_threshold,
            "threshold_strategy": threshold_strategy,
            "artifacts": {
                "best_model": portable_artifact_path(best_model_path),
                "last_model": portable_artifact_path(last_model_path),
                "tokenizer": portable_artifact_path(tokenizer_path),
                "training_history": portable_artifact_path(history_path),
            },
        },
        "training_history": {
            key: [float(value) for value in values] for key, values in history.history.items()
        },
        "evaluation": evaluations,
    }
    with (source_dir / "metadata_and_results.json").open("w", encoding="utf-8") as file:
        json.dump(model_metadata, file, ensure_ascii=False, indent=2)

    return model_metadata, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the Hybrid CNN-LSTM SQLi/XSS detector.")
    parser.add_argument("--kaggle-path", default=KAGGLE_PATH)
    parser.add_argument("--csic-path", default=CSIC_PATH)
    parser.add_argument("--obfuscation-path", default=OBFUSCATION_PATH)
    parser.add_argument("--obfu-payload-path", default=OBFU_PAYLOAD_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--split-protocol",
        choices=sorted(prep.SPLIT_PROTOCOLS),
        default=prep.DEFAULT_SPLIT_PROTOCOL,
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=["val_f1", "fixed"],
        default="val_f1",
        help="Tune threshold for attack F1 on validation data or keep fixed 0.5.",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="Optional quick-run sample size for each non-obfuscation dataset.")
    parser.add_argument("--obfu-sample-size", type=int, default=None, help="Optional quick-run sample size for the obfuscation datasets (obfu_http, obfu_payload).")
    parser.add_argument(
        "--train-sources",
        nargs="+",
        default=["all"],
        help="Datasets to train separate models for: all, kaggle, csic, obfu_http, obfu_payload.",
    )
    # Separate from --train-sources: this controls what gets *loaded*. Loading a
    # source costs a per-row serialisation pass, so skipping the ones you are
    # not using cuts start-up time. Every loaded source is still evaluated
    # against every trained model, which is what cross-source evaluation needs.
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to load at all: all, kaggle, csic, obfu_http, obfu_payload. "
             "Use a subset to skip loading sources you are not evaluating on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_splits, metadata = build_source_datasets(args)
    save_source_processed_csvs(output_dir, dataset_splits)

    print("=== DATASETS PREPARED ===")
    for dataset_name, splits in dataset_splits.items():
        # Print every split, not just the first three. The v2 dataset ships
        # three extra held-out splits; hiding them made the totals look wrong.
        total = sum(len(part) for part in splits.values())
        parts = " | ".join(f"{name}={len(part):,}" for name, part in splits.items())
        print(f"{dataset_name} (tổng {total:,}): {parts}")

    trainable_sources = [
        name
        for name, splits in dataset_splits.items()
        if splits["train"]["label"].nunique() >= 2
    ]
    evaluation_only_sources = sorted(set(dataset_splits) - set(trainable_sources))
    if evaluation_only_sources:
        print(
            "Evaluation-only source(s), skipped for standalone binary training: "
            + ", ".join(evaluation_only_sources)
        )
    train_sources = resolve_train_sources(args.train_sources, trainable_sources)
    print(f"\nTraining separate models for: {', '.join(train_sources)}")

    all_model_results = {}
    summary_rows = []
    for train_source in train_sources:
        model_metadata, model_rows = train_and_evaluate_source_model(
            train_source,
            dataset_splits,
            output_dir,
            args,
        )
        all_model_results[train_source] = model_metadata
        summary_rows.extend(model_rows)

    results_df = pd.DataFrame(summary_rows)
    results_df.to_csv(output_dir / "cross_eval_results.csv", index=False, encoding="utf-8")
    if not results_df.empty:
        results_df.pivot(
            index="train_source",
            columns="test_source",
            values="accuracy",
        ).to_csv(output_dir / "cross_eval_accuracy_matrix.csv", encoding="utf-8")

    experiment_results = {
        "dataset_metadata": metadata,
        "trained_sources": train_sources,
        "models": all_model_results,
        "artifacts": {
            "processed_data": portable_artifact_path(output_dir / "processed_data_by_dataset"),
            "models": portable_artifact_path(output_dir / "by_dataset"),
            "cross_eval_results": portable_artifact_path(output_dir / "cross_eval_results.csv"),
            "cross_eval_accuracy_matrix": portable_artifact_path(
                output_dir / "cross_eval_accuracy_matrix.csv"
            ),
        },
    }
    with (output_dir / "experiment_results.json").open("w", encoding="utf-8") as file:
        json.dump(experiment_results, file, ensure_ascii=False, indent=2)

    print(f"\nSaved by-dataset artifacts to: {(output_dir / 'by_dataset').resolve()}")
    print(f"Saved cross-evaluation table to: {(output_dir / 'cross_eval_results.csv').resolve()}")


if __name__ == "__main__":
    main()
