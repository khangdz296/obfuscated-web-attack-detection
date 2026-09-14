"""CNN-only baseline for per-dataset web attack experiments.

This script follows the same data policy as cnn_lstm/CNN_LSTM.py:
- load Kaggle, CSIC, and the custom obfuscation dataset as separate sources;
- split train/validation/test inside each source;
- fit one tokenizer per training source to avoid data leakage;
- train one CNN-only model per selected source;
- evaluate every trained model on every source test split.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn_lstm import CNN_LSTM as pipeline


CSIC_PATH = pipeline.CSIC_PATH
EMBEDDING_DIM = pipeline.EMBEDDING_DIM
KAGGLE_PATH = pipeline.KAGGLE_PATH
MAX_LEN = pipeline.MAX_LEN
OBFUSCATION_PATH = pipeline.OBFUSCATION_PATH
SEED = pipeline.SEED
DECISION_THRESHOLD = pipeline.DECISION_THRESHOLD

OUTPUT_DIR = str(MODEL_DIR / "artifacts_cnn_only_by_dataset")


def build_cnn_model(vocab_size: int, max_len: int, embedding_dim: int) -> Sequential:
    """CNN-only ablation: same Conv blocks as CNN-LSTM, no recurrent layer."""
    model = Sequential(name="CNN_Only_Web_Attack_Detector")
    model.add(Input(shape=(max_len,), name="payload_tokens"))
    model.add(Embedding(vocab_size, embedding_dim, name="char_embedding"))

    model.add(Conv1D(128, 3, padding="same", activation="relu", name="conv_k3"))
    model.add(MaxPooling1D(pool_size=4, name="pool_1"))

    model.add(Conv1D(128, 5, padding="same", activation="relu", name="conv_k5"))
    model.add(MaxPooling1D(pool_size=4, name="pool_2"))

    model.add(GlobalMaxPooling1D(name="global_max_pool"))
    model.add(Dense(64, activation="relu", name="dense_classifier"))
    model.add(Dropout(0.3, name="dropout"))
    model.add(Dense(1, activation="sigmoid", name="attack_probability"))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def train_and_evaluate_source_model(
    train_source: str,
    dataset_splits: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    tf.keras.backend.clear_session()
    pipeline.set_seed(args.seed)

    source_dir = output_dir / "by_dataset" / train_source
    source_dir.mkdir(parents=True, exist_ok=True)

    train_df = dataset_splits[train_source]["train"]
    val_df = dataset_splits[train_source]["val"]

    print(f"\n=== TRAINING CNN-ONLY MODEL FOR {train_source.upper()} ===")
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    tokenizer = pipeline.build_tokenizer(train_df["payload"])
    vocab_size = len(tokenizer.word_index) + 1

    X_train = pipeline.vectorize(tokenizer, train_df["payload"], args.max_len)
    X_val = pipeline.vectorize(tokenizer, val_df["payload"], args.max_len)
    y_train = train_df["label"].to_numpy(dtype=np.int32)
    y_val = val_df["label"].to_numpy(dtype=np.int32)
    class_weight = pipeline.class_weights_for(y_train)

    print(f"Vocabulary size: {vocab_size}")
    print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")
    print(f"Class weights: {class_weight}")

    model = build_cnn_model(vocab_size, args.max_len, args.embedding_dim)
    model.summary()

    best_model_path = source_dir / "best_cnn_only.keras"
    last_model_path = source_dir / "last_cnn_only.keras"
    tokenizer_path = source_dir / "tokenizer.pkl"
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=pipeline.EARLY_STOPPING_MIN_DELTA,
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            str(best_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]

    started = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    training_seconds = time.perf_counter() - started

    model.save(last_model_path)
    with tokenizer_path.open("wb") as file:
        pickle.dump(tokenizer, file)

    evaluations = {}
    summary_rows = []
    for test_source, splits in dataset_splits.items():
        test_split_names = sorted(name for name in splits if name.startswith("test"))
        for split_name in test_split_names:
            test_df = splits[split_name]
            X_test = pipeline.vectorize(tokenizer, test_df["payload"], args.max_len)
            y_test = test_df["label"].to_numpy(dtype=np.int32)
            result = pipeline.evaluate_model(
                model,
                X_test,
                y_test,
                f"{train_source} CNN-only model on {test_source} {split_name}",
                args.batch_size,
                DECISION_THRESHOLD,
            )
            result_key = test_source if split_name == "test" else f"{test_source}:{split_name}"
            evaluations[result_key] = result
            summary_rows.append(pipeline.evaluation_summary_row(train_source, result_key, result))

    model_metadata = {
        "train_source": train_source,
        "model": {
            "architecture": "Embedding -> Conv1D(k3) -> MaxPool(4) -> Conv1D(k5) -> MaxPool(4) -> GlobalMaxPooling1D -> Dense(64) -> Dropout -> Sigmoid",
            "architecture_note": "CNN-only ablation. It keeps the same embedding and convolution blocks as CNN-LSTM but removes the LSTM layer.",
            "max_len": args.max_len,
            "embedding_dim": args.embedding_dim,
            "vocab_size": vocab_size,
            "parameter_count": int(model.count_params()),
            "class_weight": class_weight,
            "training_seconds": float(training_seconds),
            "epochs_run": len(history.history["loss"]),
            "seconds_per_epoch_average": float(training_seconds / max(len(history.history["loss"]), 1)),
            "artifacts": {
                "best_model": str(best_model_path),
                "last_model": str(last_model_path),
                "tokenizer": str(tokenizer_path),
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
    parser = argparse.ArgumentParser(description="Train and evaluate CNN-only baselines per dataset.")
    parser.add_argument("--kaggle-path", default=KAGGLE_PATH)
    parser.add_argument("--csic-path", default=CSIC_PATH)
    parser.add_argument("--obfuscation-path", default=OBFUSCATION_PATH)
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
        choices=sorted(pipeline.prep.SPLIT_PROTOCOLS),
        default=pipeline.prep.DEFAULT_SPLIT_PROTOCOL,
        help="How to split each loaded dataset.",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="Optional quick-run sample size for Kaggle and CSIC.")
    parser.add_argument("--obfu-sample-size", type=int, default=None, help="Optional quick-run sample size for the obfuscation dataset.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["all", "kaggle", "csic", "obfu_http"],
        default=["all"],
        help="Datasets to load, split, and evaluate. Use 'kaggle' for a Kaggle-only run.",
    )
    parser.add_argument(
        "--train-sources",
        nargs="+",
        default=["all"],
        help="Datasets to train separate CNN-only models for: all, kaggle, csic, obfu_http.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline.set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_splits, metadata = pipeline.build_source_datasets(args)
    pipeline.save_source_processed_csvs(output_dir, dataset_splits)

    print("=== CNN-ONLY DATASETS PREPARED ===")
    for dataset_name, splits in dataset_splits.items():
        print(
            f"{dataset_name}: "
            f"train={len(splits['train']):,} | "
            f"val={len(splits['val']):,} | "
            f"test={len(splits['test']):,}"
        )

    train_sources = pipeline.resolve_train_sources(args.train_sources, list(dataset_splits.keys()))
    print(f"\nTraining separate CNN-only models for: {', '.join(train_sources)}")

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
            "processed_data": str(output_dir / "processed_data_by_dataset"),
            "models": str(output_dir / "by_dataset"),
            "cross_eval_results": str(output_dir / "cross_eval_results.csv"),
            "cross_eval_accuracy_matrix": str(output_dir / "cross_eval_accuracy_matrix.csv"),
        },
    }
    with (output_dir / "cross_eval_results.json").open("w", encoding="utf-8") as file:
        json.dump(experiment_results, file, ensure_ascii=False, indent=2)

    print(f"\nSaved CNN-only by-dataset artifacts to: {(output_dir / 'by_dataset').resolve()}")
    print(f"Saved cross-evaluation table to: {(output_dir / 'cross_eval_results.csv').resolve()}")


if __name__ == "__main__":
    main()
