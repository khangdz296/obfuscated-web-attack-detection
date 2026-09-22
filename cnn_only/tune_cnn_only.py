"""Tune CNN-only on the saved obfu_http train/validation split, then retrain.

The search budget, seed, stopping rules and validation F1 selection match the
CNN-LSTM tuning notebook. The LSTM-only dimension is removed from its search
space, so all eight CNN configurations are distinct. Test rows are read only
after the winning configuration has been saved.
"""

import argparse
import hashlib
import json
import pickle
import random
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import BackupAndRestore, CSVLogger, EarlyStopping, ModelCheckpoint

from train_cnn_only import build_cnn_model, pipeline


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "cnn_lstm" / "artifacts_cnn_lstm_by_dataset" / "processed_data_by_dataset" / "obfu_http"
OUTPUT_DIR = ROOT / "cnn_only" / "artifacts_cnn_only_tuning" / "obfu_http"
HP_KEYS = ("embedding_dim", "cnn_filters", "dense_units", "dropout", "learning_rate", "batch_size")
SEED = 42
MAX_LEN = 1024
THRESHOLD = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configurations(count: int) -> list[dict]:
    baseline = dict(embedding_dim=64, cnn_filters=128, dense_units=64,
                    dropout=0.3, learning_rate=1e-3, batch_size=128)
    # Replicate the CNN-LSTM search order before projecting away lstm_units.
    keys = ("embedding_dim", "cnn_filters", "lstm_units", "dense_units",
            "dropout", "learning_rate", "batch_size")
    full_baseline = dict(baseline, lstm_units=128)
    candidates = [dict(zip(keys, values)) for values in product(
        [32, 64], [64, 128], [64, 128], [32, 64],
        [0.2, 0.4], [3e-4, 1e-3], [128, 256],
    )]
    candidates = [candidate for candidate in candidates if candidate != full_baseline]
    random.Random(SEED).shuffle(candidates)
    selected = [baseline]
    seen = {tuple(baseline[key] for key in HP_KEYS)}
    for candidate in candidates:
        projected = {key: candidate[key] for key in HP_KEYS}
        signature = tuple(projected[key] for key in HP_KEYS)
        if signature not in seen:
            selected.append(projected)
            seen.add(signature)
        if len(selected) >= count:
            break
    return selected


def read_split(name: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{name}.csv"
    frame = pd.read_csv(path, usecols=["payload", "label"], dtype={"payload": str, "label": np.int32})
    if frame["payload"].isna().any() or frame["label"].isna().any():
        raise ValueError(f"Null payload or label in {path}")
    return frame


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fit_model(hp: dict, vocab_size: int, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, model_path: Path,
              history_path: Path, epochs: int, patience: int,
              backup_dir: Path | None = None,
              resume: bool = False) -> tuple[pd.DataFrame, float, int]:
    tf.keras.backend.clear_session()
    pipeline.set_seed(SEED)
    initial_epoch = 0
    if resume:
        if not model_path.is_file() or not history_path.is_file():
            raise FileNotFoundError("Both the final checkpoint and history are required to resume")
        previous_history = pd.read_csv(history_path)
        if previous_history.empty:
            raise ValueError("Cannot resume from an empty training history")
        best_epoch = int(previous_history["val_loss"].idxmin()) + 1
        if best_epoch < len(previous_history):
            # The checkpoint contains the best epoch, so replay later epochs.
            archived = history_path.with_name("training_history_before_resume.csv")
            history_path.replace(archived)
            previous_history.iloc[:best_epoch].to_csv(history_path, index=False)
        initial_epoch = best_epoch
        model = tf.keras.models.load_model(model_path)
        if model.optimizer is None or int(model.input_shape[1]) != MAX_LEN:
            raise ValueError("Saved checkpoint cannot resume the selected CNN-only fit")
        print(f"Resuming saved checkpoint after epoch {initial_epoch}", flush=True)
    else:
        model = build_cnn_model(vocab_size, MAX_LEN, int(hp["embedding_dim"]),
                                cnn_filters=int(hp["cnn_filters"]),
                                dense_units=int(hp["dense_units"]),
                                dropout=float(hp["dropout"]),
                                learning_rate=float(hp["learning_rate"]))
    callbacks = []
    if backup_dir is not None:
        callbacks.append(BackupAndRestore(backup_dir=str(backup_dir), save_freq="epoch"))
    previous_best_loss = None
    if history_path.exists():
        previous_history = pd.read_csv(history_path)
        if not previous_history.empty:
            previous_best_loss = float(previous_history["val_loss"].min())
    callbacks.extend([
        CSVLogger(str(history_path), append=history_path.exists()),
        EarlyStopping(monitor="val_loss", mode="min", patience=patience,
                      min_delta=pipeline.EARLY_STOPPING_MIN_DELTA, restore_best_weights=True),
        ModelCheckpoint(str(model_path), monitor="val_loss", mode="min", save_best_only=True,
                        initial_value_threshold=previous_best_loss),
    ])
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              initial_epoch=initial_epoch, epochs=epochs, batch_size=int(hp["batch_size"]),
              class_weight=pipeline.class_weights_for(y_train), callbacks=callbacks, verbose=2)
    history = pd.read_csv(history_path)
    best = tf.keras.models.load_model(model_path, compile=False)
    scores = best.predict(X_val, batch_size=int(hp["batch_size"]), verbose=0).ravel()
    val_f1 = float(f1_score(y_val, scores >= THRESHOLD, pos_label=1))
    parameter_count = int(best.count_params())
    del model, best
    tf.keras.backend.clear_session()
    return history, val_f1, parameter_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-trials", type=int, default=8)
    parser.add_argument("--tuning-epochs", type=int, default=12)
    parser.add_argument("--final-epochs", type=int, default=50)
    parser.add_argument("--only-trial", type=int, help="Run one tuning trial, then exit.")
    parser.add_argument("--final-only", action="store_true", help="Retrain after all trials are saved.")
    args = parser.parse_args()
    if not 1 <= args.max_trials <= 128 or args.tuning_epochs < 1 or args.final_epochs < 1:
        parser.error("Trial count and epoch limits must be positive")
    if args.only_trial is not None and not 1 <= args.only_trial <= args.max_trials:
        parser.error("--only-trial must be within --max-trials")
    if args.final_only and args.only_trial is not None:
        parser.error("--final-only and --only-trial cannot be combined")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = read_split("train")
    val = read_split("val")
    split_hashes = {name: sha256(SPLIT_DIR / f"{name}.csv") for name in ("train", "val", "test")}
    tokenizer = pipeline.build_tokenizer(train["payload"])
    vocab_size = len(tokenizer.word_index) + 1
    X_train = pipeline.vectorize(tokenizer, train["payload"], MAX_LEN)
    X_val = pipeline.vectorize(tokenizer, val["payload"], MAX_LEN)
    y_train = train["label"].to_numpy(dtype=np.int32)
    y_val = val["label"].to_numpy(dtype=np.int32)
    print(f"Train {len(train):,}; val {len(val):,}; vocab {vocab_size}; max_len {MAX_LEN}", flush=True)

    results_path = args.output_dir / "tuning_results.csv"
    saved = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    completed = {int(row["trial"]): row for row in saved}
    search = configurations(args.max_trials)
    save_json(args.output_dir / "search_protocol.json", {
        "source": "obfu_http", "split_paths": {name: str(SPLIT_DIR / f"{name}.csv") for name in split_hashes},
        "split_sha256": split_hashes, "seed": SEED, "max_len": MAX_LEN,
        "threshold": THRESHOLD, "tuning_epochs": args.tuning_epochs,
        "tuning_patience": 2, "final_epochs": args.final_epochs,
        "final_patience": 3, "selection": "validation attack F1 at 0.5 descending, then min validation loss ascending",
        "configurations": search,
    })
    for index, hp in enumerate(search, 1):
        if args.final_only or (args.only_trial is not None and index != args.only_trial):
            continue
        if index in completed:
            if any(not np.isclose(float(completed[index][key]), hp[key]) for key in HP_KEYS):
                raise ValueError(f"Saved trial {index} has different hyperparameters")
            if not Path(completed[index]["checkpoint"]).is_file():
                raise FileNotFoundError(f"Saved trial {index} checkpoint is missing")
            print(f"Skipping completed trial {index}/{len(search)}", flush=True)
            continue
        trial_dir = args.output_dir / f"trial_{index:02d}"
        trial_dir.mkdir(exist_ok=True)
        save_json(trial_dir / "hyperparameters.json", hp)
        model_path = trial_dir / "best.keras"
        history_path = trial_dir / "training_history.csv"
        print(f"Trial {index}/{len(search)}: {hp}", flush=True)
        history, val_f1, parameters = fit_model(
            hp, vocab_size, X_train, y_train, X_val, y_val,
            model_path, history_path, args.tuning_epochs, 2,
            trial_dir / "training_backup",
        )
        row = dict(trial=index, **hp, epochs_ran=len(history),
                   min_val_loss=float(history["val_loss"].min()),
                   val_attack_f1_at_0_5=val_f1, parameter_count=parameters,
                   checkpoint=str(model_path))
        saved = [item for item in saved if int(item["trial"]) != index] + [row]
        pd.DataFrame(saved).sort_values("trial").to_csv(results_path, index=False)
        completed[index] = row
        print(f"Trial {index} complete: val F1={val_f1:.6f}", flush=True)

    if args.only_trial is not None:
        return
    if len(completed) != len(search):
        raise RuntimeError(f"Expected {len(search)} completed trials; found {len(completed)}")
    for index, hp in enumerate(search, 1):
        row = completed.get(index)
        if row is None or any(not np.isclose(float(row[key]), hp[key]) for key in HP_KEYS):
            raise ValueError(f"Saved trial {index} does not match the search configuration")
        if not Path(row["checkpoint"]).is_file():
            raise FileNotFoundError(f"Saved trial {index} checkpoint is missing")

    ranked = pd.DataFrame(saved).sort_values(
        ["val_attack_f1_at_0_5", "min_val_loss"], ascending=[False, True])
    winner = ranked.iloc[0]
    best_hp = {key: (int(winner[key]) if key in ("embedding_dim", "cnn_filters", "dense_units", "batch_size")
                     else float(winner[key])) for key in HP_KEYS}
    save_json(args.output_dir / "best_hyperparameters.json", best_hp)
    print(f"Selected trial {int(winner['trial'])}: {best_hp}", flush=True)

    final_dir = args.output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    final_model_path = final_dir / "best_cnn_only.keras"
    final_history_path = final_dir / "training_history.csv"
    if not final_model_path.exists() or not (final_dir / "metadata_and_results.json").exists():
        resume_final = final_model_path.is_file() and final_history_path.is_file()
        print("Resuming final fit" if resume_final else "Retraining selected configuration from scratch", flush=True)
        # The final fit follows the CNN-LSTM final retrain: no test data until it ends.
        if final_history_path.exists() and not resume_final:
            final_history_path.unlink()
        history, val_f1, parameters = fit_model(
            best_hp, vocab_size, X_train, y_train, X_val, y_val,
            final_model_path, final_history_path, args.final_epochs, 3,
            resume=resume_final,
        )
        with (final_dir / "tokenizer.pkl").open("wb") as stream:
            pickle.dump(tokenizer, stream)
        model = tf.keras.models.load_model(final_model_path, compile=False)
        test = read_split("test")
        X_test = pipeline.vectorize(tokenizer, test["payload"], MAX_LEN)
        y_test = test["label"].to_numpy(dtype=np.int32)
        test_result = pipeline.evaluate_model(model, X_test, y_test,
                                              "tuned CNN-only obfu_http internal test",
                                              int(best_hp["batch_size"]), THRESHOLD)
        metadata = {
            "train_source": "obfu_http",
            "seed": SEED,
            "selection_data": "validation only",
            "test_used_for_selection": False,
            "split_sha256": split_hashes,
            "best_hyperparameters": best_hp,
            "selected_trial": int(winner["trial"]),
            "model": {
                "architecture": "Embedding -> Conv1D(k3) -> MaxPool(4) -> Conv1D(k5) -> MaxPool(4) -> GlobalMaxPooling1D -> Dense -> Dropout -> Sigmoid",
                "max_len": MAX_LEN,
                "vocab_size": vocab_size,
                "parameter_count": parameters,
                "epochs_run": len(history),
                "threshold_selection_on_validation": {"threshold": THRESHOLD, "strategy": "fixed"},
                "artifacts": {"best_model": str(final_model_path), "tokenizer": str(final_dir / "tokenizer.pkl")},
            },
            "validation_attack_f1_at_0_5": val_f1,
            "evaluation": {"test": test_result},
        }
        save_json(final_dir / "metadata_and_results.json", metadata)
        print(f"Final checkpoint: {final_model_path}", flush=True)
    else:
        print(f"Final checkpoint already complete: {final_model_path}", flush=True)


if __name__ == "__main__":
    main()
