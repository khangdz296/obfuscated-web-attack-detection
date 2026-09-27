"""
Flask inference app for the trained Hybrid CNN-LSTM detector.

Run:
    python app.py

Then open:
    http://127.0.0.1:8000
"""

import json
import os
import pickle
import sys
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.preprocess_data import preprocess_inference_input

DEFAULT_ARTIFACTS_DIR = (
    PROJECT_ROOT
    / "cnn_lstm"
    / "artifacts_cnn_lstm_tuning_768"
    / "obfu_http"
    / "final"
)
ARTIFACTS_DIR = Path(
    os.environ.get("CNN_LSTM_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR))
).expanduser().resolve()
MODEL_PATH = ARTIFACTS_DIR / "best_tuned_hybrid_cnn_lstm.keras"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.pkl"
METADATA_PATH = ARTIFACTS_DIR / "metadata_and_results.json"
DEFAULT_MAX_LEN = 768
DEFAULT_THRESHOLD = 0.5
MAX_INPUT_CHARS = 100_000
HOST = os.environ.get("DETECTOR_HOST", "127.0.0.1")
PORT = int(os.environ.get("DETECTOR_PORT", "8000"))

app = Flask(__name__)


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_runtime_status() -> dict:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "tensorflow_importable": find_spec("tensorflow") is not None,
        "keras_importable": find_spec("keras") is not None,
        "flask_importable": find_spec("flask") is not None,
    }


@lru_cache(maxsize=1)
def load_inference_assets():
    """Load TensorFlow model and tokenizer once, on the first prediction request."""
    missing = [str(path) for path in (MODEL_PATH, TOKENIZER_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifact(s): " + ", ".join(missing))

    try:
        import tensorflow as tf
    except ImportError as exc:
        runtime = get_runtime_status()
        raise RuntimeError(
            "TensorFlow is not installed or is broken in this Python environment. "
            "Run: python -m pip install -r requirements.txt. "
            f"Current Python: {runtime['python_executable']}"
        ) from exc

    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except ImportError:
        try:
            from keras.preprocessing.sequence import pad_sequences
        except ImportError as exc:
            runtime = get_runtime_status()
            raise RuntimeError(
                "TensorFlow is importable, but pad_sequences could not be imported "
                "from tensorflow.keras or keras. Run: "
                "python -m pip install --upgrade tensorflow keras. "
                f"Current Python: {runtime['python_executable']}"
            ) from exc

    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    metadata = load_metadata()
    model_max_len = model.input_shape[1]
    if model_max_len is None:
        model_max_len = DEFAULT_MAX_LEN
    model_max_len = int(model_max_len)

    metadata_max_len = metadata.get("model", {}).get(
        "max_len",
        metadata.get("max_len"),
    )
    if metadata_max_len is not None and int(metadata_max_len) != model_max_len:
        raise ValueError(
            "MAX_LEN in metadata does not match the model input shape: "
            f"metadata={metadata_max_len}, model={model_max_len}."
        )

    max_len = model_max_len
    return model, tokenizer, pad_sequences, max_len


def predict_payload(payload: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    if not isinstance(payload, str):
        raise ValueError("Input must be a string.")
    if len(payload) > MAX_INPUT_CHARS:
        raise ValueError(f"Input exceeds the {MAX_INPUT_CHARS:,}-character limit.")

    model_input, input_kind = preprocess_inference_input(payload)

    model, tokenizer, pad_sequences, max_len = load_inference_assets()
    sequence = tokenizer.texts_to_sequences([model_input])
    vector = pad_sequences(sequence, maxlen=max_len, padding="post", truncating="post")
    probability = float(model.predict(vector, verbose=0).flatten()[0])
    label = 1 if probability >= threshold else 0

    return {
        "label": label,
        "class_name": "Attack" if label else "Normal",
        "attack_probability": probability,
        "normal_probability": 1.0 - probability,
        "threshold": threshold,
        "normalized_payload": model_input,
        "input_kind": input_kind,
        "original_input_length": len(payload),
        "input_length": len(model_input),
        "token_count": len(sequence[0]),
        "max_len": max_len,
        "truncated": len(sequence[0]) > max_len,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    metadata = load_metadata()
    return jsonify(
        {
            "ok": True,
            "model_variant": "CNN-LSTM tuned max_len=768",
            "model_exists": MODEL_PATH.exists(),
            "tokenizer_exists": TOKENIZER_PATH.exists(),
            "model_path": str(MODEL_PATH),
            "tokenizer_path": str(TOKENIZER_PATH),
            "metadata_path": str(METADATA_PATH),
            "metadata_max_len": metadata.get("model", {}).get(
                "max_len",
                metadata.get("max_len"),
            ),
            "threshold": metadata.get("threshold", DEFAULT_THRESHOLD),
            "runtime": get_runtime_status(),
        }
    )


@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        text = payload.get("payload", "")
        threshold = float(payload.get("threshold", DEFAULT_THRESHOLD))
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")

        result = predict_payload(text, threshold)
        return jsonify({"ok": True, "result": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    print(f"Serving CNN-LSTM detector at http://{HOST}:{PORT}")
    print(f"Model    : {MODEL_PATH}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    app.run(host=HOST, port=PORT, debug=False)
