import argparse
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
KAGGLE_PATH = str(DATA_DIR / "SQLInjection_XSS_MixDataset.1.0.0.csv")
CSIC_PATH = str(DATA_DIR / "csic_database.csv")
OBFU_PATH = str(DATA_DIR / "obfuscated_http_dataset.csv")
OUTPUT_DIR = str(PROJECT_ROOT / "cnn_lstm" / "artifacts" / "processed_data")
RANDOM_STATE = 42
DEFAULT_SPLIT_PROTOCOL = "random_stratified_row"
SPLIT_PROTOCOLS = {"random_stratified_row", "family_group"}


def normalize_payload(value: object) -> str:
    """No URL decode, no HTML unescape, no lowercase: keep obfuscation evidence."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def stable_choice(value: str, choices: list[tuple[str, str, str, str]]) -> tuple[str, str, str, str]:
    """Choose a neutral HTTP wrapper deterministically, without using the label."""
    digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).digest()
    return choices[int.from_bytes(digest[:4], "big") % len(choices)]


def serialize_http_request(
    method: object = "",
    path: object = "",
    query: object = "",
    body: object = "",
    cookie: object = "",
    content_type: object = "",
    user_agent: object = "",
) -> str:
    """Serialize heterogeneous sources into one source-agnostic model input."""
    fields = (
        ("METHOD", method),
        ("PATH", path),
        ("QUERY", query),
        ("BODY", body),
        ("COOKIE", cookie),
        ("CONTENT_TYPE", content_type),
        ("USER_AGENT", user_agent),
    )
    return " ".join(
        f"[{name}] {normalize_payload(value)}"
        for name, value in fields
    ).strip()


PAYLOAD_HTTP_TEMPLATES = [
    ("POST", "/submit", "body", "input"),
    ("GET", "/search", "query", "q"),
    ("POST", "/comment", "body", "text"),
    ("GET", "/product", "query", "id"),
    ("POST", "/login", "body", "username"),
]


def wrap_payload_as_request(payload: object) -> tuple[str, str]:
    """Place payload-only samples in a neutral HTTP envelope shared by both labels."""
    raw_payload = normalize_payload(payload)
    if not raw_payload:
        return "", ""
    # Variants that differ only in encoding/numbers receive the same wrapper,
    # which keeps the wrapper itself from defeating family-based splitting.
    wrapper_key = re.sub(r"\d+", "<num>", re.sub(r"%[0-9a-fA-F]{2}", "%hh", raw_payload.lower()))
    method, path, location, parameter = stable_choice(wrapper_key, PAYLOAD_HTTP_TEMPLATES)
    parameter_value = f"{parameter}={raw_payload}"
    query = parameter_value if location == "query" else ""
    body = parameter_value if location == "body" else ""
    content_type = "application/x-www-form-urlencoded" if body else ""
    model_input = serialize_http_request(
        method=method,
        path=path,
        query=query,
        body=body,
        content_type=content_type,
    )
    return model_input, raw_payload


def canonical_payload_family(value: object) -> str:
    """Group obvious variants so near-identical payloads cannot cross splits."""
    text = normalize_payload(value).lower()
    text = re.sub(r"%[0-9a-f]{2}", "%hh", text)
    text = re.sub(r"\d+", "<num>", text)
    return text


def canonical_value_shape(value: object) -> str:
    """Keep delimiters/encoding shape while removing request-specific values."""
    text = normalize_payload(value).lower()
    text = re.sub(r"%[0-9a-f]{2}", "%hh", text)
    text = re.sub(r"[a-z0-9]+", "<text>", text)
    return re.sub(r"(?:<text>){2,}", "<text>", text)


def canonical_csic_family(method: object, path: object, query: object, body: object) -> str:
    """Build a request family without cookies/session IDs or literal values."""
    normalized_path = re.sub(r"\d+", "<num>", normalize_payload(path).lower())
    return "|".join(
        [
            normalize_payload(method).lower(),
            normalized_path,
            canonical_value_shape(query),
            canonical_value_shape(body),
        ]
    )


def to_binary_label(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    attack_words = {
        "1", "true", "attack", "attacks", "malicious", "sqli", "sql", "xss", "anomalous",
    }
    normal_words = {"0", "false", "normal", "benign", "clean", "none"}

    mapped = []
    for value in text:
        if value in attack_words:
            mapped.append(1)
        elif value in normal_words:
            mapped.append(0)
        else:
            numeric = pd.to_numeric(value, errors="coerce")
            mapped.append(1 if pd.notna(numeric) and numeric > 0 else 0)
    return pd.Series(mapped, index=series.index, dtype="int64")


def load_kaggle(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Sentence", "SQLInjection", "XSS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    wrapped = df["Sentence"].apply(wrap_payload_as_request)
    out = pd.DataFrame()
    out["payload"] = wrapped.str[0]
    out["raw_payload"] = wrapped.str[1]
    out["label"] = df[["SQLInjection", "XSS"]].max(axis=1).astype(int)
    out["source"] = "kaggle_sqli_xss"
    out["attack_type"] = "mixed"
    out["obfuscation_type"] = "original"
    out["pattern_category"] = ""
    out["difficulty_level"] = ""
    out["split_group"] = out["payload"].apply(canonical_payload_family)
    return out


def extract_form_values(payload: object) -> str:
    """Extract raw form/query values without URL-decoding obfuscation evidence."""
    if not isinstance(payload, str):
        return ""

    values = []
    for pair in payload.split("&"):
        pair = pair.strip()
        if not pair:
            continue
        if "=" in pair:
            _, value = pair.split("=", 1)
        else:
            value = pair
        if value:
            values.append(value)
    return " ".join(values)


def extract_url_query(url: object) -> str:
    if not isinstance(url, str):
        return ""

    request_url = re.sub(r"\s+HTTP/\d(?:\.\d)?\s*$", "", url.strip())
    if "?" not in request_url:
        return ""
    return request_url.split("?", 1)[1]


def split_csic_url(url: object) -> tuple[str, str]:
    """Return raw path and query without URL-decoding attack evidence."""
    if not isinstance(url, str):
        return "", ""
    request_url = re.sub(r"\s+HTTP/\d(?:\.\d)?\s*$", "", url.strip())
    request_url = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^/]*", "", request_url)
    path, separator, query = request_url.partition("?")
    return path or "/", query if separator else ""


HTTP_METHODS = {
    "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"
}
HTTP_ENVELOPE_MARKERS = (
    "[METHOD]", "[PATH]", "[QUERY]", "[BODY]",
    "[COOKIE]", "[CONTENT_TYPE]", "[USER_AGENT]",
)


def preprocess_inference_input(value: object) -> tuple[str, str]:
    """Convert WebApp input to the unified HTTP envelope used for training.

    Detect an existing envelope, a raw HTTP request, a URL/path, or a
    payload-only sample. URL encoding and letter case are deliberately kept
    because they may carry obfuscation evidence.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Input is empty after whitespace normalization.")

    raw_input = value.strip()
    normalized_input = normalize_payload(raw_input)

    if all(marker in normalized_input for marker in HTTP_ENVELOPE_MARKERS):
        return normalized_input, "unified_http_envelope"

    lines = raw_input.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    request_line = lines[0].strip()
    request_match = re.match(
        r"^([A-Za-z]+)\s+(\S+?)(?:\s+HTTP/\d(?:\.\d)?)?$",
        request_line,
    )
    if request_match and request_match.group(1).upper() in HTTP_METHODS:
        method = request_match.group(1).upper()
        target = request_match.group(2)
        headers: dict[str, str] = {}
        body_start = len(lines)
        for index, line in enumerate(lines[1:], start=1):
            if not line.strip():
                body_start = index + 1
                break
            if ":" in line:
                name, header_value = line.split(":", 1)
                headers[name.strip().lower()] = header_value.strip()

        path, query = split_csic_url(target)
        body = "\n".join(lines[body_start:]) if body_start < len(lines) else ""
        model_input = serialize_http_request(
            method=method,
            path=path,
            query=query,
            body=body,
            cookie=headers.get("cookie", ""),
            content_type=headers.get("content-type", ""),
            user_agent=headers.get("user-agent", ""),
        )
        return normalize_payload(model_input), "raw_http_request"

    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw_input) or raw_input.startswith("/"):
        path, query = split_csic_url(raw_input)
        model_input = serialize_http_request(method="GET", path=path, query=query)
        return normalize_payload(model_input), "url"

    model_input, _ = wrap_payload_as_request(raw_input)
    return normalize_payload(model_input), "payload"


def serialize_csic_row(row: pd.Series) -> tuple[str, str, str]:
    path, query = split_csic_url(row.get("URL", ""))
    body = normalize_payload(row.get("content", ""))
    model_input = serialize_http_request(
        method=row.get("Method", ""),
        path=path,
        query=query,
        body=body,
        cookie=row.get("cookie", ""),
        content_type=row.get("content-type", ""),
    )
    raw_payload = " ".join(value for value in (query, body) if value)
    split_group = canonical_csic_family(row.get("Method", ""), path, query, body)
    return model_input, raw_payload, split_group


def load_csic(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"content", "URL", "classification"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    serialized = df.apply(serialize_csic_row, axis=1)
    out = pd.DataFrame()
    out["payload"] = serialized.str[0]
    out["raw_payload"] = serialized.str[1]
    out["split_group"] = serialized.str[2]
    out["label"] = to_binary_label(df["classification"])
    out["source"] = "csic_2010"
    out["attack_type"] = "mixed"
    out["obfuscation_type"] = "original"
    out["pattern_category"] = ""
    out["difficulty_level"] = ""
    return out


def read_xlsx_first_sheet(path: str) -> pd.DataFrame:
    """Read a simple XLSX table without openpyxl."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for si in root.findall(ns + "si"):
                shared_strings.append("".join(t.text or "" for t in si.iter(ns + "t")))

        sheet_root = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet_root.find(ns + "sheetData").findall(ns + "row"):
            values = []
            for cell in row.findall(ns + "c"):
                value_node = cell.find(ns + "v")
                inline_node = cell.find(ns + "is")
                if cell.get("t") == "inlineStr" and inline_node is not None:
                    value = "".join(t.text or "" for t in inline_node.iter(ns + "t"))
                else:
                    value = "" if value_node is None else value_node.text or ""
                if cell.get("t") == "s" and value:
                    value = shared_strings[int(value)]
                values.append(value)
            rows.append(values)

    if not rows:
        return pd.DataFrame()

    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    header = normalized_rows[0]
    data = normalized_rows[1:]
    return pd.DataFrame(data, columns=header)


def load_obfuscation(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        df = read_xlsx_first_sheet(path)
    else:
        df = pd.read_csv(path)

    required = {"obfuscated_input", "label", "obfuscation_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    wrapped = df["obfuscated_input"].apply(wrap_payload_as_request)
    out = pd.DataFrame()
    out["payload"] = wrapped.str[0]
    out["raw_payload"] = wrapped.str[1]
    out["label"] = to_binary_label(df["label"])
    out["source"] = "custom_obfuscation"
    out["attack_type"] = df["label"].astype(str).str.lower()
    out["obfuscation_type"] = df["obfuscation_type"]
    out["pattern_category"] = df["pattern_category"] if "pattern_category" in df.columns else ""
    out["difficulty_level"] = df["difficulty_level"] if "difficulty_level" in df.columns else ""
    if "original_pattern" in df.columns:
        out["original_pattern"] = df["original_pattern"]
        out["split_group"] = df["original_pattern"].fillna("").astype(str).apply(
            canonical_payload_family
        )
    else:
        out["split_group"] = out["payload"].apply(canonical_payload_family)
    return out


def serialize_obfu_http_row(row: pd.Series) -> tuple[str, str, str]:
    """Serialize a CSIC-style row, keeping cookie and User-Agent as attack carriers."""
    path, query = split_csic_url(row.get("url", ""))
    body = normalize_payload(row.get("content", ""))
    cookie = normalize_payload(row.get("cookie", ""))
    user_agent = normalize_payload(row.get("user_agent", ""))
    model_input = serialize_http_request(
        method=row.get("method", ""),
        path=path,
        query=query,
        body=body,
        cookie=cookie,
        content_type=row.get("content_type", ""),
        user_agent=user_agent,
    )
    raw_payload = " ".join(value for value in (query, body, cookie, user_agent) if value)
    split_group = canonical_csic_family(row.get("method", ""), path, query, body)
    return model_input, raw_payload, split_group


def load_obfu_http(path: str, drop_second_order_triggers: bool = True) -> pd.DataFrame:
    """Load the CSIC-style obfuscated HTTP dataset (method/url/classification schema)."""
    df = pd.read_csv(path)
    required = {"method", "url", "classification"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    if drop_second_order_triggers and {"is_second_order", "context_location"} <= set(df.columns):
        # Trigger rows are labelled anomalous but carry no payload, so a
        # single-request classifier cannot learn anything from them.
        is_trigger = (
            df["is_second_order"].astype(str).str.strip().str.lower().eq("true")
            & df["context_location"].fillna("").astype(str).str.strip().eq("")
        )
        df = df[~is_trigger].reset_index(drop=True)

    serialized = df.apply(serialize_obfu_http_row, axis=1)
    out = pd.DataFrame()
    out["payload"] = serialized.str[0]
    out["raw_payload"] = serialized.str[1]
    out["split_group"] = serialized.str[2]
    out["label"] = to_binary_label(df["classification"])
    out["source"] = "obfu_http"
    for source_column, target_column in [
        ("attack_category", "attack_type"),
        ("obfuscation_type", "obfuscation_type"),
        ("context_location", "pattern_category"),
        ("difficulty_level", "difficulty_level"),
        ("obfuscation_techniques", "obfuscation_techniques"),
        # v2 columns. benign_kind tells false positives apart by the kind of
        # legitimate traffic that triggered them; attack_technique shows which
        # attack families are blind spots; seed_id ties a row back to its
        # original payload for leakage checks.
        ("benign_kind", "benign_kind"),
        ("attack_technique", "attack_technique"),
        ("seed_id", "seed_id"),
        ("split", "split"),
    ]:
        out[target_column] = (
            df[source_column].fillna("").astype(str) if source_column in df.columns else ""
        )
    return out


def split_dataset_by_column(df: pd.DataFrame, split_column: str = "split") -> dict[str, pd.DataFrame]:
    """Use the split assignment shipped with the dataset instead of re-splitting it."""
    if split_column not in df.columns:
        raise ValueError(f"Dataset has no {split_column!r} column to split on.")

    values = df[split_column].fillna("").astype(str).str.strip().str.lower()
    # v1 shipped train/val/test/test_heldout. v2 replaces the single held-out
    # set with three, each isolating a different kind of generalisation:
    #   test_unseen_technique  payload seen in training, encoding never seen
    #   test_unseen_seed       encoding seen in training, payload never seen
    #   test_unseen_both       neither seen
    known = ("train", "val", "test", "test_heldout",
             "test_unseen_technique", "test_unseen_seed", "test_unseen_both")
    splits = {
        name: df[values == name].reset_index(drop=True)
        for name in known
    }
    missing = [name for name in ("train", "val", "test") if splits[name].empty]
    if missing:
        raise ValueError(f"Column {split_column!r} produced empty split(s): {missing}")
    return {name: part for name, part in splits.items()
            if name in ("train", "val", "test") or not part.empty}


def clean(df: pd.DataFrame, deduplicate: bool = True, drop_label_conflicts: bool = True) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["payload"] = cleaned["payload"].apply(normalize_payload)
    cleaned["label"] = to_binary_label(cleaned["label"])

    for column in [
        "raw_payload",
        "split_group",
        "source",
        "attack_type",
        "obfuscation_type",
        "pattern_category",
        "difficulty_level",
    ]:
        if column not in cleaned.columns:
            cleaned[column] = ""
        cleaned[column] = cleaned[column].fillna("").astype(str)

    cleaned = cleaned[cleaned["payload"].str.len() > 0]
    if drop_label_conflicts:
        label_counts = cleaned.groupby("payload")["label"].transform("nunique")
        cleaned = cleaned[label_counts == 1]
    if deduplicate:
        cleaned = cleaned.drop_duplicates(subset=["payload", "label"])
    return cleaned.reset_index(drop=True)


def summarize(df: pd.DataFrame) -> dict:
    lengths = df["payload"].str.len()
    summary = {
        "rows": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "source_counts": {str(k): int(v) for k, v in df["source"].value_counts().items()},
        "length": {
            "mean": float(lengths.mean()) if len(df) else 0.0,
            "median": float(lengths.median()) if len(df) else 0.0,
            "p90": float(lengths.quantile(0.90)) if len(df) else 0.0,
            "p95": float(lengths.quantile(0.95)) if len(df) else 0.0,
            "p99": float(lengths.quantile(0.99)) if len(df) else 0.0,
            "max": int(lengths.max()) if len(df) else 0,
        },
    }
    if "obfuscation_type" in df.columns:
        summary["obfuscation_counts"] = {
            str(k): int(v) for k, v in df["obfuscation_type"].value_counts().head(30).items()
        }
    if "split_group" in df.columns:
        summary["unique_split_groups"] = int(df["split_group"].nunique(dropna=False))
    return summary


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
    group_column: str | None = None,
) -> dict[str, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if not 0 < val_size < 1:
        raise ValueError("--val-size must be between 0 and 1.")

    if group_column and group_column in df.columns:
        return split_dataset_by_group(
            df.reset_index(drop=True),
            test_size,
            val_size,
            seed,
            group_column,
        )
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return split_dataset_by_row(shuffled, test_size, val_size, seed)


def safe_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    stratify_column: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = df[stratify_column] if stratify_column in df.columns else None
    try:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(df, test_size=test_size, random_state=seed)


def split_dataset_by_row(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    train_val_df, test_df = safe_train_test_split(df, test_size, seed)
    train_df, val_df = safe_train_test_split(train_val_df, val_size, seed)
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def split_dataset_by_group(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
    group_column: str,
) -> dict[str, pd.DataFrame]:
    group_key = df[group_column].fillna("").astype(str)
    group_key = group_key.where(group_key.str.len() > 0, df["payload"])

    test_keys = select_balanced_group_holdout(df, group_key, test_size, seed)
    train_val_mask = ~group_key.isin(test_keys)
    train_val_df = df[train_val_mask]
    train_val_group_key = group_key[train_val_mask]
    val_keys = select_balanced_group_holdout(
        train_val_df,
        train_val_group_key,
        val_size,
        seed + 1,
    )
    train_keys = set(train_val_group_key) - val_keys

    return {
        "train": df[group_key.isin(train_keys)].reset_index(drop=True),
        "val": df[group_key.isin(val_keys)].reset_index(drop=True),
        "test": df[group_key.isin(test_keys)].reset_index(drop=True),
    }


def select_balanced_group_holdout(
    df: pd.DataFrame,
    group_key: pd.Series,
    holdout_fraction: float,
    seed: int,
) -> set[str]:
    """Greedily choose whole groups while matching row and class targets."""
    group_counts = (
        pd.DataFrame(
            {
                "group_key": group_key.to_numpy(copy=False),
                "label": df["label"].to_numpy(copy=False),
            }
        )
        .groupby(["group_key", "label"], sort=False)
        .size()
        .unstack(fill_value=0)
    )
    for label in (0, 1):
        if label not in group_counts.columns:
            group_counts[label] = 0
    group_counts = group_counts[[0, 1]]

    count_values = group_counts[[0, 1]].to_numpy(dtype=np.float64, copy=False)
    targets = count_values.sum(axis=0) * holdout_fraction
    denominators = np.maximum(targets, 1.0)
    selected_counts = np.zeros(2, dtype=np.float64)
    selected: set[str] = set()

    rng = np.random.default_rng(seed)
    shuffled_positions = rng.permutation(len(group_counts))
    totals = count_values.sum(axis=1)
    mixed = ((count_values[:, 0] > 0) & (count_values[:, 1] > 0)).astype(np.int8)
    order_within_shuffle = np.lexsort(
        (totals[shuffled_positions], mixed[shuffled_positions])
    )
    ordered_positions = shuffled_positions[order_within_shuffle]
    group_names = group_counts.index.to_numpy(copy=False)

    def cost(counts: np.ndarray) -> float:
        return float(np.square((counts - targets) / denominators).sum())

    for position in ordered_positions:
        candidate_counts = selected_counts + count_values[position]
        if cost(candidate_counts) < cost(selected_counts):
            selected.add(str(group_names[position]))
            selected_counts = candidate_counts

    if not selected and len(group_counts):
        selected.add(str(group_counts.index[0]))
    if len(selected) == len(group_counts) and len(selected) > 1:
        selected.remove(str(group_counts.index[-1]))
    return selected


DATASET_SOURCES = ("kaggle", "csic", "obfu_http")


def load_clean_datasets(
    kaggle_path: str,
    csic_path: str,
    obfu_path: str,
    sources: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load and clean the requested sources.

    Serialising a source costs a per-row pass, so loading all three when only
    one is needed wastes a couple of minutes on every run. The loaders are kept
    behind lambdas so an unrequested source is never read from disk at all.

    `sources=None` (or "all") keeps the original behaviour of loading
    everything, which is what cross-source evaluation needs.
    """
    loaders = {
        "kaggle": lambda: clean(load_kaggle(kaggle_path),
                                deduplicate=True, drop_label_conflicts=True),
        "csic": lambda: clean(load_csic(csic_path),
                              deduplicate=True, drop_label_conflicts=True),
        "obfu_http": lambda: clean(load_obfu_http(obfu_path),
                                   deduplicate=True, drop_label_conflicts=False),
    }

    if not sources or "all" in sources:
        wanted = list(DATASET_SOURCES)
    else:
        unknown = sorted(set(sources) - set(loaders))
        if unknown:
            raise ValueError(
                f"Unknown dataset source(s): {unknown}. "
                f"Available: {sorted(loaders)} (or 'all')."
            )
        wanted = [name for name in DATASET_SOURCES if name in sources]

    return {name: loaders[name]() for name in wanted}


def split_all_datasets(
    datasets: dict[str, pd.DataFrame],
    test_size: float,
    val_size: float,
    seed: int,
    split_protocol: str = DEFAULT_SPLIT_PROTOCOL,
) -> dict[str, dict[str, pd.DataFrame]]:
    if split_protocol not in SPLIT_PROTOCOLS:
        raise ValueError(
            f"Unknown split protocol {split_protocol!r}; choose from {sorted(SPLIT_PROTOCOLS)}."
        )
    output = {}
    for name, frame in datasets.items():
        ships_own_split = (
            "split" in frame.columns
            and frame["split"].fillna("").astype(str).str.strip().ne("").any()
        )
        if ships_own_split:
            # The dataset author encoded a held-out design a generic splitter cannot rebuild.
            output[name] = split_dataset_by_column(frame)
        elif split_protocol == "random_stratified_row":
            output[name] = split_dataset_by_row(frame, test_size, val_size, seed)
        else:
            output[name] = split_dataset(
                frame,
                test_size,
                val_size,
                seed,
                group_column="split_group",
            )
    return output


def build_dataset_splits(
    kaggle_path: str,
    csic_path: str,
    obfu_path: str,
    test_size: float,
    val_size: float,
    seed: int,
    split_protocol: str = DEFAULT_SPLIT_PROTOCOL,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict]:
    datasets = load_clean_datasets(kaggle_path, csic_path, obfu_path)
    dataset_splits = split_all_datasets(
        datasets, test_size, val_size, seed, split_protocol=split_protocol
    )
    metadata = {
        "preprocessing_policy": {
            "url_decode": False,
            "html_unescape": False,
            "lowercase": False,
            "whitespace_normalization_only": True,
            "deduplicate_by": ["payload", "label"],
            "input_representation": "Unified HTTP envelope with METHOD, PATH, QUERY, BODY, COOKIE and CONTENT_TYPE fields.",
            "payload_only_policy": "Wrap Kaggle and obfuscation payloads in deterministic, label-independent HTTP templates.",
            "csic_payload_policy": "Keep raw method, path, query, body, cookie and content type; do not drop requests without parameters.",
            "split_protocol": split_protocol,
            "split_protocol_note": (
                "Stratified row split: request families may cross train, validation, and test."
                if split_protocol == "random_stratified_row"
                else "Family-group split: canonical split_group never crosses train, validation, and test."
            ),
            "tokenizer_rule": "Each model fits its tokenizer on that dataset's train split only.",
        },
        "datasets": {},
    }
    for name, frame in datasets.items():
        metadata["datasets"][name] = {
            "clean": summarize(frame),
            "splits": {split_name: summarize(split_df) for split_name, split_df in dataset_splits[name].items()},
        }
    return dataset_splits, metadata


def save_dataset_splits(dataset_splits: dict[str, dict[str, pd.DataFrame]], output_dir: Path) -> None:
    for dataset_name, splits in dataset_splits.items():
        for split_name, split_df in splits.items():
            save_csv(split_df, output_dir / dataset_name / f"{split_name}.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess datasets for char-level SQLi/XSS detection.")
    parser.add_argument("--kaggle-path", default=KAGGLE_PATH)
    parser.add_argument("--csic-path", default=CSIC_PATH)
    parser.add_argument("--obfu-path", default=OBFU_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--split-protocol",
        choices=sorted(SPLIT_PROTOCOLS),
        default=DEFAULT_SPLIT_PROTOCOL,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_splits, metadata = build_dataset_splits(
        args.kaggle_path,
        args.csic_path,
        args.obfu_path,
        args.test_size,
        args.val_size,
        args.seed,
        args.split_protocol,
    )
    save_dataset_splits(dataset_splits, output_dir)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=== PREPROCESSING DONE ===")
    print(f"Output directory: {output_dir.resolve()}")
    for dataset_name, splits in dataset_splits.items():
        print(
            f"{dataset_name}: "
            f"train={len(splits['train']):,} | "
            f"val={len(splits['val']):,} | "
            f"test={len(splits['test']):,}"
        )


if __name__ == "__main__":
    main()
