"""Shared utilities for evaluation: answer normalization, data loading, etc."""

import json
import re
import string
from collections import Counter
from pathlib import Path


def normalize_answer(s: str, remove_punctuation: bool = True) -> str:
    """Lower text, optionally remove punctuation, remove articles, and fix whitespace.

    This is the standard SQuAD-style normalization used across
    AdaCAD/eval_qa.py, COIECD/evaluate.py, and literature.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    text = s.lower()
    if remove_punctuation:
        text = remove_punc(text)
    return white_space_fix(remove_articles(text))


def normalize_answer_keep_punc(s: str) -> str:
    """Normalize answers like EM, but keep punctuation."""
    return normalize_answer(s, remove_punctuation=False)


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Strict exact match after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def exact_match_keep_punc(prediction: str, ground_truth: str) -> bool:
    """Exact match after normalization without punctuation removal."""
    return normalize_answer_keep_punc(prediction) == normalize_answer_keep_punc(ground_truth)


def substring_match(ground_truth: str, prediction: str) -> bool:
    """Check if normalized gold is a substring of normalized prediction.

    This matches AdaCAD/eval_qa.py behaviour.
    """
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score after normalization."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def load_gold_data(gold_path: str, format_type: str = "auto") -> list:
    """Load gold data from JSONL.

    Returns a list of dicts, one per *input_index*.  For CoCoA/AdaCAD
    dual-process format, only assigned_process==0 rows are kept.
    """
    path = Path(gold_path)
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    if not raw:
        return []

    if "assigned_process" in raw[0]:
        return [r for r in raw if r.get("assigned_process") == 0]

    return raw


def load_pred_data(pred_path: str, format_type: str = "auto", task_type: str = "qa") -> dict:
    """Load prediction data, returning a dict keyed by input_index.

    Supports two formats:
      - CoCoA/AdaCAD output: {"input_index": ..., "string": [...]}
      - COIECD output:       {"id": ..., "coiecd_answer": ...}
    """
    path = Path(pred_path)
    index2pred = {}

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("["):
        data = json.loads(content)
        for item in data:
            idx = item.get("id", item.get("input_index"))
            pred = _extract_prediction(item, task_type=task_type)
            index2pred[idx] = pred
    else:
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            idx = item.get("input_index", item.get("id"))
            pred = _extract_prediction(item, task_type=task_type)
            index2pred[idx] = pred

    return index2pred


def _extract_prediction(item: dict, task_type: str = "qa") -> str:
    """Extract the prediction string from various output formats.

    For QA tasks, only the first line of model output is the actual answer;
    subsequent lines are explanations/notes that should be ignored during eval.
    For non-QA tasks, keep the full output.
    """
    raw = ""
    if "coiecd_answer" in item:
        raw = item["coiecd_answer"]
    elif "string" in item:
        strings = item["string"]
        if isinstance(strings, list):
            raw = strings[0]
        else:
            raw = str(strings)
    elif "prediction" in item:
        raw = item["prediction"]
    elif "pred" in item:
        raw = item["pred"]

    if task_type == "qa":
        return raw.strip().split("\n")[0].strip()
    return raw.strip()
