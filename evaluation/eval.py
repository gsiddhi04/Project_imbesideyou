"""
eval.py
-------
Basic evaluation for Smart Wellness Planner+ outputs.
"""

import json
import re
from difflib import SequenceMatcher

def extract_calories(text):
    match = re.search(r"(\d+)\s*kcal", text)
    return int(match.group(1)) if match else None

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def evaluate(pred_file, ref_file):
    preds, refs = [], []
    with open(pred_file) as p, open(ref_file) as r:
        pred_lines = json.load(p)
        ref_lines = json.load(r)

    for p, r in zip(pred_lines, ref_lines):
        preds.append(extract_calories(p.get("model_response", "")))
        refs.append(extract_calories(r.get("output", "")))

    mae = sum(abs(p - r) for p, r in zip(preds, refs) if p and r) / len(preds)
    return mae

if __name__ == "__main__":
    mae = evaluate("logs/session_2025-10-29_normal_mode.json", "data/train.jsonl")
    print(f"📉 Mean Absolute Error (Calories): {mae:.2f}")
