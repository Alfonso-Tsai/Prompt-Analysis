#!/usr/bin/env python3
# prediction.py
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

TARGETS = ["func_bizstr", "edu_stem", "entre_training_experience"]

def predict_with_thresholds(clf, thresholds, X, targets=TARGETS) -> pd.DataFrame:
    proba_list = clf.predict_proba(X)  # list of arrays, one per target
    out = {}
    for i, t in enumerate(targets):
        proba = proba_list[i]
        p1 = proba[:, 1] if proba.shape[1] > 1 else np.zeros(X.shape[0])
        thr = float(thresholds.get(t, 0.5))
        out[f"{t}_p1"] = p1
        out[f"{t}_pred"] = (p1 >= thr).astype(int)
    return pd.DataFrame(out)

STAMP_RE = re.compile(r"rf_multioutput_(\d{8}-\d{6})\.joblib$")

def find_latest_model(models_dir: Path) -> Path:
    cands = sorted(models_dir.glob("rf_multioutput_*.joblib"))
    if not cands:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    # sort by modified time (last is newest)
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]

def infer_thresholds_from_model(model_path: Path) -> Path:
    m = STAMP_RE.search(model_path.name)
    if m:
        stamp = m.group(1)
        th = model_path.with_name(f"rf_thresholds_{stamp}.json")
        if th.exists():
            return th
    # Fallback: pick latest thresholds in same dir
    cands = sorted(model_path.parent.glob("rf_thresholds_*.json"))
    if not cands:
        raise FileNotFoundError(f"No thresholds json found in {model_path.parent}")
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]

def default_output_for(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem} - Predictions.csv")

def main():
    parser = argparse.ArgumentParser(description="Predict with saved RF model and per-target thresholds.")
    parser.add_argument("-i", "--input", default="RA_analysis - Vectorized.csv",
                        help="Path to vectorized CSV (must contain emb_* columns).")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to .joblib model. If omitted, use latest in --models-dir.")
    parser.add_argument("-t", "--thresholds", default=None,
                        help="Path to thresholds .json. If omitted, infer from model or latest in --models-dir.")
    parser.add_argument("-o", "--output", default=None,
                        help="Path to output predictions CSV. Default: '<input> - Predictions.csv'.")
    parser.add_argument("--models-dir", default="models",
                        help="Directory to search for latest model/thresholds if not provided (default: models).")
    args = parser.parse_args()

    in_csv = Path(args.input)
    if in_csv.suffix.lower() != ".csv":
        raise ValueError(f"Input must be a .csv produced by vectorization. Got: {in_csv.suffix}")
    if not in_csv.exists():
        raise FileNotFoundError(f"Vectorized CSV not found: {in_csv}")

    models_dir = Path(args.models_dir)

    # Resolve model / thresholds
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_latest_model(models_dir)

    if args.thresholds:
        th_path = Path(args.thresholds)
    else:
        th_path = infer_thresholds_from_model(model_path)

    out_csv = Path(args.output) if args.output else default_output_for(in_csv)

    print(f"📦 Model:      {model_path}")
    print(f"🎚️ Thresholds: {th_path}")
    print(f"📥 Input CSV:  {in_csv}")
    print(f"📤 Output CSV: {out_csv}")

    # Load artifacts
    clf = load(model_path)
    with open(th_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    # Load embeddings
    df = pd.read_csv(in_csv)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No embedding columns found (expected columns starting with 'emb_').")
    X = df[emb_cols].to_numpy(dtype=np.float32)

    # Predict
    preds = predict_with_thresholds(clf, thresholds, X, TARGETS)

    # Keep analysis_ID if present
    keep_cols = []
    if "analysis_ID" in df.columns:
        keep_cols.append(df["analysis_ID"])
    out = pd.concat(keep_cols + [preds], axis=1)

    out.to_csv(out_csv, index=False)
    print("✅ Saved predictions.")

if __name__ == "__main__":
    main()
