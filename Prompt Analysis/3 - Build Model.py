# rf_grid_search_cv5_with_thresholds.py
import argparse
import json
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn as sk
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier

# ========= CONFIG (defaults; can be overridden by CLI where provided) =========
DEFAULT_VEC_CSV = "RA_analysis - Vectorized.csv"
TARGETS = ["func_bizstr", "edu_stem", "entre_training_experience"]
RANDOM_STATE = 42
N_SPLITS = 5  # fixed

# Search spaces
MAX_DEPTH_RANGE = (3, 15, 3)
MIN_SAMPLES_LEAF_RANGE = (1, 7, 2)
N_ESTIMATORS_LIST = [200]
MAX_FEATURES_LIST = ["sqrt", "log2"]
MIN_SAMPLES_SPLIT_LIST = [2, 4]

# Threshold scan grid (per target)
THRESH_GRID = np.linspace(0.05, 0.95, 19)

# ========= HELPERS =========
def expand_range(rng):
    start, stop_incl, step = rng
    return list(range(start, stop_incl + 1, step))

def cv_macro_f1(X, y, params, n_splits=N_SPLITS):
    """5-fold CV macro-F1 (per-target + mean) using 0.5 threshold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    per_target_scores = [[] for _ in TARGETS]
    for tr, va in kf.split(X):
        base = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            min_samples_split=params["min_samples_split"],
            class_weight="balanced",
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        clf = MultiOutputClassifier(base)
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[va])  # 0.5 default
        for i in range(y.shape[1]):
            s = f1_score(y[va, i], y_pred[:, i], average="macro", zero_division=0)
            per_target_scores[i].append(s)

    per_target_mean = {TARGETS[i]: float(np.mean(sc)) for i, sc in enumerate(per_target_scores)}
    overall = float(np.mean(list(per_target_mean.values())))
    return per_target_mean, overall

def cv_collect_probs(X, y, params, n_splits=N_SPLITS):
    """Run CV with best params and return stacked val probs (per target) and y_true."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    probs_per_target = [[] for _ in TARGETS]
    y_true_all = []
    for tr, va in kf.split(X):
        base = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            min_samples_split=params["min_samples_split"],
            class_weight="balanced",
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        clf = MultiOutputClassifier(base)
        clf.fit(X[tr], y[tr])
        proba_list = clf.predict_proba(X[va])  # list of arrays, one per target
        for i in range(len(TARGETS)):
            p1 = proba_list[i][:, 1] if proba_list[i].shape[1] > 1 else np.zeros(len(va))
            probs_per_target[i].append(p1)
        y_true_all.append(y[va])
    probs_per_target = [np.concatenate(chunks, axis=0) for chunks in probs_per_target]
    y_true_all = np.concatenate(y_true_all, axis=0)
    return probs_per_target, y_true_all

def pick_best_thresholds(probs_per_target, y_true_all):
    """For each target, pick threshold maximizing macro-F1."""
    best_thresh = {}
    best_f1 = {}
    for i, t in enumerate(TARGETS):
        y_t = y_true_all[:, i]
        p = probs_per_target[i]
        best_score = -1.0
        best_th = 0.5
        for th in THRESH_GRID:
            y_hat = (p >= th).astype(int)
            sc = f1_score(y_t, y_hat, average="macro", zero_division=0)
            if sc > best_score:
                best_score, best_th = sc, th
        best_thresh[t] = float(best_th)
        best_f1[t] = float(best_score)
    overall = float(np.mean(list(best_f1.values())))
    return best_thresh, best_f1, overall

def main():
    parser = argparse.ArgumentParser(description="RF grid-search + CV5 + threshold tuning on vectorized CSV.")
    parser.add_argument("-i", "--input", default=DEFAULT_VEC_CSV,
                        help="Path to vectorized CSV produced by 2_vectorize.py")
    parser.add_argument("-o", "--outdir", default="models",
                        help="Directory to save model/thresholds/metadata (default: models)")
    args = parser.parse_args()

    vec_csv = Path(args.input)
    if vec_csv.suffix.lower() != ".csv":
        raise ValueError(f"Input must be a .csv file (got: {vec_csv.suffix})")
    if not vec_csv.exists():
        raise FileNotFoundError(f"Vectorized CSV not found: {vec_csv}")

    print(f"📥 Loading vectors: {vec_csv}")
    df = pd.read_csv(vec_csv)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No embedding columns found (expected columns starting with 'emb_').")
    missing_targets = [t for t in TARGETS if t not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    X = df[emb_cols].to_numpy(dtype=np.float32)
    y = df[TARGETS].astype(int).to_numpy()

    # ========= GRID SEARCH (CV=5, threshold fixed at 0.5) =========
    grid_max_depth = expand_range(MAX_DEPTH_RANGE)
    grid_min_leaf = expand_range(MIN_SAMPLES_LEAF_RANGE)

    rows = []
    for n_estimators, max_depth, min_leaf, max_feat, min_split in product(
        N_ESTIMATORS_LIST, grid_max_depth, grid_min_leaf, MAX_FEATURES_LIST, MIN_SAMPLES_SPLIT_LIST
    ):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_leaf,
            "max_features": max_feat,
            "min_samples_split": min_split,
        }
        per_target_mean, overall = cv_macro_f1(X, y, params, n_splits=N_SPLITS)
        row = dict(params)
        row["F1_macro_mean@0.5"] = overall
        for t in TARGETS:
            row[f"F1_{t}@0.5"] = per_target_mean[t]
        rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        by=["F1_macro_mean@0.5", "n_estimators"], ascending=[False, False]
    ).reset_index(drop=True)

    print("\n=== Grid Search (5-fold CV, threshold=0.5) — macro-F1 per target & overall ===")
    cols = ["n_estimators", "max_depth", "min_samples_leaf", "max_features", "min_samples_split",
            "F1_macro_mean@0.5"] + [f"F1_{t}@0.5" for t in TARGETS]
    print(results[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Select best row ---
    metric_col = "F1_macro_mean@0.5" if "F1_macro_mean@0.5" in results.columns else "F1_macro_mean"
    best_idx = results[metric_col].astype(float).idxmax()
    best = results.loc[best_idx]

    best_params = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_features": best["max_features"],
        "min_samples_split": int(best["min_samples_split"]),
    }

    # --- Threshold tuning with best hyperparams ---
    probs_per_target, y_true_all = cv_collect_probs(X, y, best_params, n_splits=N_SPLITS)
    best_thresh, best_f1, tuned_overall = pick_best_thresholds(probs_per_target, y_true_all)

    print("\nOptimal per-target thresholds:")
    for t in TARGETS:
        print(f"  {t}: threshold={best_thresh[t]:.2f}, macro-F1={best_f1[t]:.4f}")
    print(f"Overall mean macro-F1 (with tuned thresholds): {tuned_overall:.4f}")

    # --- Refit on ALL data and save ---
    outdir = Path(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    base_final = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
        min_samples_split=best_params["min_samples_split"],
        class_weight="balanced",
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    final_clf = MultiOutputClassifier(base_final)
    final_clf.fit(X, y)

    model_path = outdir / f"rf_multioutput_{stamp}.joblib"
    th_path    = outdir / f"rf_thresholds_{stamp}.json"
    meta_path  = outdir / f"rf_metadata_{stamp}.json"

    dump(final_clf, model_path)
    with open(th_path, "w", encoding="utf-8") as f:
        json.dump(best_thresh, f, indent=2)

    metadata = {
        "best_params": best_params,
        "targets": TARGETS,
        "n_splits": N_SPLITS,
        "cv_metric_col": metric_col,
        "cv_f1_macro_mean_at_0.5": float(best[metric_col]),
        "cv_f1_macro_mean_tuned": float(tuned_overall),
        "sklearn_version": sk.__version__,
        "timestamp": stamp,
        "vectorized_input": str(vec_csv),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved:\n  {model_path}\n  {th_path}\n  {meta_path}")

if __name__ == "__main__":
    main()