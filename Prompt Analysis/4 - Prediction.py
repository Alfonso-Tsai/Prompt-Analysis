#!/usr/bin/env python3
# prediction.py - Revised with per-sheet renames, dual summaries, and Excel ergonomics
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

TARGETS = ["func_bizstr", "edu_stem", "entre_training_experience"]

def predict_with_thresholds(clf, thresholds, X, targets=TARGETS) -> pd.DataFrame:
    """Generate predictions using optimized thresholds."""
    proba_list = clf.predict_proba(X)  # list of arrays, one per target
    out = {}
    for i, t in enumerate(targets):
        proba = proba_list[i]
        p1 = proba[:, 1] if proba.shape[1] > 1 else np.zeros(X.shape[0])
        thr = float(thresholds.get(t, 0.5))
        out[f"{t}_p1"] = p1
        out[f"{t}_pred"] = (p1 >= thr).astype(int)
    return pd.DataFrame(out)

def predict_with_standard_threshold(clf, X, threshold=0.5, targets=TARGETS) -> pd.DataFrame:
    """Generate predictions using standard 0.5 threshold for comparison."""
    proba_list = clf.predict_proba(X)  # list of arrays, one per target
    out = {}
    for i, t in enumerate(targets):
        proba = proba_list[i]
        p1 = proba[:, 1] if proba.shape[1] > 1 else np.zeros(X.shape[0])
        out[f"{t}_p1"] = p1
        out[f"{t}_pred"] = (p1 >= threshold).astype(int)
    return pd.DataFrame(out)

def generate_summary_stats(df, targets=TARGETS) -> pd.DataFrame:
    """Generate summary statistics for all predictions (no F1)."""
    summary_data = []

    for t in targets:
        p1_col = f"{t}_p1"
        pred_col = f"{t}_pred"

        if p1_col in df.columns and pred_col in df.columns:
            stats = {
                "Target": t,
                "Total_Predictions": int(len(df)),
                "Positive_Predictions": int(df[pred_col].sum()),
                "Negative_Predictions": int(len(df) - df[pred_col].sum()),
                "Positive_Rate": float(df[pred_col].mean()) if len(df) else 0.0,
                "Avg_Confidence": float(df[p1_col].mean()) if len(df) else 0.0,
                "Min_Confidence": float(df[p1_col].min()) if len(df) else 0.0,
                "Max_Confidence": float(df[p1_col].max()) if len(df) else 0.0,
                "High_Confidence_Count": int((df[p1_col] >= 0.7).sum()),
                "Moderate_Confidence_Count": int(((df[p1_col] >= 0.3) & (df[p1_col] < 0.7)).sum()),
                "Low_Confidence_Count": int((df[p1_col] < 0.3).sum()),
            }
            summary_data.append(stats)

    if len(summary_data) == 3:
        overview_stats = {
            "Target": "OVERALL",
            "Total_Predictions": summary_data[0]["Total_Predictions"],
            "Positive_Predictions": sum(s["Positive_Predictions"] for s in summary_data),
            "Negative_Predictions": sum(s["Negative_Predictions"] for s in summary_data),
            "Positive_Rate": float(np.mean([s["Positive_Rate"] for s in summary_data])),
            "Avg_Confidence": float(np.mean([s["Avg_Confidence"] for s in summary_data])),
            "Min_Confidence": float(min(s["Min_Confidence"] for s in summary_data)),
            "Max_Confidence": float(max(s["Max_Confidence"] for s in summary_data)),
            "High_Confidence_Count": int(sum(s["High_Confidence_Count"] for s in summary_data)),
            "Moderate_Confidence_Count": int(sum(s["Moderate_Confidence_Count"] for s in summary_data)),
            "Low_Confidence_Count": int(sum(s["Low_Confidence_Count"] for s in summary_data)),
        }
        summary_data.append(overview_stats)

    return pd.DataFrame(summary_data)

STAMP_RE = re.compile(r"rf_multioutput_(\d{8}-\d{6})\.joblib$")

def find_latest_model(models_dir: Path) -> Path:
    cands = sorted(models_dir.glob("rf_multioutput_*.joblib"))
    if not cands:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]

def infer_thresholds_from_model(model_path: Path) -> Path:
    m = STAMP_RE.search(model_path.name)
    if m:
        stamp = m.group(1)
        th = model_path.with_name(f"rf_thresholds_{stamp}.json")
        if th.exists():
            return th
    cands = sorted(model_path.parent.glob("rf_thresholds_*.json"))
    if not cands:
        raise FileNotFoundError(f"No thresholds json found in {model_path.parent}")
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]

def default_output_for(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem} - Predictions.xlsx")

def main():
    parser = argparse.ArgumentParser(description="Predict with saved RF model and per-target thresholds.")
    parser.add_argument("-i", "--input", default="RA_analysis - Vectorized.csv",
                        help="Path to vectorized CSV (must contain emb_* columns).")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to .joblib model. If omitted, use latest in --models-dir.")
    parser.add_argument("-t", "--thresholds", default=None,
                        help="Path to thresholds .json. If omitted, infer from model or latest in --models-dir.")
    parser.add_argument("-o", "--output", default=None,
                        help="Path to output predictions XLSX. Default: '<input> - Predictions.xlsx'.")
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
    model_path = Path(args.model) if args.model else find_latest_model(models_dir)
    th_path = Path(args.thresholds) if args.thresholds else infer_thresholds_from_model(model_path)
    out_xlsx = Path(args.output) if args.output else default_output_for(in_csv)

    print(f"📦 Model:      {model_path}")
    print(f"🎚️ Thresholds: {th_path}")
    print(f"📥 Input CSV:  {in_csv}")
    print(f"📤 Output XLSX: {out_xlsx}")

    # Load artifacts
    clf = load(model_path)
    with open(th_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    # Load embeddings and original data
    df = pd.read_csv(in_csv)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No embedding columns found (expected columns starting with 'emb_').")
    X = df[emb_cols].to_numpy(dtype=np.float32)

    # Predictions
    threshold_preds = predict_with_thresholds(clf, thresholds, X, TARGETS)
    standard_preds = predict_with_standard_threshold(clf, X, 0.5, TARGETS)

    # Carry ID column if present
    base_cols = ["analysis_ID"] if "analysis_ID" in df.columns else []
    threshold_df = pd.concat([df[base_cols], threshold_preds], axis=1) if base_cols else threshold_preds
    standard_df  = pd.concat([df[base_cols], standard_preds], axis=1) if base_cols else standard_preds

    # Summaries
    indiv_thres_summary_df = generate_summary_stats(threshold_df, TARGETS)
    standard_summary_df    = generate_summary_stats(standard_df, TARGETS)

    # Threshold metadata
    threshold_info_df = pd.DataFrame(
        [{"Target": t, "Optimized_Threshold": thresholds.get(t, 0.5), "Standard_Threshold": 0.5}
         for t in TARGETS]
    )

    # Save to Excel with ergonomics
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        threshold_df.to_excel(writer, sheet_name="Individual_Thresholds", index=False)
        standard_df.to_excel(writer, sheet_name="Standard_0.5", index=False)
        indiv_thres_summary_df.to_excel(writer, sheet_name="Indiv_Thres_Summary", index=False)
        standard_summary_df.to_excel(writer, sheet_name="Standard_Summary", index=False)
        threshold_info_df.to_excel(writer, sheet_name="Threshold_Info", index=False)

        wb = writer.book
        for name in ["Individual_Thresholds", "Standard_0.5", "Indiv_Thres_Summary", "Standard_Summary", "Threshold_Info"]:
            ws = wb[name]
            ws.freeze_panes = "A2"  # freeze header row
            # Set number format for probability columns *_p1
            headers = {cell.column: (cell.value or "") for cell in ws[1]}
            for col_cells in ws.iter_cols(min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
                col_idx = col_cells[0].column
                header = str(headers.get(col_idx, ""))
                if header.endswith("_p1"):
                    for cell in col_cells:
                        cell.number_format = "0.000"

    print("✅ Saved predictions to XLSX with multiple tabs:")
    print("   📊 'Individual_Thresholds' - Main predictions with optimized thresholds")
    print("   📊 'Standard_0.5' - Predictions with standard 0.5 threshold")
    print("   📊 'Indiv_Thres_Summary' - Statistics from optimized thresholds")
    print("   📊 'Standard_Summary' - Statistics from standard 0.5 threshold")
    print("   📊 'Threshold_Info' - Threshold values used")

if __name__ == "__main__":
    main()
