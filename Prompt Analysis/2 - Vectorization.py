# 2_vectorize.py  (XLSX-only)
import os
import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384 dims for this model   

def default_output_for(input_path: Path) -> Path:
    # e.g., "RA_analysis - Normalized.xlsx" -> "RA_analysis - Normalized - Vectorized.csv"
    return input_path.with_name(f"{input_path.stem} - Vectorized.csv")

def save_csv(df: pd.DataFrame, out_path: Path):
    out_dir = out_path.parent
    os.makedirs(out_dir if str(out_dir) else ".", exist_ok=True)
    if out_path.suffix.lower() != ".csv":
        out_path = out_path.with_suffix(".csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path} with {df.shape[0]} rows and {df.shape[1]} columns.")

def main():
    parser = argparse.ArgumentParser(description="Vectorize normalized text from an XLSX file.")
    parser.add_argument("-i", "--input", default="RA_analysis - Normalized.xlsx",
                        help="Path to input XLSX containing a 'normalized' column.")
    parser.add_argument("-o", "--output", default=None,
                        help="Path for output CSV (default: '<input> - Vectorized.csv').")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="SentenceTransformer model name or local path.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Encoding batch size.")
    args = parser.parse_args()

    in_path = Path(args.input)
    if in_path.suffix.lower() != ".xlsx":
        raise ValueError(f"Input must be an .xlsx file, got: {in_path.suffix}")

    out_path = Path(args.output) if args.output else default_output_for(in_path)

    print(f"📥 Loading XLSX: {in_path}")
    df = pd.read_excel(in_path, engine="openpyxl")  # first sheet by default

    if "normalized" not in df.columns:
        raise ValueError("Column 'normalized' not found in input XLSX.")

    texts = df["normalized"].fillna("").astype(str).tolist()

    print(f"🧠 Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"⚙️ Encoding {len(texts)} texts (batch_size={args.batch_size})...")
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    emb_dim = emb.shape[1]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(emb, columns=emb_cols, index=df.index)

    out = pd.concat([df, emb_df], axis=1)
    save_csv(out, out_path)

if __name__ == "__main__":
    main()
