import pandas as pd
import spacy
import re
import argparse
from pathlib import Path

# Load spaCy English model
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# Updated keyword library (case-insensitive removal done via lowercasing)
KEYWORD_LIBRARY = {
    "chatgpt",
    "gpt",
    "peer",
    "gemini"
}

# Bullet point pattern (your original pattern)
BULLET_PATTERN = re.compile(r"^\s*(\d+[\.\)]|\(\d+\))\s+")

def remove_keywords(text: str) -> str:
    """Remove keyword phrases from the text before normalization."""
    lowered_text = (text or "").lower()
    for keyword in KEYWORD_LIBRARY:
        lowered_text = lowered_text.replace(keyword, "")
    return lowered_text

def remove_bullet_symbols(text: str) -> str:
    """Remove bullet point symbols from the beginning of lines but keep the rest of the text."""
    lines = (text or "").splitlines()
    cleaned_lines = [BULLET_PATTERN.sub("", line.strip()) for line in lines]
    return " ".join(cleaned_lines)

# Pre-clean each row BEFORE grouping
def preprocess_text(text: str) -> str:
    text = remove_bullet_symbols(text)
    text = remove_keywords(text)
    return text

def normalize_text(text: str, already_precleaned: bool = False) -> str:
    """
    Normalize text:
      - If not pre-cleaned, remove bullet symbols + keywords first
      - lowercase, lemmatize, remove stopwords/punctuation
    """
    if not already_precleaned:
        text = preprocess_text(text)

    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return " ".join(tokens)

def extract_and_group_data(filepath: str) -> pd.DataFrame:
    """Load Excel file, pre-clean text per row, group by analysis_ID, concatenate, and aggregate labels."""
    df = pd.read_excel(filepath, sheet_name="prediction", engine="openpyxl")
    print("✅ Loaded Excel file.")
    print("📋 Column Names and Data Types:")
    print(df.dtypes)
    print("\n🔍 Preview of First Few Rows:")
    print(df.head())

    # Ensure required columns exist
    required_cols = {"analysis_ID", "original"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure types and pre-clean BEFORE grouping
    df["original"] = df["original"].astype(str).fillna("")
    df["preclean"] = df["original"].apply(preprocess_text)

    grouped_df = df.groupby("analysis_ID", as_index=False).agg({
        "original": lambda texts: " ".join(texts),   # keep raw concatenated
        "preclean": lambda texts: " ".join(texts),   # concatenated pre-cleaned (used for normalization)
        "comparison": "max",
        "group": "max",
        "Categorization-final": "max"
    })

    # Rename preclean to a clear name (not included in final output)
    grouped_df = grouped_df.rename(columns={"preclean": "original_preclean"})

    print("\n✅ Grouped Data by analysis_ID.")
    print("🔍 Preview of Grouped Data:")
    print(grouped_df.head())

    return grouped_df

def normalize_and_save(df: pd.DataFrame, output_path: str):
    """Normalize the pre-cleaned grouped text and save with only requested columns."""
    print("\n🔄 Starting normalization of text (using pre-cleaned grouped text)...")

    source_col = "original_preclean" if "original_preclean" in df.columns else "original"
    df["normalized"] = df[source_col].apply(
        lambda t: normalize_text(t, already_precleaned=(source_col == "original_preclean"))
    )

    print("✅ Normalization complete.")
    print("🔍 Preview of Normalized Text:")
    print(df[["analysis_ID", "original", "normalized"]].head())

    # Keep only requested columns in specified order
    out_cols = [
        "analysis_ID",        
        "comparison",
        "group",
        "Categorization-final",
        "original",
        "normalized"
    ]
    df_out = df[out_cols].copy()
    df_out.to_excel(output_path, index=False, engine="openpyxl")
    print(f"\n✅ Normalized data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Group+normalize the Excel input.")
    parser.add_argument("-i", "--input", required=False, default="RA_analysis.xlsx",
                        help="Path to input Excel file.")
    parser.add_argument("-o", "--output", required=False, default=None,
                        help="Path to output Excel file.")
    args = parser.parse_args()

    input_file = args.input
    if args.output is None:
        inp = Path(input_file)
        # e.g., RA_analysis.xlsx -> RA_analysis - Normalized.xlsx
        output_file = inp.with_name(f"{inp.stem} - Normalized.xlsx")
    else:
        output_file = args.output

    grouped_df = extract_and_group_data(input_file)
    normalize_and_save(grouped_df, str(output_file))

if __name__ == "__main__":
    main()
