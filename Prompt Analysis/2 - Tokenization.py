import spacy
import os

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Specify the folder paths
folder_path = "Normalized Data"
output_folder = "Tokenized Data"

# Ensure the output folder exists and clean it
os.makedirs(output_folder, exist_ok=True)
for f in os.listdir(output_folder):
    file_path = os.path.join(output_folder, f)
    if os.path.isfile(file_path):
        os.remove(file_path)

def read_text_file(filepath: str) -> str:
    """Read text from a .txt file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def tokenize_text(text: str) -> list:
    """Tokenize text using spaCy and return a list of tokens."""
    doc = nlp(text)
    return [token.text for token in doc if not token.is_space]

def save_tokens_to_file(tokens: list, output_path: str):
    """Save tokens to a file, one token per line."""
    with open(output_path, "w", encoding="utf-8") as f:
        for token in tokens:
            f.write(token + "\n")
    print(f"✅ Tokenized output saved to: {output_path}")

if __name__ == "__main__":
    # Get a list of all .txt files in the folder
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".txt") and os.path.isfile(os.path.join(folder_path, f))
    ]

    for file in files:
        filename = os.path.splitext(os.path.basename(file))[0]
        output_file = os.path.join(output_folder, f"{filename}_tokenized.txt")
        text = read_text_file(file)
        tokens = tokenize_text(text)
        save_tokens_to_file(tokens, output_file)
