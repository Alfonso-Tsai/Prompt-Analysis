import os
from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model
model_path = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_path)

# Define input and output folders
input_folder = "Tokenized Data"
output_folder = "Vectorized Data"

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

def vectorize_text(text: str) -> list:
    """Generate embedding vector from text."""
    return model.encode(text).tolist()

def save_vector_to_file(vector: list, output_path: str):
    """Save vector to a file as comma-separated values."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(",".join(map(str, vector)))
    print(f"✅ Embedding vector saved to: {output_path}")

if __name__ == "__main__":
    # Process all .txt files in the input folder
    files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(".txt") and os.path.isfile(os.path.join(input_folder, f))
    ]

    for file in files:
        filename = os.path.splitext(os.path.basename(file))[0]
        output_file = os.path.join(output_folder, f"{filename}_vectorized.txt")
        text = read_text_file(file)
        vector = vectorize_text(text)
        save_vector_to_file(vector, output_file)
