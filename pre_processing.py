import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os

# Download NLTK tokenizer
nltk.download("punkt_tab", quiet=True)

# Get absolute path
base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")

# Load and check data
try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Print column names to debug
    print("Train columns:", train_df.columns.tolist())
    print("Test columns:", test_df.columns.tolist())
    
    # Get correct text column name
    text_col = next((col for col in train_df.columns if col.lower() == 'text'), None)
    if text_col is None:
        raise KeyError(f"Text column not found. Available columns: {train_df.columns.tolist()}")
    
    # Function definitions remain the same
    def clean_text(text):
        if pd.isnull(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"[^a-zA-Z0-9@#.,\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_text(text):
        return word_tokenize(text)

    # Process with correct column name
    print("🔹 Processing Training Data...")
    tqdm.pandas()
    train_df["clean_text"] = train_df[text_col].progress_apply(clean_text)
    train_df["tokens"] = train_df["clean_text"].progress_apply(tokenize_text)

    print("🔹 Processing Testing Data...")
    test_df["clean_text"] = test_df[text_col].progress_apply(clean_text)
    test_df["tokens"] = test_df["clean_text"].progress_apply(tokenize_text)

    # Save with correct paths
    output_train = os.path.join(base_dir, "train_processed.csv")
    output_test = os.path.join(base_dir, "test_processed.csv")
    
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)
    print(f"Processed files saved to:\n- {output_train}\n- {output_test}")

except FileNotFoundError as e:
    print(f"Error: CSV file not found\n{str(e)}")
except KeyError as e:
    print(f"Error: Column not found\n{str(e)}")
except Exception as e:
    print(f"Error: {str(e)}")
