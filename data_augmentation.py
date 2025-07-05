import pandas as pd
import numpy as np
import random
from typing import List, Dict
import spacy
import os

# Load spaCy model for better name/address detection
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def augment_names(original_text: str) -> List[str]:
    """Generate variations of name formats"""
    common_titles = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.']
    augmented = []
    
    # Remove titles if present
    clean_text = ' '.join(word for word in original_text.split() 
                         if word.rstrip('.') not in [t.rstrip('.') for t in common_titles])
    
    # Add different title variations
    for title in common_titles:
        augmented.append(f"{title} {clean_text}")
    
    # Add name variations
    name_parts = clean_text.split()
    if len(name_parts) >= 2:
        # First name + Last name initial
        augmented.append(f"{name_parts[0]} {name_parts[-1][0]}.")
        # Full name with middle initial
        if len(name_parts) >= 3:
            augmented.append(f"{name_parts[0]} {name_parts[1][0]}. {name_parts[-1]}")
    
    return augmented

def augment_addresses(original_text: str) -> List[str]:
    """Generate variations of address formats"""
    augmented = []
    
    # Parse address components
    doc = nlp(original_text)
    address_parts = original_text.split()
    
    # Common address abbreviations
    street_variants = {
        'Street': ['St.', 'Street'],
        'Avenue': ['Ave.', 'Avenue'],
        'Road': ['Rd.', 'Road'],
        'Drive': ['Dr.', 'Drive'],
        'Boulevard': ['Blvd.', 'Boulevard']
    }
    
    # Create variations
    for word in address_parts:
        for full, variants in street_variants.items():
            if word.lower() == full.lower():
                for variant in variants:
                    new_address = original_text.replace(word, variant)
                    augmented.append(new_address)
    
    return augmented

def augment_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Augment training data with variations"""
    augmented_data = []
    
    # First, check and print column names
    print("Available columns:", df.columns.tolist())
    
    # Get the correct text column name
    text_col = 'Text' if 'Text' in df.columns else 'text'
    if text_col not in df.columns:
        raise ValueError(f"Text column not found. Available columns: {df.columns.tolist()}")
    
    for _, row in df.iterrows():
        # Add original row
        augmented_data.append(row)
        
        text = row[text_col]
        spans = eval(row['True Predictions'])
        
        # Process each span
        for start, end, label in spans:
            original_entity = text[start:end]
            
            if label == 'NAME':
                variations = augment_names(original_entity)
            elif label == 'ADDRESS':
                variations = augment_addresses(original_entity)
            else:
                continue
                
            # Create augmented samples
            for variation in variations:
                new_text = text[:start] + variation + text[end:]
                new_row = row.copy()
                new_row[text_col] = new_text
                
                # Adjust span positions for new text
                new_spans = []
                curr_pos = 0
                for s, e, l in spans:
                    if s == start and l == label:
                        new_spans.append((curr_pos, curr_pos + len(variation), l))
                        curr_pos += len(variation)
                    else:
                        new_spans.append((curr_pos + s, curr_pos + e, l))
                new_row['True Predictions'] = str(new_spans)
                augmented_data.append(new_row)
    
    return pd.DataFrame(augmented_data)

if __name__ == "__main__":
    try:
        # Get file paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(current_dir, "train_processed.csv")
        output_path = os.path.join(current_dir, "train_processed_augmented.csv")
        
        print(f"Loading training data from: {train_path}")
        train_df = pd.read_csv(train_path)
        print(f"Loaded {len(train_df)} samples")
        
        print("\nAugmenting dataset...")
        augmented_df = augment_dataset(train_df)
        print(f"Created {len(augmented_df)} samples after augmentation")
        
        print(f"\nSaving augmented data to: {output_path}")
        augmented_df.to_csv(output_path, index=False)
        print("Done!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise