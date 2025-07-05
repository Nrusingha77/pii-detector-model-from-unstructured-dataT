import numpy as np
import pandas as pd
import tensorflow as tf # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
from gensim.models import Word2Vec
import os
import ast
from tqdm import tqdm
from train_model import CustomCRF

# Constants should match train_model.py
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300  # Changed to match train_model.py

def setup_paths():
    """Setup evaluation paths"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    return {
        "model_keras": os.path.join(models_dir, "bilstm_model.keras"),  # Updated extension
        "model_h5": os.path.join(models_dir, "bilstm_model.h5"),
        "label_encoder": os.path.join(models_dir, "label_encoder.pkl"),
        "word2vec": os.path.join(models_dir, "word2vec.model"),
        "test_data": os.path.join(base_dir, "data", "test_processed.csv")
    }

def text_to_embedding(tokens, word2vec_model):
    """Convert tokens to embeddings with consistent shape"""
    embeddings = []
    tokens = tokens[:MAX_SEQUENCE_LENGTH]
    
    for token in tokens:
        try:
            embeddings.append(word2vec_model.wv[token.lower()])
        except KeyError:
            embeddings.append(np.zeros(EMBEDDING_DIM))
    
    # Pad if needed
    while len(embeddings) < MAX_SEQUENCE_LENGTH:
        embeddings.append(np.zeros(EMBEDDING_DIM))
    
    return np.array(embeddings).reshape(1, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

def parse_spans(pred_str):
    """Parse span predictions safely"""
    try:
        if isinstance(pred_str, str):
            spans = ast.literal_eval(pred_str)
            return [(int(start), int(end), label.strip().upper()) 
                   for start, end, label in spans]
    except:
        pass
    return []

def spans_to_token_labels(tokens, spans):
    """Convert spans to token labels"""
    labels = ['O'] * len(tokens)
    
    for start, end, label in spans:
        for i in range(start, min(end, len(tokens))):
            labels[i] = label
    
    return labels

def evaluate_model():
    """Evaluate PII detection model"""
    try:
        MODEL_PATHS = setup_paths()
        
        # Load models with updated custom objects
        print("Loading models...")
        custom_objects = {
            'CustomCRF': CustomCRF,
            'compute_loss': CustomCRF.compute_loss,
            'compute_accuracy': CustomCRF.compute_accuracy
        }
        
        # Try loading in different formats
        try:
            print("\nAttempting to load Keras format...")
            with tf.keras.utils.custom_object_scope(custom_objects):
                bilstm_model = tf.keras.models.load_model(MODEL_PATHS["model_keras"])
        except Exception as e1:
            print(f"Keras format loading failed: {str(e1)}")
            print("\nAttempting to load H5 format...")
            try:
                with tf.keras.utils.custom_object_scope(custom_objects):
                    bilstm_model = tf.keras.models.load_model(MODEL_PATHS["model_h5"])
            except Exception as e2:
                print(f"H5 format loading failed: {str(e2)}")
                raise Exception("Failed to load model in any format")

        # Load other required models
        word2vec_model = Word2Vec.load(MODEL_PATHS["word2vec"])
        label_encoder = joblib.load(MODEL_PATHS["label_encoder"])
        print("Models loaded successfully")
        print(f"Available labels: {label_encoder.classes_}")
        
        # Load test data
        test_df = pd.read_csv(MODEL_PATHS["test_data"])
        print(f"\n Loaded {len(test_df)} test samples")
        
        # Process predictions
        all_predictions = []
        all_true_labels = []
        
        print("\nProcessing samples...")
        for idx, row in tqdm(enumerate(test_df.iterrows(), 1), total=len(test_df)):
            try:
                # Get tokens and true labels
                tokens = ast.literal_eval(row[1]['tokens'])
                true_spans = parse_spans(row[1]['True Predictions'])
                
                # Ensure consistent length by using same MAX_SEQUENCE_LENGTH
                tokens = tokens[:MAX_SEQUENCE_LENGTH]
                true_labels = spans_to_token_labels(tokens, true_spans)
                
                # Get predictions for the same length
                input_data = text_to_embedding(tokens, word2vec_model)
                pred_probs = bilstm_model.predict(input_data, verbose=0)[0]
                pred_labels = [label_encoder.classes_[np.argmax(p)] 
                             for p in pred_probs[:len(tokens)]]
                
                # Ensure same length for both lists
                min_len = min(len(true_labels), len(pred_labels))
                all_true_labels.extend(true_labels[:min_len])
                all_predictions.extend(pred_labels[:min_len])
                
            except Exception as e:
                print(f"\n Error processing sample {idx}: {str(e)}")
                continue
        
        # Verify lengths match
        assert len(all_true_labels) == len(all_predictions), "Mismatch in label lengths"
        
        # Calculate metrics
        print("\n==== PII Detection Model Evaluation ====")
        labels = sorted(list(set(all_true_labels + all_predictions)))
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(all_true_labels, all_predictions, labels=labels)
        print("\n Confusion Matrix:")
        print(pd.DataFrame(
            conf_matrix,
            index=labels,
            columns=labels
        ))
        
        # Classification Report
        print("\n Performance Metrics per Label:")
        print(classification_report(
            all_true_labels,
            all_predictions,
            labels=labels,
            zero_division=0
        ))
        
    except Exception as e:
        print(f"\n Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass
    
    evaluate_model()