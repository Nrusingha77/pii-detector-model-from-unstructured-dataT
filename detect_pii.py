import os
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from gensim.models import Word2Vec
import joblib
import numpy as np
from train_model import CustomCRF
from typing import List
# Import any additional modules if needed

router = APIRouter()

# Setup absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "bilstm_model.keras")
WORD2VEC_PATH = os.path.join(MODELS_DIR, "word2vec.model")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

def verify_model_files():
    required_files = [
        (MODEL_PATH, "BiLSTM model"),
        (WORD2VEC_PATH, "Word2Vec model"),
        (LABEL_ENCODER_PATH, "Label encoder")
    ]
    missing_files = []
    for file_path, file_desc in required_files:
        if not os.path.exists(file_path):
            missing_files.append(f"{file_desc} at {file_path}")
    if missing_files:
        raise FileNotFoundError("Missing required model files:\n" + "\n".join(missing_files))

class TextRequest(BaseModel):
    text: str

class Entity(BaseModel):
    text: str
    label: str
    confidence: float
    start: int  # Add these fields
    end: int    # Add these fields

class PIIResponse(BaseModel):
    entities: List[Entity]

# Updated patterns – note that we still use regex only for SSN and EMAIL.
PATTERNS = {
    'SSN': r'\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b',
    'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'PHONE': None,
    'ADDRESS': None,
    'COMPANY': None,
    'NAME': None
}

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300

# Confidence thresholds (if needed in future improvements)
CONFIDENCE_THRESHOLDS = {
    'SSN': 0.95,
    'EMAIL': 0.90,
    'PHONE': 0.85,
    'ADDRESS': 0.70,
    'COMPANY': 0.75,
    'NAME': 0.70,
    'O': 0.60
}

def load_models():
    """Load all required models with detailed error checking"""
    try:
        # Verify files exist first
        verify_model_files()
        print("Model files verified successfully")
        
        # Load BiLSTM model
        print("Loading BiLSTM model...")
        custom_objects = {
            'CustomCRF': CustomCRF,
            'compute_loss': CustomCRF.compute_loss,
            'compute_accuracy': CustomCRF.compute_accuracy
        }
        
        try:
            with tf.keras.utils.custom_object_scope(custom_objects):
                # Try .keras format first
                model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e1:
            print(f"Error loading .keras model: {str(e1)}")
            # Try H5 format as fallback
            h5_path = MODEL_PATH.replace('.keras', '.h5')
            try:
                with tf.keras.utils.custom_object_scope(custom_objects):
                    model = tf.keras.models.load_model(h5_path)
                print("Successfully loaded H5 model as fallback")
            except Exception as e2:
                raise Exception(f"Failed to load model in any format: {str(e2)}")
        
        # Load Word2Vec
        print("Loading Word2Vec model...")
        word2vec = Word2Vec.load(WORD2VEC_PATH)
        
        # Load Label Encoder
        print("Loading Label Encoder...")
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        
        print("All models loaded successfully!")
        return model, word2vec, label_encoder
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Model files not found: {str(e)}\nSearched in: {MODELS_DIR}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Model loading failed: {str(e)}"
        )

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text in a way similar to your training pre-processing.
    Here we use a regex-based tokenizer that captures words and punctuation.
    """
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens

def pattern_based_detection(text: str) -> List[Entity]:
    entities = []
    for label, pattern in PATTERNS.items():
        if pattern:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(),
                    label=label,
                    confidence=0.99,  # High confidence for regex matches
                    start=match.start(),
                    end=match.end()
                ))
    return entities

def text_to_embedding(tokens: List[str], word2vec_model: Word2Vec) -> np.ndarray:
    tokens = tokens[:MAX_SEQUENCE_LENGTH] if len(tokens) > MAX_SEQUENCE_LENGTH else tokens
    embeddings = []
    for token in tokens:
        try:
            embeddings.append(word2vec_model.wv[token.lower()])
        except KeyError:
            embeddings.append(np.zeros(EMBEDDING_DIM))
    padding_length = MAX_SEQUENCE_LENGTH - len(embeddings)
    if padding_length > 0:
        embeddings.extend([np.zeros(EMBEDDING_DIM)] * padding_length)
    return np.array(embeddings).reshape(1, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

def validate_entity_text(text: str, label: str) -> bool:
    if label == 'NAME':
        words = text.split()
        return len(words) >= 1 and all(word.replace('.', '').replace('-', '').isalpha() for word in words)
    elif label == 'ADDRESS':
        words = text.split()
        has_number = any(word.isdigit() for word in words)
        street_types = {'street', 'st', 'avenue', 'ave', 'road', 'rd', 'drive', 'dr', 'lane', 'ln'}
        has_street = any(word.lower() in street_types for word in words)
        return has_number and has_street and len(words) >= 3
    elif label == 'COMPANY':
        return len(text) >= 2 and text.lower() not in {'at', 'by', 'in', 'from', 'to'}
    elif label == 'PHONE':
        digits = ''.join(filter(str.isdigit, text))
        return len(digits) >= 10
    return True

@router.post("/detect", response_model=PIIResponse)
async def detect_pii(request: TextRequest):
    try:
        model, word2vec, label_encoder = load_models()
        text = request.text.strip()
        
        # Detect PII
        entities = []
        
        # First do pattern-based detection
        pattern_entities = pattern_based_detection(text)
        for entity in pattern_entities:
            # Find position in text
            start = text.find(entity.text)
            if start != -1:
                entities.append(Entity(
                    text=entity.text,
                    label=entity.label,
                    confidence=entity.confidence,
                    start=start,
                    end=start + len(entity.text)
                ))
        
        # Then do model-based detection
        tokens = tokenize_text(text)
        input_data = text_to_embedding(tokens, word2vec)
        predictions = model.predict(input_data, verbose=0)
        
        # Get model predictions with positions
        current_pos = 0
        for token, pred in zip(tokens[:MAX_SEQUENCE_LENGTH], predictions[0]):
            label_idx = np.argmax(pred)
            label = label_encoder.inverse_transform([label_idx])[0]
            if label != 'O':
                entities.append(Entity(
                    text=token,
                    label=label,
                    confidence=float(pred[label_idx]),
                    start=current_pos,
                    end=current_pos + len(token)
                ))
            current_pos += len(token) + 1  # +1 for space
            
        return PIIResponse(entities=entities)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@router.get("/health")
async def health_check():
    try:
        _, _, _ = load_models()
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
