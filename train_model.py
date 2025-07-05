import os
import re
import pandas as pd
import numpy as np
import ast
from datetime import datetime
from gensim.models import Word2Vec
import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras.models import Sequential, save_model, load_model, Model  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, Dropout, BatchNormalization, Input, LayerNormalization  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.optimizers import Adam   # type: ignore
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
import gensim.downloader as api 
from keras.src.saving import register_keras_serializable  # type: ignore
from keras.src.layers import Layer  # type: ignore

# Constants
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300  
BATCH_SIZE = 16   # Reduced for better generalization
EPOCHS = 100      # Increased for better learning
LSTM_UNITS = 256  # Increased from 128
DROPOUT_RATE = 0.3  # Increased from 0.2
LEARNING_RATE = 0.0005  # Reduced for more stable training

def setup_model_paths(create_new_version=False):
    """Setup model paths with optional versioning"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    if create_new_version:
        suffix = f"_{timestamp}"
    else:
        suffix = ""
        
    return {
        "data": os.path.join(data_dir, "train_processed.csv"),
        "word2vec": os.path.join(models_dir, f"word2vec{suffix}.model"),
        "bilstm": os.path.join(models_dir, f"bilstm_model{suffix}.h5"),
        "label_encoder": os.path.join(models_dir, f"label_encoder{suffix}.pkl")
    }

def extract_labels_from_spans(spans):
    """Extract unique labels from span annotations"""
    try:
        spans = ast.literal_eval(spans) if isinstance(spans, str) else spans
        return set(span[2].strip().upper() for span in spans if len(span) >= 3)
    except:
        return set()

@register_keras_serializable(package="CustomCRF")
class CustomCRF(Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.transition_params = None
        self.supports_masking = True

    def build(self, input_shape):
        self.transition_params = self.add_weight(
            name='transition_params',
            shape=[self.num_classes, self.num_classes],
            initializer='glorot_uniform',
            trainable=True
        )
        self.built = True

    def call(self, inputs, training=None):
        return tf.nn.softmax(inputs, axis=-1)

    def compute_loss(self, y_true, y_pred):
        log_pred = tf.math.log(tf.clip_by_value(y_pred, 1e-10, 1.0))
        sequence_loss = -tf.reduce_sum(tf.multiply(y_true, log_pred), axis=-1)
        return tf.reduce_mean(tf.abs(sequence_loss))

    def compute_accuracy(self, y_true, y_pred):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        mask = tf.not_equal(y_true, 0)
        matches = tf.cast(tf.equal(y_true, y_pred), tf.float32) * tf.cast(mask, tf.float32)
        return tf.reduce_sum(matches) / (tf.reduce_sum(tf.cast(mask, tf.float32)) + 1e-10)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

def build_enhanced_bilstm_model(max_len, embedding_dim, n_classes):
    """Build enhanced BiLSTM model"""
    # Input layer
    inputs = tf.keras.Input(shape=(max_len, embedding_dim))
    
    # First BiLSTM layer with increased units
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Second BiLSTM layer
    x = Bidirectional(LSTM(LSTM_UNITS // 2, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Multi-head attention
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=8,
        key_dim=64,
        dropout=DROPOUT_RATE
    )(x, x)
    x = tf.keras.layers.Add()([x, attention])
    x = LayerNormalization()(x)
    
    # Dense layers
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Output layer
    outputs = TimeDistributed(Dense(n_classes, activation='softmax'))(x)
    
    model = Model(inputs, outputs)
    
    # Use custom optimizer configuration
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def process_data_batch(df, word2vec_model, label_encoder, batch_size=1000):
    """Process data in batches to avoid memory issues"""
    n_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    X_batches = []
    y_batches = []
    
    print(f"Processing {n_batches} batches...")
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        # Process X
        X_batch = []
        for tokens in batch_df["tokens"]:
            try:
                # Convert string tokens to list if needed
                token_list = ast.literal_eval(tokens) if isinstance(tokens, str) else tokens
                # Process only up to MAX_SEQUENCE_LENGTH tokens
                token_list = token_list[:MAX_SEQUENCE_LENGTH]
                
                # Get embeddings
                embeddings = []
                for token in token_list:
                    try:
                        embeddings.append(word2vec_model.wv[token.lower()])
                    except KeyError:
                        embeddings.append(np.zeros(EMBEDDING_DIM))
                
                # Pad sequences
                padding_length = MAX_SEQUENCE_LENGTH - len(embeddings)
                if padding_length > 0:
                    embeddings.extend([np.zeros(EMBEDDING_DIM)] * padding_length)
                
                X_batch.append(embeddings)
            except Exception as e:
                print(f"Error processing tokens: {str(e)}")
                # Add zero embeddings if processing fails
                X_batch.append([np.zeros(EMBEDDING_DIM)] * MAX_SEQUENCE_LENGTH)
        
        # Process y
        y_batch = []
        for spans in batch_df["True Predictions"]:
            try:
                # Initialize labels array
                token_labels = np.full(MAX_SEQUENCE_LENGTH, label_encoder.transform(['O'])[0])
                
                # Process spans
                span_list = ast.literal_eval(spans) if isinstance(spans, str) else spans
                for start, end, label in span_list:
                    if start < MAX_SEQUENCE_LENGTH:
                        end = min(end, MAX_SEQUENCE_LENGTH)
                        label_idx = label_encoder.transform([label.upper()])[0]
                        token_labels[start:end] = label_idx
                
                y_batch.append(token_labels)
            except Exception as e:
                print(f"Error processing spans: {str(e)}")
                # Add default 'O' labels if processing fails
                y_batch.append(np.full(MAX_SEQUENCE_LENGTH, label_encoder.transform(['O'])[0]))
        
        # Convert batches to arrays with proper shapes
        try:
            X_batch_array = np.array(X_batch, dtype=np.float32)
            y_batch_array = np.array(y_batch, dtype=np.int32)
            y_batch_categorical = tf.keras.utils.to_categorical(
                y_batch_array, 
                num_classes=len(label_encoder.classes_)
            )
            
            X_batches.append(X_batch_array)
            y_batches.append(y_batch_categorical)
            
            print(f"Processed batch {i+1}/{n_batches}")
            print(f"X shape: {X_batch_array.shape}, y shape: {y_batch_categorical.shape}")
            
        except Exception as e:
            print(f"Error converting batch to array: {str(e)}")
            continue
    
    # Stack all batches
    try:
        X_final = np.vstack(X_batches)
        y_final = np.vstack(y_batches)
        print(f"\nFinal shapes - X: {X_final.shape}, y: {y_final.shape}")
        return X_final, y_final
        
    except Exception as e:
        raise ValueError(f"Error stacking batches: {str(e)}")

def compute_class_weights(y_data):
    """Compute balanced class weights for sequence labeling"""
    try:
        # Flatten the one-hot encoded labels into a 1-D array of class indices
        y_reshaped = y_data.reshape(-1, y_data.shape[-1])
        y_classes = np.argmax(y_reshaped, axis=1)

        # Get the unique classes present
        unique_classes = np.unique(y_classes)

        # Compute weights using sklearn's compute_class_weight
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_classes
        )

        # Build a dictionary; use .item() so each weight is a plain Python float
        weights_dict = {}
        for idx, class_val in enumerate(unique_classes):
            weights_dict[int(class_val)] = float(weights[idx].item())
        
        print(f"Computed class weights: {weights_dict}")
        return weights_dict
        
    except Exception as e:
        print(f"Error computing class weights: {str(e)}")
        n_classes = y_data.shape[-1]
        return {i: 1.0 for i in range(n_classes)}

def enhance_text_preprocessing(text: str) -> str:
    """Enhanced text preprocessing"""
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Standardize phone numbers
    text = re.sub(r'\+?(\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                  lambda m: re.sub(r'[^\d]', '', m.group(0)), text)
    
    # Standardize SSN format
    text = re.sub(r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
                  lambda m: '-'.join([m.group(0)[:3], m.group(0)[3:5], m.group(0)[5:]]), text)
    
    return text

def compute_sample_weights(y_data, class_weights):
    """
    Given y_data (one-hot with shape (num_samples, timesteps, num_classes))
    and a dictionary mapping class indices to weights,
    compute a weight for each token.
    Returns a 2D array of shape (num_samples, timesteps).
    """
    num_samples, timesteps, num_classes = y_data.shape
    sample_weights = np.zeros((num_samples, timesteps), dtype=np.float32)
    for i in range(num_samples):
        for j in range(timesteps):
            # Get actual label index for this token
            label_idx = int(np.argmax(y_data[i, j]))
            sample_weights[i, j] = class_weights.get(label_idx, 1.0)
    return sample_weights

def train_model(create_new_version=False):
    """Main training function"""
    try:
        # Setup paths first
        MODEL_PATHS = setup_model_paths(create_new_version)
        
        # Load original training data if augmented doesn't exist
        data_dir = os.path.join(os.path.dirname(MODEL_PATHS["data"]), "train_processed.csv")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Training data file not found at: {data_dir}")
        print(f"Loading training data from: {data_dir}")
        
        # Load training data first
        train_df = pd.read_csv(data_dir)
        print(f"Loaded {len(train_df)} training samples")
        
        # Modified: Check for either 'Text' or 'text' column
        text_column = 'Text' if 'Text' in train_df.columns else 'text'
        if text_column not in train_df.columns:
            raise ValueError(f"Neither 'Text' nor 'text' column found in the CSV file")
        
        # Adjust required columns based on actual column name
        required_columns = [text_column, 'tokens', 'True Predictions']
        missing_columns = [col for col in required_columns if col not in train_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Enhanced preprocessing - use the correct column name
        print("Applying text preprocessing...")
        train_df[text_column] = train_df[text_column].apply(enhance_text_preprocessing)
        
        # Train Word2Vec if not exists
        print("\nTraining/Loading Word2Vec model...")
        try:
            word2vec_model = Word2Vec.load(MODEL_PATHS["word2vec"])
        except:
            # Train new model if not exists
            print("Training new Word2Vec model...")
            texts = train_df['tokens'].apply(ast.literal_eval).tolist()
            word2vec_model = Word2Vec(
                sentences=texts,
                vector_size=EMBEDDING_DIM,
                window=5,
                min_count=1,
                workers=4
            )
            word2vec_model.save(MODEL_PATHS["word2vec"])
            print("Word2Vec model trained and saved")
        
        # Prepare label encoder
        print("\nPreparing label encoder...")
        try:
            label_encoder = joblib.load(MODEL_PATHS["label_encoder"])
            print("Loaded existing label encoder")
        except:
            print("Creating new label encoder...")
            all_labels = set()
            for spans in train_df['True Predictions']:
                all_labels.update(extract_labels_from_spans(spans))
            all_labels.add('O')  # Add outside label
            
            label_encoder = LabelEncoder()
            label_encoder.fit(sorted(list(all_labels)))
            joblib.dump(label_encoder, MODEL_PATHS["label_encoder"])
            print(f"Created label encoder with labels: {label_encoder.classes_}")
        
        # Process data in batches
        print("\nProcessing training data...")
        X, y = process_data_batch(train_df, word2vec_model, label_encoder)
        
        # Build and train model
        print("\nBuilding and training model...")
        model = build_enhanced_bilstm_model(
            MAX_SEQUENCE_LENGTH, 
            EMBEDDING_DIM, 
            len(label_encoder.classes_)
        )
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                MODEL_PATHS["bilstm"],
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=1
            )
        ]
        
        # Compute class weights
        print("\nComputing class weights...")
        class_weights = compute_class_weights(y)
        
        # Instead of passing class_weight (which is not supported for 3D outputs),
        # we compute sample weights per token.
        sample_weights = compute_sample_weights(y, class_weights)

        # Train model using sample_weight
        print("\nStarting model training...")
        history = model.fit(
            X, y,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.2,
            sample_weight=sample_weights,  # Use the computed sample weights
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model safely
        print("\nSaving trained model...")
        save_model_safely(model, MODEL_PATHS["bilstm"])
        
        return history
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def save_model_safely(model, path):
    """Safe model saving with error handling"""
    try:
        # Save the model in TF SavedModel format
        model.save(path.replace('.h5', ''), save_format='tf')
        print(f"Model saved successfully to {path.replace('.h5', '')}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        try:
            # Fallback to weights-only saving
            model.save_weights(path.replace('.h5', '_weights'))
            print(f"Saved model weights to {path.replace('.h5', '_weights')}")
        except Exception as e:
            print(f"Error saving weights: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train PII detection models')
    parser.add_argument('--new-version', action='store_true', help='Create new versioned models')
    args = parser.parse_args()
    
    train_model(create_new_version=args.new_version)
