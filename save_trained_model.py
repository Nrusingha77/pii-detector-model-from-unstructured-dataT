import os
import tensorflow as tf
from keras import Model
from keras.models import load_model # type: ignore
from train_model import CustomCRF

def setup_paths():
    """Setup paths for model loading and saving"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    return {
        "model_h5": os.path.join(models_dir, "bilstm_model.h5"),
        "model_keras": os.path.join(models_dir, "bilstm_model.keras"),
        "weights": os.path.join(models_dir, "bilstm_model.weights.h5")
    }

def save_model():
    """Load and save the model in Keras format"""
    try:
        PATHS = setup_paths()
        print("\nAttempting to load model...")
        
        # Define custom objects
        custom_objects = {
            'CustomCRF': CustomCRF,
            'compute_loss': CustomCRF.compute_loss,
            'compute_accuracy': CustomCRF.compute_accuracy
        }
        
        # Load model - use H5 format since it was successful
        print("\nLoading model from H5 format...")
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = load_model(PATHS["model_h5"])
        print("Model loaded successfully")
        
        # Save in Keras format (recommended format)
        try:
            print("\nSaving in native Keras format...")
            # Use .keras extension without explicit save_format
            model.save(PATHS["model_keras"])
            print(f"Model saved successfully to: {PATHS['model_keras']}")
            
            # Save weights separately as backup
            print("\nSaving weights as backup...")
            model.save_weights(PATHS["weights"])
            print(f"Weights saved to: {PATHS['weights']}")
            
            print("\nModel saving completed successfully!")
            
        except Exception as e:
            print(f"Error during saving: {str(e)}")
            raise
                
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass
    
    save_model()