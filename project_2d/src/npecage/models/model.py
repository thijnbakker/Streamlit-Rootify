import logging
from tensorflow.keras.models import load_model
from src.npecage.utils.helpers import f1

logger = logging.getLogger(__name__)


# Load the model
def load_trained_model(model_path):
    """
    Load a trained Keras model from the specified file path, including custom metrics.

    Args:
        model_path (str): Path to the saved Keras model file.

    Returns:
        tensorflow.keras.Model: Loaded Keras model with custom_objects (e.g., f1 metric).
    """

    logger.info(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path, custom_objects={"f1": f1})
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    return model
