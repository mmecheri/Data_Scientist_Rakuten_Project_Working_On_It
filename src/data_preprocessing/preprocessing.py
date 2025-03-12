import numpy as np
from tensorflow.keras.applications import preprocess_input

def preprocess_images(image_batch):
    """
    Applique la normalisation des images pour un modèle spécifique.
    """
    return preprocess_input(image_batch)
