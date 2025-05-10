from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3

def preprocess_image(img_path, target_size=(299, 299), model_type="Xception"):
    """
    Prépare une image pour la prédiction en redimensionnant, en la convertissant en tableau et en appliquant les pré-traitements nécessaires.

    Args:
    - img_path : str, le chemin de l'image à traiter.
    - target_size : tuple, la taille vers laquelle l'image sera redimensionnée (par défaut (299, 299) pour Xception et InceptionV3).
    - model_type : str, le type de modèle ('Xception' ou 'InceptionV3'), ce qui déterminera quel pré-traitement appliquer.

    Returns:
    - image : np.ndarray, l'image prétraitée prête pour la prédiction.
    """
    # Redimensionner l'image
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Ajoute une dimension pour batch

    # Appliquer le pré-traitement en fonction du modèle
    if model_type == "Xception":
        img_array = preprocess_input_xception(img_array)  # Prétraitement spécifique à Xception
    elif model_type == "InceptionV3":
        img_array = preprocess_input_inceptionv3(img_array)  # Prétraitement spécifique à InceptionV3
    else:
        raise ValueError("Le modèle spécifié n'est pas supporté. Utilisez 'Xception' ou 'InceptionV3'.")

    return img_array
