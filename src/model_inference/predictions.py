import os
import numpy as np

import tensorflow as tf
from sklearn.metrics import f1_score
from src.data_preprocessing.load_text_data import load_tokenized_text_data  # Import tokenizer loader
from src.modeling_image.image_preprocessing import preprocess_image
# from src.model_combination.voting_strategy import max_voting, max_voting_proba ,weighted_average_voting
from src.model_combination.voting_strategy import hard_voting, soft_voting ,weighted_soft_voting, max_confidence_voting

# Assurez-vous que is_Debug est défini en global
# is_Debug = True
is_Debug = False

def predict_text_model(text_model, x_val_text, y_val=None, max_sequence_length=500, return_proba=False):
    """
    Predict using the pre-trained text model on the validation data.

    Args:
    - text_model: Loaded text model.
    - x_val_text: Raw text data (string or list).
    - y_val: True labels (optional).
    - max_sequence_length: Sequence length for padding.
    - return_proba: If True, returns probabilities instead of class predictions.

    Returns:
    - predictions or probabilities
    - evaluation_metrics if y_val provided
    """

    if is_Debug:
        print(f"[INFO] Loading tokenizer and applying preprocessing on validation text data.")

    _, _, tokenizer = load_tokenized_text_data(num_samples=None)

    if isinstance(x_val_text, str):
        x_val_text = [x_val_text]
        if is_Debug:
            print(f"[INFO] x_val_text is a single string.")

    x_val_text_tokenized = tokenizer.texts_to_sequences(x_val_text)
    x_val_text_padded = tf.keras.preprocessing.sequence.pad_sequences(
        x_val_text_tokenized,
        padding='post',
        maxlen=max_sequence_length
    )

    if is_Debug:
        print(f"[INFO] Tokenization & Padding done.")
        print(f"[DEBUG] Sample padded data : {x_val_text_padded[:5]}")

    y_proba = text_model.predict(x_val_text_padded, verbose=0)

    if return_proba:
        return y_proba  # Direct return of probabilities

    predictions = np.argmax(y_proba, axis=-1)

    if y_val is not None:
        if is_Debug:
            print("[INFO] Evaluating predictions with true labels.")

        accuracy = np.mean(predictions == y_val)
        f1_score_value = compute_f1_score(y_val, predictions)

        if is_Debug:
            print(f"[INFO] Accuracy: {accuracy:.4f} | F1 Score: {f1_score_value:.4f}")

        evaluation_metrics = {
            "accuracy": accuracy,
            "f1_score": f1_score_value
        }

        return predictions, evaluation_metrics
    else:
        return predictions


def predict_image_model(image_model, x_val_image, image_dir, y_val=None, batch_size=32, return_proba=False):
    """
    Predict using the pre-trained image model on the validation image data.

    Args:
    - image_model: The loaded pre-trained image model (e.g., Xception, InceptionV3).
    - x_val_image: List or Series of image file names (can be a single image or multiple).
    - image_dir: The base directory where the images are stored.
    - y_val: The true target labels for the validation data (optional, used for evaluation).
    - batch_size: The batch size for processing images.
    - return_proba: If True, returns probabilities instead of class predictions.

    Returns:
    - predictions or probabilities
    - evaluation_metrics if y_val provided
    """

    if isinstance(x_val_image, str):
        x_val_image = [x_val_image]

    image_paths = [os.path.join(image_dir, img_name) for img_name in x_val_image]

    y_proba_list = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        preprocessed_images = [preprocess_image(img_path) for img_path in batch_paths]
        preprocessed_images = np.vstack(preprocessed_images)

        batch_proba = image_model.predict(preprocessed_images, verbose=0)  # Probabilities output
        y_proba_list.append(batch_proba)

    y_proba = np.vstack(y_proba_list)

    if return_proba:
        return y_proba

    predictions = np.argmax(y_proba, axis=-1)

    if y_val is not None:
        accuracy = np.mean(predictions == y_val)
        f1_score_value = compute_f1_score(y_val, predictions)

        evaluation_metrics = {
            "accuracy": accuracy,
            "f1_score": f1_score_value
        }

        return predictions, evaluation_metrics

    return predictions


def predict_combined_models(models, 
                            x_val_text=None, 
                            x_val_image=None, 
                            image_dir=None, 
                            use_text=True, 
                            use_image=True, 
                            text_max_sequence_length=500,
                            weights=None):
    """
    Predict using multiple models (Text and/or Image) and return:
    
    - Raw model outputs (probability arrays)
    - Combined predictions using:
        - Hard Voting (majority class voting)
        - Soft Voting (average of probabilities)
        - Weighted Soft Voting (probabilities weighted by model importance)
        - Max Confidence Voting (prediction with the highest single probability)
    """

    if is_Debug:
        print("=" * 60)
        print("[DEBUG] predict_combined_models - Parameters received:")
        print(f"use_text = {use_text} | use_image = {use_image}")
        print(f"text_max_sequence_length = {text_max_sequence_length}")
        print(f"Models passed: {list(models.keys())}")
        print("=" * 60)

    raw_preds_list = []

    return_proba = True  # Always get probabilities for flexibility

    # Predict Text Models
    if use_text:
        for name, model in models.items():
            if 'text' in name:
                preds = predict_text_model(
                    model, x_val_text,
                    max_sequence_length=text_max_sequence_length,
                    return_proba=return_proba
                )
                raw_preds_list.append(preds)
                if is_Debug:
                    print(f"[INFO] Text predictions done for {name}")

    # Predict Image Models
    if use_image:
        if image_dir is None:
            raise ValueError("[ERROR] image_dir must be provided when use_image=True.")
        for name, model in models.items():
            if 'image' in name:
                preds = predict_image_model(
                    model, x_val_image, image_dir,
                    return_proba=return_proba
                )
                raw_preds_list.append(preds)
                if is_Debug:
                    print(f"[INFO] Image predictions done for {name}")

    if len(raw_preds_list) < 2:
        raise ValueError("[ERROR] At least two models are required for combined prediction.")

    # Compute all voting strategies
    preds_hard = hard_voting([np.argmax(p, axis=1) for p in raw_preds_list])
    preds_soft = soft_voting(raw_preds_list)
    if weights is None:
        weights = [1 / len(raw_preds_list)] * len(raw_preds_list)
    preds_weighted = weighted_soft_voting(raw_preds_list, weights)
    preds_max_conf = max_confidence_voting(raw_preds_list)

    if is_Debug:
        print("[INFO] Voting completed (Hard, Soft, Weighted, Max Confidence).")

    return {
        'raw_preds': raw_preds_list,
        'hard_voting': preds_hard,
        'soft_voting': preds_soft,
        'weighted_voting': preds_weighted,
        'max_confidence_voting': preds_max_conf
    }



def predict_combined_models_V01(models, 
                            x_val_text=None, 
                            x_val_image=None, 
                            image_dir=None, 
                            use_text=True, 
                            use_image=True, 
                            text_max_sequence_length=500,
                            weights=None):
    """
    Predict using multiple models (Text and/or Image) and return:
    - Raw predictions (probabilities or labels)
    - Voting results: Hard, Soft, Weighted
    """

    if is_Debug:
        print("=" * 60)
        print("[DEBUG] predict_combined_models - Parameters received:")
        print(f"use_text = {use_text} | use_image = {use_image}")
        print(f"text_max_sequence_length = {text_max_sequence_length}")
        print(f"Models passed: {list(models.keys())}")
        print("=" * 60)

    raw_preds_list = []

    return_proba = True  # Always get proba here (for flexibility)

    # Predict Text Models
    if use_text:
        for name, model in models.items():
            if 'text' in name:
                preds = predict_text_model(
                    model, x_val_text,
                    max_sequence_length=text_max_sequence_length,
                    return_proba=return_proba
                )
                raw_preds_list.append(preds)
                if is_Debug:
                    print(f"[INFO] Text predictions done for {name}")

    # Predict Image Models
    if use_image:
        if image_dir is None:
            raise ValueError("[ERROR] image_dir must be provided when use_image=True.")
        for name, model in models.items():
            if 'image' in name:
                preds = predict_image_model(
                    model, x_val_image, image_dir,
                    return_proba=return_proba
                )
                raw_preds_list.append(preds)
                if is_Debug:
                    print(f"[INFO] Image predictions done for {name}")

    if len(raw_preds_list) < 2:
        raise ValueError("[ERROR] At least two models are required for combined prediction.")


    preds_hard = hard_voting([np.argmax(p, axis=1) for p in raw_preds_list]) # Majority Class Voting
    preds_soft = soft_voting(raw_preds_list)
    if weights is None:
        weights = [1 / len(raw_preds_list)] * len(raw_preds_list)
    preds_weighted = weighted_soft_voting(raw_preds_list, weights) # Weighted Soft Voting

    if is_Debug:
        print("[INFO] Voting completed (Hard, Soft, Weighted).")

    return {
        'raw_preds': raw_preds_list,
        'hard_voting': preds_hard,
        'soft_voting': preds_soft,
        'weighted_voting': preds_weighted
    }



def compute_f1_score(y_true, y_pred):
    """
    Compute F1 Score between true and predicted labels.
    """
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='weighted')


def evaluate_combined_predictions(y_val, predictions, metric="f1"):
    """
    Evaluate the performance of the combined model predictions using F1-score.

    Args:
    - y_val: True labels.
    - predictions: Predicted labels.

    Returns:
    - score: The F1-score of the combined predictions.
    """
    if is_Debug:
        print(f"[INFO] Evaluating combined model predictions.")
    
    if metric == "f1":
        score = f1_score(y_val, predictions, average='weighted')
    else:
        raise ValueError(f"[ERROR] Metric {metric} not supported.")
    
    return score



