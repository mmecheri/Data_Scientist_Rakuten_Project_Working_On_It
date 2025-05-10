# voting_strategy.py

import numpy as np
import tensorflow as tf

is_Debug = True  # Global Debug Mode


def hard_voting(predictions_list):
    """
    Hard Voting (Majority Voting) based on predicted classes.
    """
    if is_Debug:
        print(f"[INFO] Applying Max Voting (Hard Voting).")

    final_predictions = np.array([
        np.bincount(preds).argmax() for preds in zip(*predictions_list)
    ])

    return final_predictions


def soft_voting(prob_list):
    """
    Soft Voting (Average of Probabilities without weights).
    """
    if is_Debug:
        print(f"[INFO] Applying Max Voting (Soft Voting / Proba Average).")

    avg_proba = np.mean(prob_list, axis=0)
    final_predictions = np.argmax(avg_proba, axis=1)

    return final_predictions


def weighted_soft_voting(prob_list, weights):
    """
    Soft Voting with weights based on model importance.
    """
    if is_Debug:
        print(f"[INFO] Applying Weighted Average Voting.")

    weighted_sum = np.zeros_like(prob_list[0], dtype=np.float32)

    for i, prob in enumerate(prob_list):
        weighted_sum += prob * weights[i]

    final_predictions = np.argmax(weighted_sum, axis=1)

    return final_predictions

def max_confidence_voting(prob_list):
    """
    Max Confidence Voting:
    For each sample, selects the class associated with the highest single probability
    across all models.
    """
    if is_Debug:
        print(f"[INFO] Applying Max Confidence Voting.")

    num_samples = prob_list[0].shape[0]
    num_classes = prob_list[0].shape[1]
    final_predictions = []

    for i in range(num_samples):
        best_class = -1
        best_prob = -1
        for model_probs in prob_list:
            for cls in range(num_classes):
                if model_probs[i, cls] > best_prob:
                    best_prob = model_probs[i, cls]
                    best_class = cls
        final_predictions.append(best_class)

    return np.array(final_predictions)
