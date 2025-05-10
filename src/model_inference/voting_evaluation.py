# import numpy as np
# from src.model_inference.predictions import (
#     max_voting,
#     weighted_average_voting,
#     max_voting_proba,  # Soft Voting = Max Voting on averaged probabilities
#     compute_f1_score
# )

# def evaluate_voting_strategies(predictions_list, y_val, weights=None):
#     """
#     Evaluate voting strategies based on raw predictions.

#     Args:
#     - predictions_list : list of proba np.array from models.
#     - y_val : true labels.
#     - weights : optional weights for weighted voting.
#     """

#     results = {}

#     # Hard Voting
#     hard_preds = max_voting([np.argmax(p, axis=1) for p in predictions_list])
#     results['Max Voting (Hard)'] = compute_f1_score(y_val, hard_preds)

#     # Soft Voting
#     soft_preds = max_voting_proba(predictions_list)
#     results['Max Voting (Soft)'] = compute_f1_score(y_val, soft_preds)

#     # Weighted Voting
#     if weights is None:
#         weights = [1 / len(predictions_list)] * len(predictions_list)

#     weighted_preds = weighted_average_voting(predictions_list, weights)
#     results[f'Weighted Voting (weights={weights})'] = compute_f1_score(y_val, weighted_preds)

#     return results


# src/evaluation/voting_evaluation.py

from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate_voting_strategies(strategies_dict, y_true, verbose=True):
    """
    Evaluate multiple voting strategies on the same ground truth labels.

    Parameters:
        strategies_dict (dict): Keys = strategy names, Values = predicted class arrays
        y_true (array-like): True labels
        verbose (bool): If True, prints the scores

    Returns:
        dict: Dictionary with accuracy and F1-score for each strategy
    """
    results = {}

    for name, y_pred in strategies_dict.items():
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        results[name] = {'accuracy': acc, 'f1_weighted': f1}
        
        if verbose:
            print(f"\n{name}")
            print("-" * len(name))
            print(f"Accuracy       : {acc:.4f}")
            print(f"F1 Score (weighted): {f1:.4f}")
            print(classification_report(y_true, y_pred, zero_division=0))

    return results
