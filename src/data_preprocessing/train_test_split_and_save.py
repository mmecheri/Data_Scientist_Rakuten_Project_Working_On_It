import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
import config  # Import du fichier config.py pour gérer les chemins

# ------------------ PARAMÈTRES ------------------
TEST_SIZE = 0.2  # Pourcentage du dataset utilisé pour le test
RANDOM_STATE = 1234  # Pour garantir un split identique à chaque exécution

# ------------------ CHARGEMENT DES DONNÉES ------------------
print(" Loading full training dataset...")

# Charger le DataFrame contenant le texte + image
train_full_path = Path(config.XTRAIN_FINAL_PATH)
train_full = pd.read_pickle(train_full_path)

# Vérification de la structure des données
print(f"Dataset chargé | Shape: {train_full.shape}")
print(train_full.head(3))  # Afficher un aperçu des données

# ------------------ SPLIT TRAIN / TEST ------------------
print("Splitting dataset into train and test sets...")

X_train_full, X_test_full = train_test_split(train_full, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Vérifier la répartition
print(f"Taille X_train_full : {X_train_full.shape}")
print(f"Taille X_test_full  : {X_test_full.shape}")

# ------------------ SAUVEGARDE ------------------
print("Saving split datasets...")

# Sauvegarde en pickle
X_train_full.to_pickle(Path(config.PROCESSED_DATA_DIR) / "X_train_split.pkl")
X_test_full.to_pickle(Path(config.PROCESSED_DATA_DIR) / "X_test_split.pkl")

print("Splitting & saving completed successfully!")
