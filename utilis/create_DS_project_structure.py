# Data_Scientist_Rakuten_Project-main/
# ├── README.md
# ├── LICENSE
# ├── data/
# │   ├── raw/                # Données brutes non modifiées
# │   │   ├── text/           # Données textuelles brutes
# │   │   └── images/         # Images brutes
# │   └── processed/          # Données traitées prêtes pour l'analyse
# │       ├── text/           # Données textuelles traitées
# │       └── images/         # Images traitées
# │   └── interim/          #  Données temporaires ou intermédiaires
# │       ├── .pickle 
# │       
# │     
# │── notebooks/              # Notebooks Jupyter pour les différentes étapes du projet
# │   ├── 1_Data_Acquisition.ipynb       # Notebook pour l'acquisition des données
# │   ├── 2_Data_Exploration_and_Visualization.ipynb  # Notebook pour l'exploration et la visualisation des données
# │   ├── 3_Data_Preprocessing.ipynb     # Notebook pour le prétraitement des données
# │   ├── 4_Text_Modeling.ipynb          # Notebook pour la modélisation des données textuelles
# │   ├── 5_Image_Modeling.ipynb         # Notebook pour la modélisation des données images
# │   └── 6_Model_Combination.ipynb      # Notebook pour la combinaison des modèles texte et image
# ├── src/                   # Code source structuré du projet
# │   ├── data_acquisition/   # Scripts pour l'acquisition des données
# │   ├── data_preprocessing/ # Scripts pour le prétraitement des données
# │   ├── modeling_text/      # Scripts pour l'entraînement et l'évaluation des modèles textuels
# │   ├── modeling_image/     # Scripts pour l'entraînement et l'évaluation des modèles d'images
# │   ├── model_combination/  # Scripts pour la combinaison des modèles texte et image
# │   └── app/                # Application interactive développée avec Streamlit
# ├── reports/               # Rapports et présentations
# │   ├── figures/            # Figures et graphiques générés pour les rapports
# │   └── final_report.md     # Rapport technique final du projet
# ├── models/                # Modèles entraînés et artefacts associés
# ├── references/            # Documents de référence et ressources supplémentaires
# └── requirements.txt       # Liste des dépendances du projet

# ===============================================
# Code Python
# ==============================================
import os

# Définir la structure des répertoires
directories = [
    "Data_Scientist_Rakuten_Project-main/data/raw/text",
    "Data_Scientist_Rakuten_Project-main/data/raw/images",
    "Data_Scientist_Rakuten_Project-main/data/processed/text",
    "Data_Scientist_Rakuten_Project-main/data/processed/images",
    "Data_Scientist_Rakuten_Project-main/notebooks",
    "Data_Scientist_Rakuten_Project-main/src/data_acquisition",
    "Data_Scientist_Rakuten_Project-main/src/data_preprocessing",
    "Data_Scientist_Rakuten_Project-main/src/modeling_text",
    "Data_Scientist_Rakuten_Project-main/src/modeling_image",
    "Data_Scientist_Rakuten_Project-main/src/model_combination",
    "Data_Scientist_Rakuten_Project-main/src/app",
    "Data_Scientist_Rakuten_Project-main/reports/figures",
    "Data_Scientist_Rakuten_Project-main/models",
    "Data_Scientist_Rakuten_Project-main/references"
]

# Créer les répertoires
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Définir la structure des fichiers
files = [
    "Data_Scientist_Rakuten_Project-main/README.md",
    "Data_Scientist_Rakuten_Project-main/LICENSE",
    "Data_Scientist_Rakuten_Project-main/reports/final_report.md",
    "Data_Scientist_Rakuten_Project-main/requirements.txt",
    "Data_Scientist_Rakuten_Project-main/notebooks/1_Data_Acquisition.ipynb",
    "Data_Scientist_Rakuten_Project-main/notebooks/2_Data_Exploration_and_Visualization.ipynb",
    "Data_Scientist_Rakuten_Project-main/notebooks/3_Data_Preprocessing.ipynb",
    "Data_Scientist_Rakuten_Project-main/notebooks/4_Text_Modeling.ipynb",
    "Data_Scientist_Rakuten_Project-main/notebooks/5_Image_Modeling.ipynb",
    "Data_Scientist_Rakuten_Project-main/notebooks/6_Model_Combination.ipynb"
]

# Créer les fichiers
for file in files:
    with open(file, 'w') as f:
        pass  # Crée un fichier vide
