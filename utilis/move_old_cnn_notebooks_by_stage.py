import os
import shutil

# 📌 Dossier source où se trouvent les anciens notebooks
old_notebooks_dir = "old/"

# 📌 Dossier destination où on va copier les notebooks
new_notebooks_dir = "notebooks/modeling/image/"

# 📌 Mapping des notebooks par étape
notebooks_mapping = {
    "Etape_1": {
        "InceptionResNetV2": "Iteration_#3_InceptionResNetV2_All_Train_data_16122021_299x299_40Epochs_Accuracy_62.35%.ipynb",
        "DenseNet121": "Iteration_#3_DenseNet121_All_Train_data_02122021_40Epochs_Accuracy_61.45%.ipynb",
        "Xception": "Iteration_#3_Xception_All_Train_data_16122021_299x299_40Epochs_Accuracy_61.43%.ipynb",
        "InceptionV3": "Iteration_#3_InceptionV3_All_Train_data_299x299_02122021_40Epochs_Accuracy_59.91%.ipynb",
        "MobileNetV2": "Iteration_#3_MobileNetV2_All_Train_data_02122021_40Epochs_Accuracy_59.48%.ipynb",
    },
    "Etape_2": {
        "InceptionResNetV2": "Iteration_#4_1_InceptionResNetV2_All_Train_data_299x299_18122021_40Epochs_Accuracy_58.80%.ipynb",
        "Xception": "Iteration_#4_1_Xception_All_Train_data_299x299_19122021_40Epochs_Accuracy_58.96%.ipynb",
        "DenseNet121": "Iteration_#4_1_DenseNet121_All_Train_data_03122021_40Epochs_Accuracy_57.75%.ipynb",
        "InceptionV3": "Iteration_#4_1_InceptionV3_All_Train_data_299x299_03122021_40Epochs_Accuracy_57.94%.ipynb",
        "MobileNetV2": "Iteration_#4_1_MobileNetV2_All_Train_data_04122021_40Epochs_Accuracy_56.42%.ipynb",
    },
    "Etape_3": {
        "Xception": "Iteration_#6_Xception_All_Train_data_299x299_16012022_100Epochs_Accuracy_66.36%.ipynb",
        "InceptionV3": "Iteration_#6_InceptionV3_All_Train_data_299x299_13022022_100Epochs_Accuracy_TBE.ipynb",
        "MobileNetV2": "Iteration_#6_MobileNetV2_All_Train_data_13022022_100Epochs_Accuracy_TBE.ipynb",
        "InceptionResNetV2": "Iteration_#6_InceptionResNetV2_All_Train_data_299x299_14022022_100Epochs_Accuracy_62.19%.ipynb",
        "DenseNet121": "Iteration_#6_DenseNet121_All_Train_data_15022022_100Epochs_Accuracy_60.22%.ipynb",
    },
    "Etape_5": {
        "Xception": "Iteration_#7_Xception_299x299_26022022_OverFittting_Solving_6.ipynb",
        "InceptionV3": "Iteration_#7_InceptionV3_299x299_26022022_OverFittting_Solving_1.ipynb",
        "InceptionResNetV2": "Iteration_#4_2_InceptionResNetV2_All_Train_data_299x299_19122021_450Epochs_Accuracy_63.83%.ipynb",
        "DenseNet121": "Iteration_#4_3_DenseNet121_All_Train_data_21122021_50Epochs_Accuracy_60.10%.ipynb",
        "MobileNetV2": "Iteration_#4_3_MobileNetV2_All_Train_data_21122021_50Epochs_Accuracy_60.63%.ipynb",
    }
}

# 📌 Fonction pour rechercher un fichier dans tous les sous-dossiers
def find_notebook(model_keyword):
    for root, _, files in os.walk(old_notebooks_dir):
        for file in files:
            if model_keyword in file and file.endswith(".ipynb"):
                return os.path.join(root, file)  # Retourne le chemin complet du fichier trouvé
    return None  # Retourne None si aucun fichier n'a été trouvé

# 📌 Fonction pour organiser les notebooks
def organize_notebooks():
    for etape, models in notebooks_mapping.items():
        etape_folder = os.path.join(new_notebooks_dir, etape)
        os.makedirs(etape_folder, exist_ok=True)  # Créer le dossier de l'étape s'il n'existe pas

        for model_name, keyword in models.items():
            source_path = find_notebook(keyword)
            if source_path:
                dest_path = os.path.join(etape_folder, f"{model_name}.ipynb")  # Renommer le fichier
                shutil.copy(source_path, dest_path)
                print(f"✅ Copied {source_path} → {dest_path}")
            else:
                print(f"⚠️ WARNING: No notebook found for {model_name} in {old_notebooks_dir}")

# 📌 Exécuter la fonction
organize_notebooks()