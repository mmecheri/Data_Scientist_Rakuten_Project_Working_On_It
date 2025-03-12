import os

# 📌 Base directory of the project (local execution)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 📂 Data directory
DATA_DIR = os.path.join(BASE_DIR, "data")

# 📂 Raw Data Paths
RAW_CSV_DIR = os.path.join(DATA_DIR, "raw_csv")  # Raw CSV files (metadata + text)
RAW_IMAGES_DIR = os.path.join(DATA_DIR, "raw_images")  # Raw images (train/test)
RAW_IMAGE_TRAIN_DIR = os.path.join(RAW_IMAGES_DIR, "image_train")
RAW_IMAGE_TEST_SUB_DIR = os.path.join(RAW_IMAGES_DIR, "image_test")

# 📂 Processed Data Paths
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# 📂 Interim Text Data
INTERIM_DIR = os.path.join(DATA_DIR, "interim")

# 📂 Processed Text Data
PROCESSED_TEXT_DIR = os.path.join(PROCESSED_DIR, "text")



# Text_Vectorization_TF-IDF
XTRAIN_MATRIX_PATH = os.path.join(PROCESSED_TEXT_DIR, "Xtrain_matrix.pkl")
XTEST_MATRIX_PATH = os.path.join(PROCESSED_TEXT_DIR, "Xtest_matrix.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(PROCESSED_TEXT_DIR, "tfidf_vectorizer.pkl")

#The encoded labels (0-26) and their original product code classes
YTRAIN_ENCODED_PATH = os.path.join(PROCESSED_TEXT_DIR, "y_train_encoded.pkl")
PRDTYPECODE_MAPPING_PATH = os.path.join(PROCESSED_TEXT_DIR, "prdtypecode_mapping.pkl")

# Text Tokenization_and_Sequencing
TOKENIZER_PATH =  os.path.join(PROCESSED_TEXT_DIR, "tokenizer.pkl")
X_TRAIN_SPLIT_TOKENIZED_PATH =  os.path.join(PROCESSED_TEXT_DIR, "X_train_TokenizationSequencing.pkl")
X_TEST_SPLIT_TOKENIZED_PATH =  os.path.join(PROCESSED_TEXT_DIR, "X_test_TokenizationSequencing.pkl")
Y_TRAIN_SPLIT_PATH =  os.path.join(PROCESSED_TEXT_DIR, "y_train_split.pkl")
Y_TEST_SPLIT_PATH =  os.path.join(PROCESSED_TEXT_DIR, "y_test_split.pkl")

X_SUBMISSION_TOKENIZED_PATH =  os.path.join(PROCESSED_TEXT_DIR, "X_submission_TokenizationSequencing.pkl")



# 📂 Processed Image Data
PROCESSED_IMAGES_DIR = os.path.join(PROCESSED_DIR, "images")
RESIZED_IMAGE_DIR = os.path.join(PROCESSED_IMAGES_DIR, "resized")
RESIZED_IMAGE_TRAIN_DIR = os.path.join(RESIZED_IMAGE_DIR, "image_train")
RESIZED_IMAGE_TEST_DIR = os.path.join(RESIZED_IMAGE_DIR, "image_test")

IMAGE_FEATURES_DIR = os.path.join(PROCESSED_IMAGES_DIR, "features")
CNN_FEATURES_TRAIN = os.path.join(IMAGE_FEATURES_DIR, "cnn_features_train.pkl")
CNN_FEATURES_TEST = os.path.join(IMAGE_FEATURES_DIR, "cnn_features_test.pkl")
HOG_FEATURES_PATH = os.path.join(IMAGE_FEATURES_DIR, "hog_features.pkl")
SIFT_FEATURES_PATH = os.path.join(IMAGE_FEATURES_DIR, "sift_features.pkl")
PCA_IMAGES_PATH = os.path.join(IMAGE_FEATURES_DIR, "pca_images.pkl")

# 📄 Final Datasets
XTRAIN_FINAL_PATH = os.path.join(PROCESSED_DIR, "X_train_final.pkl") 
XTRAIN_FINAL_ENCODED_PATH = os.path.join(PROCESSED_DIR, "X_train_final_encoded.pkl") 
XTEST_SUB_FINAL_PATH = os.path.join(PROCESSED_DIR, "X_test_Submission_final.pkl")
YTRAIN_FINAL_PATH = os.path.join(PROCESSED_DIR, "y_train_final.pkl")


# 📂 Models directory
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 📂 Text Models directory
TEXT_MODELS_DIR = os.path.join(MODELS_DIR, "text")
# 📂 Classical models
CLASSICAL_MODELS_DIR = os.path.join(TEXT_MODELS_DIR, "classical")

# 📂 Neural models
NEURAL_MODELS_DIR = os.path.join(TEXT_MODELS_DIR, "neural")

# 📂 Image Models directory
IMAGE_MODELS_DIR = os.path.join(MODELS_DIR, "image")

# 📂 Notebooks directory
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
EDA_NOTEBOOKS_DIR = os.path.join(NOTEBOOKS_DIR, "eda_and_processing")
TEXT_MODELING_NOTEBOOKS_DIR = os.path.join(NOTEBOOKS_DIR, "modeling/text")
IMAGE_MODELING_NOTEBOOKS_DIR = os.path.join(NOTEBOOKS_DIR, "modeling/image")
BIMODAL_MODELING_NOTEBOOKS_DIR = os.path.join(NOTEBOOKS_DIR, "modeling/bimodal")

# 📂 Reports Directory
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CLASSIFICATION_REPORTS_DIR = os.path.join(REPORTS_DIR, "classification_reports")

# 📂 Classification Reports for Different Modalities
TEXT_REPORTS_DIR = os.path.join(CLASSIFICATION_REPORTS_DIR, "text")
IMAGE_REPORTS_DIR = os.path.join(CLASSIFICATION_REPORTS_DIR, "image")
BIMODAL_REPORTS_DIR = os.path.join(CLASSIFICATION_REPORTS_DIR, "bimodal")


# # 📂 AWS S3 Configuration (for Cloud Deployment)
# S3_BUCKET_NAME = "my-ds-project-bucket"  # Replace with your S3 bucket name
# S3_RAW_CSV_DIR = f"s3://{S3_BUCKET_NAME}/raw_csv/"
# S3_RAW_IMAGES_DIR = f"s3://{S3_BUCKET_NAME}/raw_images/"
# S3_PROCESSED_TEXT_DIR = f"s3://{S3_BUCKET_NAME}/processed/text/"
# S3_PROCESSED_IMAGES_DIR = f"s3://{S3_BUCKET_NAME}/processed/images/"
# S3_MODELS_DIR = f"s3://{S3_BUCKET_NAME}/models/"

# 📌 Display loaded paths for verification
if __name__ == "__main__":
    print("✅ Configuration loaded successfully!")
    print(f"🔹 Project Root: {BASE_DIR}")
    print(f"📂 Data Directory: {DATA_DIR}")
    print(f"📂 Raw CSV Path: {RAW_CSV_DIR}")
    print(f"📂 Raw Images Path: {RAW_IMAGES_DIR}")
    print(f"📂 Processed Text Path: {PROCESSED_TEXT_DIR}")
    print(f"📂 Processed Image Features Path: {IMAGE_FEATURES_DIR}")
    print(f"📂 Final Train Dataset Path: {XTRAIN_FINAL_PATH}")
    print(f"📂 Models Directory: {MODELS_DIR}")
    # print('X_TRAIN_TOKENIZED_PATH',X_TRAIN_TOKENIZED_PATH)  # This should print the correct path for X_train
    # print(f"☁️ AWS S3 Bucket: {S3_BUCKET_NAME}")
