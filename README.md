# Data Scientist Rakuten Project - Work in Progress

## Streamlit Application  
The first version of the **interactive Streamlit application**, which presents the project, different modeling steps, and results, is hosted on Hugging Face:  
[**Rakuten Streamlit App**](https://huggingface.co/spaces/mmecheri/Rakuten_Streamlit)  

In the **Demo** section of the application, users can make their own predictions using **text, images, or a combination of both**.  

## Project Overview  
This project focuses on **multimodal product classification** for Rakuten France. The goal is to predict product category codes (`prdtypecode`) using both **text** (product title & description) and **image** (product images) data.  

Originally developed as part of a **challenge** in a Data Scientist training program, this repository is a **work in progress** aimed at improving previous implementations and optimizing the classification model.  

**Current Status:** The project is undergoing improvements, with updates coming soon.  

## Objectives  
- Improve classification accuracy beyond the initial baseline models.  
- Optimize **text-based** classification using machine learning and deep learning.  
- Enhance **image-based** classification using convolutional neural networks (CNNs) and transfer learning.  
- Develop a robust **bimodal approach** combining both text and image models.  

## Dataset  
Rakuten France provides a dataset of approximately **99,000 product listings**, split into:  
- `X_train_update.csv`: Training data with product descriptions and image references.  
- `y_train_CVw08PX.csv`: Target labels (`prdtypecode`).  
- `X_test_update.csv`: Test data for evaluation.  
- `images.zip`:  
  - `image_train/` (84,916 images)  
  - `image_test/` (13,812 images)  

## Modeling Approach  

### Text-Based Classification  
- **Traditional Machine Learning**: Initial tests with **TF-IDF + ML classifiers**.  
- **Deep Learning**: Models tested include **Conv1D, LSTMs, and DNNs**.  
- **Best Performance**: Conv1D and DNN achieved the highest F1-score.  

### Image-Based Classification  
- **CNNs & Transfer Learning**: Models trained using **ResNet, Xception, and InceptionV3**.  
- **Best Performance**: Identified top models for further fine-tuning.  

### Multimodal (Text + Image) Classification  
- **Voting Ensemble Methods**: Max Voting and Weighted Average.  
- **Best Configuration**: A combination of **Conv1D, Simple DNN, and InceptionV3** achieved the highest accuracy.  

## Work in Progress & Next Steps  
- Improving text-based feature engineering.  
- Fine-tuning image models and exploring data augmentation.  
- Enhancing bimodal fusion strategies for better classification.  
- Deploying an updated interactive **Streamlit demo** (coming soon).  

## Baseline Results  
| Model | Data Type | Weighted F1-score |  
|------------|------------|----------------|  
| **RNN** | Text | 0.8113 |  
| **ResNet** | Images | 0.5534 |  
| **Best Multimodal (Initial Submission)** | Text + Image | **0.8349** (Rank #25/83) |  

**Goal:** Improve beyond **0.8349** F1-score with optimized models.  

## How to Use (Once Ready)  
Since this project is still under development, the full pipeline is not yet finalized. Once the improvements are completed, the repository will include:  
- Code for training and evaluating models.  
- Scripts for preprocessing and feature engineering.  
- Instructions to reproduce experiments.  
- Deployment guide for the **Streamlit app**.  

## Stay Updated  
This project is actively being improved. Check back for updates or contribute if interested.  

For any questions or suggestions, feel free to open an issue or reach out.
