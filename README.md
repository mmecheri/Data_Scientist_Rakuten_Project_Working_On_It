# 🛒 Multimodal Product Classification – Rakuten France  
**Advanced Deep Learning Project | Text + Image | Voting Ensemble | AWS-Ready Deployment**

---

## 🚀 Overview

This project addresses the challenge of classifying products into categories (`prdtypecode`) based on **textual descriptions and product images**, using advanced **multimodal deep learning techniques**.

It began as part of a data science challenge and evolved into a production-ready, AWS-compatible pipeline with modules for preprocessing, modeling, ensemble voting, and prediction.

👉 Try it live: [**Rakuten Streamlit Demo**](https://huggingface.co/spaces/mmecheri/Rakuten_Streamlit)

---

## 🎯 Objectives

- Achieve high product classification accuracy using real-world multimodal data  
- Leverage deep learning for both **text** and **image** inputs  
- Combine models using ensemble voting strategies  
- Prepare for scalable **cloud deployment (AWS-ready)**  
- Deliver reusable, modular, and well-documented code

---

## 🧠 Approach Summary

### 📄 Text-Based Models
- Classical: TF-IDF + XGBoost, Logistic Regression  
- Deep Learning: Simple DNN, Conv1D, GRU, LSTM  
- ✅ Best results: Conv1D and Simple DNN (F1 > 0.81)

### 🖼️ Image-Based Models
- Transfer learning with ResNet, Xception, InceptionV3, EfficientNet  
- Fine-tuning and data augmentation applied  
- ✅ Best model: Xception (F1 ≈ 0.66)

### 🔁 Multimodal Voting Ensemble
- Fuses predictions from best text and image models  
- Voting strategies:
  - Hard Voting  
  - Soft Voting  
  - Weighted Soft Voting *(based on per-model F1)*  
  - Max Confidence Voting  
- ✅ Best configuration: **DNN + Conv1D + Xception** using **Weighted Soft Voting**

---

## 📊 Results

| Model Setup                        | Modality     | Weighted F1-score |
|-----------------------------------|--------------|-------------------|
| TF-IDF + XGBoost                  | Text         | 0.802             |
| Simple DNN                        | Text         | 0.810             |
| Xception                          | Image        | 0.660             |
| **DNN + Conv1D + Xception (Ensemble)** | Text + Image | **0.8349**        |

📌 Initial leaderboard: **Ranked #25 / 83 submissions**

---

## 🗂️ Project Structure

```bash
Data_Scientist_Rakuten_Project/
├── config.py                      # Centralized configuration paths
├── data/                          # Raw, interim, and processed datasets
├── models/                        # Pretrained models for text and image (final, tuned, benchmark)
├── notebooks/                     # Jupyter notebooks (EDA, modeling, ensemble, submission)
├── reports/                       # Model evaluation reports (XLSX, charts)
├── src/                            # Core Python modules for data loading, model loading/training, ensemble voting, reporting, and CSV export
├── submissions/                   # Final CSV submission files
└── requirements.txt               # Project dependencies

---

## ⚙️ How to Reproduce

**Clone the repository:**
```bash
git clone https://github.com/your-username/Data_Scientist_Rakuten_Project.git
cd Data_Scientist_Rakuten_Project

**Install dependencies:**
pip install -r requirements.txt

**Ensure the data folder structure is complete (see config.py)**
**Run the notebooks in order:**

- `1_Project_and_Data_Overview.ipynb` – Project presentation and dataset structure  
- `09_Benchmark_Text_Model.ipynb` – First text model baselines  
- `14_Simple_DNN_for_Text_Classification.ipynb` – Deep learning model for text  
- `16_Image_Benchmark_Model.ipynb` – First image model evaluation  
- `22_Model_Combination.ipynb` – Multimodal ensemble with voting  
- `23_Submission.ipynb` – Final predictions and CSV export

## ☁️ AWS-Ready Deployment (In Progress)
The project is being restructured for deployment on AWS, including:

Model hosting via S3

Inference via FastAPI / AWS Lambda

Packaging with Docker

(Future) CI/CD and monitoring integration
