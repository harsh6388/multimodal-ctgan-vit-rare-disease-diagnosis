# Multimodal CTGAN + Vision Transformer for Early Diagnosis of Rare Diseases

## 📌 Overview
This project proposes a multimodal deep learning framework for early diagnosis of rare diseases.
The approach integrates:
- Conditional Tabular GAN (CTGAN) for tabular medical data synthesis
- Vision Transformer (ViT) for medical image feature extraction
- Fusion Probability Model for robust multimodal decision making

The model is evaluated on Lung Cancer data.

## 🎯 Objectives
- Improve sensitivity of lung cancer diagnosis models
- Address data imbalance using synthetic tabular data
- Fuse image and tabular modalities for enhanced performance

## 🧠 Methodology
- CTGAN generates high-quality synthetic tabular data
- Vision Transformer extracts global image features
- Fusion Probability Model combines both modalities
  
## Models
Pretrained model weights are not included due to size constraints.


## 📊 Results
| Metric | Before | After |
|------|--------|-------|
| Sensitivity | 80% | 89% |

## 🧪 Dataset
- Tabular clinical features
- Lung CT scan images
*(Dataset not shared due to privacy concerns)*
Due to privacy and size constraints, the dataset is not uploaded to GitHub.
Researchers may contact the author for access or use publicly available lung cancer datasets.

## 📸 Project Screenshots

### 🔹 Multimodal Model Architecture
![Model Architecture](screenshotsscreenshots/1.png)

![Model Architecture](screenshotsscreenshots/2.png)


![Model Architecture](screenshotsscreenshots/3.png)

![Model Architecture](screenshotsscreenshots/4.png)

### 🔹 Results and Evaluation
![Sensitivity Improvement](screenshotsscreenshots/Evalutation.png)

![Fusion Output](screenshotsscreenshots/Evaluation2.png)





## ⚙️ Installation
```bash
pip install -r requirements.txt
