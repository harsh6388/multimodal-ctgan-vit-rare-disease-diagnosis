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

## 📊 Results
| Metric | Before | After |
|------|--------|-------|
| Sensitivity | 80% | 89% |

## 🧪 Dataset
- Tabular clinical features
- Lung CT scan images
*(Dataset not shared due to privacy concerns)*

## ⚙️ Installation
```bash
pip install -r requirements.txt
