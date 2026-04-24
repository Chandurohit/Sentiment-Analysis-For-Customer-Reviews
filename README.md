# Sentiment-Analysis-For-Customer-Reviews
Academic project focused on sentiment classification of customer reviews.
# RoBERTa + Capsule Network Ensemble for Sentiment Analysis

## 📌 Overview

This project implements a hybrid deep learning architecture combining **Twitter-RoBERTa** and a **Capsule Network (CapsNet)** for sentiment classification on the SST-2 dataset.

## 🚀 Key Features

* Transformer-based feature extraction (Twitter-RoBERTa)
* Capsule Network for advanced feature modeling
* Back-translation data augmentation (EN → FR → EN)
* Weighted ensemble learning
* SHAP-based explainability

## 📊 Results

* RoBERTa Accuracy: ~93.5%
* CapsNet Accuracy: ~92%
* **Ensemble Accuracy: ~94%+**

## 📈 Outputs

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

### Training Curves

![Training Curves](training_curves.png)

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python complete_implementation.py
```

## 📚 Dataset

* GLUE SST-2 dataset

## 💡 Highlights

* Hybrid model combining Transformers + Capsule Networks
* Data augmentation improves class balance
* Ensemble boosts performance over individual models
