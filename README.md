#  Breast Cancer Histopathology Classification

A deep learning project that classifies breast tumor histopathology images into **8 subtypes** (4 benign, 4 malignant) using **DenseNet121** transfer learning on **7,900+ images** from the BreakHis dataset.

> Developed during my Data Science internship at the Innovation & Entrepreneurship Center (Jun – Aug 2025).

##  Approach: Hierarchical Classification

Instead of one model for all 8 classes, the project uses three models in two stages:

```
                 Histopathology image
                          │
            ┌─────────────▼─────────────┐
            │  Model 1: Binary          │
            │  Benign vs. Malignant     │
            └──────┬─────────────┬──────┘
                   │             │
      ┌────────────▼───┐   ┌─────▼────────────────┐
      │ Model 2:       │   │ Model 3:             │
      │ Benign subtype │   │ Malignant subtype    │
      │ (4 classes)    │   │ (4 classes)          │
      └────────────────┘   └──────────────────────┘
```

- **Benign:** Adenosis · Fibroadenoma · Phyllodes Tumor · Tubular Adenoma
- **Malignant:** Ductal Carcinoma · Lobular Carcinoma · Mucinous Carcinoma · Papillary Carcinoma

##  Pipeline

- Image loading and preprocessing with OpenCV
- Normalization and image augmentation
- **DenseNet121** (ImageNet weights) as the feature extractor, with custom dense layers and L2 regularization
- Separate evaluation for the binary, benign and malignant models (accuracy, classification report, confusion matrix)

##  Tech Stack

Python · TensorFlow · Keras · OpenCV · NumPy · Pandas · Matplotlib · Scikit-learn

##  Dataset

[BreakHis – Breast Cancer Histopathological Database](https://www.kaggle.com/datasets/ambarish/breakhis) (Kaggle).
The script downloads it automatically with `kagglehub`.

##  How to Run

Recommended: run on **Kaggle** or **Google Colab with a GPU**.

```bash
pip install -r requirements.txt
python breast_cancer_classification.py
```
