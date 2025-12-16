# uber-
# Booking Status Prediction (Multi-Class) — Model Benchmarking

This repository contains an end-to-end machine learning workflow to predict **Booking Status** (multi-class) from a ride/booking dataset. It includes data exploration, leakage-aware preprocessing, class imbalance handling, model training, cross-validation, and probability-based evaluation (ROC/PR curves + LogLoss).

---

## Project Goals

- Predict **Booking Status** as a multi-class classification problem
- Handle **class imbalance** with SMOTE-based resampling
- Compare multiple models using consistent evaluation metrics
- Evaluate both:
  - **Classification quality** (Accuracy, Macro F1, Balanced Accuracy)
  - **Probability quality** (LogLoss + calibration)

---

## Dataset Overview

- **Rows:** 150,000  
- **Columns:** 48 (numeric + engineered features + one-hot encoded flags)  
- **Target:** `Booking Status` (multi-class)

> Note: The notebook removes outcome-leaking features (e.g., cancellation/incomplete indicators and “reason” fields) to ensure models learn from valid signals instead of direct outcome hints.

---

## Workflow Summary

1. **EDA**
   - `df.info()`, descriptive statistics, target distribution
2. **Leakage-aware preprocessing**
   - Drop columns that directly reveal the target outcome (leakage)
   - Encode target with `LabelEncoder`
   - Stratified train/test split
3. **Imbalance handling**
   - SMOTE-family techniques (SMOTE, BorderlineSMOTE)
4. **Model training & evaluation**
   - Confusion matrix + classification report
   - Accuracy, Macro F1, Balanced Accuracy, LogLoss
   - 5-fold Stratified Cross-Validation
   - Multi-class ROC & Precision–Recall curves
5. **Probability calibration**
   - Isotonic calibration to improve probability reliability (lower LogLoss)

---

## Models Implemented

- Logistic Regression (+ SMOTE)
- Logistic Regression + **Isotonic Calibration**
- Decision Tree
- Random Forest
- CatBoost
- XGBoost
- LightGBM
- KNN
- Gaussian Naive Bayes
- Neural Network (MLPClassifier)
- Custom “precision-focused” pipeline using **BorderlineSMOTE** (to control Class-3 false positives)

---

## Key Findings (High-Level)

- **Boosted trees (LightGBM / XGBoost / CatBoost)** produced the strongest overall performance.
- **LightGBM** achieved the best **probability quality** (lowest LogLoss).
- **CatBoost** delivered strong balanced classification performance and stable results.
- **Random Forest / Decision Tree** performed well on balanced recall, but probabilities were less reliable than boosting.
- **KNN / Naive Bayes** struggled with overlapping classes and had poor probability estimates (high LogLoss).
- **Calibration** significantly improved Logistic Regression probability quality.

---
