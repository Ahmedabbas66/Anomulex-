# 🤖 ML Models in Anomulex
**Project:** Anomulex – AI-based Intrusion Detection System (IDS)  
**Version:** v0.1.0-alpha  
**Author:** Ahmed Adel  
**Date:** 2025-09-30  

---

## 1. Introduction
Anomulex leverages **Machine Learning (ML)** to detect anomalies in network traffic.  
Unlike traditional IDS/IPS that rely only on signatures, Anomulex detects **zero-day attacks** and **unknown threats** using statistical learning.  

This document explains the ML models used, training methodology, and evaluation strategies.  

---

## 2. Model Categories

### 2.1 Supervised Models
Used when labeled datasets (e.g., CICIDS2017, UNSW-NB15) are available.  

- **Random Forest (RF)**  
  - Ensemble of decision trees.  
  - Strong at handling tabular data and categorical flags.  
  - Resistant to overfitting.  
  - Outputs feature importance.  

- **Gradient Boosting (XGBoost / LightGBM)**  
  - Boosted decision trees optimized for speed and accuracy.  
  - Effective for imbalanced data (attack vs benign).  
  - Handles large datasets efficiently.  

- **Support Vector Machine (SVM)**  
  - Effective for binary classification (attack vs normal).  
  - Works well with small-to-medium feature sets.  
  - Computationally expensive for large flows.  

---

### 2.2 Unsupervised Models
Used for **anomaly detection** when only normal traffic is available.  

- **Autoencoder (Deep Learning)**  
  - Neural network trained to reconstruct normal flows.  
  - High reconstruction error → anomaly detected.  
  - Captures non-linear patterns in data.  

- **Isolation Forest**  
  - Randomly partitions data space.  
  - Anomalies are isolated faster than normal points.  
  - Lightweight, interpretable.  

- **K-Means Clustering**  
  - Groups flows into clusters.  
  - Outliers far from clusters → anomalies.  
  - Useful for traffic profiling.  

---

## 3. Training Workflow

1. **Preprocessing**  
   - Normalize continuous features.  
   - Encode categorical features (protocol, flags).  
   - Handle missing values.  

2. **Feature Selection**  
   - Recursive Feature Elimination (RFE).  
   - Correlation pruning.  
   - Domain-driven selection (IAT, packet size, flag ratios).  

3. **Model Training**  
   - Train multiple candidate models.  
   - Use **cross-validation (k-fold)** to avoid overfitting.  
   - Optimize hyperparameters (grid search, Bayesian optimization).  

4. **Evaluation Metrics**  
   - Accuracy (baseline metric).  
   - Precision, Recall, F1-score.  
   - ROC-AUC (binary classification).  
   - Detection Rate (TPR) vs False Alarm Rate (FPR).  

---

## 4. Model Deployment in Anomulex

- **Offline Training**  
  - Train ML models on historical datasets (CICIDS2017, UNSW-NB15).  
  - Save best-performing models in serialized format (`.pkl` for scikit-learn, `.h5` for TensorFlow).  

- **Online Inference**  
  - Collector exports **flows (IPFIX/NetFlow/CICFlowMeter)**.  
  - Features are mapped to the trained model input.  
  - ML model predicts: **Normal / Attack (with confidence score)**.  

- **Integration with IPS**  
  - If attack is detected with high confidence, send mitigation signal (block IP, drop flow).  
  - Feedback loop updates the model with new labeled data.  

---

## 5. Model Comparison Table

| Model            | Type        | Pros | Cons | Use Case |
|------------------|------------|------|------|----------|
| Random Forest    | Supervised | High accuracy, interpretable | Slower for very large data | General-purpose detection |
| XGBoost/LightGBM | Supervised | Scalable, fast, handles imbalance | More complex tuning | Real-time classification |
| SVM              | Supervised | Strong binary classifier | Not scalable to millions of flows | Small-scale IDS |
| Autoencoder      | Unsupervised | Detects novel attacks | Requires GPU for training | Zero-day anomaly detection |
| Isolation Forest | Unsupervised | Lightweight, fast | Limited feature interaction | Online anomaly filtering |
| K-Means          | Unsupervised | Simple, good for profiling | Poor with high-dimensional data | Traffic clustering |

---

## 6. Future Enhancements
- Add **Graph Neural Networks (GNNs)** for relationship-based attack detection.  
- Integrate **Federated Learning** for privacy-preserving collaborative training.  
- Adaptive online learning (continuous retraining with new flows).  

---

📌 **End of Document**
