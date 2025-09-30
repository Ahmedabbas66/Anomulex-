# Anomulex  
**AI-based Network Intrusion Detection System (IDS)**  

Anomulex is an AI-powered Intrusion Detection System designed to detect anomalies and malicious activities in network traffic using flow-level metadata and machine learning.  
It leverages NetFlow/IPFIX/CICFlowMeter features, advanced ML models, and explainability techniques to provide accurate and interpretable intrusion detection.  

---

## 🚀 Overview
- **Problem**: Traditional IDS struggle against modern, evolving attacks.  
- **Solution**: Anomulex applies AI/ML to detect anomalies in real-time with high accuracy.  
- **Scope**: Intrusion **Detection** only (IDS), not prevention.  
- **Users**: SOC analysts, cybersecurity researchers, and engineers.  

---

## 🏗️ Architecture

[Network Traffic]
→
[Collector: nProbe / CICFlowMeter]
→
[Parser & Feature Extractor]
→
[AI/ML Engine: Classification]
→
[Alert Manager → Dashboard (ntopng/Custom UI)]

---

## 📋 Features
- Flow collection via **NetFlow, IPFIX, or CICFlowMeter**.  
- Feature extraction from bidirectional flows.  
- ML classification: benign vs multiple attack types (DoS, PortScan, Botnet, DDos, etc.).  
- Real-time alerting and visualization.  
- Explainability with **SHAP / LIME**.  

---

## ⚙️ Requirements
- **Software**:  
  - Python 3.10+  
  - Docker & Docker Compose  
  - Kafka / Elasticsearch (for real-time pipeline)  
  - ntopng (optional, for visualization)  

- **Datasets**:  
  - CIC-IDS2017, UNSW-NB15, or custom NetFlow/IPFIX exports.  

---

## 📊 Machine Learning
- Algorithms: Random Forest, XGBoost, LSTM (experimental).  
- Metrics: Accuracy, Precision, Recall, F1, AUC.  
- Example performance:  
  - Precision: 97%  
  - Recall: 95%  
  - F1-score: 96%  

---

## 🔧 Deployment
- **Offline mode**: batch analysis of PCAP or CSV flow datasets.  
- **Online mode**: real-time detection with Kafka + nProbe + ntopng.  
- Scalable inference pipeline with distributed workers.  

---

## 📑 Documentation
- [System Requirements Specification (SRS)](docs/SRS.md)  
- [System Design Document (SDD)](docs/SDD.md)
- [System Architecture](docs/SYSTEM_ARCHITECTURE.md) 
- [Data & Features](docs/DATA_FEATURES.md)  
- [Model Training & Evaluation](docs/MODELS.md)  
- [Testing Plan](docs/TESTING.md)  
- [Runbook (Ops Guide)](docs/RUNBOOK.md)
 

---

## 🔒 Security & Privacy
- Metadata only (no raw payload storage).  
- TLS-encrypted flow export.  
- Role-based access control (RBAC).  
- Logs anonymization.  

---

## 📈 Monitoring
- Flow ingestion rate.  
- Model inference latency.  
- Alert frequency.  
- Automatic retraining for concept drift.  

---

## 📌 Versioning
- Semantic Versioning: `v0.1.0-alpha`  
- Models tagged separately: `ml-v20250930`  

---







