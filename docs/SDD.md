# 🛠️ System Design Document (SDD)  
**Project:** Anomulex – AI-based Intrusion Detection System (IDS)  
**Version:** v0.1.0-alpha   
**Date:** 2025-09-30  

---

## 1. Introduction

### 1.1 Purpose
This document describes the architecture, components, and design choices of **Anomulex**.  
It ensures that the system is **scalable, modular, and maintainable**, while meeting the requirements defined in the SRS.  

### 1.2 Scope
Anomulex detects anomalies in **network traffic flows** using **Machine Learning**.  
It integrates with existing flow exporters (nProbe, NetFlow/IPFIX devices) and provides dashboards for analysts.  

---

## 2. System Overview

            ┌───────────────────┐
            │ Traffic Sources   │
            │ (Routers, Switch) │
            └─────────┬─────────┘
                      │
               NetFlow/IPFIX
                      │
            ┌─────────▼─────────┐
            │ Flow Collector     │
            │ (nProbe/CICFM)    │
            └─────────┬─────────┘
                      │
             Feature Extraction
                      │
            ┌─────────▼─────────┐
            │ ML Engine          │
            │ - Training         │
            │ - Inference        │
            └─────────┬─────────┘
                      │
               Alerts & Reports
                      │
            ┌─────────▼─────────┐
            │ Dashboard/Analyzer │
            │ (ntopng/ELK)      │
            └───────────────────┘

---

## 3. Architecture

### 3.1 High-Level Architecture
- **Collector Layer**: Receives NetFlow/IPFIX data.  
- **Processing Layer**: Parses flows, extracts metadata.  
- **Detection Layer**: ML-based detection (classification + anomaly detection).  
- **Presentation Layer**: Visual dashboards, alerting, REST APIs.  

### 3.2 Deployment View
- Supports **Dockerized microservices**.  
- Each component (Collector, ML Engine, Dashboard) runs as a container.  
- Kafka bus connects Collector → ML Engine → Analyzer.  

---

## 4. Module Design

### 4.1 Flow Collector
- **Responsibilities:** Capture NetFlow/IPFIX, normalize into JSON.  
- **Tech Stack:** nProbe, CICFlowMeter.  

### 4.2 Feature Extractor
- **Responsibilities:** Convert raw flow to statistical features.  
- **Examples:** duration, bytes, packets, inter-arrival time.  
- **Tech Stack:** Python, Pandas, Scikit-learn preprocessing.  

### 4.3 ML Engine
- **Responsibilities:**  
  - Train supervised models (RandomForest, XGBoost).  
  - Support anomaly detection (Autoencoder, Isolation Forest).  
  - Expose REST API for inference.  
- **Tech Stack:** Python, PyTorch, Scikit-learn.  

### 4.4 Dashboard / Analyzer
- **Responsibilities:** Display alerts, charts, reports.  
- **Tech Stack:** ntopng / Kibana / custom React dashboard.  

---

## 5. Data Flow Design

### 5.1 Offline Training
1. Import datasets (CIC-IDS2017, UNSW-NB15).  
2. Preprocess & feature engineer.  
3. Train ML models, evaluate accuracy.  
4. Save models (`.pkl` or ONNX format).  

### 5.2 Online Detection
1. Router exports NetFlow/IPFIX.  
2. Collector parses and forwards flows.  
3. Feature Extractor prepares ML input.  
4. ML Engine classifies → Benign / Attack.  
5. Alerts sent to dashboard.  

---

## 6. Security Considerations
- TLS encryption for flow export.  
- RBAC for dashboard access.  
- Logs anonymized to avoid sensitive data exposure.  
- Secure model update pipeline.  

---

## 7. Scalability and Performance
- Horizontal scaling via Kafka partitions.  
- Multiple ML workers for inference.  
- Load balancing for dashboard queries.  
- Target throughput: ≥ 50K flows/sec.  

---

## 8. Future Enhancements
- Extend from IDS → IPS (inline blocking).  
- Add federated learning for distributed environments.  
- Expand protocol parsers (DNS, HTTP/2, QUIC).  

---

📌 **End of Document**
