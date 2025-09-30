# 📑 Software Requirements Specification (SRS)  
**Project:** Anomulex – AI-based Intrusion Detection System (IDS)  
**Version:** v0.1.0-alpha  
**Date:** 2025-09-30  

---

## 1. Introduction

### 1.1 Purpose
The purpose of this SRS is to define the functional and non-functional requirements of **Anomulex**, an AI-powered Intrusion Detection System.  
The document ensures clarity between stakeholders (researchers, SOC teams, and developers) and provides a baseline for development and validation.  

### 1.2 Scope
- Detect **intrusions and anomalies** in network traffic.  
- Support flow-based monitoring using **NetFlow, IPFIX, CICFlowMeter**.  
- Apply **Machine Learning** to classify traffic into *benign* and *attack categories*.  
- Provide **alerts, dashboards, and reports** for analysts.  
- Operate in both **offline (batch)** and **online (real-time)** modes.  

### 1.3 Definitions, Acronyms, Abbreviations
- **IDS** – Intrusion Detection System  
- **SOC** – Security Operations Center  
- **ML** – Machine Learning  
- **DoS/DDoS** – Denial of Service / Distributed Denial of Service  
- **RBAC** – Role-Based Access Control  

---

## 2. Overall Description

### 2.1 Product Perspective
Anomulex integrates into a network monitoring ecosystem.  
It collects metadata flows, extracts features, applies ML models, and generates alerts.  

[Traffic Source] → [Flow Collector] → [Feature Extractor] → [ML Engine] → [Alerts/Dashboard]

### 2.2 Product Functions
- Collect bidirectional flow records.  
- Extract features (packet size, duration, inter-arrival time, etc.).  
- Train ML models using labeled datasets.  
- Perform inference in real-time.  
- Generate alerts & send them to dashboards.  

### 2.3 User Classes and Characteristics
- **SOC Analysts** – Monitor dashboards, investigate alerts.  
- **Researchers** – Test models, tune features.  
- **System Administrators** – Manage deployment, scaling, security.  

### 2.4 Constraints
- Must run on **Linux (Ubuntu 22.04+)** or Docker.  
- Requires high throughput support (≥ 10K flows/sec).  
- Metadata only (no payload inspection).  

### 2.5 Assumptions and Dependencies
- Traffic exported via NetFlow/IPFIX is accurate.  
- Training datasets are representative of real-world attacks.  
- Relies on Kafka/Elasticsearch for online mode.  

---

## 3. System Features

### 3.1 Flow Collection
- **Description:** Collect traffic metadata using nProbe or CICFlowMeter.  
- **Priority:** High  

### 3.2 Feature Extraction
- **Description:** Extract statistical features from flows.  
- **Priority:** High  

### 3.3 Machine Learning Detection
- **Description:** Classify flows into benign or attack categories.  
- **Priority:** High  

### 3.4 Alerts & Dashboard
- **Description:** Display detection results via UI (ntopng/custom).  
- **Priority:** Medium  

### 3.5 Explainability
- **Description:** Provide SHAP/LIME explanations for decisions.  
- **Priority:** Medium  

---

## 4. Non-Functional Requirements

- **Performance:** Detect anomalies within ≤ 1 sec latency.  
- **Scalability:** Support distributed workers.  
- **Security:** Encrypt flow export (TLS). Enforce RBAC.  
- **Usability:** Clear dashboard for non-expert users.  
- **Reliability:** 99.5% uptime target.  

---

## 5. External Interface Requirements

### 5.1 User Interfaces
- Web-based dashboard (alerts, graphs, reports).  

### 5.2 Hardware Interfaces
- x86_64 server with minimum: 8 cores, 16GB RAM, 200GB storage.  

### 5.3 Software Interfaces
- Python, Docker, Kafka, Elasticsearch, ntopng.  

### 5.4 Communication Interfaces
- NetFlow v9, IPFIX, gRPC/REST APIs.  

---

## 6. Other Requirements
- Comply with **GDPR** (no storage of personal payloads).  
- Log anonymization for privacy.  
- Future extension: support NIDS → NIPS.  

---

## 7. Appendices
### Dataset References
- CIC-IDS2017  
- UNSW-NB15  
- Custom enterprise traffic  

---

📌 **End of Document**
