# 🏗️ System Architecture – Anomulex
**Project:** Anomulex – AI-based Intrusion Detection System (IDS)  
**Version:** v0.1.0-alpha  
**Author:** Ahmed Adel  
**Date:** 2025-09-30  

---

## 1. Introduction
The system architecture of **Anomulex** defines how raw network traffic is collected, processed, analyzed, and acted upon.  
It combines **flow-based monitoring** (NetFlow/IPFIX/CICFlowMeter) with **AI-powered anomaly detection**, enabling real-time detection of malicious traffic.  

---

## 2. High-Level Architecture

(4) Benign Flow (5) Malicious Flow
Allow / Log traffic Alert / Drop / Block


---

## 3. Components

### 3.1 Exporter (Probe)
- Role: Captures traffic and generates flow metadata.  
- Examples:  
  - **nProbe** (software probe for NetFlow/IPFIX).  
  - **CICFlowMeter** (research-focused, ML-ready).  
  - Routers/Switches with NetFlow/IPFIX export enabled.  
- Output: Flow records containing **L3/L4/L7 metadata**.  

---

### 3.2 Collector
- Role: Receives flow records from exporters.  
- Functions:  
  - Store flows in a database (Elasticsearch, InfluxDB, or CSV).  
  - Forward to pipeline (Kafka, RabbitMQ).  
  - Apply preliminary filtering (e.g., sampling).  
- Example tools: **ntopng**, ELK Stack, custom Kafka consumer.  

---

### 3.3 Parser & Feature Extractor
- Role: Transforms raw flow records into ML-ready features.  
- Tasks:  
  - Normalize timestamps, packet lengths, rates.  
  - Derive additional statistical features (IAT, packet variance).  
  - Encode categorical variables (protocol, flags).  
- Input: Raw NetFlow/IPFIX or CICFlowMeter CSV.  
- Output: Feature vectors → ML model input.  

---

### 3.4 ML/AI Engine
- Role: Core intelligence of Anomulex.  
- Functions:  
  - Supervised classification (RF, XGBoost, DL).  
  - Unsupervised anomaly detection (Autoencoder, Isolation Forest).  
  - Assigns confidence score to each flow.  
- Output: **Normal / Attack** decision.  

---

### 3.5 Decision Engine
- Role: Converts ML outputs into security actions.  
- Actions:  
  - **Benign** → forward traffic, log event.  
  - **Malicious** → generate alerts, trigger mitigation.  
- Integration:  
  - Firewalls (iptables, Windows Firewall API).  
  - SIEM systems (Splunk, ELK, Graylog).  
  - SOAR automation (blocking, ticketing).  

---

## 4. Data Flow
1. **Traffic Capture** – Packets flow through a router/switch/nProbe.  
2. **Flow Export** – Metadata sent via NetFlow/IPFIX to Collector.  
3. **Storage/Queue** – Collector saves flows in DB or streams via Kafka.  
4. **Feature Extraction** – Parser transforms flows into ML feature vectors.  
5. **Inference** – ML model classifies the flow.  
6. **Action** – Decision Engine decides: allow, alert, or block.  

---

## 5. Deployment Options

- **Offline Mode**  
  - Input: PCAP or CSV flow datasets.  
  - Use case: Research, model training, retrospective analysis.  

- **Online Mode**  
  - Input: Real-time NetFlow/IPFIX streams.  
  - Use case: SOC operations, continuous anomaly detection.  

---

## 6. Future Enhancements
- Support for **federated learning** across multiple collectors.  
- Real-time **streaming analytics** with Apache Flink.  
- Integration with **Zeek** for deep packet inspection (DPI).  
- Adaptive response: automatic firewall rules injection.  

---

📌 **End of Document**
