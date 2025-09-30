# Testing Guide for Anomulex

This document describes the testing strategy, test cases, and validation process for **Anomulex (AI-based IDS).**

---

## 1. Objectives

- Validate that Anomulex detects anomalies and intrusions accurately.  
- Ensure system stability under different traffic loads.  
- Verify false positive/false negative rates remain within acceptable limits.  
- Confirm integration with dashboards and alerting systems.  

---

## 2. Types of Tests

- **Unit Testing**  
  - Validate data parsing and feature extraction modules.  
  - Ensure ML model input/output integrity.  

- **Integration Testing**  
  - Validate communication between Collector → Parser → Analyzer → Dashboard.  
  - Test database writes and alert persistence.  

- **System Testing**  
  - Run end-to-end tests simulating real network traffic.  
  - Validate alerts in the dashboard.  

- **Performance Testing**  
  - Stress-test with high-volume NetFlow/IPFIX/PCAP data.  
  - Measure throughput and latency.  

- **Security Testing**  
  - Simulate attacks (DoS, port scanning, SQL injection).  
  - Ensure system flags anomalies and raises alerts.  

---

## 3. Test Environment

- **OS:** Ubuntu 22.04 LTS  
- **Tools:**
  - CICFlowMeter (for labeled datasets)  
  - tcpreplay (for replaying PCAP traffic)  
  - Wireshark/tcpdump (for packet verification)  
  - Locust or JMeter (for performance load testing)  
- **Datasets:**  
  - CIC-IDS2017  
  - UNSW-NB15  
  - Custom lab-generated attack traffic  

---

## 4. Test Cases

### 4.1 Collector Tests
- Inject synthetic traffic.  
- Verify packet/flow ingestion.  

### 4.2 Parser Tests
- Input: PCAP/NetFlow file.  
- Output: Extracted features in `features.csv`.  
- Validation: Compare against ground-truth dataset.  

### 4.3 Analyzer Tests
- Input: Features dataset.  
- Output: Alert classification (normal vs anomaly).  
- Validation: Compare predicted vs expected labels.  

### 4.4 Dashboard Tests
- Simulate alerts.  
- Verify alerts visible in UI with correct metadata (IP, port, timestamp).  

---

## 5. Acceptance Criteria

- Detection accuracy ≥ 95%.  
- False positive rate ≤ 5%.  
- System handles ≥ 1 Gbps flow data without packet loss.  
- Dashboard updates in real-time (< 2s delay).  

---

## 6. Regression Testing

- Re-run test suite after every code change.  
- Validate no new bugs are introduced.  
- Automate with CI/CD (GitHub Actions, GitLab CI, Jenkins).  

---

## 7. Reporting

- Test results logged in `/var/log/anomulex/testing/`.  
- Weekly summary reports generated automatically.  
- Critical failures escalated to Security Engineering team.  
