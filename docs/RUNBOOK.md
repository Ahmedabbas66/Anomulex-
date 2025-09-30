# Runbook for Anomulex

Operational procedures for deploying, managing, and troubleshooting **Anomulex (AI-based IDS).**

---

## 1. System Overview

- **Purpose:** Detect anomalous/malicious traffic using AI.  
- **Components:**
  - **Collector:** Ingests network traffic (PCAP, NetFlow, IPFIX).  
  - **Parser:** Extracts features from traffic.  
  - **Analyzer:** ML-based anomaly detection.  
  - **Dashboard:** Displays alerts, logs, health.  

---

## 2. Prerequisites

- **OS:** Ubuntu 22.04 LTS  
- **Dependencies:**
  - Docker & Docker Compose  
  - Python 3.10+  
  - Redis  
  - PostgreSQL or MongoDB  
- **Access:**
  - Root/sudo privileges  
  - Promiscuous mode enabled NIC  

---

## 3. Starting the System

### 3.1 Deployment
```bash
git clone https://github.com/your-org/anomulex.git
cd anomulex
docker-compose up -d
````

### 3.2 Verify

```bash
docker ps
```

**Expected containers:**

* anomulex-collector
* anomulex-parser
* anomulex-analyzer
* anomulex-dashboard

### 3.3 Dashboard

* **URL:** [http://localhost:8080](http://localhost:8080)
* **Credentials:** `admin / admin` (change immediately)

---

## 4. Normal Operations

* **Check Health**

```bash
docker logs anomulex-analyzer --tail 20
```

* **Restart Service**

```bash
docker-compose restart <service-name>
```

* **Update System**

```bash
git pull origin main
docker-compose build
docker-compose up -d
```

---

## 5. Incident Handling

### 5.1 Responding to Alerts

1. Open dashboard.
2. Identify IPs, ports, timestamps.
3. Cross-check with SIEM/firewall.
4. Escalate if attack confirmed.

### 5.2 False Positives

1. Mark alert as false positive.
2. Review training dataset.
3. Retrain with benign traffic if needed.

---

## 6. Troubleshooting

* **Collector**

  * Symptom: No traffic ingested.
  * Check:

    ```bash
    tcpdump -i eth0
    ```

* **Parser**

  * Symptom: Features missing.
  * Fix: Check `config/features.yaml`.

* **Analyzer**

  * Symptom: No alerts.
  * Fix:

    * Check model path: `models/anomulex_model.pkl`.
    * Restart analyzer.

* **Dashboard**

  * Symptom: UI not loading.
  * Fix:

    ```bash
    docker-compose logs anomulex-dashboard
    ```

---

## 7. Maintenance

* **Logs:** Stored in `/var/log/anomulex`, rotated weekly.
* **Database Backup:**

  ```bash
  pg_dump anomulex_db > backup.sql
  ```
* **Retrain Model:** Every 3 months with new data.

---

## 8. Emergency Shutdown

```bash
docker-compose down
```

---

## 9. Contacts

* **Owner:** Security Operations Team
* **Escalation Path:**

  * Security Engineer On-Call
  * SOC Manager
  * CTO

```

