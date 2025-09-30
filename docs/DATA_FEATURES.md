# 📊 Data Features Specification  
**Project:** Anomulex – AI-based Intrusion Detection System (IDS)  
**Version:** v0.1.0-alpha  
**Date:** 2025-09-30  

---

## 1. Introduction
The **feature set** is the foundation of the ML-based anomaly detection in Anomulex.  
Features are extracted from network traffic flows (NetFlow/IPFIX, CICFlowMeter, or nProbe).  
They represent **metadata/statistics** about communications between hosts, rather than raw packet data.  

---

## 2. Feature Categories

### 2.1 Basic Flow Identifiers
| Feature | Description |
|---------|-------------|
| `src_ip` | Source IP address |
| `dst_ip` | Destination IP address |
| `src_port` | Source port |
| `dst_port` | Destination port |
| `protocol` | Transport protocol (TCP/UDP/ICMP) |
| `flow_duration` | Time between first and last packet in the flow |

---

### 2.2 Packet and Byte Counters
| Feature | Description |
|---------|-------------|
| `tot_fwd_pkts` | Number of packets sent in forward direction |
| `tot_bwd_pkts` | Number of packets sent in backward direction |
| `tot_fwd_bytes` | Total bytes in forward direction |
| `tot_bwd_bytes` | Total bytes in backward direction |
| `pkt_len_min` | Minimum packet length |
| `pkt_len_max` | Maximum packet length |
| `pkt_len_mean` | Average packet length |
| `pkt_len_std` | Standard deviation of packet lengths |

---

### 2.3 Time-based Features
| Feature | Description |
|---------|-------------|
| `fwd_pkt_iat_min` | Minimum inter-arrival time forward |
| `fwd_pkt_iat_max` | Maximum inter-arrival time forward |
| `fwd_pkt_iat_mean` | Average inter-arrival time forward |
| `fwd_pkt_iat_std` | Std deviation of inter-arrival times forward |
| `bwd_pkt_iat_min` | Minimum inter-arrival time backward |
| `bwd_pkt_iat_mean` | Average inter-arrival time backward |
| `flow_iat_mean` | Average inter-arrival time across flow |

---

### 2.4 TCP Flag Counters
| Feature | Description |
|---------|-------------|
| `fwd_psh_flags` | Count of PSH flags in forward direction |
| `bwd_psh_flags` | Count of PSH flags in backward direction |
| `fwd_urg_flags` | Count of URG flags forward |
| `bwd_urg_flags` | Count of URG flags backward |
| `fwd_fin_flags` | FIN flags count |
| `fwd_syn_flags` | SYN flags count |
| `fwd_rst_flags` | RST flags count |
| `fwd_ack_flags` | ACK flags count |

---

### 2.5 Flow Behavior Ratios
| Feature | Description |
|---------|-------------|
| `pkt_size_avg` | Average packet size in the flow |
| `fwd_pkt_len_mean` | Avg length of forward packets |
| `bwd_pkt_len_mean` | Avg length of backward packets |
| `fwd_bwd_ratio` | Ratio of forward to backward packets |
| `bwd_fwd_ratio` | Ratio of backward to forward packets |
| `pkt_rate` | Packets per second |
| `byte_rate` | Bytes per second |

---

### 2.6 Advanced Statistical Features
| Feature | Description |
|---------|-------------|
| `active_mean` | Average time a flow was active |
| `idle_mean` | Average idle time between flows |
| `flow_bytes_s` | Number of bytes per second |
| `flow_pkts_s` | Number of packets per second |
| `subflow_fwd_pkts` | Packets in sub-flows forward |
| `subflow_bwd_pkts` | Packets in sub-flows backward |

---

## 3. Dataset Comparison

- **CICFlowMeter Features** → ~77 features (rich statistics, TCP flags, IATs).  
- **NetFlow v5** → ~20 fixed fields (basic counters only).  
- **IPFIX (NetFlow v9 standard)** → flexible, supports vendor-specific fields (e.g., HTTP method, TLS SNI).  

Anomulex uses **CICFlowMeter features** for ML training and testing.  
In real-time, **IPFIX/NetFlow** features are mapped to a **reduced compatible subset**.  

---

## 4. Feature Selection Strategy
- Use **statistical correlation + mutual information** to reduce redundancy.  
- Apply **Recursive Feature Elimination (RFE)** during training.  
- Keep ~20–30 features that maximize detection accuracy.  

---

## 5. Future Enhancements
- Add **Layer 7 metadata** (HTTP method, DNS query, TLS JA3 fingerprint).  
- Leverage **IPFIX custom fields** for richer ML input.  
- Dynamic feature selection per protocol.  

---

📌 **End of Document**
