# IndusPredict AI – Intelligent, Explainable & Adaptive Predictive Maintenance System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![GUI-PySide6](https://img.shields.io/badge/GUI-PySide6%20(Qt6)-teal.svg)](https://www.qt.io/)
[![DeepLearning-PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%20LSTM-orange.svg)](https://pytorch.org/)
[![ML-Scikit--Learn%20%7C%20XGBoost%20%7C%20LightGBM](https://img.shields.io/badge/ML-RF%20%7C%20XGB%20%7C%20LGBM-brightgreen.svg)](https://scikit-learn.org/)
[![XAI-SHAP](https://img.shields.io/badge/XAI-SHAP%20TreeExplainer-purple.svg)](https://shap.readthedocs.io/)
[![Web-Streamlit](https://img.shields.io/badge/Web%20Portal-Streamlit-red.svg)](https://streamlit.io/)
[![Standard-ISO%2013374](https://img.shields.io/badge/Standard-ISO%2013374%20CBM-navy.svg)](https://www.iso.org/standard/41870.html)
[![Tests-72%20Passed](https://img.shields.io/badge/Tests-72%20Passed-success.svg)](tests/)

---

## 🌟 Executive Overview

**IndusPredict AI** is a state-of-the-art, research-grade industrial predictive maintenance platform engineered to prevent catastrophic machinery breakdowns in heavy manufacturing and continuous process plants. 

Unlike traditional condition monitoring systems that merely flag threshold exceedances *after* severe damage has occurred, IndusPredict AI combines **cyber-physical IoT telemetry**, **probabilistic failure classification**, **deep recurrent sequence prognostics (PyTorch LSTM)**, **game-theoretic explainability (SHAP)**, **parametric digital twin simulation**, and **streaming concept drift monitoring (ADWIN & Page-Hinkley)**.

The solution provides both:
1. **A Full-Featured Python Desktop Application** (PySide6 / Qt6) with a dark industrial cyber-physical aesthetic (`main.py`).
2. **An Executive Web Portal & Showcase** (Streamlit) for browser-based asset health monitoring and report generation (`web_app.py`).

---

## 🏗️ 7-Layer Decoupled System Architecture

The codebase strictly enforces modular separation of concerns:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   1. PRESENTATION LAYER (GUI & Web)                    │
│   PySide6 Desktop Application (10 Pages)  │  Streamlit Web Showcase   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                      2. APPLICATION LOGIC LAYER                        │
│   AuthService │ MachineService │ TelemetryService │ PredictionService │
│   PrognosticsService │ DigitalTwinService │ XAIService │ MaintService │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    3. MACHINE LEARNING & PROGNOSTICS                   │
│   • Random Forest (98.6% Acc, 0.797 F1)  • XGBoost (0.792 F1)          │
│   • LightGBM (0.787 F1)                  • Logistic Regression         │
│   • PyTorch 2-Layer LSTM RUL Engine (NASA C-MAPSS FD001)               │
│   • SHAP TreeExplainer & KernelExplainer Local/Global Attributions     │
│   • Streaming Concept Drift: ADWIN & Page-Hinkley & KS-Test            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                  4. INDUSTRIAL DATA & TELEMETRY LAYER                  │
│   • Physics Multi-Sensor Simulator (5 Fault Profiles)                  │
│   • Read-Only Paho MQTT Client with Signal Forwarding                  │
│   • OPC UA Client (asyncua) for PLC / SCADA Ingestion                  │
│   • CSV & Excel Imputation Engine (Data Leakage Guards)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                  5. DATABASE & PERSISTENCE LAYER                       │
│   • SQLite with SQLAlchemy 2.0 ORM (16 Normalized Tables)              │
│   • Write-Ahead Logging (WAL) Mode & Foreign Key Enforcement           │
│   • Zero-Downtime Hot Online Backup (backup_database)                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                6. REPORTING & COMPLIANCE EXPORT LAYER                  │
│   • ReportLab PDF Generator (ISO 13374 Sign-offs & Certifications)     │
│   • XlsxWriter Multi-Sheet Formatted Excel Workbooks                   │
│   • Standardized CSV Streams                                           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ 10 Complete Desktop Application Pages

| # | Page Name | Core Functionality | Database Integration |
|---|---|---|---|
| **0** | **Login & Authentication** | Secure bcrypt hashing, RBAC (Administrator, Maintenance Engineer, Researcher, Viewer), session token expiration. | `users`, `roles`, `audit_logs` |
| **1** | **Operations Dashboard** | 6 dynamic KPI cards, rolling health curves, failure risk breakdown, critical alert logs, machine filters. | `machines`, `predictions`, `anomaly_events` |
| **2** | **Machine Registry** | Full CRUD, sub-component wear tracking, multi-sensor calibration, criticality matrix, location tagging. | `machines`, `machine_components`, `sensors` |
| **3** | **Live Sensor Monitoring** | 7 physical variables (Temp, Vib, Press, Speed, Current, Torque, Hours), 2 Hz waveforms, simulator with 5 fault profiles, MQTT subscriber, CSV upload. | `sensors`, `sensor_readings`, `anomaly_events` |
| **4** | **Failure Prediction Engine** | Multi-model classifier (RF, XGBoost, LightGBM, Logistic Regression), uncertainty intervals, physical diagnosis (HDF, PWF, OSF, TWF). | `predictions`, `model_versions`, `model_evaluations` |
| **5** | **RUL Prognostics (LSTM)** | PyTorch deep recurrent LSTM, sliding window $W=30$, piecewise degradation, 95% MC Dropout uncertainty, NASA asymmetric score comparison. | `rul_estimates`, `model_versions` |
| **6** | **Digital Twin Simulator** | Virtual machine replicas, what-if parametric stress perturbation, Arrhenius thermal & Archard mechanical wear models, scenario archiving. | `simulation_scenarios`, `machines` |
| **7** | **Explainable AI (SHAP)** | Game-theoretic Shapley attributions ($\phi_i$), local waterfall plots, global feature importance rankings, normal vs abnormal benchmarks, physical disclaimer. | `predictions`, background datasets |
| **8** | **Adaptive Maintenance Planner** | Automated risk-prioritized work orders, lifecycle approval, downtime estimates, ADWIN and Page-Hinkley streaming drift detection. | `work_orders`, `maintenance_history`, `drift_events` |
| **9** | **Certified Reports & Analytics** | Executive PDF reports (ReportLab), multi-sheet formatted Excel (XlsxWriter), CSV datasets, certified engineer sign-off blocks. | All database tables |

---

## 🔬 Genuine Datasets & Empirical Benchmarks

### 1. UCI AI4I 2020 Predictive Maintenance (Classification)
- **10,000 real industrial machine records** (339 failure events, 3.39% class imbalance).
- Evaluated with stratified 80/20 train/test split.
- **Empirical Results on Held-Out Test Set:**

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|---|---|
| **Random Forest** | **98.55%** | **76.00%** | **83.82%** | **0.7972** | **0.9852** | **0.8705** | **0.7907** |
| **XGBoost** | 98.50% | 75.00% | 83.82% | 0.7917 | 0.9795 | 0.8832 | 0.7852 |
| **LightGBM** | 98.40% | 71.95% | 86.76% | 0.7867 | 0.9754 | 0.8962 | 0.7821 |
| **Logistic Regression** | 87.05% | 19.09% | 86.76% | 0.3130 | 0.9437 | 0.4720 | 0.3702 |

### 2. NASA C-MAPSS Turbofan Degradation (Prognostics - FD001)
- **100 run-to-failure engines** (20,631 cycles).
- Strict engine-wise split: Units 1–80 for training, Units 81–100 strictly held-out test fleet.
- Sequence window $W=30$ consecutive cycles; piecewise linear RUL clipping at 125 cycles.
- **Empirical Prognostics Benchmark on Test Fleet:**

| Architecture | MAE (Cycles) | RMSE (Cycles) | NASA Score ($S$) | 95% PICP Coverage |
|---|---|---|---|---|
| **PyTorch 2-Layer LSTM (IndusPredict)** | **12.05** | **15.51** | **17,106.4** | **85.8%** |
| **Random Forest Baseline** | 19.41 | 25.11 | 110,067.1 | 71.2% |

*Note: The asymmetric NASA scoring function penalizes optimistic/late predictions exponentially more than early predictions.*

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation
Ensure Python 3.11 or higher is installed:
```bash
git clone https://github.com/your-org/induspredict-ai.git
cd "indusPredict AI"

# Install dependencies
pip install -r requirements.txt
```

### 2. Default Login Credentials (RBAC)
The database auto-seeds with 4 certified role profiles:
- **Administrator:** `admin` / `Admin@123` (Full system configuration, user management, work order approvals)
- **Maintenance Engineer:** `engineer` / `Engineer@123` (Work order execution, sensor thresholds, telemetry monitoring)
- **Researcher:** `researcher` / `Researcher@123` (Model benchmarks, SHAP explanations, RUL prognostics)
- **Viewer:** `viewer` / `Viewer@123` (Read-only dashboard and fleet status inspection)

### 3. Launching the Desktop Application
```bash
python main.py
```

### 4. Launching the Web Portal / Website
```bash
streamlit run web_app.py
```
Open your browser at `http://localhost:8501`.

### 5. Running the Automated Test Suite
```bash
pytest -v
```
**72 automated unit, integration, and headless GUI tests verify every phase with 100% passing results.**

---

## 🔒 Industrial Safety & Human-in-the-Loop Policy
1. **Read-Only Telemetry:** IndusPredict AI exclusively reads industrial telemetry via MQTT and OPC UA; it never issues autonomous physical control commands to machinery.
2. **Model Retraining Sign-Off:** Concepts drifts detected by ADWIN or Page-Hinkley are flagged for engineering inspection; unvalidated models are never deployed automatically into production.
3. **Physical Explainability Disclaimer:** Statistical SHAP attributions illuminate mathematical model reasoning but do not replace certified on-site mechanical and metallurgical inspections.

---
*Developed under ISO 13374 Condition Based Maintenance (CBM) & Prognostics Standards.*
