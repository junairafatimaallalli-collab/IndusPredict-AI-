# IndusPredict AI – Research Methodology and System Architecture

## 1. Executive Summary & Research Foundations
**IndusPredict AI** is an intelligent, explainable, and adaptive predictive maintenance (PdM) platform engineered for industrial cyber-physical systems (CPS), rotational machinery, and critical industrial assets (e.g., centrifugal pumps, induction motors, screw compressors, and multi-axis industrial robots).

Traditional industrial maintenance methodologies operate under either:
1. **Corrective / Reactive Maintenance (Run-to-Failure)**: Generates catastrophic unplanned downtime, secondary structural damage, and severe safety hazards.
2. **Preventive / Periodic Maintenance (Time-Based)**: Incurs unnecessary downtime, premature replacement of functional components, and high labor overhead without preventing early unexpected infant mortality failures.

IndusPredict AI implements **Condition-Based Predictive Maintenance (CBM/PdM)** compliant with **ISO 13374** (*Condition monitoring and diagnostics of machine systems*), structured around four scientific pillars:
- **Real-Time Anomaly Detection & Failure Classification**: Supervised and semi-supervised classifiers (Random Forest, XGBoost, LightGBM, Logistic Regression) predicting fault probability across mechanical and electrical failure modes.
- **Prognostics & Remaining Useful Life (RUL) Estimation**: Deep recurrent neural networks (PyTorch Bidirectional LSTM / GRU) with sequence windowing, mapping sensor degradation trajectories to remaining operational cycles.
- **Explainable Artificial Intelligence (XAI)**: Game-theoretic feature attribution via **SHAP (SHapley Additive exPlanations)**, translating complex black-box model decisions into actionable engineering insights for maintenance crews without confusing mathematical attribution with confirmed physical root-cause mechanics.
- **Adaptive Drift Monitoring & Closed-Loop Planning**: Continuous monitoring of streaming sensor distributions and model performance via **ADWIN (Adaptive Windowing)** and **Page-Hinkley** statistical drift tests, coupled with an automated work order management system incorporating strict human-in-the-loop validation.

---

## 2. Literature Review & Theoretical Foundations

### 2.1 Anomaly Detection & Failure Classification in Industrial Machinery
- **ISO 13374 Architecture**: Establishes standard processing blocks: Data Acquisition (DA), Data Manipulation (DM), State Detection (SD), Health Assessment (HA), Prognostic Assessment (PA), and Advisory Generation (AG). IndusPredict AI mirrors this standard within its modular software architecture.
- **UCI AI4I 2020 Predictive Maintenance Dataset**: Synthetic dataset reflecting real-world industrial milling machine operations (Matzka, 2020). Features include Air Temperature [K], Process Temperature [K], Rotational Speed [rpm], Torque [Nm], and Tool Wear [min]. Crucially, it models distinct failure modes: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), and Random Failure (RNF). Class imbalance (typically ~3.4% failure rate) requires optimization using Precision-Recall Area Under Curve (PR-AUC), Matthews Correlation Coefficient (MCC), and cost-sensitive loss functions rather than deceptive raw accuracy.
- **MetroPT-3 Dataset**: Real-world industrial compressor telemetry gathered from Lisbon Metro trains (Brás & Soares, 2021). Evaluates vibration, oil temperature, motor current, and reservoir air pressure, documenting actual failure events (air leakages, valve blockages, compressor shutdowns).

### 2.2 Deep Learning for Prognostics (RUL Estimation)
- **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**: Established by Saxena et al. (NASA Ames Research Center). Contains run-to-failure turbofan engine trajectories under diverse operating regimes.
- **Degradation Modeling**: Equipment remains healthy during initial operating cycles; wear accelerates past an inflection point $t_{inflection}$. Piecewise linear RUL targets ($RUL_{target} = \min(RUL_{actual}, RUL_{max})$) prevent premature penalization of stable healthy states.
- **Recurrent Prognostic Architectures**: Multi-layer Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks capture temporal dependencies across sliding windows $W = [x_{t-w+1}, \dots, x_t]$, overcoming the memory degradation of traditional Markovian models.
- **Performance Metrics**: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and asymmetric NASA Scoring Metric:
  $$S = \sum_{i=1}^{N} s_i, \quad s_i = \begin{cases} e^{-\frac{d_i}{13}} - 1 & \text{if } d_i < 0 \text{ (late prediction)} \\ e^{\frac{d_i}{10}} - 1 & \text{if } d_i \ge 0 \text{ (early prediction)} \end{cases}$$
  where $d_i = \hat{RUL}_i - RUL_i$. Late predictions are penalized significantly more heavily due to catastrophic failure risk.

### 2.3 Explainability via Cooperative Game Theory (SHAP)
- **Lundberg & Lee (2017)**: Unified framework connecting LIME, DeepLIFT, and Shapley values. For a model prediction $f(x)$, the explanation model $g(z')$ is linear:
  $$g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i$$
  where $\phi_i$ denotes the Shapley attribution value for feature $i$.
- **Industrial Safety Consideration**: SHAP provides statistical attribution regarding which sensor deviations pushed the model probability across a risk threshold. The application interface must clearly present these attributions as model feature importance rather than definitive physical metallurgical or thermodynamic root-causes, prompting physical inspection rather than blind action.

### 2.4 Streaming Concept Drift & Model Degradation
- **Distribution Shift in Industrial IoT**: Caused by mechanical retooling, seasonal ambient temperature shifts, sensor calibration decay, or operational load adjustments.
- **ADWIN (Adaptive Windowing)** (Bifet & Gavaldà, 2007): Dynamically resizes observation windows when sub-window means differ beyond a Hoeffding-bound threshold:
  $$\epsilon_{cut} = \sqrt{\frac{1}{2m} \cdot \ln\left(\frac{4n}{\delta}\right)}$$
- **Page-Hinkley Test**: Sequential analysis technique monitoring the cumulative deviation of sensor streams from their running mean, detecting abrupt mean increases indicative of abrupt mechanical degradation.

---

## 3. System Architecture Specification

IndusPredict AI is organized into **seven decoupled architectural layers**, guaranteeing testability, separation of concerns, and clean industrial extensibility:

```
+-----------------------------------------------------------------------------------+
|                            1. PRESENTATION LAYER (PySide6)                        |
|  - Modern Dark Navy & Charcoal Theme (#0D1117, #161B22, #00F2FE, #00C9A7)         |
|  - QStackedWidget Page Navigation:                                                |
|    * Login & RBAC       * Main Dashboard         * Machine Management             |
|    * Live Sensors       * Failure Prediction     * RUL Prognostics (LSTM)         |
|    * Digital Twin       * Explainable AI (SHAP)  * Adaptive Maintenance & Drift   |
|    * Reports & Analytics                                                          |
+-----------------------------------------------------------------------------------+
                                         | (Events, Signals, Data Binds)
                                         v
+-----------------------------------------------------------------------------------+
|                           2. APPLICATION LOGIC LAYER                              |
|  - AuthService (RBAC, Session tokens)      - MachineService (CRUD, specs)         |
|  - TelemetryService (Streams, thresholds)  - PredictionService (Multi-model)      |
|  - PrognosticsService (PyTorch RUL)        - DigitalTwinService (What-if sim)     |
|  - XAIService (SHAP explanations)          - MaintenanceService (Work orders)     |
|  - DriftMonitorService (ADWIN / P-H)       - ReportService (PDF, CSV, Excel)      |
+-----------------------------------------------------------------------------------+
      |                      |                             |                    |
      v                      v                             v                    v
+---------------+  +---------------------+  +--------------------+  +---------------+
| 3. ML/DL LAYER|  | 4. DATA PROCESSING  |  | 5. DATABASE LAYER  |  | 6. IoT LAYER  |
| - Scikit-learn|  | - Window Generator  |  | - SQLAlchemy 2.0   |  | - Paho MQTT   |
| - XGBoost     |  | - Imputation        |  | - SQLite (WAL mode)|  | - OPC UA Client|
| - LightGBM    |  | - Robust Scaler     |  | - Repositories     |  | - Simulated    |
| - PyTorch LSTM|  | - Feature Extractor |  | - DB Backups       |  |   Sensor Gen  |
| - SHAP Engine |  | - Anomaly Scorer    |  | - Audit Trail Logs |  | - Safe Read-only|
+---------------+  +---------------------+  +--------------------+  +---------------+
                                                                            |
+---------------------------------------------------------------------------+-------+
|                                7. REPORTING LAYER                                 |
|  - ReportLab PDF Engine (Tabular summaries, KPI badges, maintenance certifications)|
|  - OpenPyXL / XlsxWriter (Detailed multi-sheet sensor dumps & telemetry logs)    |
|  - CSV Streaming Exporter (High-throughput time-series export)                    |
+-----------------------------------------------------------------------------------+
```

---

## 4. Security & Industrial Safety Governance

1. **Read-Only Industrial Interfacing**: The industrial communication adapters (`paho-mqtt` and `asyncua`) are strictly configured with read-only scopes. Under no circumstances does the desktop software emit control commands, start/stop actuators, or override Programmable Logic Controller (PLC) safety interlocks.
2. **Human-in-the-Loop Maintenance Authorization**: Automated failure predictions and RUL countdowns generate *recommended* draft work orders. Only authenticated users with the `Maintenance Engineer` or `Administrator` role can approve, schedule, modify, or close a work order.
3. **Role-Based Access Control (RBAC)**:
   - **Administrator**: User management, machine configuration, model deployment approval, audit trail inspection.
   - **Maintenance Engineer**: Live sensor monitoring, prediction reviews, work order creation and execution, what-if digital twin simulations.
   - **Researcher**: Model training, RUL hyperparameter tuning, SHAP explainability analysis, drift threshold calibration, dataset uploads.
   - **Viewer**: Read-only observation of dashboard KPIs, health trends, and generated reports.
4. **Cryptographic Security**: Passwords hashed using `bcrypt` with adaptive salt cost factor. Database includes an immutable `AuditLog` recording every login, machine modification, work order approval, and model retraining event with timestamps and user identifiers.
