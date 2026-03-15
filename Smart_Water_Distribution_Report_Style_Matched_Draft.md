# Smart Water Distribution Management System - ANN

## STEP 1: Dataset Preparation and Processing

### 1. Data Collection
This project uses four datasets from Kaggle, each mapped to one module of the smart water system.

| Module | Dataset | Link | Samples |
|---|---|---|---|
| Demand Forecasting | United Utilities Water Management | https://www.kaggle.com/datasets/muzammalnawaz/united-utilities-water-management-water-demand | 140,086 |
| Distribution Risk | BATADAL | https://www.kaggle.com/datasets/minhbtnguyen/batadal-a-dataset-for-cyber-attack-detection | 4,177 |
| Leak Detection | Smart Water Leak Detection Dataset | https://www.kaggle.com/datasets/talha97s/smart-water-leak-detection-dataset | 5,000 |
| Water Quality | Water Potability | https://www.kaggle.com/datasets/adityakadiwal/water-potability | 3,276 |

Each dataset captures a different operational view of water infrastructure, enabling a modular AI pipeline with one integrated decision output.

### 2. Data Cleaning
Before model training, the following cleaning steps were applied:
- Missing value handling (numeric median, categorical most frequent)
- Binary label normalization (0/1, -1/1, and string variants)
- Invalid target row removal
- Datetime conversion for model compatibility
- Duplicate/irrelevant field exclusion where needed

### 3. Feature Selection
#### 3.1 Demand Module Features
- Timestamp-derived: hour, day of week, month
- Lag features: target_lag_1, target_lag_2, target_lag_3
- Netflow and related meter features from source CSV

#### 3.2 Distribution Module Features
- Tank levels: L_T1 to L_T7
- Pump flow and status: F_PU*, S_PU*
- Valve flow/status: F_V2, S_V2
- Junction pressure variables: P_J*
- Target: ATT_FLAG

#### 3.3 Leak Module Features
- Pressure
- Flow_Rate
- Vibration
- RPM
- Additional location and telemetry fields
- Target: Leakage_Flag

#### 3.4 Quality Module Features
- ph
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic_carbon
- Trihalomethanes
- Turbidity
- Target: Potability

### 4. Label Handling
- Distribution, leak, and quality tasks are binary classification.
- Demand is regression (continuous target).
- For classification tasks, labels are standardized to binary values before training.

### 5. Data Balancing Strategy
Class balancing experiments were implemented through oversampling support in the training pipeline. This option is available for classification modules and was evaluated during optimization.

### 6. Train-Test Split
- Standard split: 80% train, 20% test
- Stratified split for classification modules
- Random seed fixed for reproducibility

### 7. Feature Scaling and Encoding
A shared preprocessing pipeline is applied:
- Numeric features: median imputation + StandardScaler
- Categorical features: most-frequent imputation + OneHotEncoder
- Datetime features: Unix timestamp conversion

---

## STEP 2: Model Training

### 1. Overview
The system is designed as a multi-module ANN pipeline:
- Demand module predicts future water demand (regression)
- Distribution module predicts operational risk
- Leak module predicts leakage probability
- Quality module predicts safe water probability

These outputs are fused by a rule-based decision engine to produce one final operational action.

### 2. Model Type
| Module | Model |
|---|---|
| Demand | MLPRegressor |
| Distribution | MLPClassifier |
| Leak | MLPClassifier |
| Quality | MLPClassifier |

### 3. ANN Hyperparameters (Best Selected)
| Module | Hidden Layers | Learning Rate | Alpha | Max Iter | Solver |
|---|---:|---:|---:|---:|---|
| Demand | (128, 64) | 0.001 | 0.0001 | 400 | adam |
| Distribution | (256, 128, 64) | 0.0007 | 0.001 | 600 | adam |
| Leak | (256, 128, 64) | 0.0007 | 0.001 | 600 | adam |
| Quality | (128, 64) | 0.001 | 0.0001 | 300 | adam |

### 4. Training and Validation
- Backpropagation-based ANN training
- Train-test evaluation for all modules
- Cross-validation support enabled via CLI (`--cv-folds`)

### 5. Evaluation Metrics
#### 5.1 Regression (Demand)
- RMSE
- MAE
- R2

#### 5.2 Classification (Distribution/Leak/Quality)
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### 6. Final Results
| Module | Key Results |
|---|---|
| Demand | RMSE 1.9480, MAE 1.2239, R2 0.9150 |
| Distribution | Accuracy 0.9318, F1 0.6369, ROC-AUC 0.8395 |
| Leak | Accuracy 0.9720, F1 0.7705, ROC-AUC 0.9912 |
| Quality | Accuracy 0.6204, F1 0.5108, ROC-AUC 0.6338 |

### 7. Cross-Validation Example (Quality, 5-Fold)
- Accuracy mean ± std: 0.6374 ± 0.0174
- F1 mean ± std: 0.5208 ± 0.0350
- ROC-AUC mean ± std: 0.6622 ± 0.0224

### 8. Model Saving
Trained modules are persisted as joblib artifacts:
- demand_model.joblib
- distribution_model.joblib
- leak_model.joblib
- quality_model.joblib

Metrics are saved as JSON for traceability:
- demand_metrics.json
- distribution_metrics.json
- leak_metrics.json
- quality_metrics.json

---

## STEP 3: Integrated Decision Logic

### 1. Why Rule-Based Fusion
A single labeled dataset for final supervisory action is unavailable. Therefore, module outputs are combined with transparent expert rules.

### 2. Rule Set
| Rule ID | Condition | Action | Priority |
|---|---|---|---|
| R1_QUALITY_CRITICAL | quality_probability < 0.50 | HOLD_SUPPLY_AND_TRIGGER_TREATMENT | critical |
| R2_LEAK_HIGH | leak_probability >= 0.70 | ISOLATE_LEAK_ZONE_AND_DISPATCH_MAINTENANCE | high |
| R3_DISTRIBUTION_RISK_HIGH | distribution_risk >= 0.70 | REDUCE_PRESSURE_AND_REROUTE_FLOW | high |
| R4_DEMAND_SURGE | demand_score >= 0.75 | INCREASE_SUPPLY_TO_HIGH_DEMAND_ZONES | medium |
| R5_NORMAL | otherwise | NORMAL_OPERATION | low |

The app displays rule ID and rationale for explainability.

---

## STEP 4: Dashboard and Visualization

The Streamlit app provides:
- Training metrics table and charts
- Classification diagnostics:
  - confusion matrix
  - ROC curve
  - precision-recall curve
  - threshold slider
- Live parameter-to-prediction flow (module inputs, not manual module scores)
- Final action with rule and reason
- Scenario presets:
  - Normal Operation
  - High Demand
  - Leak Emergency
  - Poor Quality Alert

---

## What To Add (To Match Reference Report Style Better)

Add these items to make your report look very close to the IKS reference style:
1. College certificate/declaration pages (formatted as per department template)
2. Table of abbreviations (ANN, ROC-AUC, RMSE, etc.)
3. Per-module input table with short feature descriptions
4. Train-test split and preprocessing screenshots/code snippets
5. Confusion matrix screenshots for Distribution/Leak/Quality
6. Dashboard screenshots for at least 3 scenario presets
7. One page of limitations and future scope

---

## Conclusion
This report presents a complete ANN-driven smart water management framework trained on real datasets and integrated through explainable decision rules. The modular architecture, measurable performance, and interactive dashboard make the system suitable for academic demonstration and practical extension.
