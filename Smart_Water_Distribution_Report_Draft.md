# Smart Water Distribution Management System Using Artificial Neural Networks

## Certificate Page
(Use your college format for guide name, HOD signature, and department seal)

## Declaration
I hereby declare that this project report titled Smart Water Distribution Management System Using Artificial Neural Networks is an original work carried out by me under the guidance of my faculty mentor. The content presented is based on implementation, experimentation, and analysis performed for academic purposes.

## Acknowledgement
I would like to express sincere gratitude to my project guide, department faculty, and classmates for their support, feedback, and motivation throughout this work.

## Abstract
This project presents a Smart Water Distribution Management System using Artificial Neural Networks (ANN) with a multi-module architecture. The system addresses key urban water management challenges through four functional modules: water demand forecasting, distribution risk detection, leak detection, and water quality potability classification. Real-world and benchmark datasets were used from Kaggle for training and validation. The demand module was implemented as ANN regression, while distribution, leak, and quality modules were implemented as ANN classification models.

A rule-based decision fusion engine integrates module outputs to provide one final operational action such as normal operation, pressure rerouting, maintenance dispatch, or treatment alert. The system includes preprocessing pipelines, cross-validation support, diagnostics, and an interactive dashboard for scenario-based simulation. Experimental results show strong performance in demand and leak modules, while distribution and quality modules provide moderate but useful predictive capability. The work demonstrates a practical AI-driven framework for smart utility management with explainable decision flow.

Keywords: Smart Water, ANN, Demand Forecasting, Leak Detection, Potability, Distribution Risk, Rule-based Fusion

## Table of Contents
1. Introduction
2. Problem Statement
3. Objectives
4. Literature Review
5. System Architecture
6. Datasets
7. Data Cleaning and Preprocessing
8. Model Development
9. Decision Fusion Logic
10. Experimental Results
11. Dashboard and Visualization
12. Conclusion
13. Future Scope
14. References

## 1. Introduction
Water distribution systems face multiple operational challenges such as demand fluctuation, pressure instability, leak losses, and quality degradation. Conventional monitoring approaches are mostly reactive and require manual intervention. With recent improvements in sensor data availability and machine learning, ANN-based systems can provide predictive and near real-time support for utility operations.

This project proposes an integrated architecture where each critical task is modeled separately and then fused for a single actionable output.

## 2. Problem Statement
The key problem is to design an intelligent system that can:
- Forecast near-future water demand
- Detect distribution-level operational risk
- Detect leak events from sensor patterns
- Classify water quality safety
- Generate one final operational recommendation in a transparent way

## 3. Objectives
- Build four ANN-based modules for major water management tasks
- Train using real Kaggle datasets
- Apply standard preprocessing and model evaluation pipelines
- Integrate module outputs through explainable rule-based fusion
- Provide an interactive dashboard for practical demonstration

## 4. Literature Review
Recent research in smart water systems highlights separate advances in demand forecasting, leak detection, and water quality prediction. Many studies focus on one task only, while real deployment requires integrated decisions. This project bridges that gap by combining multiple ANN modules under one decision engine and demonstrating practical decision policies.

## 5. System Architecture
The proposed architecture has six stages:
1. Data Ingestion
2. Module 1: Demand Forecasting (ANN Regression)
3. Module 2: Distribution Risk Detection (ANN Classification)
4. Module 3: Leak Detection (ANN Classification)
5. Module 4: Quality Potability Classification (ANN Classification)
6. Rule-based Decision Engine for final action

Data Flow:
Sensor and historical data -> Module predictions -> Rule evaluation -> Final action and priority

## 6. Datasets
### 6.1 Demand Forecasting Dataset
- Source: United Utilities Water Management
- Link: https://www.kaggle.com/datasets/muzammalnawaz/united-utilities-water-management-water-demand
- Samples: 140,086

### 6.2 Distribution Risk Dataset
- Source: BATADAL
- Link: https://www.kaggle.com/datasets/minhbtnguyen/batadal-a-dataset-for-cyber-attack-detection
- Samples: 4,177
- Target: ATT_FLAG

### 6.3 Leak Detection Dataset
- Source: Smart Water Leak Detection Dataset
- Link: https://www.kaggle.com/datasets/talha97s/smart-water-leak-detection-dataset
- Samples: 5,000
- Target: Leakage_Flag

### 6.4 Quality Dataset
- Source: Water Potability
- Link: https://www.kaggle.com/datasets/adityakadiwal/water-potability
- Samples: 3,276
- Target: Potability

## 7. Data Cleaning and Preprocessing
The following preprocessing pipeline is applied:
- Missing-value handling:
  - Numeric features: median imputation
  - Categorical features: most-frequent imputation
- Feature scaling: StandardScaler for numeric columns
- Categorical encoding: OneHotEncoder with unknown-safe handling
- Datetime handling: converted to numeric Unix time where needed
- Demand feature engineering:
  - Hour, day-of-week, month
  - Lag features: lag1, lag2, lag3
- Label normalization for binary targets (including support for 0, 1, -1 and textual forms)

## 8. Model Development
### 8.1 Model Family
- Demand: MLPRegressor
- Distribution, Leak, Quality: MLPClassifier

### 8.2 Hyperparameters (Best selected)
- Demand: hidden layers (128, 64), learning rate 0.001, alpha 0.0001, max_iter 400
- Distribution: hidden layers (256, 128, 64), learning rate 0.0007, alpha 0.001, max_iter 600
- Leak: hidden layers (256, 128, 64), learning rate 0.0007, alpha 0.001, max_iter 600
- Quality: hidden layers (128, 64), learning rate 0.001, alpha 0.0001, max_iter 300

### 8.3 Validation
- Train-test split used for all modules
- Cross-validation support added (k-fold option)

## 9. Decision Fusion Logic
The final system output is produced by rule-based fusion over module probabilities and demand score:

- Rule R1_QUALITY_CRITICAL:
  If quality probability < 0.50 -> HOLD_SUPPLY_AND_TRIGGER_TREATMENT (critical)

- Rule R2_LEAK_HIGH:
  Else if leak probability >= 0.70 -> ISOLATE_LEAK_ZONE_AND_DISPATCH_MAINTENANCE (high)

- Rule R3_DISTRIBUTION_RISK_HIGH:
  Else if distribution risk >= 0.70 -> REDUCE_PRESSURE_AND_REROUTE_FLOW (high)

- Rule R4_DEMAND_SURGE:
  Else if demand score >= 0.75 -> INCREASE_SUPPLY_TO_HIGH_DEMAND_ZONES (medium)

- Rule R5_NORMAL:
  Else -> NORMAL_OPERATION (low)

This design was selected because no integrated supervisory dataset exists for end-to-end fusion labels.

## 10. Experimental Results
### 10.1 Final Module Metrics

Demand Module:
- RMSE: 1.9480
- MAE: 1.2239
- R2: 0.9150

Distribution Module:
- Accuracy: 0.9318
- Precision: 0.8475
- Recall: 0.5102
- F1: 0.6369
- ROC-AUC: 0.8395

Leak Module:
- Accuracy: 0.9720
- Precision: 0.8246
- Recall: 0.7231
- F1: 0.7705
- ROC-AUC: 0.9912

Quality Module:
- Accuracy: 0.6204
- Precision: 0.5138
- Recall: 0.5078
- F1: 0.5108
- ROC-AUC: 0.6338

### 10.2 Cross-Validation (Quality Example, 5-Fold)
- Accuracy mean ± std: 0.6374 ± 0.0174
- F1 mean ± std: 0.5208 ± 0.0350
- ROC-AUC mean ± std: 0.6622 ± 0.0224

## 11. Dashboard and Visualization
The Streamlit dashboard includes:
- Module result tables and charts
- Diagnostics panel:
  - Confusion matrix
  - ROC curve
  - Precision-Recall curve
  - Threshold slider
- Live parameter-based prediction (no manual module score input)
- Explainable final decision showing:
  - Action
  - Priority
  - Rule ID
  - Rationale
- Demo scenario presets:
  - Normal Operation
  - High Demand
  - Leak Emergency
  - Poor Quality Alert

## 12. Conclusion
The project successfully demonstrates a multi-module ANN-based smart water management framework trained on real datasets. The architecture supports demand forecasting, distribution risk detection, leak prediction, and water quality assessment under one operational decision layer. The system is practical, explainable, and suitable for academic demonstration and future extension.

## 13. Future Scope
- Replace rule-based fusion with learned weighted fusion when integrated labels become available
- Add geographic zone-level optimization for routing
- Add reinforcement learning for pump-valve control policies
- Improve quality module performance with advanced feature engineering and hybrid models
- Integrate real IoT stream ingestion for online inference

## 14. References
1. United Utilities Water Management Dataset, Kaggle
2. BATADAL Dataset, Kaggle
3. Smart Water Leak Detection Dataset, Kaggle
4. Water Potability Dataset, Kaggle
5. Scikit-learn Documentation
6. Streamlit Documentation
