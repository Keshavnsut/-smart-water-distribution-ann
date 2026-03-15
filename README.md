# Smart Water Distribution Management System (ANN)

End-to-end machine learning project for intelligent water-network monitoring and decision support.

The system trains and serves four ANN-based modules:
- Demand forecasting (regression)
- Distribution risk detection (classification)
- Leak detection (classification)
- Water potability classification (classification)

It includes:
- A unified training CLI
- Saved models and metrics
- A rule-based integrated recommendation engine
- A Streamlit dashboard for visualization and live inference

## Features

- Multi-module ANN training pipeline using scikit-learn MLP models
- Shared preprocessing with imputation, scaling, and categorical handling
- Classification diagnostics: confusion matrix, ROC, PR curves
- Integrated action recommendation from module outputs
- Streamlit-ready deployment configuration

## Repository Structure

- `app.py` - Streamlit dashboard entrypoint
- `main.py` - Training, evaluation, and integrated decision CLI
- `data/` - Input datasets used by training and dashboard diagnostics
- `models/` - Trained model files and metrics JSON outputs
- `requirements.txt` - Python dependencies
- `runtime.txt` - Streamlit Cloud Python runtime pin
- `.python-version` - Local/runtime version hint

## Prerequisites

- Python 3.11+ recommended
- pip

## Install

```bash
pip install -r requirements.txt
```

## Run Dashboard Locally

```bash
streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

Then open:

http://127.0.0.1:8501

## Training Commands

Train each module individually:

```bash
python main.py train-demand --csv ./data/demand/netbase_inlet-outlet-cont_logged_user_April2018.csv
python main.py train-distribution --csv ./data/distribution/training_dataset_2.csv --target-col ATT_FLAG
python main.py train-leak --csv ./data/leak/location_aware_gis_leakage_dataset.csv --target-col Leakage_Flag
python main.py train-quality --csv ./data/quality/water_potability.csv --target-col Potability
```

Train all modules in one run:

```bash
python main.py train-all \
	--demand-csv ./data/demand/netbase_inlet-outlet-cont_logged_user_April2018.csv \
	--distribution-csv ./data/distribution/training_dataset_2.csv \
	--leak-csv ./data/leak/location_aware_gis_leakage_dataset.csv \
	--quality-csv ./data/quality/water_potability.csv \
	--distribution-target-col ATT_FLAG \
	--leak-target-col Leakage_Flag \
	--quality-target-col Potability
```

Run integrated decision from command line:

```bash
python main.py recommend \
	--demand-score 0.70 \
	--distribution-risk 0.35 \
	--leak-probability 0.20 \
	--quality-probability 0.85
```

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Open Streamlit Community Cloud.
3. Create new app with:
	 - Repository: your fork/repo
	 - Branch: main
	 - Main file path: app.py
4. Deploy.

This repository already contains deployment essentials:
- `requirements.txt`
- `runtime.txt` (python-3.11)

## Troubleshooting

- If app fails while loading models, check package compatibility in `requirements.txt`.
- If files are reported missing, verify `data/` and `models/` are committed.
- If deployment is slow, large model/data files are the main reason.

## Security Notes

- Never commit secrets or API keys.
- Use Streamlit Secrets for sensitive configuration.
