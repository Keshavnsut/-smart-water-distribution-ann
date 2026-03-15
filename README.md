# Smart Water Distribution Management System (ANN)

A 4-module AI system for smart water operations:
- Demand forecasting (MLPRegressor)
- Distribution risk detection (MLPClassifier)
- Leak detection (MLPClassifier)
- Water quality/potability classification (MLPClassifier)

Includes model training CLI, metrics, decision fusion, and a Streamlit dashboard.

## Project Structure

- `app.py`: Streamlit dashboard (deployment entrypoint)
- `main.py`: Training and module pipelines
- `models/`: Trained model files + metrics JSON
- `data/`: Input datasets

## Local Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start app:

```bash
streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

3. Open:

`http://127.0.0.1:8501`

## GitHub Setup

Run these commands from project root:

```bash
git init
git add .
git commit -m "Initial commit: Smart Water Distribution ANN"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Open Streamlit Community Cloud and click **New app**.
3. Select repository and branch `main`.
4. Set main file path to `app.py`.
5. Deploy.

This repo already includes:
- `requirements.txt`
- `runtime.txt` (`python-3.11`)

## Notes

- If deployment fails due to very large files, move datasets/models to cloud storage and load on startup.
- Keep secrets in Streamlit Cloud Secrets, not in source files.
