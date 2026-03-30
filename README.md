# MSME Failure Risk Dashboard

This repository contains a Streamlit dashboard and the model training pipeline for predicting whether an MSME is likely to be **Active** or **Failed** from historical business data.

## Repository structure

```text
msme-dashboard/
├── app.py
├── dashboard.py
├── msme_training_pipeline.py
├── requirements.txt
├── data/
│   └── final_dataset.csv
├── msme_outputs/
│   └── msme_model_bundle.joblib   # created after training, or add your saved bundle here
├── notebooks/
│   ├── model_training.ipynb
│   └── dashboard.ipynb
└── .streamlit/
    └── config.toml
```

## What each file does

- `app.py`: the main Streamlit dashboard for local use and Streamlit Community Cloud.
- `dashboard.py`: same dashboard code under an alternate name.
- `msme_training_pipeline.py`: trains the models, selects the best one, optionally calibrates it, and saves the final bundle.
- `notebooks/`: notebook versions of the training and dashboard code.
- `msme_outputs/msme_model_bundle.joblib`: the saved trained model bundle used by the dashboard.

## Option 1: Run the dashboard with an existing model bundle

Place your trained model bundle at:

```text
msme_outputs/msme_model_bundle.joblib
```

Then install the dependencies and run the dashboard:

```bash
git clone https://github.com/YOUR_USERNAME/msme-dashboard.git
cd msme-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Option 2: Train the model first, then run the dashboard

If you want others to reproduce the training step, add your dataset file as:

```text
final_dataset.csv
```

Then run:

```bash
python msme_training_pipeline.py
```

This will create:

```text
msme_outputs/msme_model_bundle.joblib
```

After that, start the dashboard:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create a new app from this repository.
4. Set the main file path to `app.py`.
5. Deploy.

If your repository already includes `msme_outputs/msme_model_bundle.joblib`, the app should run immediately.

## Important notes

- Do **not** upload `final_dataset.csv` publicly if it contains confidential or sensitive business data.
- If the model bundle is too large for normal GitHub upload, use **Git LFS** or upload the model bundle separately and place it in `msme_outputs/` before running the app.
- The dashboard uses the saved model bundle, so the predictions and what-if analysis stay consistent with the trained model.

## Suggested Git commands

```bash
git init
git add .
git commit -m "Initial MSME dashboard repository"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/msme-dashboard.git
git push -u origin main
```
