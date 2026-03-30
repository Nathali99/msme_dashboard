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

## Option 2: Run the code in GitHub CodeSpaces

First install the required libraries

```bash
pip install -r requirements.txt
```
Add your datset to a separate folder called data  as:

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
msme_outputs/test_predictions.csv
msme_outputs/training_summary.json
```

After that, start the dashboard:

```bash
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

