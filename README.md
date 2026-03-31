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
│   └── msme_model_bundle.joblib  
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

### Run the code in GitHub CodeSpaces

In GitHub CodeSpace, create a new codespace and select Nathali99/msme_dashboard as the repository.

In the terminal,

You can install the required libraries

```bash
pip install -r requirements.txt
```

Then run to train the model (This step is not mandatory as the final model is already saved in the msme_outputs folder):

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
