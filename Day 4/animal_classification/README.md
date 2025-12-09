# Animal Classifier (FastAPI + Classical ML)

## Overview
Train classical ML models (Logistic Regression, KNN, GaussianNB) on an animals image dataset, serve predictions with FastAPI, and use a simple frontend to upload images and show results.

Install:
```bash
pip install -r requirements.txt

python training/train_model.py --data_dir data/Animals-10 --img_size 64 64

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000