# training/train_model.py
"""
Train three classical ML models (LogisticRegression, KNN, GaussianNB)
on an ImageFolder-style dataset (each class in its own folder).
Saves models and a label mapping to ../app/models/
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import argparse

# --- Config ---
IMG_SIZE = (64, 64)   # resize images to this size (you can increase)
DATA_DIR = Path("../data/Animals-10")  # expected data path (change if needed)
OUT_DIR = Path("../app/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset(root_dir: Path, img_size=(64,64), max_per_class=None):
    X, y, classes = [], [], []
    class_dirs = [d for d in sorted(root_dir.iterdir()) if d.is_dir()]
    label_map = {}
    for idx, d in enumerate(class_dirs):
        label_map[idx] = d.name
        classes.append(d.name)
        images = list(d.glob("*"))
        if max_per_class:
            images = images[:max_per_class]
        for p in images:
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize(img_size)
                arr = np.asarray(img).astype(np.float32) / 255.0  # normalize
                X.append(arr.flatten())
                y.append(idx)
            except Exception as e:
                print("skipping", p, "due to", e)
    X = np.array(X)
    y = np.array(y)
    return X, y, label_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--img_size", type=int, nargs=2, default=list(IMG_SIZE))
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max_per_class", type=int, default=None)
    args = parser.parse_args()

    print("Loading dataset from", args.data_dir)
    X, y, label_map = load_dataset(Path(args.data_dir), tuple(args.img_size), args.max_per_class)
    print("Loaded", X.shape[0], "images with shape", args.img_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # Scale for models that benefit from it (Logistic & KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler & label_map
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    joblib.dump(label_map, OUT_DIR / "label_map.joblib")

    # 1) Logistic Regression
    print("Training LogisticRegression...")
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    print("Logistic acc:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    joblib.dump(lr, OUT_DIR / "logistic.pkl")

    # 2) KNN
    print("Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    print("KNN acc:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    joblib.dump(knn, OUT_DIR / "knn.pkl")

    # 3) GaussianNB (works best on unscaled or scaled data; we'll use scaled)
    print("Training GaussianNB...")
    gnb = GaussianNB()
    gnb.fit(X_train_scaled, y_train)
    y_pred = gnb.predict(X_test_scaled)
    print("GaussianNB acc:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    joblib.dump(gnb, OUT_DIR / "gaussian_nb.pkl")

    print("All models saved to", OUT_DIR)

if __name__ == "__main__":
    main()