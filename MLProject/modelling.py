"""
modelling.py
============
Pelatihan model Credit Risk dengan MLflow Autolog.

Jalankan MLflow UI terlebih dahulu di terminal:
    mlflow ui

Kemudian jalankan script ini:
    python modelling.py

Akses dashboard MLflow di: http://127.0.0.1:5000
Kriteria 2 — Basic | Sistem Machine Learning (Dicoding)
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ── Konfigurasi MLflow ────────────────────────────────────────────────────────
import os
if "GITHUB_ACTIONS" not in os.environ:
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("CreditRisk-Basic")

# ── Load Dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv("credit_risk_preprocessing.csv")

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Dataset dimuat: {df.shape}")
print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Training Model 1: Random Forest ──────────────────────────────────────────
print("\n🚀 Training Random Forest ...")
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest-Basic"):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Tidak Default", "Default"]))

print("✅ Random Forest selesai — cek MLflow UI di http://127.0.0.1:5000")

# ── Training Model 2: Logistic Regression ─────────────────────────────────────
print("\n🚀 Training Logistic Regression ...")
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="LogisticRegression-Basic"):
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    lr_model.fit(X_train, y_train)

    y_pred_lr = lr_model.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr  = f1_score(y_test, y_pred_lr, average="weighted")
    auc_lr = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])

    print(f"   Accuracy : {acc_lr:.4f}")
    print(f"   F1-Score : {f1_lr:.4f}")
    print(f"   ROC-AUC  : {auc_lr:.4f}")
    print(classification_report(y_test, y_pred_lr,
                                target_names=["Tidak Default", "Default"]))

print("✅ Logistic Regression selesai — cek MLflow UI di http://127.0.0.1:5000")
print("\n🎯 Semua model berhasil dilatih dan dicatat di MLflow!")
