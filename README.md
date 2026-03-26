# 🚀 Fraud ML System

A **production-grade, GPU-first, config-driven fraud detection platform** designed for large-scale datasets (50GB+) with support for:

* Multi-model training (XGBoost, LightGBM, CatBoost, etc.)
* Extreme class imbalance handling (1:20,000+)
* Hyperparameter tuning (Optuna / Ray)
* SHAP explainability
* Ensemble modeling
* CPU-distributed inference (Polars)
* MLflow experiment tracking

---

# 🧠 System Overview

This system is designed as a **modular fraud ML platform**, not just a training script.

It supports:

* Offline training & evaluation
* Large-scale batch inference
* Explainability & auditability
* Config-driven experimentation

---

# 🏗️ Architecture

Data → Sampling → Models → Tuning → Evaluation → Ensemble → Explainability → Inference

### Key Design Principles

* Fully **config-driven**
* GPU-first, CPU fallback
* Fault-tolerant (per-model failure isolation)
* Scalable to large datasets
* Fraud-specific (precision-recall optimized)

---

# 📁 Project Structure

    fraud_ml_system/
    ├── config/                # All configuration files
    ├── data/                  # Input datasets
    ├── logs/                  # Logs (app + errors)
    ├── outputs/               # Models, predictions, reports
    ├── mlruns/                # MLflow tracking
    ├── src/
    │   ├── data/              # Loader, splitter, sampler
    │   ├── models/            # Model implementations
    │   ├── training/          # Trainer, tuning, threshold
    │   ├── evaluation/        # Metrics
    │   ├── inference/         # Distributed inference
    │   ├── explainability/    # SHAP + reports
    │   ├── ensemble/          # Model blending
    │   ├── mlops/             # MLflow integration
    │   ├── resource/          # GPU/CPU management
    │   ├── observability/     # logging, run summary
    │   └── pipeline/          # main pipeline

---

# ⚙️ Installation

## 1. Clone Repo

    git clone <repo>
    cd fraud_ml_system

## 2. Install Dependencies

    pip install -r requirements.txt

⚠️ For GPU support, ensure RAPIDS + CUDA are correctly installed.

---

# ▶️ Running the Pipeline

    python src/pipeline/run_pipeline.py

Or via CLI:

    python src/cli.py

---

# 🔧 Configuration Guide (IMPORTANT)

All behavior is controlled via:

    config/config.yaml

---

## 🧾 Data

    train_path: data/train/
    test_path: data/test/
    has_separate_test: false
    target: label

---

## 🔀 Split

    split:
        method: stratified
        ratios:
        train: 0.7
        validation: 0.15
        test: 0.15

---

## ⚖️ Imbalance Handling

    imbalance:
    enabled: true
    strategy: hybrid

Supports:

* undersampling
* oversampling
* class weighting
* threshold optimization

---

## 🤖 Models

    models:
    xgb: {enabled: true}
    lgbm: {enabled: true}
    catboost: {enabled: true}
    sklearn_rf: {enabled: true}

---

## 🎯 Tuning

    tuning:
    enabled: true
    n_trials: 20

---

## ⚡ Inference

    inference:
    enabled: true
    engine: polars
    cpu_cores: 16

---

## 🧠 Explainability

    explainability:
    shap:
    enabled: true
    sample_size: 10000

---

# 🧪 Output Artifacts

outputs/

---

## 📊 metrics.csv

    Model performance:

    * AUC-PR (primary)
    * ROC-AUC
    * threshold

---

## 🧠 model_explanations.md

Human-readable insights:

* Best model
* Trade-offs
* Threshold recommendation

---

## 🔍 SHAP

outputs/shap/

---

## 📈 Feature Importance

outputs/feature_importance/

---

## 📦 Models

outputs/models/<model_name>/

Includes:

* model.pkl
* metrics.json

---

## 🔮 Predictions

outputs/predictions/

---

# 🧠 Imbalance Strategy (Fraud-Specific)

This system handles extreme imbalance via:

1. Hybrid sampling
2. Class weighting
3. Threshold tuning

---

# 📬 Logging & Debugging

## Logs

logs/app.log
logs/errors.log

---

## Run Summary

outputs/run_summary.json

---

## Debug Failures

* Check logs/errors.log
* Check model-specific failures
* Pipeline continues even if one model fails

---

# ⚡ Performance & Scaling

## GPU Training

* XGBoost → GPU
* LightGBM → GPU
* CatBoost → GPU

---

## CPU Inference

* Polars lazy execution
* Multiprocessing

---

## Large Data Handling

* Batch loading
* File-level control
* Memory-safe operations

---

# 🔁 How Inference Works

If test_path exists:

test → inference

Else:

train → split → test → inference

---

# 🧩 Extending the System

## Add New Model

1. Create file in src/models/
2. Implement fit() and predict()
3. Register in model_factory.py

---

## Add Features

Modify:

src/data/preprocessor.py

---

# ⚠️ Known Limitations

* No real-time scoring (batch only)
* No feature store
* No graph modeling (yet)

---

# 🚀 Future Enhancements

* Real-time API scoring
* Graph-based fraud detection
* Online feature store

---

# 👨‍💻 Author

Designed by **amit binjola** for large-scale production use.

---
