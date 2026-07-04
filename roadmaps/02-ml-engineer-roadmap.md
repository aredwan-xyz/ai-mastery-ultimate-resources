# ⚡ ML Engineer Roadmap
### Python Developer → Production ML Systems in 16 Weeks

> **By Abid Redwan & CodeBeez** | [AI Magic Mastery](https://aimagicmastery.codebeez.ai)

---

## 📋 Prerequisites

- [ ] Comfortable with Python (functions, classes, list/dict comprehensions)
- [ ] Basic command line / Git
- [ ] High-school math (we refresh the rest)
- [ ] ~10–12 hours/week

**Coming from the [AI Foundations Roadmap](01-ai-foundations-roadmap.md)?** You're perfectly set up — skip to Module 2.

---

## 🎯 Outcomes

By week 16 you will:
- ✅ Build, evaluate, and tune ML models like a pro
- ✅ Master the modern ML stack (scikit-learn, XGBoost, PyTorch)
- ✅ Ship models to production with FastAPI, Docker & CI/CD
- ✅ Track experiments, version data, and monitor models
- ✅ Have 3 portfolio projects — including one end-to-end deployed system
- ✅ Be ready for **ML Engineer** interviews ([prep here](../interview-prep/README.md))

---

## 📅 Roadmap

### Module 1 — ML Foundations Refresher (Weeks 1–3)

#### Week 1: Python for ML & the Data Stack
**Resources:**
- 📗 [Kaggle: Pandas](https://www.kaggle.com/learn/pandas) + [Data Cleaning](https://www.kaggle.com/learn/data-cleaning)
- 🎥 [NumPy for ML – Patrick Loeber](https://www.youtube.com/watch?v=9JUAPgtkKpI)
- 📗 [Polars Guide](https://docs.pola.rs/) — the fast modern alternative to Pandas

```python
import numpy as np, pandas as pd
# Vectorize everything — loops are the enemy of ML performance
X = df[['feature_a', 'feature_b']].to_numpy()
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)   # standardize
```

#### Week 2: Math That Actually Matters
Focus on **intuition for the 20% you use 80% of the time.**
- 🎥 [3Blue1Brown: Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (1–7)
- 🎥 [StatQuest: Statistics Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)
- 📄 [Matrix Calculus for Deep Learning](https://explained.ai/matrix-calculus/)

#### Week 3: scikit-learn Fluency
- 📗 [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- 🎥 [scikit-learn Crash Course](https://www.youtube.com/watch?v=0B5eIE_1vpU)

**Milestone:** Full EDA + baseline model on a [Kaggle Playground](https://www.kaggle.com/competitions?searchQuery=playground) dataset.

---

### Module 2 — Core Machine Learning (Weeks 4–7)

#### Week 4: Supervised Learning Deep Dive
```
Regression      → Linear, Ridge, Lasso, ElasticNet
Classification  → Logistic, SVM, k-NN, Naive Bayes
Metrics         → RMSE/MAE (reg), Precision/Recall/F1/AUC (clf)
```
- 🎥 [Andrew Ng ML Specialization](https://www.coursera.org/specializations/machine-learning-introduction) (audit free)

#### Week 5: Feature Engineering — The Real Skill
> "Applied ML is basically feature engineering." — Andrew Ng
- 📗 [Feature Engineering – Kaggle](https://www.kaggle.com/learn/feature-engineering)
- 📗 [Feature Engineering and Selection (free book)](http://www.feat.engineering/)
```
Techniques: encoding (target/one-hot/ordinal), binning, interactions,
            datetime features, aggregations, scaling, handling missing/outliers
```

#### Week 6: Trees & Gradient Boosting (Kaggle's Weapon)
- 🎥 [StatQuest: Gradient Boost](https://www.youtube.com/playlist?list=PLblh5JKOoLUJjeb04ojG4pXQxxa4c5R3-)
- 📗 [XGBoost](https://xgboost.readthedocs.io/) · [LightGBM](https://lightgbm.readthedocs.io/) · [CatBoost](https://catboost.ai/)
```python
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50)])
```

#### Week 7: Model Selection & Hyperparameter Tuning
- 📗 [Optuna](https://optuna.org/) — modern hyperparameter optimization
```python
import optuna
def objective(trial):
    params = {'num_leaves': trial.suggest_int('num_leaves', 20, 300),
              'learning_rate': trial.suggest_float('lr', 1e-3, 0.1, log=True)}
    return cross_val_score(lgb.LGBMClassifier(**params), X, y, cv=5).mean()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

**Milestone Project 🏆:** Enter a live [Kaggle competition](https://www.kaggle.com/competitions) — aim for top 50%.

---

### Module 3 — Deep Learning for Engineers (Weeks 8–10)

#### Week 8: Neural Networks & PyTorch
- 🎥 [Karpathy: Neural Networks Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) (Ep. 1–2)
- 📗 [Our PyTorch Cheatsheet](../cheatsheets/pytorch-cheatsheet.md)

#### Week 9: CNNs & Transfer Learning
- 🎥 [CS231n Lectures](http://cs231n.stanford.edu/)
```python
import timm  # 1000+ pretrained models
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)
```

#### Week 10: Tabular DL & When *Not* to Use It
Reality check: **gradient boosting still beats deep learning on most tabular data.** Know when each wins.
- 📄 [Why do tree-based models still outperform DL on tabular data?](https://arxiv.org/abs/2207.08815)

---

### Module 4 — MLOps & Production (Weeks 11–14)

> This module is what separates a **data scientist** from an **ML engineer**.

#### Week 11: Experiment Tracking & Data Versioning
```
Experiment tracking → MLflow, Weights & Biases
Data versioning     → DVC
```
- 📗 [MLflow Tutorial](https://mlflow.org/docs/latest/getting-started/index.html)
- 🎥 [Weights & Biases Crash Course](https://www.youtube.com/watch?v=krWjJcW80_A)

#### Week 12: Serving Models as APIs
```python
from fastapi import FastAPI
import joblib
app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(features: dict):
    return {"prediction": model.predict([list(features.values())]).tolist()}
```
- 📗 [FastAPI](https://fastapi.tiangolo.com/) · [BentoML](https://www.bentoml.com/) — production ML serving

#### Week 13: Docker & CI/CD for ML
- 📗 [Our Docker for ML Cheatsheet](../cheatsheets/docker-for-ml.md)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- Set up **GitHub Actions** to test + build + deploy on every push.

#### Week 14: Monitoring, Drift & Retraining
```
Data drift    → input distribution changes over time
Concept drift → the X→y relationship changes
Tools         → Evidently AI, NannyML, Arize
```
- 📗 [Evidently AI](https://www.evidentlyai.com/) — open-source ML monitoring

**Best full course:** [Made With ML by Goku Mohandas](https://madewithml.com/) (free, MLOps gold standard) · [MLOps Specialization](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)

---

### Module 5 — Capstone (Weeks 15–16)

**Build one end-to-end production system.** Pick one:

| Project | Stack |
|---------|-------|
| 🏦 Loan default risk API | LightGBM + FastAPI + Docker + monitoring |
| 🛒 Product recommender service | implicit/LightFM + Redis + FastAPI |
| 📸 Image classifier SaaS | timm + Gradio + Hugging Face Spaces |
| 📊 Churn prediction pipeline | XGBoost + MLflow + Airflow + dashboard |

**Ship checklist:**
- [ ] Reproducible training pipeline (DVC + MLflow)
- [ ] REST API with input validation & tests
- [ ] Dockerized + deployed (Railway / Fly.io / HF Spaces)
- [ ] Monitoring dashboard
- [ ] README with architecture diagram
- [ ] Live demo link in your portfolio

---

## 🛠️ The ML Engineer Stack

```python
# Core
pip install numpy pandas polars scikit-learn matplotlib seaborn
# Boosting
pip install xgboost lightgbm catboost optuna
# Deep learning
pip install torch torchvision timm
# MLOps
pip install mlflow dvc wandb evidently
# Serving
pip install fastapi uvicorn bentoml pydantic
```

---

## 📊 Progress Tracker

```
M1 Foundations:  [ ] Data stack  [ ] Math intuition  [ ] sklearn fluency
M2 Core ML:      [ ] Supervised  [ ] Feature eng.  [ ] Boosting  [ ] Kaggle top 50%
M3 Deep Learning:[ ] PyTorch  [ ] CNNs  [ ] Tabular DL judgment
M4 MLOps:        [ ] Tracking  [ ] Serving  [ ] Docker/CI  [ ] Monitoring
M5 Capstone:     [ ] Deployed end-to-end system  [ ] In portfolio
```

## Next Steps
- → [LLM Engineer Roadmap](03-llm-engineer-roadmap.md) — go to the frontier
- → [AI Agent Builder Roadmap](06-ai-agent-builder-roadmap.md) — build autonomous systems
- → [Interview Prep](../interview-prep/README.md) — land the job

---

*Part of the [AI Magic Mastery](https://aimagicmastery.codebeez.ai) ecosystem by CodeBeez × Abid Redwan*
