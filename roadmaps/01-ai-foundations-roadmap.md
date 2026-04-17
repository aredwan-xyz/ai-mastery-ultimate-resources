# 🌱 AI Foundations Roadmap
### Complete Beginner → First AI Model in 12 Weeks

> **By Abid Redwan & CodeBeez** | [AI Magic Mastery](https://aimagicmastery.codebeez.ai)

---

## 📋 Prerequisites Checklist

Before starting, you need:
- [ ] A computer (any OS — Windows, Mac, Linux)
- [ ] Internet connection
- [ ] ~10 hours/week commitment
- [ ] Curiosity and patience 🧠

**You do NOT need:**
- ❌ Prior programming experience
- ❌ Math beyond high-school algebra
- ❌ Expensive hardware (we use free cloud GPUs)

---

## 🎯 What You'll Achieve

By the end of 12 weeks you will:
- ✅ Write Python confidently
- ✅ Understand the math behind ML (intuitively)
- ✅ Build and train your first ML model
- ✅ Complete 3 portfolio projects
- ✅ Know how to keep learning independently
- ✅ Be ready for the [ML Engineer Roadmap](02-ml-engineer-roadmap.md)

---

## 📅 Week-by-Week Plan

### Phase 1: Python & Tools (Weeks 1–3)

#### Week 1: Python Fundamentals
**Goal:** Write basic Python programs confidently.

**Resources:**
- 📗 [Python for Everybody – Course](https://www.coursera.org/specializations/python) (Chapters 1–5, audit free)
- 🎥 [Corey Schafer Python Beginners Series](https://www.youtube.com/playlist?list=PL-osiE80TeTskrapNbzXhwoFUiLCjGgY7) — Chapters 1–8
- 🛠️ [Repl.it](https://replit.com/) — Code in browser, no setup needed

**Topics:**
```
Variables and Data Types    → int, float, str, bool
Collections                 → list, dict, tuple, set
Control Flow                → if/elif/else, loops
Functions                   → def, args, return
File I/O                    → read/write text files
Error Handling              → try/except
```

**Daily Practice:** 30 minutes on [Exercism Python Track](https://exercism.org/tracks/python)

**Milestone Project:** Build a CLI quiz game on any topic you love 🎮

---

#### Week 2: Python for Data
**Goal:** Manipulate and explore data with Python.

**Resources:**
- 📗 [Kaggle's Python Course](https://www.kaggle.com/learn/python) — Free, 7 lessons
- 📗 [Kaggle's Pandas Course](https://www.kaggle.com/learn/pandas) — Free, 6 lessons
- 🎥 [Corey Schafer NumPy](https://www.youtube.com/watch?v=GB9ByFAIAH4)

**Topics:**
```python
import numpy as np       # Arrays, math operations
import pandas as pd      # DataFrames, data wrangling
import matplotlib.pyplot as plt  # Basic plotting
```

**Milestone Project:** Load a CSV dataset, explore it, create 3 charts 📊

---

#### Week 3: Dev Environment Setup
**Goal:** Work like a real AI engineer.

**Setup:**
- Install [VS Code](https://code.visualstudio.com/) with Python extension
- Set up [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for environments
- Create a [GitHub account](https://github.com) and learn basic git

**Resources:**
- 🎥 [VS Code Python Setup](https://www.youtube.com/watch?v=W--_EOzdTHk)
- 📗 [Git for Beginners](https://rogerdudler.github.io/git-guide/)
- 🛠️ [GitHub Desktop](https://desktop.github.com/) — Visual git client

**Topics:**
```bash
git init          # Start tracking
git add .         # Stage changes
git commit -m ""  # Save snapshot
git push          # Upload to GitHub
```

---

### Phase 2: Math for AI (Weeks 4–5)

> Don't panic. We focus on intuition, not proofs.

#### Week 4: Linear Algebra (Visually)
**Goal:** Understand what matrices and vectors mean for AI.

**Resources:**
- 🎥 [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — Videos 1–9 (2h total, pure gold)
- 📝 [Linear Algebra Review for ML](https://www.deeplearning.ai/ai-notes/linear-algebra/index.html) — DeepLearning.AI notes

**Key Concepts:**
```
Vectors           → Direction and magnitude in space
Matrices          → Transformations
Dot Product       → Similarity between vectors
Matrix Multiply   → Compose transformations
Eigenvectors      → PCA, dimensionality reduction
```

---

#### Week 5: Probability & Statistics
**Goal:** Think probabilistically about data.

**Resources:**
- 🎥 [StatQuest – Statistics Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) — All playlists
- 🎥 [3Blue1Brown: Bayes Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)
- 📗 [Think Stats](https://greenteapress.com/wp/think-stats-2e/) — Free PDF

**Key Concepts:**
```
Mean, Median, Variance    → Summarize data
Probability Distributions → Normal, Binomial, Poisson
Bayes' Theorem            → Update beliefs with evidence
Hypothesis Testing         → Is this result real or luck?
Correlation vs Causation   → The most important distinction
```

---

### Phase 3: Machine Learning (Weeks 6–9)

#### Week 6: The Big Picture of ML
**Goal:** Understand what ML is and how it works conceptually.

**Resources:**
- 🎥 [Machine Learning Specialization W1–W2](https://www.coursera.org/specializations/machine-learning-introduction) — Andrew Ng (audit free)
- 🎥 [StatQuest: Machine Learning](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)

**Concepts:**
```
Supervised Learning    → Learn from labeled examples
Unsupervised Learning  → Find patterns without labels
Training vs Test Set   → Generalization is the goal
Features & Labels      → Inputs & Outputs
Model = function(inputs) → outputs
```

---

#### Week 7: Regression & Classification
**Goal:** Build your first real ML models.

**Resources:**
- 🎥 [ML Specialization Course 1 Week 2–3](https://www.coursera.org/specializations/machine-learning-introduction)
- 🎥 [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)

**Build:**
```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Train your first model in 10 lines of code!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, predictions):.2f}")
```

**Milestone Project:** Predict house prices on the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) 🏠

---

#### Week 8: More Algorithms
**Goal:** Learn the ML toolkit — know when to use what.

**Resources:**
- 🎥 [ML Specialization Course 2](https://www.coursera.org/specializations/machine-learning-introduction)
- 🎥 [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk) + [Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)

**Algorithms:**
```
Decision Trees        → Interpretable, easy to understand
Random Forests        → Ensemble of trees, very robust
Gradient Boosting     → XGBoost, the Kaggle champion
KNN                   → Simplest classifier
SVM                   → Powerful for small datasets
K-Means               → Clustering unlabeled data
PCA                   → Reduce dimensions
```

---

#### Week 9: Model Evaluation & Improvement
**Goal:** Build models that actually generalize well.

**Resources:**
- 🎥 [ML Specialization Course 2 Week 3](https://www.coursera.org/specializations/machine-learning-introduction)
- 📗 [Scikit-learn Model Selection Guide](https://scikit-learn.org/stable/model_selection.html)

**Topics:**
```
Cross-Validation      → k-fold, stratified k-fold
Confusion Matrix      → TP, FP, TN, FN
Precision / Recall    → Tradeoff, F1 Score
ROC-AUC Curve         → Discriminative power
Bias vs Variance      → Underfitting vs overfitting
Grid Search / Optuna  → Hyperparameter tuning
```

---

### Phase 4: Neural Networks & Projects (Weeks 10–12)

#### Week 10: Introduction to Neural Networks
**Goal:** Understand how deep learning works.

**Resources:**
- 🎥 [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — Essential, visual
- 🎥 [Neural Networks: Zero to Hero Ep.1](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy
- 🎥 [ML Specialization Course 2 Week 1](https://www.coursera.org/specializations/machine-learning-introduction)

**Concepts:**
```
Neuron                → Weighted sum + activation
Activation Functions  → ReLU, Sigmoid, Tanh
Forward Pass          → Data flows through layers
Backpropagation       → Gradients flow backward
Loss Function         → How wrong are we?
Optimizer             → How to get less wrong
```

---

#### Week 11: Your First Deep Learning Model
**Goal:** Build and train a neural network with PyTorch.

**Resources:**
- 🎥 [PyTorch Tutorial for Beginners](https://www.youtube.com/watch?v=EMXfZB8FVUA)
- 📗 [Official PyTorch 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

model = SimpleNet()
```

**Milestone Project:** MNIST Digit Classifier — 99%+ accuracy 🔢

---

#### Week 12: Capstone Projects & Next Steps
**Goal:** Ship 2 portfolio projects, plan your next roadmap.

**Project Options (pick 2):**
1. 🎭 **Sentiment Analyzer** — Fine-tune a model on movie reviews
2. 🌸 **Image Classifier** — Train on a dataset you care about
3. 📰 **Spam Detector** — Email/SMS classification
4. 📈 **Time Series Forecaster** — Predict sales or stocks

**Next Steps:**
- → [ML Engineer Roadmap](02-ml-engineer-roadmap.md) — Go deeper on production ML
- → [LLM Engineer Roadmap](03-llm-engineer-roadmap.md) — Jump to cutting-edge AI
- → [Prompt Engineer Roadmap](05-prompt-engineer-roadmap.md) — Quickest career pivot

---

## 📊 Progress Tracker

Copy this to your notes app and check off as you go:

```
PHASE 1: Python & Tools
  Week 1: [ ] Python basics  [ ] Quiz game project
  Week 2: [ ] NumPy/Pandas   [ ] Data exploration project
  Week 3: [ ] VS Code setup  [ ] First GitHub repo

PHASE 2: Math
  Week 4: [ ] 3B1B Linear Algebra videos  [ ] NumPy matrix operations
  Week 5: [ ] StatQuest stats              [ ] Probability notebook

PHASE 3: Machine Learning
  Week 6: [ ] Big picture of ML      [ ] Kaggle account setup
  Week 7: [ ] Linear/Logistic Reg.   [ ] House price predictor
  Week 8: [ ] Trees & Forests        [ ] Titanic challenge
  Week 9: [ ] Evaluation metrics     [ ] Beat baseline model

PHASE 4: Neural Networks
  Week 10: [ ] Understand backprop  [ ] Can explain NNs to a friend
  Week 11: [ ] PyTorch basics       [ ] MNIST classifier
  Week 12: [ ] 2 capstone projects  [ ] Published to GitHub
```

---

## 💬 Community & Support

- 💬 [Discord: AI Magic Mastery](https://discord.gg/codebeez) — Ask questions, share progress
- 🐦 [Twitter/X: @abidlit](https://x.com/abidlit) — Follow Abid for tips
- 📧 Email: hello@codebeez.ai

---

*Part of the [AI Magic Mastery](https://aimagicmastery.codebeez.ai) learning ecosystem by CodeBeez × Abid Redwan*
