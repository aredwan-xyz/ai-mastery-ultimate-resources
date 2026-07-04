# 🎯 AI/ML Interview Prep
### 300+ Questions from Top AI Companies — AI Magic Mastery by CodeBeez × Abid Redwan

> Used by engineers who've landed roles at Google, Meta, OpenAI, Anthropic, Apple, Amazon, and top AI startups.

---

## 📋 Interview Categories

| Category | Questions | File |
|----------|-----------|------|
| 🧮 ML Theory & Statistics | 60 | [ml-theory.md](ml-theory-questions.md) |
| 🧠 Deep Learning & Transformers | 60 | [deep-learning.md](deep-learning-questions.md) |
| 🤖 LLMs & Generative AI | 70 | [llm-questions.md](llm-questions.md) |
| ⚙️ MLOps & Systems Design | 50 | [mlops-systems.md](mlops-questions.md) |
| 💻 Coding Challenges | 60 | [coding-challenges.md](coding-challenges.md) |

---

## 🏢 Company-Specific Patterns

### Google / DeepMind
- Heavy emphasis on ML theory (Bayesian methods, probability theory)
- Systems design for large-scale ML infrastructure
- Paper discussions (know 3–5 key papers deeply)
- Coding in Python + pseudo-code for algorithms

### Meta AI
- Applied ML with business impact metrics
- Large-scale recommendation systems
- A/B testing and experimentation
- PyTorch internals

### OpenAI / Anthropic
- Deep knowledge of transformer architecture
- Alignment and safety research understanding
- LLM training and fine-tuning
- Novel problem-solving and research intuition

### Startups (Series A–C)
- Can you ship fast?
- End-to-end ML ownership
- Product thinking + ML thinking combined
- "Tell me about a project you owned fully"

---

## 🧮 ML Theory — 60 Questions

### Fundamentals (Q1–Q20)

**Q1: Explain the bias-variance tradeoff with an example.**
> **Answer:** Bias = error from wrong assumptions (model too simple = underfitting). Variance = error from sensitivity to data fluctuations (model too complex = overfitting). 
> 
> Example: Linear regression on a non-linear dataset has high bias. A 100th-degree polynomial has high variance. 
> 
> Total Error = Bias² + Variance + Irreducible Noise. Goal: find the sweet spot.
> 
> Reduce bias: more complex model, more features, reduce regularization.
> Reduce variance: simpler model, regularization (L1/L2), more training data, dropout, ensemble methods.

**Q2: What is gradient descent? Compare SGD, mini-batch, and full-batch.**
> **Answer:**
> Gradient descent is an optimization algorithm that iteratively updates parameters θ in the direction that decreases the loss L:
> `θ = θ - α * ∇L(θ)`
> 
> | Method | Batch Size | Pros | Cons |
> |--------|-----------|------|------|
> | Full-batch GD | All data | Stable gradients, guaranteed convergence | Slow, can't fit large data in memory |
> | SGD | 1 sample | Fast updates, escapes local minima | Noisy, slow convergence |
> | Mini-batch | 32–512 | Best of both worlds, GPU parallelism | Hyperparameter to tune |
> 
> In practice: always use mini-batch with Adam/AdamW.

**Q3: Explain L1 vs L2 regularization. When would you use each?**
> **Answer:**
> - **L1 (Lasso)**: Adds `λ * Σ|wᵢ|` to loss. Produces sparse weights (many → exactly 0). Use when you want feature selection or believe most features are irrelevant.
> - **L2 (Ridge)**: Adds `λ * Σwᵢ²` to loss. Shrinks all weights towards 0 but rarely to exactly 0. Use for most cases — handles correlated features better, differentiable everywhere.
> - **Elastic Net**: Combines both. Best when you want some sparsity + ridge's stability.

**Q4: How does a Random Forest reduce variance compared to a single decision tree?**
> **Answer:** Random Forest uses two sources of randomness to decorrelate trees:
> 1. **Bootstrap sampling**: Each tree trains on a random subset (~63%) of training data with replacement.
> 2. **Feature randomness**: At each split, only a random subset of features (√n_features typically) is considered.
> 
> Because trees are trained on different data and use different features, their errors are uncorrelated. When you average uncorrelated predictions, variance decreases by a factor of 1/n_trees (approximately), while bias stays roughly the same.

**Q5: What is the difference between generative and discriminative models?**
> **Answer:**
> - **Discriminative**: Learn P(y|x) — the decision boundary directly. Examples: Logistic Regression, SVM, Neural Networks, Random Forest. Generally higher accuracy for classification.
> - **Generative**: Learn P(x, y) or P(x|y) and use Bayes' theorem. Examples: Naïve Bayes, LDA, VAEs, GANs, language models. Can generate new samples, handle missing features, work with less data.

**Q6: Explain precision, recall, F1, and AUC-ROC. When does each matter?**
> **Answer:**
> - **Precision** = TP / (TP + FP) — "Of predicted positives, how many were correct?" Use when false positives are costly (spam detection: don't block legit emails).
> - **Recall** = TP / (TP + FN) — "Of actual positives, how many did we catch?" Use when false negatives are costly (cancer detection: don't miss sick patients).
> - **F1** = 2 * (P * R) / (P + R) — Harmonic mean. Use when you need balance between P and R.
> - **AUC-ROC** — Area under ROC curve (TPR vs FPR at different thresholds). 0.5 = random, 1.0 = perfect. Use when you need to compare models independent of threshold, or classes are balanced.
> - **AUC-PR** (Precision-Recall AUC) — Better for imbalanced datasets. Use when positive class is rare.

**Q7: How do you handle class imbalance?**
> **Answer:** Multiple strategies, often combined:
> 1. **Resampling**: Oversample minority (SMOTE, ADASYN) or undersample majority.
> 2. **Class weights**: Pass `class_weight='balanced'` or compute weights inversely proportional to frequency.
> 3. **Threshold tuning**: Default 0.5 threshold isn't optimal — tune on validation set using F1 or business metric.
> 4. **Metric choice**: Use F1, AUC-PR, or MCC — not accuracy (which is misleading with imbalance).
> 5. **Algorithm choice**: Tree-based models and SVMs work better than logistic regression on imbalanced data with appropriate weights.

**Q8: What is cross-validation? Explain k-fold vs stratified k-fold.**
> **Answer:** CV estimates model performance on unseen data by rotating the validation set. Reduces dependence on a single train/val split.
> 
> - **K-Fold**: Split data into k folds. Train on k-1, validate on 1. Repeat k times. Average the scores.
> - **Stratified K-Fold**: Like k-fold but ensures each fold has the same class distribution as the original. **Use this for classification**. Especially important with imbalanced classes.
> - **Time Series CV**: Walk-forward validation. Never let future data leak into training.

**Q9: Explain the curse of dimensionality.**
> **Answer:** As the number of features (dimensions) increases:
> 1. **Data becomes sparse**: Volume of space grows exponentially. To maintain density, data requirements grow exponentially.
> 2. **Distance metrics lose meaning**: In high dimensions, the difference between nearest and farthest neighbor shrinks (everyone becomes "far" and "similar").
> 3. **Model complexity needed**: More features can mean more parameters needed.
> 
> Mitigations: Feature selection, dimensionality reduction (PCA, t-SNE, UMAP), regularization, more data.

**Q10: What is data leakage and how do you prevent it?**
> **Answer:** Data leakage occurs when information from outside the training dataset is used to create the model, leading to overly optimistic results that don't generalize.
> 
> Types:
> - **Target leakage**: Features contain information about the target that wouldn't be available at prediction time.
> - **Train-test contamination**: Preprocessing (normalization, imputation) done on full dataset before splitting.
> 
> Prevention:
> - Always split first, then preprocess
> - Use sklearn Pipelines to ensure transforms fit only on train set
> - Temporal splits for time-series data
> - Carefully examine features that have surprisingly high importance

---

### Advanced Topics (Q11–Q20)

**Q11: Explain bagging vs boosting.**
> - **Bagging**: Train models in parallel on random subsets, aggregate predictions (majority vote / average). Reduces variance. Examples: Random Forest.
> - **Boosting**: Train models sequentially, each correcting the previous one's errors. Reduces bias + variance. Examples: AdaBoost, XGBoost, LightGBM, CatBoost. Generally better performance but more prone to overfitting.

**Q12: How does XGBoost work?**
> XGBoost builds an ensemble of decision trees sequentially. Each tree fits the residuals (pseudo-residuals = negative gradient of loss) of the previous ensemble. Key innovations: regularization in the objective function (L1+L2 on leaf scores), second-order Taylor approximation for the loss, tree pruning via gain threshold, parallel and distributed training via histogram-based approximate tree construction.

**Q13: What is the kernel trick in SVMs?**
> The kernel trick implicitly maps data into a higher-dimensional space where it becomes linearly separable, without actually computing the high-dimensional representation. Instead, we only compute the dot product in the transformed space using a kernel function K(xᵢ, xⱼ). Common kernels: Linear (K = xᵢᵀxⱼ), RBF/Gaussian (K = exp(-γ||xᵢ-xⱼ||²)), Polynomial.

**Q14: Explain PCA. How do you choose the number of components?**
> PCA finds orthogonal directions of maximum variance in the data. Steps: center data → compute covariance matrix → eigendecomposition → sort eigenvectors by eigenvalue → project data onto top k eigenvectors.
> 
> Choose k: plot cumulative explained variance ratio. Common heuristic: 95% explained variance. Or use elbow method on scree plot. Or cross-validate with downstream task performance.

**Q15: What is the difference between MLE and MAP estimation?**
> - **MLE (Maximum Likelihood Estimation)**: Find parameters θ that maximize P(data|θ). No prior assumptions about θ. Can overfit with small data.
> - **MAP (Maximum A Posteriori)**: Find θ that maximizes P(θ|data) = P(data|θ) × P(θ). Incorporates a prior P(θ). A Gaussian prior = L2 regularization. Laplace prior = L1 regularization. With uniform prior, MAP = MLE.

**Q16–Q20**: [See full 60 questions →](ml-theory-questions.md)

---

## 🧠 Deep Learning — Top 15 Questions

**Q1: Explain backpropagation mathematically.**
> Backpropagation applies the chain rule to efficiently compute gradients of the loss with respect to all parameters.
> 
> For output layer: δL = ∂L/∂z_L
> For hidden layer l: δl = (W_{l+1}ᵀ δ_{l+1}) ⊙ σ'(z_l)
> Gradient w.r.t. weights: ∂L/∂W_l = δl × a_{l-1}ᵀ
> Gradient w.r.t. bias: ∂L/∂b_l = δl

**Q2: Why does batch normalization work?**
> BN normalizes layer activations to have zero mean and unit variance, then applies learnable scale (γ) and shift (β). Benefits: reduces internal covariate shift (unstable activation distributions during training); enables higher learning rates; acts as regularizer (reduces need for dropout); faster convergence. In inference, uses running mean/variance from training.

**Q3: What is the vanishing gradient problem? How do modern architectures address it?**
> In deep networks, gradients can become exponentially small as they propagate backward through many layers (especially with sigmoid/tanh activations), making early layers learn extremely slowly.
> 
> Solutions: 
> - ReLU activation (gradients = 1 or 0, no squashing)
> - Residual connections (gradients flow directly through skip connections)
> - Layer normalization (normalizes before activation)
> - Better initialization (Xavier/He initialization)
> - Gradient clipping

**Q4: Explain the self-attention mechanism in Transformers.**
> Self-attention allows each token to attend to all other tokens. Given input X:
> 1. Project X to Q (query), K (key), V (value): Q=XWQ, K=XWK, V=XWV
> 2. Compute attention scores: scores = QKᵀ / √d_k
> 3. Apply softmax: weights = softmax(scores) (optionally with mask)
> 4. Weighted sum: output = weights × V
> 
> Multi-head = run h attention operations in parallel with different projections, concatenate results.
> 
> Complexity: O(n²d) where n=sequence length, d=dimension. Bottleneck for long sequences.

**Q5: What is dropout and why does it help?**
> Randomly zeros out p fraction of neurons during each forward pass in training. At inference, no dropout but scale activations by (1-p). 
> Why it works: (1) Forces the network to learn redundant representations (ensemble effect) — equivalent to training 2^n different networks and averaging. (2) Prevents co-adaptation of features. (3) Acts as regularization.

---

## 🤖 LLM Questions — Top 20

**Q1: Explain how GPT models are pre-trained and what RLHF adds.**
> **Pre-training**: Next-token prediction on massive text corpora (trillions of tokens). Objective: maximize P(token_t | token_{1:t-1}). This teaches language, factual knowledge, reasoning patterns.
> 
> **RLHF (3 stages)**:
> 1. Supervised Fine-Tuning (SFT): Train on curated instruction-following examples
> 2. Reward Model: Train a model to predict human preference between two responses
> 3. PPO Optimization: Fine-tune the SFT model to maximize reward model score, with KL penalty to prevent drift from SFT model

**Q2: When would you use RAG vs fine-tuning?**
> | Scenario | RAG | Fine-Tuning |
> |---------|-----|------------|
> | Private knowledge base | ✅ Best | ⚠️ Expensive |
> | Dynamic/changing info | ✅ Best | ❌ Needs retraining |
> | New task format/style | ❌ | ✅ Best |
> | Domain vocabulary | ⚠️ | ✅ Best |
> | Cost constraints | ✅ Cheaper | ❌ Training cost |
> | Hallucination reduction | ✅ Better | ⚠️ Helps some |
> 
> **Rule of thumb**: Try prompting first → RAG if knowledge needed → Fine-tuning only if behavior change needed.

**Q3: How does LoRA work? Explain the math.**
> LoRA (Low-Rank Adaptation) decomposes weight updates into two low-rank matrices. Instead of updating W ∈ ℝ^{d×k}, we learn W + ΔW where ΔW = BA, B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, with r << min(d,k).
> 
> Training: only A and B are trainable (typically 0.1% of parameters). Inference: merge W+BA for zero overhead.
> 
> Why it works: weight updates for pre-trained models have low intrinsic rank — the model doesn't need to change in many directions to adapt to new tasks.

**Q4: What causes hallucination in LLMs and how do you mitigate it?**
> Causes: LLMs are trained to be fluent, not accurate. They interpolate training patterns even when no relevant knowledge exists.
> 
> Mitigations:
> - RAG: Ground responses in retrieved facts
> - Prompt engineering: "Only answer if you're confident. Say 'I don't know' otherwise."
> - Temperature: Lower temperature for factual queries
> - Citations: Ask model to cite sources (verifiable)
> - Consistency sampling: Multiple runs + compare
> - Fact-checking tools: LLM + search as validator

**Q5: Explain the difference between BPE, WordPiece, and SentencePiece tokenization.**
> - **BPE** (GPT, Llama, Mistral): Starts with characters, iteratively merges most frequent pairs. Deterministic, vocabulary built from data.
> - **WordPiece** (BERT): Similar to BPE but merges pairs that maximize likelihood rather than frequency. Marks sub-words with ## prefix.
> - **SentencePiece** (T5, ALBERT): Language-agnostic. Works directly on raw text (no pre-tokenization). Treats spaces as characters. Best for multilingual.

**Q6: What is the purpose of KV Cache in LLM inference?**
> During autoregressive generation, keys (K) and values (V) for previously generated tokens don't change. KV cache stores these computations to avoid recomputing them on every new token generation.
> 
> Without cache: O(n²) time for n tokens.
> With cache: O(n) time for n tokens (only compute new token's attention to cached KV).
> 
> Memory: KV cache grows with sequence length. At 32K context with Llama 3.1 70B: ~80GB just for KV cache. PagedAttention (vLLM) manages this efficiently.

---

## 💻 Coding Challenges

**C1: Implement attention from scratch**
```python
import torch
import torch.nn.functional as F
import math

def attention(Q, K, V, mask=None):
    """Scaled dot-product attention.
    Q, K, V: (batch, seq_len, d_k)
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights
```

**C2: Implement gradient descent from scratch**
```python
import numpy as np

def gradient_descent(X, y, lr=0.01, n_iter=1000):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(n_iter):
        predictions = X @ theta
        errors = predictions - y
        gradient = (2/m) * X.T @ errors
        theta -= lr * gradient
    
    return theta
```

**C3: Implement k-means clustering**
```python
import numpy as np

def kmeans(X, k, n_iter=100):
    # Random initialization
    centers = X[np.random.choice(len(X), k, replace=False)]
    
    for _ in range(n_iter):
        # Assign each point to nearest center
        distances = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update centers
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    
    return centers, labels
```

**C4: Implement binary cross-entropy loss**
```python
import numpy as np

def binary_cross_entropy(y_true, y_pred, epsilon=1e-7):
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

---

## 📅 Interview Prep Timeline

```
1 Week Out:
  Day 1-2: ML Theory (Q1-Q20)
  Day 3-4: Deep Learning fundamentals
  Day 5-6: LLM questions relevant to role
  Day 7:   Coding practice + review weak spots

2 Weeks Out:
  Week 1: Core theory + implement algorithms from scratch
  Week 2: System design + LLM/production questions

1 Month Out:
  Week 1: All theory, code every algorithm
  Week 2: Deep learning + transformers
  Week 3: LLM/GenAI + MLOps
  Week 4: Mock interviews + review
```

---

*AI Magic Mastery by CodeBeez × Abid Redwan | [aimagicmastery.codebeez.ai](https://aimagicmastery.codebeez.ai)*
