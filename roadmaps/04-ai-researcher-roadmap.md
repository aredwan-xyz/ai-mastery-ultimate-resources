# 🔬 AI Researcher Roadmap
### Strong CS/Math Background → Publishing Original Research in 24 Weeks

> **By Abid Redwan & CodeBeez** | [AI Magic Mastery](https://aimagicmastery.codebeez.ai)

---

## 📋 Prerequisites

- [ ] Solid programming (Python + PyTorch or willingness to master it fast)
- [ ] Comfort with university-level math (linear algebra, calculus, probability)
- [ ] Ability to read technical writing patiently
- [ ] ~15–20 hours/week — research is deep work
- [ ] Genuine curiosity — you're going to live at the edge of the unknown

> This is the most demanding roadmap. It assumes you've done something like the [ML Engineer](02-ml-engineer-roadmap.md) or [LLM Engineer](03-llm-engineer-roadmap.md) track, or equivalent.

---

## 🎯 Outcomes

By week 24 you will:
- ✅ Have deep mathematical foundations for modern AI
- ✅ Read, understand, and critique cutting-edge papers
- ✅ Reproduce published results from scratch
- ✅ Identify a genuine research gap
- ✅ Run rigorous experiments and write them up
- ✅ Submit to a workshop, arXiv, or conference

---

## 📅 Roadmap

### Module 1 — Mathematical Foundations (Weeks 1–4)

You can't push the frontier with borrowed intuition. Build the real thing.

#### Week 1–2: Linear Algebra & Matrix Calculus
- 📗 [Mathematics for Machine Learning (free book)](https://mml-book.github.io/) — Ch. 2–5
- 🎥 [MIT 18.06 – Gilbert Strang](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- 📄 [The Matrix Calculus You Need for Deep Learning](https://arxiv.org/abs/1802.01528)

#### Week 3: Probability & Information Theory
- 📗 [MML Book Ch. 6](https://mml-book.github.io/)
- 📄 [Deep Learning Book (Goodfellow) Ch. 3](https://www.deeplearningbook.org/) — probability
- 🎥 [Information Theory – Mutual Information channel](https://www.youtube.com/@mutualinformation)

#### Week 4: Optimization
```
Convex optimization → the theory
SGD, Momentum, Adam → the practice
Second-order        → why we usually don't
```
- 📗 [Convex Optimization – Boyd (free)](https://web.stanford.edu/~boyd/cvxbook/)
- 📄 [An Overview of Gradient Descent Optimization](https://arxiv.org/abs/1609.04747)

---

### Module 2 — Deep Learning Theory (Weeks 5–9)

#### Week 5–6: Build Everything From Scratch
The single best way to *understand* is to *build*.
- 🎥 [Karpathy: Neural Networks Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — all of it
- 📗 [Deep Learning Book](https://www.deeplearningbook.org/) — Part II

**Do:** implement backprop, a CNN, and a Transformer in pure NumPy/PyTorch — no high-level APIs.

#### Week 7: The Transformer, Deeply
- 📄 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 📄 [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — implement line by line
- 🎥 [Stanford CS224N](https://web.stanford.edu/class/cs224n/)

#### Week 8: Vision & Multimodal
- 🎥 [Stanford CS231n](http://cs231n.stanford.edu/)
- 📄 [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)
- 📄 [CLIP](https://arxiv.org/abs/2103.00020)

#### Week 9: Generative Models
```
VAEs        → latent variable models
GANs        → adversarial training
Diffusion   → the dominant paradigm (DDPM, score-based, flow matching)
```
- 📄 [DDPM](https://arxiv.org/abs/2006.11239) · [Score-Based Models](https://arxiv.org/abs/2011.13456)
- 🎥 [MIT 6.S191: Deep Generative Modeling](https://www.youtube.com/watch?v=Dmm4UG-6jxA)

---

### Module 3 — Reading & Reproducing Research (Weeks 10–14)

#### Week 10: How to Read a Paper (a real skill)
> Use the **three-pass method**: (1) title/abstract/figures, (2) main claims + method, (3) full detail + math.
- 📄 [How to Read a Paper – S. Keshav](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf)
- 🛠️ Tools: [arXiv](https://arxiv.org/) · [Papers With Code](https://paperswithcode.com/) · [Semantic Scholar](https://www.semanticscholar.org/) · [Connected Papers](https://www.connectedpapers.com/) · [alphaXiv](https://www.alphaxiv.org/)

#### Week 11–12: Reproduce a Paper End-to-End
Pick one influential paper with public data and **reproduce its headline result from scratch.** This is the single highest-signal thing you can do.
- Good candidates: a GAN, a small Transformer LM, a diffusion model, an RL agent
- Track everything in [Weights & Biases](https://wandb.ai/)

#### Week 13: Join the Research Conversation
- Follow [@_akhaliq](https://x.com/_akhaliq) (Daily Papers), [Hugging Face Daily Papers](https://huggingface.co/papers)
- 🎥 [Yannic Kilcher paper breakdowns](https://www.youtube.com/@YannicKilcher)
- Read [The Batch](https://www.deeplearning.ai/the-batch/) & [Import AI](https://importai.substack.com/)

#### Week 14: Pick Your Specialization
```
LLMs & reasoning     · Alignment & interpretability
Generative models    · Reinforcement learning
Multimodal           · Efficient ML / systems
ML theory            · AI for science
```

---

### Module 4 — Frontier Depth (Weeks 15–19)

Go deep in your chosen area. For each week: read 3–5 seminal + recent papers, and implement one idea.

**Example — LLMs & reasoning track:**
- [InstructGPT / RLHF](https://arxiv.org/abs/2203.02155)
- [Chain-of-Thought](https://arxiv.org/abs/2201.11903)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Scaling Laws](https://arxiv.org/abs/2001.08361)

**Example — Interpretability track:**
- [Anthropic: Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- [Sparse Autoencoders / Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)

> Keep a **research log**: every paper → 3 bullets (idea, method, what's missing). The "what's missing" column becomes your research ideas.

---

### Module 5 — Original Research (Weeks 20–24)

#### Week 20: Find the Gap
Great research questions come from:
- A limitation stated in a paper's "future work"
- A result that surprised you / seems wrong
- Combining two ideas nobody has combined
- A simpler explanation for a known phenomenon

#### Week 21–23: Run Rigorous Experiments
```
✓ Strong baselines (this is where most projects fail)
✓ Controlled ablations (change ONE thing at a time)
✓ Multiple seeds + report variance
✓ Honest negative results
✓ Reproducible code (seed, config, environment)
```

#### Week 24: Write It Up & Share
- 📄 [How to Write a Great Research Paper – Simon Peyton Jones](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/)
- Use the [NeurIPS](https://neurips.cc/) / [ICML](https://icml.cc/) LaTeX template (Overleaf)
- **Where to submit as a newcomer:** a workshop (lower bar, great feedback), [arXiv](https://arxiv.org/), or open-review venues
- Post a thread + blog explainer — communication *is* part of research

**Top venues:** NeurIPS · ICML · ICLR · ACL · EMNLP · CVPR · AAAI

---

## 🛠️ The Researcher's Toolkit

```
Compute      → university cluster, Lambda Labs, RunPod, Modal, TPU Research Cloud
Tracking     → Weights & Biases, Hydra (configs)
Reading      → arXiv, Papers With Code, Connected Papers, Zotero
Writing      → Overleaf (LaTeX), Typst
Frameworks   → PyTorch, JAX, Hugging Face, Lightning
Community    → EleutherAI Discord, ML Collective, local reading groups
```

---

## 📊 Progress Tracker

```
M1 Math:          [ ] Linear algebra  [ ] Probability/Info  [ ] Optimization
M2 DL Theory:     [ ] From scratch  [ ] Transformer  [ ] Vision  [ ] Generative
M3 Reproduce:     [ ] Read fluently  [ ] Reproduced a paper  [ ] In the conversation
M4 Frontier:      [ ] Specialization chosen  [ ] Research log with gaps
M5 Research:      [ ] Question  [ ] Experiments  [ ] Write-up  [ ] Submitted
```

## Realistic Expectations
Research is non-linear. Most experiments fail. A "result" can take months. This roadmap gets you to the **starting line of real research** — the rest is persistence, mentorship, and taste. Consider a research-focused MS/PhD, a residency (e.g. industry AI residencies), or open-source collectives like [EleutherAI](https://www.eleuther.ai/).

---

*Part of the [AI Magic Mastery](https://aimagicmastery.codebeez.ai) ecosystem by CodeBeez × Abid Redwan*
