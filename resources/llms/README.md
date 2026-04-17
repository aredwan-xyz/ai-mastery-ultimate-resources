# 🤖 Large Language Models — Complete Resource Guide
### AI Magic Mastery by CodeBeez × Abid Redwan

---

## 🚀 Quick Start

New to LLMs? Start here in this order:
1. 🎥 [Karpathy: Let's build GPT from scratch](https://youtu.be/kCc8FmEb1nY) (2h, watch fully)
2. 📗 [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
3. 📗 [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) (chapters 1-4)
4. 🛠️ [Call your first API](https://platform.openai.com/docs/quickstart)

---

## 📚 Learning Resources

### Courses
| Resource | Level | Cost | Stars |
|----------|-------|------|-------|
| [LLM Engineering Masterclass – AI Magic Mastery](https://aimagicmastery.codebeez.ai) | Advanced | Pro | ⭐⭐⭐⭐⭐ |
| [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) | Intermediate | Free | ⭐⭐⭐⭐⭐ |
| [LLM Bootcamp – FSDL](https://fullstackdeeplearning.com/llm-bootcamp/) | Intermediate | Free | ⭐⭐⭐⭐⭐ |
| [DeepLearning.AI Short Courses](https://www.deeplearning.ai/short-courses/) | Varies | Free | ⭐⭐⭐⭐ |
| [fast.ai Practical Deep Learning](https://course.fast.ai/) | Intermediate | Free | ⭐⭐⭐⭐⭐ |

### Books
| Book | Author | Level | Free? |
|------|--------|-------|-------|
| Build a Large Language Model (From Scratch) | Sebastian Raschka | Intermediate | No |
| Generative Deep Learning | David Foster | Advanced | No |
| Natural Language Processing with Transformers | Lewis Tunstall et al. | Intermediate | [Free online](https://transformersbook.com/) |

---

## 🤗 Open-Source Models (2025)

### Text Models
| Model | Params | Context | Strengths | License |
|-------|--------|---------|-----------|---------|
| Llama 3.3 70B | 70B | 128K | Best overall open model | Llama 3.3 |
| Llama 3.2 3B/11B | 3B/11B | 128K | Efficient, edge deployment | Llama 3.2 |
| Mistral Small 3 | 24B | 128K | Fast, strong reasoning | Apache 2.0 |
| Qwen 2.5 72B | 72B | 128K | Coding, math, multilingual | Apache 2.0 |
| DeepSeek-R1 | 671B/7B | 64K | Reasoning, STEM | MIT |
| Phi-4 | 14B | 16K | Tiny + powerful | MIT |
| Gemma 2 27B | 27B | 8K | Google quality, open | Gemma |

### Code Models
| Model | Strengths |
|-------|-----------|
| Qwen2.5-Coder 32B | Best open-source coding model |
| DeepSeek-Coder-V2 | Strong on HumanEval |
| CodeLlama 70B | Meta's coding specialist |
| Starcoder2 | BigCode, multi-language |

### Multimodal Models
| Model | Capabilities |
|-------|-------------|
| LLaVA-1.6 | Image + text, open source |
| InternVL 2.5 | Strong vision-language |
| Qwen2-VL | Document, chart, math vision |
| Phi-3.5-Vision | Compact multimodal |

---

## 🔧 Fine-Tuning Guide

### When to Fine-Tune
```
✅ Fine-tune when:
  - You need consistent output format/style
  - Domain-specific terminology/knowledge
  - Small model must match big model quality
  - Reducing prompt length at inference

❌ Don't fine-tune when:
  - RAG can solve it
  - Prompting can solve it
  - You don't have 100+ high-quality examples
  - Budget is tight (try prompting/RAG first)
```

### Methods Ranked by Resource Requirement
```
Method          GPU RAM    Training Time    Quality
────────────────────────────────────────────────────
Full FT          80GB+      Days             Best
DoRA             24GB+      Hours            ≈FullFT
LoRA r=64        16GB+      Hours            Great
QLoRA r=64       8GB+       Hours            Very Good
QLoRA r=16       6GB+       Hours            Good
ORPO/DPO         16GB+      Hours            Alignment
```

### Best Tools
```python
# 1. Unsloth — Fastest, most memory-efficient
pip install unsloth

# 2. LLaMA Factory — GUI + CLI
pip install llamafactory

# 3. Axolotl — Production-grade, YAML config
pip install axolotl

# 4. HuggingFace TRL — Official RL library
pip install trl
```

---

## 🔍 RAG Patterns

### Basic RAG
```
Query → Embed → Search VectorDB → Retrieve → Prompt → LLM → Answer
```

### Advanced RAG Techniques
```
1. Hybrid Search      → BM25 + Dense (best retrieval)
2. Reranking         → Cross-encoder after initial retrieval
3. HyDE              → Generate hypothetical document, use to search
4. Multi-Query       → Generate N variants of query, merge results
5. Contextual Chunks → Add chunk + surrounding context
6. Parent-Child Retrieval → Search small, return large
7. Sentence Window   → Index sentences, return surrounding window
```

### RAG Evaluation with RAGAS
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

score = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

---

## 🚀 Serving & Deployment

### Local Inference
```bash
# Ollama (easiest — any model in one command)
ollama run llama3.2

# LM Studio (GUI)
# Download from lmstudio.ai

# llama.cpp (fastest CPU inference)
./llama-cli -m model.gguf -p "Hello world"
```

### Production Serving
```bash
# vLLM (best for production, PagedAttention)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16

# TGI (Hugging Face Text Generation Inference)
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference \
  --model-id meta-llama/Llama-3.1-8B-Instruct
```

### Managed APIs (No Infra)
| Provider | Models | Speed | Cost |
|---------|--------|-------|------|
| [Groq](https://groq.com) | Llama, Mixtral | Fastest (700+ t/s) | Low |
| [Together AI](https://together.ai) | 100+ models | Fast | Low |
| [Fireworks AI](https://fireworks.ai) | 50+ models | Fast | Low |
| [Replicate](https://replicate.com) | 1000+ models | Medium | Pay-per-use |
| [OpenRouter](https://openrouter.ai) | 200+ models | Varies | Varies |

---

## 📄 Essential Papers

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | The Transformer |
| [GPT-3 (Brown et al.)](https://arxiv.org/abs/2005.14165) | 2020 | Scale + few-shot |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | 2022 | RLHF alignment |
| [LoRA](https://arxiv.org/abs/2106.09685) | 2021 | PEFT |
| [QLoRA](https://arxiv.org/abs/2305.14314) | 2023 | 4-bit fine-tuning |
| [RAG (Lewis et al.)](https://arxiv.org/abs/2005.11401) | 2020 | RAG |
| [Chain-of-Thought](https://arxiv.org/abs/2201.11903) | 2022 | CoT prompting |
| [Flash Attention](https://arxiv.org/abs/2205.14135) | 2022 | Memory-efficient attn |
| [Llama 3 (Meta)](https://arxiv.org/abs/2407.21783) | 2024 | Best open model |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 2025 | Open reasoning model |

---

*AI Magic Mastery by CodeBeez × Abid Redwan | [aimagicmastery.codebeez.ai](https://aimagicmastery.codebeez.ai)*
