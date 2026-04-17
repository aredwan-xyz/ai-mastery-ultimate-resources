# 🤖 LLM Engineer Roadmap
### From ML Basics → Production LLM Systems in 12 Weeks

> **By Abid Redwan & CodeBeez** | [AI Magic Mastery](https://aimagicmastery.codebeez.ai)

---

## 📋 Prerequisites

- [ ] Python proficiency (functions, classes, decorators)
- [ ] Basic ML knowledge (training loops, loss functions)
- [ ] Familiarity with PyTorch or TensorFlow
- [ ] Comfortable with APIs and HTTP requests
- [ ] Git basics

---

## 🎯 Outcomes

By week 12 you will:
- ✅ Understand Transformer architecture deeply
- ✅ Build RAG systems from scratch
- ✅ Fine-tune open-source LLMs (Llama, Mistral, Qwen)
- ✅ Deploy LLM applications to production
- ✅ Build AI agents with tool use
- ✅ Have 5 portfolio-worthy LLM projects

---

## 📅 Roadmap

### Module 1: Understanding LLMs (Weeks 1–2)

#### Week 1: The Transformer Architecture
**Resources:**
- 🎥 [Andrej Karpathy: Let's build GPT from scratch](https://youtu.be/kCc8FmEb1nY) — **THE essential video, 2h**
- 🎥 [3Blue1Brown: Attention Mechanism](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- 📄 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Read the original paper
- 📗 [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/) — Best visual explanation

**Concepts to Master:**
```
Tokenization                  → BPE, SentencePiece, Tiktoken
Token Embeddings              → Dense vector representations
Positional Encoding           → RoPE, ALiBi, learned
Multi-Head Self-Attention     → Q, K, V matrices
Feed-Forward Layers           → SwiGLU, GeLU activations
Layer Normalization           → RMSNorm (modern models)
KV Cache                      → Inference efficiency trick
```

**Build It:**
```python
# Implement attention from scratch
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights
```

---

#### Week 2: The LLM Landscape
**Resources:**
- 📗 [Hugging Face NLP Course Ch. 1–3](https://huggingface.co/learn/nlp-course)
- 🎥 [LLM Bootcamp: State of GPT – Karpathy](https://www.youtube.com/watch?v=bZQun8Y4L2A)
- 📄 [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

**Understanding the Model Zoo:**
```
Encoder-only    → BERT, DeBERTa, RoBERTa  (classification, NER)
Decoder-only    → GPT family, Llama, Mistral (generation)
Encoder-Decoder → T5, BART, mT5           (translation, summarization)

Closed Source:
  GPT-4o / o3       → OpenAI's best
  Claude 3.7 Sonnet → Anthropic's best
  Gemini 2.5 Pro    → Google's best

Open Weights (2025 Best):
  Llama 3.3 70B     → Meta, all-rounder
  DeepSeek-R1       → Reasoning tasks
  Qwen2.5 72B       → Coding + math
  Mistral Large 2   → Efficient, multilingual
  Phi-4 14B         → Tiny but powerful
```

---

### Module 2: Working with LLMs (Weeks 3–4)

#### Week 3: LLM APIs & Prompting
**Resources:**
- 📗 [OpenAI API Docs](https://platform.openai.com/docs)
- 📗 [Anthropic API Docs](https://docs.anthropic.com)
- 📗 [Prompt Engineering Guide – DAIR.AI](https://www.promptingguide.ai/)

**Hands-On:**
```python
# OpenAI
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful AI expert."},
        {"role": "user", "content": "Explain attention mechanisms."}
    ],
    temperature=0.7,
    max_tokens=1000
)
print(response.choices[0].message.content)

# Anthropic Claude
import anthropic
client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain chain-of-thought prompting."}]
)

# Run locally with Ollama
import ollama
response = ollama.chat(model='llama3.2', messages=[
    {'role': 'user', 'content': 'Why is the sky blue?'}
])
```

**Prompt Engineering Techniques:**
```
Zero-Shot           → Direct question
Few-Shot            → 2-5 examples
Chain-of-Thought    → "Think step by step"
System Prompts      → Persona, constraints, format
Output Formatting   → JSON mode, structured output
Self-Consistency    → Sample + vote
```

**Milestone:** Build a multi-turn chatbot with conversation history 💬

---

#### Week 4: Embeddings & Semantic Search
**Resources:**
- 📗 [Hugging Face: Embeddings & Semantic Search](https://huggingface.co/learn/nlp-course/chapter5/2)
- 🎥 [Sentence Transformers Tutorial](https://www.youtube.com/watch?v=OATCgQtNX2o)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('BAAI/bge-m3')  # 2025 SOTA embedding model

sentences = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "What is artificial intelligence?"
]

embeddings = model.encode(sentences)

# Cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Query similarity
query = model.encode("What is ML?")
similarities = [cosine_sim(query, emb) for emb in embeddings]
```

**Best Embedding Models (2025):**
```
text-embedding-3-large  → OpenAI, best overall
BGE-M3                  → Open source, multilingual, SOTA
Nomic Embed             → Open source, long context
Cohere Embed v3         → Commercial, multilingual
```

---

### Module 3: RAG Systems (Weeks 5–6)

#### Week 5: Building RAG from Scratch
**Goal:** Understand every component of a RAG pipeline.

```
Documents → Load → Chunk → Embed → Store in Vector DB
                                         ↓
Query → Embed → Similarity Search → Retrieve Top-K
                                         ↓
Prompt = "Context: {docs}\n\nQuestion: {query}" → LLM → Answer
```

**Resources:**
- 🎥 [Building RAG from Scratch – LlamaIndex](https://www.youtube.com/watch?v=TRjq7t2Ms5I)
- 📗 [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)

**Code:**
```python
# Full RAG pipeline with ChromaDB + LangChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# 1. Load documents
loader = WebBaseLoader("https://docs.python.org/3/tutorial/")
docs = loader.load()

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# 3. Embed + Store
vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings())

# 4. RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
)
answer = qa_chain.invoke("How do I use list comprehensions?")
```

**Milestone Project:** RAG system over your own PDF library 📚

---

#### Week 6: Advanced RAG
**Topics:**
```
Chunking Strategies    → Semantic, hierarchical, sentence window
Hybrid Search          → Dense + sparse (BM25) retrieval
Reranking              → Cross-encoder reranking (BGE, Cohere)
Query Rewriting        → HyDE, multi-query, step-back prompting
Contextual Compression → Only return relevant parts
Evaluation             → RAGAS framework
```

**Resources:**
- 📗 [Advanced RAG Guide – LlamaIndex](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/advanced_retrieval/)
- 🎥 [RAGAs Evaluation Framework](https://www.youtube.com/watch?v=E6rY_UWGahw)

---

### Module 4: Fine-Tuning (Weeks 7–8)

#### Week 7: When & How to Fine-Tune

**Decision Framework:**
```
Prompting (try first)   → Fastest, no training needed
RAG                     → Knowledge-intensive tasks
Fine-Tuning             → Style, format, domain, small model
  └── Full FT           → Need 8+ A100s, rare
  └── LoRA / QLoRA      → The standard approach, 1-2 GPUs
  └── DPO / ORPO        → Alignment, preference learning
```

**Resources:**
- 🎥 [Fine-Tuning LLMs – Maxime Labonne](https://www.youtube.com/watch?v=eC6Hd1hFvos)
- 📗 [Unsloth Docs](https://docs.unsloth.ai/) — 2-5x faster, easiest fine-tuning
- 📗 [LLaMA Factory Guide](https://github.com/hiyouga/LLaMA-Factory)

```python
# Fine-tuning with Unsloth (fastest method, 2025)
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,          # LoRA rank
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
```

---

#### Week 8: Training & Evaluation
**Resources:**
- 📗 [Hugging Face TRL Documentation](https://huggingface.co/docs/trl/)
- 📗 [LLM Evaluation Guide](https://huggingface.co/docs/evaluate)

**Evaluation Frameworks:**
```
LM-Eval-Harness    → Eleuther AI, academic benchmarks
RAGAS              → RAG pipeline evaluation
PromptBench        → Robustness evaluation
MT-Bench           → Instruction following
OpenAI Evals       → Custom evaluation framework
```

**Milestone Project:** Fine-tune Llama 3.2 3B on a domain-specific dataset 🦙

---

### Module 5: AI Agents (Week 9)

**Resources:**
- 🎥 [Building AI Agents – DeepLearning.AI](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)
- 📗 [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

# Simple ReAct Agent
llm = ChatOpenAI(model="gpt-4o")
search = DuckDuckGoSearchRun()

tools = [search]
agent = llm.bind_tools(tools)

# Agent state
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def agent_node(state):
    response = agent.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
# ... add tool execution, routing logic
```

---

### Module 6: Production Deployment (Weeks 10–11)

#### Week 10: Serving LLMs
**Resources:**
- 📗 [vLLM Documentation](https://docs.vllm.ai/)
- 📗 [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- 🎥 [Deploying LLMs with vLLM](https://www.youtube.com/watch?v=ACank_sS_mE)

```python
# FastAPI + vLLM serving
from vllm import LLM, SamplingParams
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: GenerateRequest):
    params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    outputs = llm.generate([request.prompt], params)
    return {"text": outputs[0].outputs[0].text}
```

**Deployment Options:**
```
Local         → Ollama (easiest), LM Studio
Cloud         → Modal (serverless), Replicate, Runpod
Managed       → Together AI, Fireworks AI, Groq (fastest inference)
Enterprise    → AWS Bedrock, GCP Vertex AI, Azure AI
```

---

#### Week 11: Monitoring & Guardrails
**Topics:**
- LLM observability with [LangFuse](https://langfuse.com/)
- Prompt injection protection
- Output validation with [Guardrails AI](https://www.guardrailsai.com/)
- Cost optimization strategies
- A/B testing prompts

---

### Module 7: Capstone Projects (Week 12)

**Project Ideas:**
1. 🏗️ **Full RAG App** — PDF chat, deployed to Hugging Face Spaces
2. 🕵️ **AI Research Agent** — Searches web, reads papers, writes reports
3. 🤝 **Code Review Bot** — GitHub Actions integration
4. 🎙️ **Voice AI Assistant** — Whisper + LLM + TTS pipeline
5. 📊 **AI Data Analyst** — Natural language → SQL → visualization

---

## 🛠️ Full Tech Stack for This Roadmap

```python
# Install everything
pip install openai anthropic
pip install langchain langchain-openai langchain-community
pip install llama-index llama-index-llms-openai
pip install chromadb qdrant-client
pip install sentence-transformers transformers
pip install unsloth  # Fine-tuning
pip install vllm     # Serving
pip install fastapi uvicorn
pip install langfuse # Observability
pip install ragas    # RAG evaluation
```

---

*Part of the [AI Magic Mastery](https://aimagicmastery.codebeez.ai) ecosystem by CodeBeez × Abid Redwan*
