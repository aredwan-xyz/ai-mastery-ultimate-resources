# 🤖 LLM & AI Glossary
### 150+ Terms — AI Magic Mastery by CodeBeez × Abid Redwan

> From "Attention" to "Zero-Shot" — every term you'll encounter in AI, clearly defined.

---

## A

**Activation Function** — A mathematical function applied to each neuron's output (e.g., ReLU, Sigmoid, GELU, SwiGLU). Introduces non-linearity into neural networks.

**Agent** — An AI system that can perceive its environment, reason, use tools, and take actions autonomously over multiple steps to achieve a goal.

**Alignment** — The research field focused on ensuring AI systems behave in accordance with human values and intentions. Key challenge for advanced AI.

**Attention Mechanism** — The core innovation of Transformers. Allows each token to "attend to" (weigh the importance of) all other tokens in the context. Foundation of modern LLMs.

**Autoregressive** — Generating output one token at a time, where each token depends on all previous tokens. How GPT-style models work.

---

## B

**BERT** (Bidirectional Encoder Representations from Transformers) — Google's 2018 encoder-only model. Reads text in both directions, excellent for classification and understanding tasks.

**BPE** (Byte Pair Encoding) — A tokenization algorithm used by most modern LLMs (GPT, Llama, Mistral). Compresses common character sequences into single tokens.

**Benchmark** — A standardized test used to evaluate AI model performance. Examples: MMLU, HumanEval, MATH, HellaSwag, GSM8K.

---

## C

**Chain-of-Thought (CoT)** — A prompting technique where the model reasons step-by-step before giving a final answer. Dramatically improves performance on reasoning tasks.

**Constitutional AI (CAI)** — Anthropic's technique for training safe, helpful AI using a set of principles ("constitution") to guide self-critique and revision.

**Context Window** — The maximum number of tokens an LLM can process at once (input + output). Modern models: Claude 3.7 (200K), Gemini 1.5 Pro (1M+), GPT-4o (128K).

**Cross-Entropy Loss** — The standard loss function for language model training. Measures how well the model predicts the next token.

---

## D

**Decoder-Only** — Transformer architecture used by GPT, Llama, Mistral. Only uses the decoder stack. Best for text generation.

**Diffusion Model** — A generative model trained to denoise data. Foundation of Stable Diffusion, DALL-E 2/3, Sora. Learns to reverse a noising process.

**DPO** (Direct Preference Optimization) — An alignment technique that directly optimizes for human preferences without a separate reward model. Simpler than RLHF.

**Dropout** — A regularization technique that randomly zeros out neurons during training, preventing overfitting.

---

## E

**Embedding** — A dense vector representation of text, images, or other data in a high-dimensional space. Semantically similar items have similar embeddings.

**Encoder-Only** — Transformer architecture (BERT-style) that processes text bidirectionally. Best for understanding/classification tasks.

**Emergent Behavior** — Capabilities that appear in large models that weren't explicitly trained for and don't appear in smaller models (e.g., arithmetic, reasoning chains).

---

## F

**Few-Shot Learning** — Prompting a model with a few input-output examples to teach it a task without any weight updates.

**Fine-Tuning** — Updating a pre-trained model's weights on a new, task-specific dataset. Expensive but powerful.

**Foundation Model** — A large model trained on broad data that can be adapted to many tasks (GPT-4, Llama, CLIP, etc.).

**Flash Attention** — An efficient attention algorithm that reduces memory usage from O(n²) to O(n) by computing attention in blocks. Essential for long contexts.

---

## G

**GAN** (Generative Adversarial Network) — Architecture with two competing networks: a generator (creates fake data) and discriminator (detects fakes). They compete to improve each other.

**GELU** (Gaussian Error Linear Unit) — An activation function used by BERT, GPT-2/3. Smoother than ReLU.

**GPT** (Generative Pre-trained Transformer) — OpenAI's family of decoder-only language models. GPT-4o is currently among the most capable.

**Gradient Descent** — Optimization algorithm that iteratively moves model parameters in the direction that reduces loss.

**Guardrails** — Safety mechanisms around LLMs that validate, filter, or constrain model inputs and outputs.

---

## H

**Hallucination** — When an LLM generates confident-sounding but factually incorrect information. A key challenge in production LLM systems.

**HHMC** (Helpful, Harmless, Honest) — Anthropic's framework for evaluating AI system quality. Foundation of modern alignment research.

**Hugging Face** — The company and platform that hosts 500K+ open-source AI models, datasets, and tools. The "GitHub of AI."

---

## I

**In-Context Learning (ICL)** — The ability of LLMs to learn new tasks purely from examples in the prompt, without weight updates. Enables few-shot prompting.

**Instruction Tuning** — Fine-tuning a base language model on instruction-following data (question-answer pairs) to make it helpful for conversations.

**RLHF** (Reinforcement Learning from Human Feedback) — Training technique using human preferences to align LLM behavior. Used to create ChatGPT from GPT-4.

---

## J

**JAX** — Google's ML framework combining NumPy-like API with automatic differentiation and GPU/TPU acceleration. Used for research at Google/DeepMind.

**JSON Mode** — A feature of some LLM APIs (OpenAI, Anthropic) that forces the model to output valid JSON. Essential for structured data extraction.

---

## K

**KV Cache** — (Key-Value Cache) Stores intermediate attention computations during autoregressive generation to avoid recomputing them. Makes LLM inference faster.

**Knowledge Distillation** — Training a smaller "student" model to mimic the behavior of a larger "teacher" model. Used to create efficient smaller models.

---

## L

**LangChain** — The most popular framework for building LLM applications. Provides chains, agents, memory, and integrations with 100+ tools.

**LangGraph** — LangChain's framework for building stateful, multi-actor agent applications using a graph-based approach.

**LoRA** (Low-Rank Adaptation) — Parameter-efficient fine-tuning technique that adds small trainable rank-decomposition matrices to each layer, requiring 1000x fewer parameters.

**LLM** (Large Language Model) — A language model with billions of parameters, pre-trained on massive text datasets. Examples: GPT-4, Claude, Llama.

---

## M

**MMLU** (Massive Multitask Language Understanding) — Benchmark testing knowledge across 57 subjects from elementary to professional level.

**MoE** (Mixture of Experts) — Architecture where different "expert" subnetworks handle different inputs. Used by Mixtral, GPT-4. Efficient at scale.

**Multi-Head Attention** — Attention mechanism with multiple parallel attention "heads" that can attend to different aspects of the input simultaneously.

**Multimodal** — Models that process multiple data types (text + images + audio + video). Examples: GPT-4o, Claude 3, Gemini 1.5.

---

## N

**NLP** (Natural Language Processing) — The field of AI focused on understanding and generating human language.

**Neural Network** — A computational system loosely inspired by biological neurons. Composed of layers of connected "neurons" that learn representations.

---

## O

**Open Weights** — Models whose weights are publicly released (vs. closed APIs). Examples: Llama 3, Mistral, Qwen. Allows local deployment and fine-tuning.

**ORPO** (Odds Ratio Preference Optimization) — A recent alignment technique that combines supervised fine-tuning and preference alignment in one step. No reference model needed.

---

## P

**Perplexity** — A measure of how well a language model predicts a text sample. Lower perplexity = better model. Used to evaluate pre-trained LMs.

**Positional Encoding** — Information added to token embeddings to tell the model each token's position. Types: sinusoidal (original), RoPE (Llama, Mistral), ALiBi.

**Prompt** — The input text given to an LLM to guide its generation.

**Prompt Injection** — An attack where malicious instructions are hidden in input data to override the model's system prompt. A key security concern.

---

## Q

**Quantization** — Reducing model precision from 32-bit/16-bit floats to INT8 or INT4. Dramatically reduces memory and increases inference speed with minimal quality loss.

**QLoRA** (Quantized LoRA) — Combines 4-bit quantization with LoRA for fine-tuning large models (13B–70B) on consumer GPUs. The standard approach in 2024–2025.

---

## R

**RAG** (Retrieval-Augmented Generation) — Architecture that retrieves relevant documents at inference time and provides them as context to the LLM, reducing hallucination.

**ReLU** (Rectified Linear Unit) — f(x) = max(0, x). Most common activation function. Simple, effective, fast.

**RLHF** (Reinforcement Learning from Human Feedback) — Training LLMs using human preference data + PPO optimizer. Created ChatGPT, Claude 1, etc.

**RMSNorm** (Root Mean Square Layer Normalization) — A simpler, more efficient variant of Layer Norm used by Llama, Mistral, and most modern LLMs.

**RoPE** (Rotary Position Embedding) — A positional encoding technique used by Llama and most modern open-source models. Can be extended to longer contexts.

---

## S

**Self-Attention** — The mechanism where each token in a sequence attends to all other tokens. The core of Transformer models.

**Softmax** — A function that converts a vector of numbers into a probability distribution. Used in attention and classification heads.

**System Prompt** — Instructions given to an LLM at the start of a conversation that define its persona, capabilities, and constraints.

**SwiGLU** — An activation function used in the FFN layers of Llama, PaLM. Outperforms ReLU/GeLU in language models.

---

## T

**Temperature** — A sampling parameter controlling randomness. 0 = deterministic (greedy), >1 = chaotic. Typical use: 0.7 for balanced generation.

**Token** — The basic unit of text for LLMs. Roughly 3/4 of a word on average. "Hello world" = 2 tokens. 1000 words ≈ 1,333 tokens.

**Top-p (Nucleus Sampling)** — Sample from the smallest set of tokens whose cumulative probability exceeds p. Top-p=0.9 is a common setting.

**Transformer** — The neural network architecture introduced in "Attention Is All You Need" (2017). Foundation of all modern LLMs.

---

## V

**VAE** (Variational Autoencoder) — A generative model with encoder/decoder. Used in latent diffusion models (Stable Diffusion uses a VAE).

**Vector Database** — A database optimized for storing and querying high-dimensional embeddings. Examples: Qdrant, Pinecone, Weaviate, Chroma.

**Vision Transformer (ViT)** — Applying the Transformer architecture to images by splitting them into patches. Foundation of modern vision models.

**vLLM** — A fast LLM serving library with PagedAttention for efficient memory management. Standard for production LLM serving.

---

## W–Z

**Weights** — The learnable parameters of a neural network. "Model weights" = the trained parameters. Same as "parameters."

**Zero-Shot** — Asking an LLM to perform a task without any examples in the prompt, relying purely on its pre-training knowledge.

**Zero-Shot CoT** — Adding "Let's think step by step" to enable chain-of-thought reasoning without providing examples.

---

*AI Magic Mastery by CodeBeez × Abid Redwan | [aimagicmastery.codebeez.ai](https://aimagicmastery.codebeez.ai)*
