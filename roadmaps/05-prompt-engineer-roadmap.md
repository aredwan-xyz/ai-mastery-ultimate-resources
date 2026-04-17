# ✨ Prompt Engineering Roadmap
### Master LLM Outputs in 3 Weeks — Any Background Welcome

> **By Abid Redwan & CodeBeez** | [AI Magic Mastery](https://aimagicmastery.codebeez.ai)

---

## What Is Prompt Engineering?

Prompt engineering is the practice of designing, optimizing, and structuring inputs to AI language models to reliably get high-quality, accurate, and useful outputs.

> "Prompt engineering is the most important skill in the AI era that nobody is teaching properly." — *Abid Redwan*

---

## 📅 3-Week Plan

### Week 1: Foundations & Core Techniques

#### Day 1–2: Understanding How LLMs Work
You don't need to be an ML expert, but you need to understand:

```
LLMs predict the next token based on all previous tokens.
They don't "understand" — they pattern-match at massive scale.
Temperature controls randomness (0=deterministic, 1=creative).
Context window = how much the model can "see" at once.
System vs User vs Assistant messages = different roles.
```

#### Day 3–5: Core Prompting Techniques

**Zero-Shot Prompting**
```
❌ Bad: "Summarize this"
✅ Good: "Summarize the following article in exactly 3 bullet points,
         each under 20 words, focusing on business implications:"
```

**Few-Shot Prompting**
```
Provide examples to teach the format you want:

User: Classify sentiment: "I love this product!"
Assistant: Positive

User: Classify sentiment: "Worst purchase ever."
Assistant: Negative

User: Classify sentiment: "It works, I guess."
Assistant: [Model will follow the pattern] → Neutral
```

**Chain-of-Thought (CoT)**
```
Add "Let's think step by step" or show reasoning steps.

Without CoT: "What is 17 × 24?" → Often wrong
With CoT: "What is 17 × 24? Let's think step by step." → Usually correct

Why it works: Forces the model into a reasoning mode before answering.
```

**Role Prompting**
```
"You are a senior software engineer at Google with 10 years of 
experience in distributed systems. Review the following code and 
provide specific, actionable feedback..."
```

#### Day 6–7: System Prompts
```
System prompts set the model's persona, constraints, and behavior.

Components of a great system prompt:
1. Role/Persona        → Who the model is
2. Context             → What situation it's in
3. Task Description    → What it should do
4. Output Format       → How to structure responses
5. Constraints         → What NOT to do
6. Tone/Style          → How to communicate

Example:
"You are Alex, an expert AI tutor at AI Magic Mastery by CodeBeez. 
Your goal is to explain complex AI concepts in simple, engaging language 
suitable for beginners. Always use analogies and concrete examples.
Format your responses with clear headings. Never use jargon without 
explaining it first. Keep explanations under 300 words unless asked 
to go deeper."
```

---

### Week 2: Advanced Techniques

#### Structured Output (JSON Mode)
```python
import openai, json

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[{
        "role": "user",
        "content": """Extract the following from this job posting and return as JSON:
        - job_title (string)
        - company (string)  
        - required_skills (array of strings)
        - salary_range (object with min, max, currency)
        - remote_ok (boolean)
        
        Job posting: "Senior ML Engineer at TechCorp. $150K-$200K USD.
        Required: PyTorch, Python, 5+ years. Remote friendly."
        """
    }]
)
data = json.loads(response.choices[0].message.content)
```

#### Tree-of-Thought
```
Instead of one chain of reasoning, explore multiple paths:

"Solve this problem using 3 different approaches, then select the best:
Approach 1: [explore]
Approach 2: [explore]  
Approach 3: [explore]
Best approach: [decision with reasoning]"
```

#### Self-Consistency
```
For important decisions, sample multiple completions and take the majority:

import openai

def self_consistent_answer(question, n=5):
    answers = []
    for _ in range(n):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}],
            temperature=0.7
        )
        answers.append(response.choices[0].message.content)
    # Use majority vote or LLM to pick the best
    return answers
```

#### ReAct (Reason + Act)
```
Interleave reasoning and tool use:

Thought: I need to find the current price of NVIDIA stock
Action: search("NVIDIA NVDA stock price today")
Observation: NVDA is trading at $875.23 as of 2:30 PM EST
Thought: Now I can answer the question
Answer: NVIDIA (NVDA) is currently trading at $875.23
```

#### Meta-Prompting
```
Use the LLM to write better prompts for itself:

"I need to write a prompt that will help an LLM [TASK].
The LLM should [OUTPUT FORMAT] and should avoid [ANTI-PATTERNS].
Write me the optimal system prompt for this use case."
```

#### Prompt Chaining
```python
# Break complex tasks into a chain of focused prompts

def analyze_document(document: str) -> dict:
    # Step 1: Extract key facts
    facts = llm.complete(f"Extract 5 key facts from:\n{document}")
    
    # Step 2: Identify sentiment
    sentiment = llm.complete(f"Analyze sentiment of:\n{document}")
    
    # Step 3: Generate summary using extracted facts
    summary = llm.complete(
        f"Write a 2-sentence summary using these facts:\n{facts}"
    )
    
    return {"facts": facts, "sentiment": sentiment, "summary": summary}
```

---

### Week 3: Production Prompt Engineering

#### Prompt Templates & Versioning
```python
# Use Langchain PromptTemplate or Jinja2 for reusable prompts

from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} specializing in {domain}. "
               "Always respond in {language}. "
               "Format: {output_format}"),
    ("user", "{user_query}")
])

# Version your prompts in code:
PROMPT_V1 = "Summarize in 3 bullets:"
PROMPT_V2 = "Summarize in exactly 3 bullet points, each under 15 words:"  # +18% quality
PROMPT_V3 = "You are a concise technical writer. Summarize in 3 bullets (max 15 words each):"  # +12% more
```

#### Prompt Evaluation
```python
# Don't guess — measure prompt quality

import json
from openai import OpenAI

def evaluate_prompt(prompt_a: str, prompt_b: str, test_cases: list) -> dict:
    """A/B test two prompts on identical test cases."""
    results = {"prompt_a": [], "prompt_b": []}
    
    for test in test_cases:
        # Get outputs from both prompts
        out_a = get_llm_output(prompt_a, test["input"])
        out_b = get_llm_output(prompt_b, test["input"])
        
        # Use LLM-as-judge to evaluate
        judgment = judge_llm(out_a, out_b, test["criteria"])
        results["prompt_a"].append(judgment["a_score"])
        results["prompt_b"].append(judgment["b_score"])
    
    return {
        "prompt_a_avg": sum(results["prompt_a"]) / len(results["prompt_a"]),
        "prompt_b_avg": sum(results["prompt_b"]) / len(results["prompt_b"]),
        "winner": "A" if results["prompt_a"] > results["prompt_b"] else "B"
    }
```

#### Anti-Patterns to Avoid
```
❌ Vague instructions          → "Make it better" → "Improve clarity by 20% and fix grammar"
❌ Asking too many things      → Break into separate prompts
❌ Inconsistent format         → Always specify exact output format
❌ No examples for complex tasks → Use few-shot for complex formatting
❌ Ignoring context limits      → Chunk long documents properly
❌ Prompt injection vulnerable  → Sanitize user inputs in system prompts
❌ Temperature 0 for creativity → Use 0.7–1.0 for creative tasks
❌ Temperature 1 for factual    → Use 0–0.2 for factual tasks
```

---

## 📋 Prompt Engineering Cheatsheet

```
TASK TYPE → OPTIMAL SETTINGS
─────────────────────────────────────────────────────
Factual Q&A          → temp=0.0,  system="Be precise and concise"
Code generation      → temp=0.1,  system="You are an expert programmer"
Creative writing     → temp=0.8,  system="You are a creative writer"
Data extraction      → temp=0.0,  response_format=json_object
Summarization        → temp=0.2,  specify length explicitly
Translation          → temp=0.0,  specify target language + formality
Classification       → temp=0.0,  list all possible labels explicitly
Brainstorming        → temp=0.9,  "Generate N diverse ideas"

TECHNIQUE → USE WHEN
─────────────────────────────────────────────────────
Zero-Shot     → Simple, clear tasks
Few-Shot      → Specific format or style needed
CoT           → Math, logic, multi-step reasoning
Self-Consistency → High-stakes single answer needed
ReAct         → Tasks needing tool use + reasoning
Role Prompting → Domain-specific expertise needed
Structured Output → Extracting structured data
```

---

## 🎯 Career Paths for Prompt Engineers

```
Prompt Engineer             → $120K–$200K
AI Product Manager          → $150K–$250K  
LLM Application Developer   → $140K–$220K
AI Content Strategist       → $80K–$140K
AI Automation Specialist    → $100K–$180K
```

---

*Part of the [AI Magic Mastery](https://aimagicmastery.codebeez.ai) ecosystem by CodeBeez × Abid Redwan*
