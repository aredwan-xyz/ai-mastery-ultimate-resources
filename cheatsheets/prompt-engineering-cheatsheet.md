# ✨ Prompt Engineering Cheatsheet
### The Ultimate Quick Reference — AI Magic Mastery by CodeBeez × Abid Redwan

---

## 🎛️ Model Settings

```
Parameter     Range    Description                    Recommendation
─────────────────────────────────────────────────────────────────────
temperature   0.0–2.0  Randomness/creativity          0.0 = factual, 0.7 = balanced, 1.0+ = creative
top_p         0.0–1.0  Nucleus sampling               Use with temperature (not both at max)
top_k         1–100    Top-k sampling                 Lower = more focused
max_tokens    1–...    Max output length              Set explicitly to avoid waste
frequency_penalty 0–2  Penalize repeated tokens       0.3–0.5 for diverse outputs
presence_penalty  0–2  Penalize mentioned topics      0.3 for varied responses
seed          int      Reproducibility                Set for deterministic outputs
```

---

## 🧩 Core Techniques

### 1. Zero-Shot
```
Give clear instructions, no examples needed for simple tasks.

Template:
  [ROLE] [TASK] [FORMAT] [CONSTRAINTS]

Example:
  "You are a senior data scientist. Explain gradient boosting 
   in 3 bullet points. Use plain English, no jargon."
```

### 2. Few-Shot
```
Provide 2–5 examples to teach the format.

Input: "I love this movie!" → Output: Positive
Input: "Total waste of time." → Output: Negative  
Input: "It was okay I guess." → Output: Neutral
Input: "Absolutely incredible!" → Output: [model follows pattern]

✅ Best for: classification, extraction, formatting
```

### 3. Chain-of-Thought (CoT)
```
Magic phrase: "Let's think step by step."

Variants:
  Standard CoT  → "Explain your reasoning step by step before answering"
  Zero-Shot CoT → Just add "Let's think step by step" to any prompt
  Auto-CoT      → Let the model generate its own examples
  
✅ Best for: math, logic, multi-step reasoning, analysis
```

### 4. Tree-of-Thought (ToT)
```
Explore multiple reasoning branches, select the best.

"Solve this problem using 3 different approaches:
 Approach A: [reasoning] → Answer: [X]
 Approach B: [reasoning] → Answer: [Y]
 Approach C: [reasoning] → Answer: [Z]
 Best approach is [A/B/C] because [reasoning]"

✅ Best for: creative problems, planning, strategy
```

### 5. ReAct
```
Reason → Act → Observe → Repeat

Thought: I need to find the current weather in Tokyo
Action: search("Tokyo weather today")
Observation: Sunny, 22°C, humidity 65%
Thought: I have the information needed
Answer: Tokyo is currently sunny at 22°C

✅ Best for: agents, tool use, research tasks
```

### 6. Self-Consistency
```
Sample 5–10 responses with temperature > 0.
Take the majority answer or use LLM-as-judge.

✅ Best for: high-stakes answers, math, QA
```

### 7. Generated Knowledge
```
Step 1: "Generate 5 key facts about [TOPIC]"
Step 2: "Using these facts: [FACTS], answer: [QUESTION]"

✅ Best for: questions requiring background knowledge
```

### 8. Least-to-Most
```
Decompose complex problem into sub-problems.
Solve each sub-problem, use results in next.

✅ Best for: complex multi-step tasks
```

---

## 📐 Prompt Templates

### General Assistant
```
You are [PERSONA], an expert in [DOMAIN] with [EXPERIENCE].
Your communication style is [STYLE: concise/detailed/friendly/formal].
Your goal is to [OBJECTIVE].
Always [REQUIREMENT 1].
Never [CONSTRAINT 1].
Format your responses as [FORMAT].
```

### Code Generation
```
You are an expert [LANGUAGE] developer following [COMPANY] best practices.
Requirements:
- [REQUIREMENT 1]
- [REQUIREMENT 2]

Generate [WHAT] that:
1. [SPEC 1]
2. [SPEC 2]

Include:
- Type hints
- Docstrings
- Error handling
- Example usage
```

### Data Extraction (JSON)
```
Extract the following fields from the text below.
Return ONLY valid JSON, no explanation.

Schema:
{
  "field1": "string",
  "field2": "integer", 
  "field3": ["array", "of", "strings"],
  "field4": {"nested": "object"}
}

If a field is not found, use null.

Text:
[INSERT TEXT]
```

### Summarization
```
Summarize the following [CONTENT TYPE] in exactly [N] [bullet points/sentences/paragraphs].
Focus on: [ASPECT 1], [ASPECT 2], [ASPECT 3].
Target audience: [AUDIENCE DESCRIPTION].
Tone: [FORMAL/CASUAL/TECHNICAL].

[CONTENT]
```

### Code Review
```
You are a senior software engineer conducting a thorough code review.
Review the following [LANGUAGE] code for:

1. 🐛 Bugs and logic errors
2. ⚡ Performance issues  
3. 🔒 Security vulnerabilities
4. 📖 Readability and maintainability
5. ✅ Best practices and style

For each issue:
- Identify the exact line/section
- Explain the problem
- Provide the fixed version

Code:
[INSERT CODE]
```

### Debate / Adversarial
```
You will evaluate [IDEA/CLAIM] from both sides.

Arguments FOR [IDEA]:
1. [Strong argument]
2. [Strong argument]
3. [Strong argument]

Arguments AGAINST [IDEA]:
1. [Strong argument]
2. [Strong argument]  
3. [Strong argument]

Balanced conclusion:
[Nuanced synthesis]
```

---

## 🚫 Anti-Patterns

```
❌ Vague → ✅ Specific
  ❌ "Make it better"
  ✅ "Rewrite this paragraph to be 30% shorter while keeping all key information"

❌ Overloading → ✅ Chaining
  ❌ "Summarize, translate, reformat, and rate this text"
  ✅ Separate prompts for each task

❌ No format spec → ✅ Explicit format
  ❌ "List the steps"  
  ✅ "List exactly 5 steps, numbered 1–5, each under 20 words"

❌ Negative only → ✅ Positive constraints  
  ❌ "Don't be verbose"
  ✅ "Be concise. Each sentence maximum 15 words."

❌ Assuming knowledge → ✅ Provide context
  ❌ "Fix the bug"
  ✅ "Fix the bug. Context: This is a Flask API handling user auth. The bug causes a 500 error when..."
```

---

## 🎯 Task → Technique Mapping

```
Task                          Best Technique(s)
──────────────────────────────────────────────────────────
Factual Q&A                → Zero-shot, temp=0
Multi-step reasoning       → CoT, temp=0.2
Math problems              → CoT + Self-Consistency
Creative writing           → Few-shot style + temp=0.8
Data extraction            → Few-shot JSON, temp=0
Code generation            → Zero-shot structured, temp=0.1
Code debugging             → CoT + show error
Classification             → Few-shot examples
Translation                → Zero-shot, temp=0
Summarization              → Zero-shot with format spec
Brainstorming              → High temp, few constraints
Research / Planning        → ToT or ReAct
Production agents          → ReAct + tools
High-stakes decisions      → Self-consistency + judge
```

---

## 💰 Cost Optimization

```
Strategy                         Savings
─────────────────────────────────────────────────────
Use GPT-4o-mini for simple tasks     90% cheaper than GPT-4o
Cache identical prompts              100% for repeated calls
Use structured output (JSON mode)    ~20% fewer tokens
Compress prompts (remove redundancy) 10–40% savings
Batch API requests                   50% discount (OpenAI)
Use Prompt Caching (Anthropic)       90% for cached prefixes
Use smaller open models (Ollama)     ~99% for local inference
```

---

## 🔬 Evaluation Checklist

Before shipping a prompt to production:
- [ ] Tested on 20+ diverse inputs
- [ ] Handles edge cases (empty input, wrong format, long input)
- [ ] Measured latency under load
- [ ] Compared against baseline (previous prompt)
- [ ] Cost estimated per 1,000 calls
- [ ] Output validated with automated tests
- [ ] Prompt injection tested
- [ ] Human evaluation on 10 samples

---

*AI Magic Mastery by CodeBeez × Abid Redwan | [aimagicmastery.codebeez.ai](https://aimagicmastery.codebeez.ai)*
