# 🕵️ AI Agent Builder Roadmap
### Build Autonomous AI Systems in 6 Weeks

> **By Abid Redwan & CodeBeez** | [AI Magic Mastery](https://aimagicmastery.codebeez.ai)

---

## What Are AI Agents?

AI agents are **autonomous systems** that can perceive their environment, reason about it, take actions using tools, and iterate until they complete complex multi-step goals — with little or no human intervention.

```
Traditional LLM:  Input → LLM → Output  (single pass)

AI Agent:
  Goal
   ↓
  Observe (environment, tools, memory)
   ↓
  Think/Plan (LLM as the "brain")
   ↓
  Act (use tools: search, code, APIs, files)
   ↓
  Observe results
   ↓
  Reflect & Repeat → until goal is achieved
```

---

## 📋 Prerequisites
- [ ] Python (intermediate level)
- [ ] Basic LLM API usage (OpenAI, Anthropic, or local Ollama)
- [ ] Understanding of prompting basics

---

## Week 1: Agent Fundamentals

### What Makes an Agent
```
4 Core Components:
1. Brain (LLM)       → Reasoning, planning, decision-making
2. Memory            → Short-term (context), long-term (vector DB), episodic
3. Tools             → Web search, code execution, APIs, file system
4. Execution Loop    → Observe → Think → Act → Observe → ...
```

### Your First Agent (From Scratch)
```python
import json
from openai import OpenAI

client = OpenAI()

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code and return result",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    }
]

def run_agent(goal: str, max_iterations: int = 10):
    messages = [
        {"role": "system", "content": "You are a helpful AI agent. Use tools to accomplish goals."},
        {"role": "user", "content": goal}
    ]
    
    for i in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        # No tool calls = agent is done
        if not message.tool_calls:
            return message.content
        
        # Execute tools
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            # Dispatch to actual tool functions
            result = dispatch_tool(tool_name, tool_args)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
    
    return "Max iterations reached"
```

---

## Week 2: LangChain & LangGraph

### LangGraph: The 2025 Standard for Production Agents
```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, Annotated
import operator

# State definition
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_action: str

# Tools
tools = [TavilySearchResults(max_results=3)]
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)
tool_node = ToolNode(tools)

# Nodes
def agent(state: AgentState):
    result = llm.invoke(state["messages"])
    return {"messages": [result]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

app = workflow.compile()

# Run
result = app.invoke({"messages": [("user", "What are the top AI news stories today?")]})
```

---

## Week 3: Memory Systems

### Short-Term Memory (Context Window)
```python
from langchain.memory import ConversationBufferWindowMemory

# Keep last K exchanges
memory = ConversationBufferWindowMemory(k=10, return_messages=True)
```

### Long-Term Memory (Vector Store)
```python
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory

# Semantic memory — retrieve relevant past conversations
embeddings = OpenAIEmbeddings()
vectorstore = Qdrant.from_texts([], embeddings, location=":memory:")
retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))
memory = VectorStoreRetrieverMemory(retriever=retriever)

# Store interaction
memory.save_context(
    {"input": "My name is Abid and I like Python"},
    {"output": "Nice to meet you, Abid!"}
)

# Later, retrieve relevant memories
relevant = memory.load_memory_variables({"prompt": "What's my name?"})
# Returns: "My name is Abid and I like Python"
```

### Episodic Memory (Structured)
```python
# Store structured episodes for long-running agents
import json
from datetime import datetime
from pathlib import Path

class EpisodicMemory:
    def __init__(self, path: str = "agent_memory.json"):
        self.path = Path(path)
        self.episodes = self._load()
    
    def _load(self):
        if self.path.exists():
            return json.loads(self.path.read_text())
        return []
    
    def save_episode(self, task: str, steps: list, outcome: str, success: bool):
        episode = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "steps": steps,
            "outcome": outcome,
            "success": success
        }
        self.episodes.append(episode)
        self.path.write_text(json.dumps(self.episodes, indent=2))
    
    def get_similar_episodes(self, task: str, n: int = 3) -> list:
        # In production: use embeddings to find similar past tasks
        return self.episodes[-n:]
```

---

## Week 4: Multi-Agent Systems with CrewAI

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileWriterTool

# Define specialized agents
researcher = Agent(
    role="AI Research Analyst",
    goal="Find and synthesize the latest AI developments",
    backstory="""You are a seasoned AI researcher who stays on top of 
    every major development in the field. You have a talent for finding 
    signal in the noise and identifying what truly matters.""",
    tools=[SerperDevTool()],
    verbose=True,
    llm="gpt-4o"
)

writer = Agent(
    role="Technical Content Writer",
    goal="Write clear, engaging technical content",
    backstory="""You are a technical writer who makes complex AI concepts 
    accessible to both beginners and experts. You write for the AI Magic 
    Mastery blog by CodeBeez.""",
    tools=[FileWriterTool()],
    verbose=True,
    llm="gpt-4o"
)

editor = Agent(
    role="Senior Editor",
    goal="Ensure content quality, accuracy, and SEO optimization",
    backstory="You are a meticulous editor who ensures every piece is world-class.",
    verbose=True,
    llm="gpt-4o"
)

# Define tasks
research_task = Task(
    description="Research the top 5 AI developments in the past week. Include links and key insights.",
    agent=researcher,
    expected_output="A structured report with 5 AI developments, each with summary, impact, and source URL"
)

writing_task = Task(
    description="Write a 1000-word blog post based on the research findings for AI Magic Mastery.",
    agent=writer,
    expected_output="A complete blog post with title, intro, sections, and conclusion",
    context=[research_task]
)

editing_task = Task(
    description="Edit the blog post for clarity, accuracy, SEO, and save as weekly-ai-news.md",
    agent=editor,
    expected_output="Polished, SEO-optimized blog post saved to file",
    context=[writing_task]
)

# Run the crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
```

---

## Week 5: Production-Grade Agents

### Error Handling & Retry Logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

class ResilientAgent:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def execute_with_retry(self, task: str):
        try:
            return await self.run(task)
        except RateLimitError:
            logging.warning("Rate limited, retrying...")
            raise
        except Exception as e:
            logging.error(f"Agent failed: {e}")
            raise
    
    def run_with_fallback(self, task: str, fallback_model: str = "gpt-4o-mini"):
        try:
            return self.run_with_model("gpt-4o", task)
        except Exception:
            logging.warning("Primary model failed, using fallback")
            return self.run_with_model(fallback_model, task)
```

### Agent Monitoring with LangFuse
```python
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)

# Attach to any LangChain/LangGraph agent
result = app.invoke(
    {"messages": [("user", "Research AI news")]},
    config={"callbacks": [langfuse_handler]}
)
# Now monitor traces, costs, latency at langfuse.cloud
```

---

## Week 6: Capstone Projects

### Project 1: AI Research Assistant
```
Agent capabilities:
├── Search web (Tavily/Serper)
├── Read PDFs and web pages
├── Search arxiv papers
├── Take notes (vector memory)
├── Write research reports
└── Create citations
```

### Project 2: Personal AI Automation
```
Agent capabilities:
├── Read/write files
├── Send emails
├── Create calendar events
├── Browse websites
├── Execute Python code
└── Summarize daily information
```

### Project 3: AI Software Engineer
```
Agent capabilities:
├── Read codebases (file system)
├── Write and edit code
├── Execute code and tests
├── Search documentation
├── Create PRs via GitHub API
└── Debug and fix errors
```

---

## 🛠️ Agent Framework Comparison 2025

| Framework | Stars | Best For | Learning Curve |
|-----------|-------|---------|---------------|
| LangGraph | ⭐ 10k+ | Production, stateful graphs | Medium |
| CrewAI | ⭐ 25k+ | Multi-agent teams | Easy |
| AutoGen | ⭐ 35k+ | Conversational agents | Medium |
| Smolagents | ⭐ 15k+ | Minimal, code-first | Easy |
| OpenAI Agents SDK | ⭐ New | OpenAI native, simple | Very Easy |
| Agno | ⭐ 8k+ | Fast, multimodal | Easy |

**Recommendation:**
- 🥇 Learning → **OpenAI Agents SDK** or **Smolagents** (simplest)
- 🥇 Production → **LangGraph** (most control and observability)
- 🥇 Multi-agent → **CrewAI** (best DX for teams of agents)

---

*Part of the [AI Magic Mastery](https://aimagicmastery.codebeez.ai) ecosystem by CodeBeez × Abid Redwan*
