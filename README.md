# Tool-Using Research Agent

This project implements an autonomous tool-using research agent built with [LangGraph](https://python.langchain.com/docs/langgraph) and [LangChain](https://python.langchain.com). It fulfills all requirements for Assignment 1.

## Question Answered
The agent answers the following open-ended question that requires reasoning across multiple steps and pieces of information:
> *"What's the best caching strategy for a read-heavy API with 10k req/sec, and how much memory would we need assuming 5KB objects for 1 hour?"*

## Tools Used
1. **`web_search`**: A mock web search tool that returns information about caching strategies, Redis, and Memcached. It includes a constraint to simulate a network timeout/failure if the query contains "FAIL_SEARCH".
2. **`calculator`**: A tool to evaluate mathematical expressions, used by the agent to calculate memory requirements for caching.

## Features & Constraints Met
- **Plans its own steps**: The agent uses an LLM to decide what to search, what to calculate, and when it has enough info.
- **2+ distinct tools**: It uses web search and a calculator.
- **Decides when to stop**: The LangGraph structure allows the agent to return a final answer instead of a tool call when it's satisfied.
- **Max 6 tool calls constraint**: The graph state tracks `tool_call_count`. The `call_model` node enforces stopping if this count reaches 6.
- **Reasoning trace**: Every step, tool invocation, and decision is logged in a `reasoning_trace` list in the graph state, which is printed at the end.
- **Handles failure gracefully**: The agent catches exceptions from the `web_search` tool (when "FAIL_SEARCH" is passed) and adapts its strategy instead of crashing.

## Deliverables Included
- `agent.py`: Core LangGraph implementation.
- `main.py`: Runner script.
- `transcript_1.txt`: Clean run with final answer.
- `transcript_2.txt`: Run with mocked tool failure and graceful handling.
- `README.md`: This file.

## How to Run

1. **Install dependencies**:
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute the Agent**:
   The script is configured to use Google AI Studio's `gemini-1.5-flash` model, as recommended by the assignment instructions.
   
   First, set your `GOOGLE_API_KEY` as an environment variable (or create a `.env` file in the same directory):
   ```bash
   # Windows (PowerShell)
   $env:GOOGLE_API_KEY="your_api_key_here"
   
   # Mac/Linux
   export GOOGLE_API_KEY="your_api_key_here"
   ```

   Then, run the following command:
   ```bash
   python main.py
   ```
   
   This will execute the LangGraph agent and automatically generate `transcript_1.txt` and `transcript_2.txt` in the root directory.
