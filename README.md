# Autonomous AI Agent Assignments

This repository contains the take-home assignments for the Junior AI Engineer role. It demonstrates the ability to build, reason about, and control autonomous agents using modern frameworks.

## Project Structure

The project is divided into separate folders for each submitted assignment, satisfying the core framework requirements and constraints.

### [Assignment 1: Tool-Using Research Agent](./assignment-1/)
A single, autonomous research agent built to plan its own steps, reason about state, and gracefully handle tool failures.
- **Framework:** LangGraph / LangChain
- **Key Features:** Dynamic tool calling (web search, calculator), explicit reasoning traces, strict iteration limits, and adaptive failure handling.
- **Deliverables:** Includes source code, setup instructions, and transcripts for both a clean execution and a forced-failure adaptation run.

### [Assignment 2: Multi-Agent Task with Review](./assignment-2/)
A dual-agent system where a worker drafts content and a reviewer strictly evaluates it against predefined criteria without a continuous revision loop.
- **Framework:** LangChain LCEL (worker/reviewer chain)
- **Key Features:** Structured JSON outputs, strict constraint evaluation, and clear acceptance/rejection verdicts.
- **Deliverables:** Includes source code and transcripts demonstrating both an approved workflow and a deliberately rejected workflow with specific reviewer feedback.

## Requirements & Setup

Both assignments require Python 3.9+ and the `langchain` ecosystem. 

To run any of the assignments locally, you will need a Google Gemini API Key:
```bash
# Windows
$env:GOOGLE_API_KEY="your_api_key"

# Mac/Linux
export GOOGLE_API_KEY="your_api_key"
