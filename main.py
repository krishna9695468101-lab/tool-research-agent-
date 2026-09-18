import os
from langchain_core.messages import HumanMessage
from agent import create_agent_graph
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def run_scenario(name: str, question: str, filename: str):
    """Runs a scenario through the LangGraph agent and saves the transcript."""
    print(f"\n{'='*50}\n--- Running {name} ---\n{'='*50}")
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        print("Please set it in a .env file or environment variables to run.")
        return
        
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    agent = create_agent_graph(llm)
    
    state = {
        "messages": [HumanMessage(content=question)],
        "tool_call_count": 0,
        "reasoning_trace": []
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Scenario: {name}\n")
        f.write(f"Question: {question}\n\n")
        print(f"Question: {question}\n")
        
        for event in agent.stream(state, stream_mode="values"):
            messages = event.get('messages', [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content") and last_message.type == "ai":
                    output = f"Agent Thought: {last_message.content}\n"
                    if getattr(last_message, "tool_calls", None):
                        output += f"Tool Calls Requested: {[tc['name'] for tc in last_message.tool_calls]}\n"
                    print(output.strip())
                    f.write(output + "\n")
                elif last_message.type == "tool":
                    output = f"Tool Result ({getattr(last_message, 'name', 'unknown')}): {last_message.content}\n"
                    print(output.strip())
                    f.write(output + "\n")
                    
        # Log reasoning trace at the end
        final_trace = "\n" + "-"*40 + "\n Final Reasoning Trace \n" + "-"*40 + "\n"
        final_trace += "\n".join(event.get('reasoning_trace', []))
        print(final_trace)
        f.write(final_trace + "\n")
        print(f"\n Transcript saved to {filename}")

if __name__ == "__main__":
    # Generate Transcript 1
    run_scenario(
        name="Scenario 1: Clean Run", 
        question="What's the best caching strategy for a read-heavy API with 10k req/sec, and how much memory would we need assuming 5KB objects for 1 hour?", 
        filename="transcript_1.txt"
    )
                 
    # Generate Transcript 2
    run_scenario(
        name="Scenario 2: Failure Handling", 
        question="What's the best caching strategy for a read-heavy API with 10k req/sec? IMPORTANT: To test your error handling, your VERY FIRST step must be to call the web_search tool with the exact query 'FAIL_SEARCH caching strategy read heavy API'. When it fails, acknowledge the failure and proceed with a real search without that keyword.", 
        filename="transcript_2.txt"
    )

