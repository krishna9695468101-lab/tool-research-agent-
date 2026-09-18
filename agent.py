import operator
from typing import TypedDict, Annotated, List, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

class AgentState(TypedDict):
    """The state of our Tool-Using Research Agent."""
    messages: Annotated[list[AnyMessage], operator.add]
    tool_call_count: int
    reasoning_trace: list[str]

@tool
def web_search(query: str) -> str:
    """Searches the web for information."""
    if "FAIL_SEARCH" in query:
        raise Exception("Timeout: Web search API is currently unavailable.")
    
    # Mocking some search results for our specific questions
    if "caching strategy" in query.lower() or "read heavy" in query.lower():
        return "For read-heavy APIs (10k req/sec), Redis or Memcached are recommended. Strategies include Write-Through, Cache-Aside (Lazy Loading)."
    if "redis vs memcached" in query.lower():
        return "Redis supports data persistence, replication, and complex data types. Memcached is multithreaded but simple and lacks persistence."
    
    return f"Search results for: {query} - General information found."

@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        # NOTE: eval is used here for simplicity in a mocked environment.
        # In a real production system, use a safe math evaluator (like numexpr or ast.literal_eval).
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

# Tools available to the agent
tools = [web_search, calculator]

def call_model(state: AgentState, llm):
    """Invokes the LLM to decide the next step."""
    messages = state['messages']
    
    # Constraint: Agent must stop if it hits the tool call limit
    if state.get('tool_call_count', 0) >= 6:
        # Assuming the LLM is instructed to output final answer or we restrict it.
        # By not binding tools, the LLM is forced to output text (final answer).
        llm_with_tools = llm 
    else:
        llm_with_tools = llm.bind_tools(tools)
    
    # In a real app, we'd inject a system prompt here if not already part of messages
    response = llm_with_tools.invoke(messages)
    
    # Extract reasoning for our trace
    trace = state.get('reasoning_trace', [])
    reason = response.content if response.content else f"Decided to invoke tools: {[tc['name'] for tc in response.tool_calls]}"
    new_trace = trace + [f"Agent planned step: {reason}"]
    
    return {"messages": [response], "reasoning_trace": new_trace}

def execute_tools(state: AgentState):
    """Executes tools requested by the LLM."""
    messages = state['messages']
    last_message = messages[-1]
    
    new_messages = []
    traces = state.get('reasoning_trace', [])
    count = state.get('tool_call_count', 0)
    
    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            count += 1
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            traces.append(f"Executing tool '{tool_name}' (Call #{count}) with args: {tool_args}")
            
            # Find the corresponding tool
            tool_fn = next((t for t in tools if t.name == tool_name), None)
            if tool_fn:
                try:
                    result = tool_fn.invoke(tool_args)
                    traces.append(f"Tool '{tool_name}' succeeded. Result snippet: {str(result)[:50]}...")
                except Exception as e:
                    # Mock Failure Constraint handled here: errors are caught and returned to the agent
                    result = f"Execution Error: {e}"
                    traces.append(f"Tool '{tool_name}' failed with error: {e}")
            else:
                result = f"Error: Tool {tool_name} not found"
                traces.append(f"Tool '{tool_name}' not found.")
                
            # Append ToolMessage to feed back into the LLM
            new_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_name))
            
    return {"messages": new_messages, "tool_call_count": count, "reasoning_trace": traces}

def should_continue(state: AgentState):
    """Determines whether to execute tools or end the loop."""
    messages = state['messages']
    last_message = messages[-1]
    
    # If there are tool calls, go to the tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"
    # Otherwise, it's a final answer
    return END

def create_agent_graph(llm):
    """Constructs the LangGraph."""
    workflow = StateGraph(AgentState)
    
    # Define the two main nodes
    workflow.add_node("agent", lambda state: call_model(state, llm))
    workflow.add_node("execute_tools", execute_tools)
    
    # Define edges and entry point
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"execute_tools": "execute_tools", END: END})
    workflow.add_edge("execute_tools", "agent")
    
    return workflow.compile()
