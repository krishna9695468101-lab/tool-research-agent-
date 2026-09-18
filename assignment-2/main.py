import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Initialize the LLM
# In LangChain, we must ensure we use the model properly
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

# Define criteria
criteria = (
    "1. Under 60 words.\n"
    "2. Mentions 'temperature control'.\n"
    "3. Mentions 'battery life'."
)

# Agent A (Worker) Prompt
worker_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert copywriter. Write a product description for a smart coffee mug. "
               "You must include exactly what the user asks for."),
    ("user", "{instructions}")
])
worker_chain = worker_prompt | llm

# Agent B (Reviewer) Output Schema
class ReviewResult(BaseModel):
    verdict: str = Field(description="Must be exactly 'approved' or 'rejected'")
    reasons: str = Field(description="Specific reasons for the verdict based on criteria.")

parser = JsonOutputParser(pydantic_object=ReviewResult)

# Agent B (Reviewer) Prompt
reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a reviewer. Review the following product description against these criteria:\n"
               "{criteria}\n\n"
               "Return a JSON object with 'verdict' ('approved' or 'rejected') and 'reasons'.\n"
               "{format_instructions}"),
    ("user", "Product Description to review:\n{description}")
])
reviewer_chain = reviewer_prompt | llm | parser

def run_scenario(name: str, instructions: str, filename: str):
    print(f"\n{'='*50}\n--- Running {name} ---\n{'='*50}")
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Scenario: {name}\n")
        f.write(f"Task: Draft a product description for a smart coffee mug.\n")
        f.write(f"Agent B Criteria:\n{criteria}\n\n")
        
        # Agent A
        f.write("--- AGENT A (WORKER) ---\n")
        f.write(f"Instructions to Worker: {instructions}\n\n")
        
        # We manually track LLM calls
        llm_calls = 0
        
        # Run Agent A
        worker_res = worker_chain.invoke({"instructions": instructions})
        llm_calls += 1
        draft = worker_res.content
        f.write(f"Draft Generated:\n{draft}\n\n")
        
        # Agent B
        f.write("--- AGENT B (REVIEWER) ---\n")
        reviewer_res = reviewer_chain.invoke({
            "criteria": criteria,
            "format_instructions": parser.get_format_instructions(),
            "description": draft
        })
        llm_calls += 1
        
        f.write(f"Verdict: {reviewer_res['verdict'].upper()}\n")
        f.write(f"Reasons: {reviewer_res['reasons']}\n\n")
        
        f.write(f"Total LLM Calls for this chain: {llm_calls}\n")
        
        print(f"Transcript generated and saved to {filename}")

if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        exit(1)

    # Run 1: Approved
    run_scenario(
        name="Transcript 1: Clean Run (Approved)",
        instructions="Draft a 40-word description of a smart coffee mug. Make sure to highlight its temperature control feature and its 12-hour battery life.",
        filename="transcript_1.txt"
    )
    
    # Run 2: Rejected
    run_scenario(
        name="Transcript 2: Forced Rejection",
        instructions="Draft a 40-word description of a smart coffee mug. Highlight its temperature control, but DO NOT mention battery life. Focus on its sleek design instead.",
        filename="transcript_2.txt"
    )
