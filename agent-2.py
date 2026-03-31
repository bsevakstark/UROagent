"""
agent.py
--------
UroAgent — the core ReAct agentic workflow.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor
from langchain_core.agents import create_react_agent
from langchain import hub

from tools import query_ehr_database, query_aua_guidelines

load_dotenv()

UROAGENT_SYSTEM_PROMPT = """You are UroAgent, an expert AI clinical assistant
specialising in Urology. You support clinicians and researchers at the
USC Institute of Urology by answering questions about patient records and
evidence-based clinical guidelines.

You have access to two tools:
1. query_ehr_database   – query structured patient EHR data (SQL)
2. query_aua_guidelines – retrieve AUA clinical guideline passages (RAG)

Decision rules:
- If the question is about specific patients, PSA values, diagnoses, or
  treatment records  →  use query_ehr_database.
- If the question is about clinical protocols, staging, screening criteria,
  or treatment recommendations  →  use query_aua_guidelines.
- For complex questions that need BOTH patient data AND guideline context,
  use BOTH tools and synthesise a comprehensive answer.

Always reason step-by-step before acting (ReAct pattern).
Always cite which tool(s) provided the information.
Never fabricate patient data or clinical facts.
Respond in clear, professional clinical language suitable for a physician audience.
"""


def build_agent() -> AgentExecutor:
    """Construct and return a LangChain ReAct AgentExecutor."""

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    tools = [query_ehr_database, query_aua_guidelines]

    base_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools, prompt=base_prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    return executor


def run_query(question: str, agent_executor: AgentExecutor = None) -> dict:
    """Run a single clinical query through UroAgent."""
    if agent_executor is None:
        agent_executor = build_agent()

    result = agent_executor.invoke({"input": question})

    return {
        "input":  result["input"],
        "output": result["output"],
        "steps":  result.get("intermediate_steps", []),
    }


if __name__ == "__main__":
    agent = build_agent()
    demo_questions = [
        "Which patients have a PSA level above 10 ng/mL?",
        "What are the AUA recommendations for active surveillance in prostate cancer?",
    ]
    for q in demo_questions:
        print("\n" + "=" * 70)
        print(f"QUERY: {q}")
        response = run_query(q, agent)
        print(f"\nFINAL ANSWER:\n{response['output']}")
