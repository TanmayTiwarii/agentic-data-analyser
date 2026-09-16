# ============================================================
# agents/planner_agent.py
# ============================================================
# PURPOSE: The "Planner Agent" — the first agent in the workflow.
#
# WHAT IT DOES:
#   Looks at the dataset structure and decides what analysis
#   steps should be taken. It outputs a simple plan (list of steps)
#   that guides the rest of the pipeline.
#
# WHY THIS IS "AGENTIC":
#   A traditional script has hardcoded steps.
#   An agent *reasons* about what to do based on the data.
#   The planner reads column names, data types, and size,
#   then intelligently decides: "this dataset needs X, Y, Z."
#
# INTERVIEW TIP:
#   "The Planner Agent acts as the orchestrator. It reads dataset
#    metadata and generates a dynamic analysis plan using the LLM,
#    rather than following a hardcoded sequence of steps."
# ============================================================

from langchain_core.messages import HumanMessage, SystemMessage
from agents.llm_client import get_llm
import pandas as pd


def run_planner_agent(df: pd.DataFrame, dataset_context: str) -> str:
    """
    Planner Agent: Analyzes dataset metadata and creates an analysis plan.

    Args:
        df: The uploaded DataFrame
        dataset_context: A pre-built text summary of the dataset (from data_loader.py)

    Returns:
        str: A numbered analysis plan as text
    """

    # Get the LLM (low temperature = focused, analytical output)
    llm = get_llm(temperature=0.2)

    # ── System Prompt ─────────────────────────────────────────
    # The system prompt defines the agent's "role" and behavior.
    # This is one of the most important parts of prompt engineering.
    system_prompt = """You are a senior data scientist and AI planning agent.

Your job is to analyze a dataset's structure and create a clear, numbered analysis plan.

The plan should include:
1. What type of analysis is appropriate
2. Which columns to focus on
3. What visualizations to generate
4. What cleaning steps might be needed
5. What business insights to look for

Keep the plan concise — 6 to 10 numbered steps.
Be specific to the actual data described, not generic.
"""

    # ── Human Prompt ──────────────────────────────────────────
    # The human prompt is the actual request sent to the LLM.
    human_prompt = f"""
Based on this dataset, create a specific analysis plan:

{dataset_context}

Generate a numbered list of analysis steps (6-10 steps).
Each step should be 1-2 sentences, specific to this dataset.
"""

    # ── LLM Call ──────────────────────────────────────────────
    # We use the message format that LangChain expects for chat models.
    # SystemMessage = instructions for how the AI should behave
    # HumanMessage = the actual user request
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    response = llm.invoke(messages)

    # response.content contains the actual text response from the LLM
    return response.content
