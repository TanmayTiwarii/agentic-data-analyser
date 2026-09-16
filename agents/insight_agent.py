# ============================================================
# agents/insight_agent.py
# ============================================================
# PURPOSE: The "Insight Agent" — the final agent in the workflow.
#
# WHAT IT DOES:
#   1. Takes all findings from the Analysis Agent
#   2. Uses LLM to generate deeper narrative insights
#   3. Answers natural language questions about the data
#   4. Maintains conversation memory (chat history)
#
# WHY THIS IS THE MOST "AGENTIC" PART:
#   This agent uses "memory" — it remembers previous questions
#   and answers in the conversation, allowing follow-up questions
#   like "tell me more about that" to work correctly.
#
# INTERVIEW TIP:
#   "The Insight Agent implements conversational memory using
#    LangChain's message history. Each new question is sent with
#    the full conversation history, allowing the LLM to maintain
#    context across multiple turns — a key feature of agentic systems."
# ============================================================

import pandas as pd
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from agents.llm_client import get_llm


def run_insight_agent(
    df: pd.DataFrame,
    dataset_context: str,
    analysis_results: dict,
    plan: str
) -> str:
    """
    Insight Agent: Generates a comprehensive narrative insight report.

    This is called once after all other agents run, to produce
    a high-level "story" about the data.

    Args:
        df: The uploaded DataFrame
        dataset_context: Text summary of the dataset
        analysis_results: Output from the Analysis Agent (stats + suggestions)
        plan: Output from the Planner Agent

    Returns:
        str: A detailed insight report as markdown text
    """
    llm = get_llm(temperature=0.5)

    # Compile cleaning suggestions into text
    suggestions_str = "\n".join(analysis_results.get("cleaning_suggestions", []))

    system_prompt = """You are a senior data scientist writing an executive insight report.

Your report should:
- Be written in clear, professional language
- Use markdown formatting (headers, bullet points, bold)
- Include specific numbers and column names from the data
- Identify trends, patterns, anomalies, and opportunities
- Conclude with actionable recommendations

Do NOT make up data. Only reference what is provided to you.
"""

    human_prompt = f"""
Write a comprehensive data insights report based on this analysis:

DATASET CONTEXT:
{dataset_context}

ANALYSIS PLAN:
{plan}

DATA QUALITY ISSUES:
{suggestions_str}

Format your response as a structured markdown report with these sections:
## 📊 Executive Summary
## 🔍 Key Findings  
## ⚠️ Data Quality Issues
## 💡 Actionable Recommendations
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    response = llm.invoke(messages)
    return response.content


def answer_question(
    question: str,
    df: pd.DataFrame,
    dataset_context: str,
    chat_history: List[dict]
) -> str:
    """
    Answers a user's natural language question about the dataset.

    This function implements CONVERSATIONAL MEMORY by including
    previous messages in the prompt. This allows the LLM to
    understand follow-up questions like "tell me more" or "why?"

    Args:
        question: The user's current question
        df: The uploaded DataFrame
        dataset_context: Text summary of the dataset
        chat_history: List of {"role": "user"/"assistant", "content": "..."} dicts

    Returns:
        str: The AI's answer
    """
    llm = get_llm(temperature=0.4)

    # ── Build the conversation history for LangChain ──────────
    # LangChain uses typed message objects, not plain dicts.
    # We convert our chat_history list into LangChain messages.
    messages = []

    # System message defines the agent's behavior for the entire conversation
    messages.append(SystemMessage(content=f"""You are an intelligent data analyst assistant.

You have access to a dataset with the following information:

{dataset_context}

Your job is to answer questions about this dataset accurately.

IMPORTANT RULES:
1. Only answer based on the provided dataset information
2. If you need to calculate something, state your reasoning clearly
3. Use exact column names from the dataset
4. Be concise but complete
5. If a question is unclear, ask for clarification
6. Format numbers nicely (use commas, round to 2 decimal places)
"""))

    # Add chat history (previous turns) so the LLM has memory
    for msg in chat_history[-6:]:  # Keep last 6 messages to avoid token overflow
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Add the current question
    messages.append(HumanMessage(content=question))

    # ── Call the LLM ──────────────────────────────────────────
    response = llm.invoke(messages)
    return response.content


def generate_quick_insights(df: pd.DataFrame, dataset_context: str) -> List[str]:
    """
    Generates 3-5 quick, one-sentence insights about the data.
    These appear as "cards" in the UI for quick scanning.

    Args:
        df: The uploaded DataFrame
        dataset_context: Text summary of the dataset

    Returns:
        List of insight strings
    """
    llm = get_llm(temperature=0.5)

    messages = [
        SystemMessage(content="You are a data analyst. Generate brief, specific insights."),
        HumanMessage(content=f"""
Based on this dataset, generate exactly 5 quick insights.

{dataset_context}

Format: Return ONLY a numbered list (1-5), one insight per line.
Each insight should be 1 sentence, specific, and mention actual column names or numbers.
Example format:
1. The average sales value is $45,230 with a standard deviation of $12,000.
""")
    ]

    response = llm.invoke(messages)

    # Parse the numbered list into individual strings
    lines = response.content.strip().split("\n")
    insights = []
    for line in lines:
        line = line.strip()
        # Remove numbering (1. 2. etc.) and keep the content
        if line and line[0].isdigit() and ". " in line:
            insight = line.split(". ", 1)[1]
            insights.append(insight)
        elif line and len(line) > 10:  # Fallback: take non-empty lines
            insights.append(line)

    return insights[:5]  # Return maximum 5 insights
