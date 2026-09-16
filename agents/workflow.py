# ============================================================
# agents/workflow.py
# ============================================================
# PURPOSE: Orchestrates the full agent pipeline using LangGraph.
#
# WHAT IS LANGGRAPH?
#   LangGraph is a framework for building stateful, multi-step
#   AI workflows as a directed graph (like a flowchart).
#
#   Nodes = individual agents or processing steps
#   Edges = the flow between steps
#   State = shared data passed between all nodes
#
# OUR WORKFLOW GRAPH:
#
#   [START]
#      |
#      v
#   [load_data]        -- Prepares the dataset context
#      |
#      v
#   [planner]          -- Planner Agent creates analysis plan
#      |
#      v
#   [analyze]          -- Analysis Agent performs EDA
#      |
#      v
#   [visualize]        -- Visualization Agent creates charts
#      |
#      v
#   [generate_insights] -- Insight Agent writes final report
#      |
#      v
#   [END]
#
# INTERVIEW TIP:
#   "I used LangGraph to model the agent workflow as a directed
#    acyclic graph (DAG). Each node is an isolated agent function
#    that reads and writes to a shared state object. This makes
#    the pipeline easy to debug, extend, and explain."
# ============================================================

from typing import TypedDict, Annotated, Any
import pandas as pd
from langgraph.graph import StateGraph, START, END

from utils.data_loader import get_basic_info, prepare_llm_context
from agents.planner_agent import run_planner_agent
from agents.analysis_agent import run_analysis_agent
from agents.visualization_agent import run_visualization_agent
from agents.insight_agent import run_insight_agent, generate_quick_insights


# ── State Definition ──────────────────────────────────────────
# The "State" is a TypedDict — a Python dict with fixed keys and types.
# Every node in the graph reads from and writes to this shared state.
# This is how agents communicate with each other.

class AgentState(TypedDict):
    """
    Shared state object that flows through the entire pipeline.

    Think of this as the "conversation" between agents —
    each agent reads what previous agents wrote, adds its own
    results, and passes the enriched state to the next agent.
    """
    # Input
    df: Any                         # The actual pandas DataFrame
    dataset_context: str            # Text summary for LLM prompts

    # Outputs from each agent (populated as pipeline runs)
    basic_info: dict                # Dataset metadata (rows, cols, types)
    plan: str                       # Analysis plan from Planner Agent
    analysis_results: dict          # EDA results from Analysis Agent
    visualization_results: dict     # Chart list from Visualization Agent
    insight_report: str             # Narrative report from Insight Agent
    quick_insights: list            # 5 quick bullet insights


# ── Node Functions ────────────────────────────────────────────
# Each function below is a "node" in the LangGraph.
# They all take state as input and return a dict to update the state.

def node_load_data(state: AgentState) -> dict:
    """
    Node 1: Prepares dataset metadata and LLM context.
    This runs first before any AI agents.
    """
    df = state["df"]

    # Compute basic statistics using pandas
    basic_info = get_basic_info(df)

    # Build the text summary that all LLM agents will use
    dataset_context = prepare_llm_context(df, basic_info)

    return {
        "basic_info": basic_info,
        "dataset_context": dataset_context
    }


def node_plan(state: AgentState) -> dict:
    """
    Node 2: Planner Agent — creates the analysis plan.
    """
    plan = run_planner_agent(
        df=state["df"],
        dataset_context=state["dataset_context"]
    )
    return {"plan": plan}


def node_analyze(state: AgentState) -> dict:
    """
    Node 3: Analysis Agent — performs EDA and generates interpretation.
    """
    analysis_results = run_analysis_agent(
        df=state["df"],
        dataset_context=state["dataset_context"]
    )
    return {"analysis_results": analysis_results}


def node_visualize(state: AgentState) -> dict:
    """
    Node 4: Visualization Agent — generates charts.
    """
    visualization_results = run_visualization_agent(df=state["df"])
    return {"visualization_results": visualization_results}


def node_generate_insights(state: AgentState) -> dict:
    """
    Node 5: Insight Agent — writes the final insight report.
    """
    insight_report = run_insight_agent(
        df=state["df"],
        dataset_context=state["dataset_context"],
        analysis_results=state["analysis_results"],
        plan=state["plan"]
    )

    # Also generate quick insight bullets for the UI
    quick_insights = generate_quick_insights(
        df=state["df"],
        dataset_context=state["dataset_context"]
    )

    return {
        "insight_report": insight_report,
        "quick_insights": quick_insights
    }


# ── Graph Builder ─────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """
    Assembles the LangGraph workflow by:
    1. Creating a graph with our state schema
    2. Adding nodes (agent functions)
    3. Adding edges (connecting nodes in sequence)
    4. Compiling to an executable graph

    Returns:
        A compiled LangGraph that can be invoked with .invoke(state)
    """

    # Create the graph, telling it what our state looks like
    graph = StateGraph(AgentState)

    # ── Add Nodes ──────────────────────────────────────────────
    graph.add_node("load_data", node_load_data)
    graph.add_node("planner", node_plan)   # renamed from "plan" — cannot share name with state key
    graph.add_node("analyze", node_analyze)
    graph.add_node("visualize", node_visualize)
    graph.add_node("generate_insights", node_generate_insights)

    # ── Add Edges (define the flow) ────────────────────────────
    graph.add_edge(START, "load_data")
    graph.add_edge("load_data", "planner")
    graph.add_edge("planner", "analyze")
    graph.add_edge("analyze", "visualize")
    graph.add_edge("visualize", "generate_insights")
    graph.add_edge("generate_insights", END)

    # Compile the graph into an executable object
    return graph.compile()


def run_pipeline(df: pd.DataFrame) -> dict:
    """
    Main entry point: runs the complete agent pipeline on a DataFrame.

    Args:
        df: The uploaded pandas DataFrame

    Returns:
        dict: The final state containing all agent outputs
    """

    # Build the graph
    app = build_agent_graph()

    # Create the initial state with just the DataFrame
    # All other keys will be populated by the agents as they run
    initial_state = {
        "df": df,
        "dataset_context": "",
        "basic_info": {},
        "plan": "",
        "analysis_results": {},
        "visualization_results": {},
        "insight_report": "",
        "quick_insights": []
    }

    # Run the pipeline — LangGraph executes each node in order
    final_state = app.invoke(initial_state)

    return final_state
