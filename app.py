# ============================================================
# app.py
# ============================================================
# PURPOSE: Main Streamlit application — the entry point.
#
# This file creates the entire web UI and orchestrates
# the agent pipeline. It handles:
# - File uploads
# - Session state management
# - Running the agent pipeline
# - Displaying results (tables, charts, insights)
# - Chat interface for Q&A
#
# HOW STREAMLIT WORKS:
#   Streamlit reruns the entire script every time the user
#   interacts with the UI. st.session_state is used to
#   persist data between reruns (like a database in memory).
#
# TO RUN: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import traceback

# ── Page Configuration ────────────────────────────────────────
# This MUST be the first Streamlit command in the script
st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Import project modules ────────────────────────────────────
from utils.config import Config
from utils.data_loader import load_csv, get_basic_info, prepare_llm_context
from agents.workflow import run_pipeline
from agents.insight_agent import answer_question


# ── Custom CSS ────────────────────────────────────────────────
# Inject custom styles to make the UI look modern and professional
def inject_css():
    st.markdown("""
    <style>
    /* ── Main font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header gradient ── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }

    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }

    /* ── Insight cards ── */
    .insight-card {
        background-color: var(--secondary-background-color);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--text-color);
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* ── Agent step badges ── */
    .agent-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* ── Chat messages ── */
    .chat-user {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px 12px 2px 12px;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        margin-left: 20%;
    }

    .chat-ai {
        background-color: var(--secondary-background-color);
        border-left: 3px solid #22c55e;
        border-radius: 12px 12px 12px 2px;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        margin-right: 20%;
    }

    /* ── Sidebar styling ── */
    .sidebar-info {
        background-color: var(--secondary-background-color);
        border-radius: 8px;
        padding: 0.8rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }

    /* ── Cleaning suggestion items ── */
    .suggestion-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--secondary-background-color);
        font-size: 0.9rem;
    }

    /* ── Streamlit button overrides ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.85;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Session State Initialization ─────────────────────────────
# Streamlit reruns the script on every interaction.
# We use session_state to remember things between reruns.

def init_session_state():
    """Initialize all session state variables if they don't exist."""
    if "df" not in st.session_state:
        st.session_state.df = None             # Uploaded DataFrame
    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None   # Agent outputs
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []     # Conversation memory
    if "file_name" not in st.session_state:
        st.session_state.file_name = None


# ── Sidebar ───────────────────────────────────────────────────

def render_sidebar():
    """Renders the sidebar with file upload and dataset info."""

    with st.sidebar:
        st.markdown("## 🤖 AI Data Analyst")
        st.markdown("---")

        # ── File Upload ───────────────────────────────────────
        st.markdown("### 📂 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload any CSV dataset to begin AI analysis."
        )

        if uploaded_file is not None:
            # Only reload if it's a new file
            if uploaded_file.name != st.session_state.file_name:
                with st.spinner("Loading dataset..."):
                    df = load_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.pipeline_results = None  # Reset results
                    st.session_state.chat_history = []        # Reset chat
                st.success(f"✅ Loaded: **{uploaded_file.name}**")

        # ── Dataset Quick Stats ───────────────────────────────
        if st.session_state.df is not None:
            df = st.session_state.df
            info = get_basic_info(df)

            st.markdown("### 📊 Dataset Info")
            st.markdown(f"""
            <div class="sidebar-info">
            📁 <b>File:</b> {st.session_state.file_name}<br>
            📏 <b>Rows:</b> {info['rows']:,}<br>
            📋 <b>Columns:</b> {info['columns']}<br>
            🔢 <b>Numeric cols:</b> {len(info['numeric_cols'])}<br>
            🔤 <b>Categorical cols:</b> {len(info['categorical_cols'])}<br>
            ⚠️ <b>Missing values:</b> {sum(info['missing_counts'].values()) if info['missing_counts'] else 0}<br>
            💾 <b>Memory:</b> {info['memory_usage_kb']} KB
            </div>
            """, unsafe_allow_html=True)

            # ── Run Analysis Button ───────────────────────────
            st.markdown("---")
            if st.button("🚀 Run AI Analysis", use_container_width=True):
                _run_analysis_pipeline()

        # ── About Section ─────────────────────────────────────
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        <div class="sidebar-info">
        <b>AI Data Analyst Agent</b><br><br>
        Powered by:<br>
        • 🦙 LLaMA 3 (Groq)<br>
        • 🔗 LangGraph<br>
        • 🐼 Pandas<br>
        • 📊 Plotly<br><br>
        <i>Agents: Planner → Analysis → Visualization → Insight</i>
        </div>
        """, unsafe_allow_html=True)


def _run_analysis_pipeline():
    """Runs the full agent pipeline and saves results to session state."""
    df = st.session_state.df

    with st.spinner("🤖 AI agents are analyzing your dataset... (this may take 30-60 seconds)"):
        try:
            # Validate API key before running
            Config.validate()

            # Run the full LangGraph pipeline
            results = run_pipeline(df)

            # Save to session state for display
            st.session_state.pipeline_results = results
            st.success("✅ Analysis complete!")
            st.rerun()  # Refresh the page to show results

        except ValueError as e:
            st.error(f"⚙️ Configuration Error: {e}")
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            with st.expander("🔧 Error Details"):
                st.code(traceback.format_exc())


# ── Main Content Tabs ─────────────────────────────────────────

def render_main_content():
    """Renders the main content area with all tabs."""

    # ── App Header ────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Data Analyst Agent</h1>
        <p>Upload a CSV → AI agents analyze, visualize, and explain your data</p>
    </div>
    """, unsafe_allow_html=True)

    # ── No file uploaded yet ──────────────────────────────────
    if st.session_state.df is None:
        _render_welcome_screen()
        return

    df = st.session_state.df

    # ── Create tabs ───────────────────────────────────────────
    tabs = st.tabs([
        "📋 Dataset Preview",
        "🤖 Agent Workflow",
        "📊 Visualizations",
        "💡 Insights",
        "🔧 Cleaning",
        "💬 Ask AI"
    ])

    with tabs[0]:
        render_dataset_preview(df)

    with tabs[1]:
        render_agent_workflow()

    with tabs[2]:
        render_visualizations()

    with tabs[3]:
        render_insights()

    with tabs[4]:
        render_cleaning_suggestions()

    with tabs[5]:
        render_chat_interface()


def _render_welcome_screen():
    """Shows a friendly welcome screen when no file is uploaded."""

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 0;">
            <div style="font-size: 5rem;">📂</div>
            <h2 style="color: #667eea;">Upload a CSV to get started</h2>
            <p style="color: #64748b; font-size: 1.05rem;">
                Use the sidebar to upload your dataset.<br>
                The AI will automatically analyze it for you.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Show example capabilities
    st.markdown("---")
    st.markdown("### 🎯 What this app does")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("**🔍 Auto EDA**\nAutomatic exploratory data analysis with pandas")
    with c2:
        st.info("**📊 Smart Charts**\nAI selects and generates the right visualizations")
    with c3:
        st.info("**💡 AI Insights**\nLLM explains findings in plain English")
    with c4:
        st.info("**💬 Q&A Chat**\nAsk natural language questions about your data")

    st.markdown("### 🗂️ Try with sample data")
    st.markdown("""
    Don't have a CSV? Try these free datasets:
    - [Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) — passenger survival data
    - [Iris Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) — flower measurements
    - [Tips Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv) — restaurant tips
    """)


# ── Tab 1: Dataset Preview ────────────────────────────────────

def render_dataset_preview(df: pd.DataFrame):
    """Shows a preview of the uploaded dataset with summary statistics."""

    st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)

    # ── Summary Metric Cards ───────────────────────────────────
    info = get_basic_info(df)
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.metric("📏 Rows", f"{info['rows']:,}")
    with m2:
        st.metric("📋 Columns", info['columns'])
    with m3:
        missing_total = sum(info['missing_counts'].values()) if info['missing_counts'] else 0
        st.metric("⚠️ Missing Cells", missing_total)
    with m4:
        st.metric("🔁 Duplicates", info['duplicate_rows'])
    with m5:
        st.metric("💾 Memory", f"{info['memory_usage_kb']} KB")

    st.markdown("---")

    # ── Data Table ────────────────────────────────────────────
    display_rows = st.slider(
        "Rows to display", 5, min(100, len(df)), 10,
        help="Drag to show more or fewer rows"
    )
    st.dataframe(df.head(display_rows), use_container_width=True)

    # ── Column Types ──────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**🔢 Numeric Columns**")
        if info['numeric_cols']:
            st.dataframe(
                df[info['numeric_cols']].describe().round(2),
                use_container_width=True
            )
        else:
            st.info("No numeric columns found.")

    with col_right:
        st.markdown("**🔤 Categorical Columns**")
        if info['categorical_cols']:
            cat_summary = pd.DataFrame({
                col: {
                    "Unique": df[col].nunique(),
                    "Most Common": df[col].mode().iloc[0] if not df[col].mode().empty else "N/A",
                    "Missing": df[col].isnull().sum()
                }
                for col in info['categorical_cols']
            }).T
            st.dataframe(cat_summary, use_container_width=True)
        else:
            st.info("No categorical columns found.")


# ── Tab 2: Agent Workflow ─────────────────────────────────────

def render_agent_workflow():
    """Shows the agent pipeline plan and analysis results."""

    st.markdown('<div class="section-header">🤖 Agent Workflow</div>', unsafe_allow_html=True)

    results = st.session_state.pipeline_results

    if results is None:
        st.info("👈 Click **'Run AI Analysis'** in the sidebar to start the agent pipeline.")

        # Show the pipeline diagram even before running
        st.markdown("### 🔄 How the Pipeline Works")
        st.markdown("""
        ```
        📂 CSV Upload
              ↓
        🔍 Planner Agent     → Reads dataset structure, creates analysis plan
              ↓
        📊 Analysis Agent    → Performs EDA with pandas, interprets with LLM
              ↓
        📈 Visualization Agent → Selects and generates charts
              ↓
        💡 Insight Agent     → Writes comprehensive insights + answers questions
        ```
        """)
        return

    # ── Planner Agent Output ───────────────────────────────────
    st.markdown('<span class="agent-badge">🔍 Planner Agent</span>', unsafe_allow_html=True)
    st.markdown("**Analysis Plan Generated:**")
    with st.container():
        st.markdown(results.get("plan", "Plan not available."))

    st.markdown("---")

    # ── Analysis Agent Output ─────────────────────────────────
    st.markdown('<span class="agent-badge">📊 Analysis Agent</span>', unsafe_allow_html=True)
    analysis = results.get("analysis_results", {})

    if analysis:
        st.markdown("**AI Interpretation:**")
        st.markdown(analysis.get("analysis_text", "Analysis not available."))

    st.markdown("---")

    # ── Visualization Agent Summary ────────────────────────────
    st.markdown('<span class="agent-badge">📈 Visualization Agent</span>', unsafe_allow_html=True)
    viz = results.get("visualization_results", {})
    if viz:
        chart_plan = viz.get("chart_plan", {})
        charts = viz.get("charts", [])
        st.success(f"✅ Generated **{len(charts)} charts** based on dataset structure.")
        if chart_plan.get("notes"):
            for note in chart_plan["notes"]:
                st.caption(f"ℹ️ {note}")

    st.markdown("---")

    # ── Insight Agent Summary ──────────────────────────────────
    st.markdown('<span class="agent-badge">💡 Insight Agent</span>', unsafe_allow_html=True)
    quick_insights = results.get("quick_insights", [])
    if quick_insights:
        st.markdown("**Quick Insights:**")
        for i, insight in enumerate(quick_insights, 1):
            st.markdown(f"""
            <div class="insight-card">
            💡 {insight}
            </div>
            """, unsafe_allow_html=True)


# ── Tab 3: Visualizations ─────────────────────────────────────

def render_visualizations():
    """Renders all generated charts."""

    st.markdown('<div class="section-header">📊 Visualizations</div>', unsafe_allow_html=True)

    results = st.session_state.pipeline_results

    if results is None:
        st.info("👈 Click **'Run AI Analysis'** in the sidebar to generate charts.")
        return

    viz = results.get("visualization_results", {})
    charts = viz.get("charts", [])

    if not charts:
        st.warning("No charts were generated.")
        return

    # Render each chart with its title and description
    for title, fig, description in charts:
        st.markdown(f"**{title}**")
        st.caption(description)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")


# ── Tab 4: Insights ───────────────────────────────────────────

def render_insights():
    """Shows the full AI-generated insight report."""

    st.markdown('<div class="section-header">💡 AI Insights Report</div>', unsafe_allow_html=True)

    results = st.session_state.pipeline_results

    if results is None:
        st.info("👈 Click **'Run AI Analysis'** in the sidebar to generate insights.")
        return

    insight_report = results.get("insight_report", "")

    if insight_report:
        st.markdown(insight_report)
    else:
        st.warning("Insight report not available.")


# ── Tab 5: Cleaning Suggestions ───────────────────────────────

def render_cleaning_suggestions():
    """Shows data cleaning recommendations."""

    st.markdown('<div class="section-header">🔧 Data Cleaning Suggestions</div>', unsafe_allow_html=True)

    results = st.session_state.pipeline_results

    if results is None:
        st.info("👈 Click **'Run AI Analysis'** in the sidebar to get cleaning suggestions.")
        return

    analysis = results.get("analysis_results", {})
    suggestions = analysis.get("cleaning_suggestions", [])

    if not suggestions:
        st.info("Run the analysis to see cleaning suggestions.")
        return

    st.markdown("The Analysis Agent identified the following issues and recommendations:")
    st.markdown("")

    for suggestion in suggestions:
        st.markdown(f"""
        <div class="suggestion-item">
        {suggestion}
        </div>
        """, unsafe_allow_html=True)

    # ── Generated cleaning code ────────────────────────────────
    st.markdown("---")
    st.markdown("### 🐍 Sample Cleaning Code")
    st.markdown("You can use this as a starting point to clean your data:")

    df = st.session_state.df
    info = get_basic_info(df)

    code_lines = ["import pandas as pd", "import numpy as np", "",
                  "# Load your dataset", "df = pd.read_csv('your_file.csv')", ""]

    if info['duplicate_rows'] > 0:
        code_lines += ["# Remove duplicate rows", "df = df.drop_duplicates()", ""]

    numeric_cols = info['numeric_cols']
    cat_cols = info['categorical_cols']

    if info['missing_counts']:
        code_lines.append("# Fill missing values")
        for col, count in info['missing_counts'].items():
            if col in numeric_cols:
                code_lines.append(f"df['{col}'].fillna(df['{col}'].median(), inplace=True)")
            elif col in cat_cols:
                code_lines.append(f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)")
        code_lines.append("")

    code_lines += ["# Verify cleaning", "print(df.isnull().sum())", "print(df.shape)"]

    st.code("\n".join(code_lines), language="python")


# ── Tab 6: Chat Interface ─────────────────────────────────────

def render_chat_interface():
    """
    Renders the natural language Q&A chat interface.
    Implements conversational memory using session state.
    """

    st.markdown('<div class="section-header">💬 Ask AI About Your Data</div>', unsafe_allow_html=True)

    df = st.session_state.df
    results = st.session_state.pipeline_results

    # Need at least the dataset context to answer questions
    if results is None:
        st.info("👈 Click **'Run AI Analysis'** first for best results, or ask basic questions directly.")

    # Get the dataset context (or build it if pipeline hasn't run)
    if results and results.get("dataset_context"):
        dataset_context = results["dataset_context"]
    else:
        from utils.data_loader import get_basic_info, prepare_llm_context
        info = get_basic_info(df)
        dataset_context = prepare_llm_context(df, info)

    # ── Suggested Questions ────────────────────────────────────
    st.markdown("**💡 Suggested questions:**")
    suggestions = [
        "What are the main trends in this dataset?",
        "Which column has the most missing values?",
        "Are there any strong correlations?",
        "What are the outliers in the numeric columns?",
        "Give me a summary in simple terms."
    ]

    cols = st.columns(len(suggestions))
    for i, (col, suggestion) in enumerate(zip(cols, suggestions)):
        with col:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                _process_chat_question(suggestion, df, dataset_context)

    st.markdown("---")

    # ── Chat History Display ───────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-user">
                <b>You:</b> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-ai">
                <b>🤖 AI Analyst:</b><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)

    # ── Chat Input ────────────────────────────────────────────
    st.markdown("")
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question about your data...",
            placeholder="e.g., What is the average value of the sales column?",
            label_visibility="collapsed"
        )
        col_submit, col_clear = st.columns([4, 1])
        with col_submit:
            submitted = st.form_submit_button("Send 📤", use_container_width=True)
        with col_clear:
            if st.form_submit_button("Clear 🗑️", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    if submitted and user_input.strip():
        _process_chat_question(user_input.strip(), df, dataset_context)


def _process_chat_question(question: str, df: pd.DataFrame, dataset_context: str):
    """Sends a question to the Insight Agent and updates chat history."""

    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.spinner("🤔 AI is thinking..."):
        try:
            # Call the Insight Agent's Q&A function with memory
            answer = answer_question(
                question=question,
                df=df,
                dataset_context=dataset_context,
                chat_history=st.session_state.chat_history[:-1]  # Exclude current question
            )

            # Add AI response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"❌ Sorry, I encountered an error: {str(e)}"
            })

    st.rerun()


# ── Entry Point ───────────────────────────────────────────────

def main():
    """Main function — sets up the app and renders all components."""

    # Apply custom CSS
    inject_css()

    # Initialize session state
    init_session_state()

    # Render sidebar (file upload + run button)
    render_sidebar()

    # Render main content
    render_main_content()


if __name__ == "__main__":
    main()
