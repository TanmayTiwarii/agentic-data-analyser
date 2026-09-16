# ============================================================
# agents/llm_client.py
# ============================================================
# PURPOSE: Creates and returns the LLM (Language Model) client.
#
# Think of this as the "brain connector" — every agent imports
# from here to get access to the LLM instead of creating their
# own client (DRY principle: Don't Repeat Yourself).
#
# We use Groq because it's:
# - Free to use
# - Extremely fast (they use custom LPU hardware)
# - Supports powerful open-source models like LLaMA 3
# ============================================================

from langchain_groq import ChatGroq
from utils.config import Config


def get_llm(temperature: float = 0.3) -> ChatGroq:
    """
    Creates and returns a Groq LLM client.

    Args:
        temperature (float):
            Controls randomness of LLM responses.
            0.0 = very focused/deterministic (good for analysis)
            1.0 = very creative/random (good for storytelling)
            We default to 0.3 for balanced analytical responses.

    Returns:
        ChatGroq: A LangChain-compatible LLM object.

    How it works:
        LangChain's ChatGroq wraps the Groq API.
        When you call llm.invoke("some prompt"), it sends the
        prompt to Groq's servers and returns the AI's response.
    """
    # Validate that the API key is set before creating the client
    Config.validate()

    return ChatGroq(
        api_key=Config.GROQ_API_KEY,
        model_name=Config.GROQ_MODEL,
        temperature=temperature,
        max_tokens=2048,           # Max response length
    )
