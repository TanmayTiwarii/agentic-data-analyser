# ============================================================
# utils/config.py
# ============================================================
# PURPOSE: Central configuration loader for the entire project.
# Reads settings from the .env file so you never hardcode secrets.
# ============================================================

import os
from dotenv import load_dotenv

# Load all variables from .env into the environment
load_dotenv()


class Config:
    """
    Single place to store all application settings.
    Any file in the project imports from here instead of
    reading os.getenv() directly — keeps things tidy.
    """

    # ── LLM Settings ──────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ── App Settings ──────────────────────────────────────────
    APP_TITLE: str      = os.getenv("APP_TITLE", "AI Data Analyst Agent")
    MAX_ROWS_DISPLAY: int = int(os.getenv("MAX_ROWS_DISPLAY", 100))

    # ── Chart Defaults ────────────────────────────────────────
    # Max columns used in correlation heatmap (performance limit)
    MAX_HEATMAP_COLS: int = 12

    # Max unique category values shown in bar chart
    MAX_CATEGORIES: int = 15

    @classmethod
    def validate(cls) -> bool:
        """Check that critical environment variables are set."""
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Please add it to your .env file.\n"
                "Get a free key at https://console.groq.com/"
            )
        return True
