"""Shared tools for the zone validation agents."""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from deep_researcher_agent.prompts import SUMMARIZE_WEB_SEARCH

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _build_summarization_model() -> ChatOpenAI:
    """Configure summarization model using OpenAI credentials."""
    kwargs = {"model": OPENAI_MODEL, "temperature": 0.4}
    if OPENAI_API_KEY:
        kwargs["api_key"] = OPENAI_API_KEY
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)


summarization_model = _build_summarization_model()


class Summary(BaseModel):
    """Schema for webpage content summarization."""

    filename: str = Field(description="Name of the file to store.")
    summary: str = Field(description="Key learnings from the webpage.")


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")


@tool(parse_docstring=True)
def get_today_str_tool() -> str:
    """Return today's date in a human-readable format (e.g., Mon Jan 1, 2024).

    Returns:
        Current date string for quick timestamping.
    """
    return get_today_str()


def summarize_webpage_content(webpage_content: str) -> Summary:
    """Summarize webpage content using the configured summarization model."""
    try:
        structured_model = summarization_model.with_structured_output(Summary)
        summary_and_filename = structured_model.invoke(
            [
                HumanMessage(
                    content=SUMMARIZE_WEB_SEARCH.format(
                        webpage_content=webpage_content, date=get_today_str()
                    )
                )
            ]
        )
        return summary_and_filename
    except Exception:
        return Summary(
            filename="search_result.md",
            summary=(
                webpage_content[:1000] + "..."
                if len(webpage_content) > 1000
                else webpage_content
            ),
        )


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Record strategic reflection on zone validation progress.

    Args:
        reflection: Detailed notes on findings, gaps, and next steps for the zone check.

    Returns:
        Confirmation message that the reflection was recorded.
    """
    return f"Reflection recorded: {reflection}"
