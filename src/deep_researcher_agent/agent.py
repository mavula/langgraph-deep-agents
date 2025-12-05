"""Supply and demand zone validation agent assembly.

This module wires together the tools, prompts, and state configuration required
for a zone-validation coordinator agent. The structure mirrors the tutorial
agent from ``docs/4_full_agent.py`` while adapting imports to the backend
package layout.
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from deep_researcher_agent.file_tools import ls, read_file, write_file
from deep_researcher_agent.prompts import (
    DEMAND_ZONE_INSTRUCTIONS,
    DATA_PREP_INSTRUCTIONS,
    FILE_USAGE_INSTRUCTIONS,
    SUPPLY_ZONE_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
)
from deep_researcher_agent.market_tools import MARKET_TOOLS
from deep_researcher_agent.research_tools import (
    get_today_str,
    get_today_str_tool,
    think_tool,
)
from deep_researcher_agent.sandbox_tool import pyodide_sandbox
from deep_researcher_agent.state import DeepAgentState
from deep_researcher_agent.task_tool import _create_task_tool
from deep_researcher_agent.todo_tools import write_todos, read_todos

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _build_model() -> ChatOpenAI:
    """Configure the chat model using OpenAI credentials and optional base URL."""
    kwargs = {"model": OPENAI_MODEL, "temperature": 0}
    if OPENAI_API_KEY:
        kwargs["api_key"] = OPENAI_API_KEY
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)


model = _build_model()

# Limits for delegation and sub-agent coordination.
max_concurrent_validation_units = 3
max_validator_iterations = 3

# Tools available to the validation and data prep sub-agents.
sub_agent_tools = [think_tool, get_today_str_tool, pyodide_sandbox, *MARKET_TOOLS]

# Core tools available to the primary agent.
built_in_tools = [
    ls,
    read_file,
    write_file,
    write_todos,
    read_todos,
    get_today_str_tool,
    pyodide_sandbox,
    *MARKET_TOOLS,
]

# Define the specialized zone validation sub-agents.
demand_zone_agent = {
    "name": "demand-zone-analyst",
    "description": "Analyze demand zones using price structure plus POC, CVD, and EMA confluence.",
    "prompt": DEMAND_ZONE_INSTRUCTIONS.format(date=get_today_str()),
    "tools": ["think_tool", "pyodide_sandbox"],
}

supply_zone_agent = {
    "name": "supply-zone-analyst",
    "description": "Analyze supply zones using price structure plus POC, CVD, and EMA confluence.",
    "prompt": SUPPLY_ZONE_INSTRUCTIONS.format(date=get_today_str()),
    "tools": ["think_tool", "pyodide_sandbox"],
}

data_prep_agent = {
    "name": "data-prep",
    "description": (
        "Fetch and prepare candles, EMA, POC, and CVD for a symbol/timeframe/date range. "
        "Ensure the DeepAgentState has all required series before zone analysis."
    ),
    "prompt": DATA_PREP_INSTRUCTIONS,
    "tools": [
        "get_candles",
        "get_ema",
        "get_current_date",
        "compare_dates",
        "think_tool",
    ],
}

# Task delegation tool for spawning isolated sub-agent contexts.
task_tool = _create_task_tool(
    sub_agent_tools,
    [data_prep_agent, demand_zone_agent, supply_zone_agent],
    model,
    DeepAgentState,
)

delegation_tools = [task_tool]


def _unique_tools(tools):
    """Return tools with unique names while preserving order."""
    seen = set()
    unique = []
    for tool_ in tools:
        name = getattr(tool_, "name", getattr(tool_, "__name__", str(tool_)))
        if name in seen:
            continue
        seen.add(name)
        unique.append(tool_)
    return unique


all_tools = _unique_tools(sub_agent_tools + built_in_tools + delegation_tools)

# Assemble the full instruction block for the primary agent.
SUBAGENT_INSTRUCTIONS = SUBAGENT_USAGE_INSTRUCTIONS.format(
    max_concurrent_validation_units=max_concurrent_validation_units,
    max_validator_iterations=max_validator_iterations,
    date=datetime.now().strftime("%a %b %-d, %Y"),
)

SECTION_SEPARATOR = "\n\n" + "=" * 80 + "\n\n"

INSTRUCTIONS = SECTION_SEPARATOR.join(
    [
        "# TODO MANAGEMENT\n" + TODO_USAGE_INSTRUCTIONS,
        "# FILE SYSTEM USAGE\n" + FILE_USAGE_INSTRUCTIONS,
        "# SUB-AGENT DELEGATION\n" + SUBAGENT_INSTRUCTIONS,
    ]
)

# Create the primary agent during module import. The caller can import
# ``agent`` directly without invoking a factory function.
agent = create_react_agent(
    model,
    all_tools,
    prompt=INSTRUCTIONS,
    state_schema=DeepAgentState,
).with_config({"recursion_limit": 50})

__all__ = ["agent"]
