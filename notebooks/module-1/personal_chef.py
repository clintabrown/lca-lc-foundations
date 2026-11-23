"""Personal chef example.

This module is import-safe: building the agent and running the workflow only
happens through the explicit builder functions or ``main()``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_tavily import TavilySearch
from langgraph.graph.state import CompiledStateGraph

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

log = logging.getLogger(__name__)

# Pin a specific tool-capable free model. Override with OPENROUTER_MODEL if
# this model is retired.
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3.5-lightning:free"

CHEF_RECURSION_LIMIT = 25
DEFAULT_USER_REQUEST = (
    "I have leftover chicken, rice, and broccoli. What can I make for dinner?"
)

CHEF_PROMPT = """You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.
"""


def configure_logging() -> None:
    """Configure application logging without doing work at import time."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"Copy example.env to .env and set it."
        )
    return value


def message_text(message: AnyMessage) -> str:
    """Extract text from a message whose content may be a string or content blocks."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_text = block.get("text")
                if isinstance(block_text, str):
                    parts.append(block_text)
        if parts:
            return "\n".join(parts)

    # Fall back to the model's canonical text rendering for other block types.
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    return str(content)


def build_model() -> BaseChatModel:
    """Build the chat model."""
    model_name = os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    model_name = model_name.removeprefix("openrouter:")
    log.info("Using OpenRouter model %s", model_name)
    return cast(
        BaseChatModel,
        init_chat_model(f"openrouter:{model_name}", temperature=0),
    )


def build_web_search() -> TavilySearch:
    """Build the Tavily web-search tool."""
    _require_env("TAVILY_API_KEY")
    return TavilySearch(
        name="web_search",
        description=(
            "Search the web for recipes that can be made with the given "
            "leftover ingredients."
        ),
        max_results=5,
        search_depth="basic",
        include_raw_content=False,
        handle_tool_error=True,
    )


def build_chef_agent(model: BaseChatModel) -> CompiledStateGraph:
    """Build the personal chef agent."""
    return create_agent(
        model=model,
        tools=[build_web_search()],
        system_prompt=CHEF_PROMPT,
        name="personal_chef",
    )


async def main() -> int:
    configure_logging()
    try:
        agent = build_chef_agent(build_model())
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=DEFAULT_USER_REQUEST)]},
            config={
                "recursion_limit": CHEF_RECURSION_LIMIT,
                "tags": ["PC"],
            },
        )
    except Exception as exc:  # noqa: BLE001 - surface a readable failure
        print(f"Personal chef failed: {exc}", file=sys.stderr)
        log.debug("Personal chef failed", exc_info=True)
        return 1

    messages = response.get("messages") or []
    text = message_text(messages[-1]).strip() if messages else ""
    if text:
        print(text)
        return 0

    print(
        "The agent returned no text. Re-run with LOG_LEVEL=DEBUG for details.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
