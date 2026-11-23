"""Email agent with authentication-gated tools.

This module is import-safe: building the agent and running the workflow only
happens through the explicit builder functions or ``main()``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, NotRequired, cast

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, ToolMessage
from langchain.tools import ToolException, ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

log = logging.getLogger(__name__)

# Pin a specific tool-capable free model. Override with OPENROUTER_MODEL if
# this model is retired.
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3.5-lightning:free"

EMAIL_RECURSION_LIMIT = 25
DEFAULT_USER_REQUEST = "Check my inbox and reply to the most recent email."

AUTHENTICATED_PROMPT = (
    "You are a helpful assistant that can check the inbox and send emails."
)
UNAUTHENTICATED_PROMPT = (
    "You are a helpful assistant that can authenticate users."
)


class AuthenticatedState(AgentState):
    """Mutable workflow state shared by the agent and its tools."""

    authenticated: NotRequired[bool]


class EmailContext(BaseModel):
    """Immutable per-run context with the demo account credentials."""

    email_address: str = Field(default="julie@example.com", min_length=1)
    password: str = Field(default="password123", min_length=1)


def configure_logging() -> None:
    """Configure application logging without doing work at import time."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


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


@tool
def check_inbox() -> str:
    """Check the inbox for recent emails."""
    return """
    Hi Julie,
    I'm going to be in town next week and was wondering if we could grab a coffee?
    - best, Jane (jane@example.com)
    """


@tool
def send_email(
    to: Annotated[str, Field(min_length=1)],
    subject: Annotated[str, Field(min_length=1)],
    body: Annotated[str, Field(min_length=1)],
) -> str:
    """Send a response email."""
    return f"Email sent to {to} with subject {subject} and body {body}"


@tool
def authenticate(
    email: Annotated[str, Field(min_length=1)],
    password: Annotated[str, Field(min_length=1)],
    runtime: ToolRuntime[EmailContext, AuthenticatedState],
) -> Command:
    """Authenticate the user with the given email and password."""
    if runtime.tool_call_id is None:
        raise ToolException("Missing tool_call_id for authentication.")
    authenticated = (
        email == runtime.context.email_address
        and password == runtime.context.password
    )
    return Command(
        update={
            "authenticated": authenticated,
            "messages": [
                ToolMessage(
                    content=(
                        "Successfully authenticated"
                        if authenticated
                        else "Authentication failed"
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@wrap_model_call
async def dynamic_tool_call(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse:
    """Allow check_inbox and send_email only once the user is authenticated."""
    if request.state.get("authenticated"):
        tools = [check_inbox, send_email]
    else:
        tools = [authenticate]
    request = request.override(tools=tools)
    return await handler(request)


@dynamic_prompt
def dynamic_prompt_func(request: ModelRequest) -> str:
    """Generate the system prompt based on authentication status."""
    if request.state.get("authenticated"):
        return AUTHENTICATED_PROMPT
    return UNAUTHENTICATED_PROMPT


def build_email_agent(model: BaseChatModel) -> CompiledStateGraph:
    """Build the email agent with authentication-gated tools."""
    return create_agent(
        model=model,
        tools=[authenticate, check_inbox, send_email],
        state_schema=AuthenticatedState,
        context_schema=EmailContext,
        checkpointer=InMemorySaver(),
        middleware=[
            dynamic_tool_call,
            dynamic_prompt_func,
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "authenticate": False,
                    "check_inbox": False,
                    "send_email": True,
                }
            ),
        ],
        name="email_agent",
    )


def build_agent() -> CompiledStateGraph:
    """Build the email agent with default settings.

    Zero-argument entry point for the LangGraph dev server, referenced from
    ``langgraph.json``. Building still happens only when this is called.
    """
    return build_email_agent(build_model())


async def main() -> int:
    configure_logging()
    try:
        agent = build_email_agent(build_model())
        config = {
            "recursion_limit": EMAIL_RECURSION_LIMIT,
            "configurable": {
                "thread_id": os.environ.get("EMAIL_THREAD_ID", "email-demo")
            },
        }
        context = EmailContext()

        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=DEFAULT_USER_REQUEST)]},
            config=config,
            context=context,
        )

        # send_email requires human approval. Approve every pending tool call
        # from the interrupt payload so the demo can continue unattended.
        interrupts = response.get("__interrupt__") or ()
        if interrupts:
            hitl_request = interrupts[0].value
            pending = (
                len(hitl_request.get("action_requests", ()))
                if isinstance(hitl_request, dict)
                else 1
            )
            log.info("Approving %d pending send_email call(s).", pending)
            response = await agent.ainvoke(
                Command(resume={"decisions": [{"type": "approve"}] * pending}),
                config=config,
                context=context,
            )
    except Exception as exc:  # noqa: BLE001 - surface a readable failure
        print(f"Email agent failed: {exc}", file=sys.stderr)
        log.debug("Email agent failed", exc_info=True)
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
