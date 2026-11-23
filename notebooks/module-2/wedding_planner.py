"""Multi-agent wedding planner example.

This module is import-safe: building the agents and running the workflow only
happens through the explicit builder functions or ``main()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, cast

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, ToolMessage
from langchain.tools import BaseTool, ToolException, ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, Field
from sqlalchemy import create_engine

if TYPE_CHECKING:
    from langchain_community.utilities import SQLDatabase

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

log = logging.getLogger(__name__)

# Keep this list small and explicit. Add codes only after confirming that the
# remote MCP server treats them as transient.
RETRYABLE_MCP_CODES = frozenset({-32603})
MAX_WEB_SEARCHES = 12
MAX_SQL_ROWS = 200
COORDINATOR_RECURSION_LIMIT = 40
SPECIALIST_RECURSION_LIMIT = 25
SPECIALIST_TIMEOUT_SECONDS = 120.0
DEFAULT_USER_REQUEST = (
    "I'm from London and I'd like a wedding in Paris for 100 guests, jazz genre."
)

# Pin a specific tool-capable free model. The ``openrouter/free`` router can
# select a model that does not support tool calling, which leaves the agent with
# no usable response. Override with OPENROUTER_MODEL if this model is retired.
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3.5-lightning:free"

TRAVEL_PROMPT = """You are a travel agent. Search for flights to the desired destination wedding location.
You are not allowed to ask any more follow-up questions; you must find the best flight options based on:
- Price (lowest, economy class)
- Duration (shortest)
- Date (the time of year you believe is best for a wedding at this location)
Only look for one ticket, one way. You may need to make multiple searches to iteratively find the best options.
You will be given only the origin and destination. Think critically about the best options.
The MCP layer retries transient failures automatically. If a tool still fails, explain the failure briefly
instead of repeating the same call indefinitely. Once you have found the best options, return a shortlist.
"""

VENUE_PROMPT = """You are a venue specialist. Search for venues in the desired location with the desired capacity.
You are not allowed to ask any more follow-up questions; you must find the best venue options based on:
- Price (lowest)
- Capacity (exact match)
- Reviews (highest)
You may need to make multiple searches to iteratively find the best options.
You have a hard limit of 12 web_search calls, enforced by the system. Count every web_search call.
After 12 searches, stop searching and summarize the best options you have found.
"""

PLAYLIST_PROMPT = """You are a playlist specialist. Query the SQL database and curate the perfect playlist for a wedding
given a genre. Once you have your playlist, calculate the total duration and cost; each song has an associated price.
If a query fails, fix the query and try again. Do not come back empty handed.

This is a read-only SQLite database. Work in this order:
1. Call list_tables to discover the available tables.
2. Call get_schema for the tables you need.
3. Call query_playlist_db with SELECT-only queries. Never attempt to modify data.
"""

COORDINATOR_PROMPT = """You are a wedding coordinator.
First collect the origin, destination, guest count, and genre from the user.
When you have all four values, call update_state alone, without any other tool calls. Wait for it to return.
Only after update_state succeeds, delegate to your specialists for flights, venues, and playlists.
Once you have received their answers, coordinate the perfect wedding for the user.
"""

class WeddingState(AgentState):
    """Mutable workflow state shared by the coordinator and its tools."""

    origin: NotRequired[str]
    destination: NotRequired[str]
    guest_count: NotRequired[int]
    genre: NotRequired[str]

class WeddingContext(BaseModel):
    """Immutable per-run context."""

    user_id: str = Field(default="demo-user", min_length=1)
    budget: int | None = Field(default=None, ge=0)

class WeddingPlan(BaseModel):
    """Structured final response from the coordinator."""

    flights: list[str] = Field(
        default_factory=list,
        description="Shortlisted flight options with price and duration.",
    )
    venue: str = Field(
        default="",
        description="Recommended venue, including capacity and price when available.",
    )
    playlist: list[str] = Field(
        default_factory=list,
        description="Recommended tracks, including duration and price when available.",
    )
    total_playlist_duration: str | None = Field(
        default=None,
        description="Total playlist duration, for example '84 minutes'.",
    )
    total_playlist_cost: str | None = Field(
        default=None,
        description="Total playlist cost, including currency.",
    )
    notes: str | None = Field(
        default=None,
        description="Any important caveats, assumptions, or next steps.",
    )

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

def _is_transient_mcp_exception(exc: BaseException) -> bool:
    """Return True for transport-style failures that are worth retrying."""
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    # MCP transport errors surface from httpx/httpcore/anyio. Checking the
    # module avoids importing those packages directly in this example.
    module = type(exc).__module__.split(".", 1)[0]
    return module in {"httpx", "httpcore", "anyio"}

def _mcp_error_result(tool_name: str, error: object) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=(
                    f"Tool {tool_name} failed: {error}. "
                    "Do not retry this tool; report the failure to the user."
                ),
            )
        ],
        isError=True,
    )

@dataclass(frozen=True, slots=True)
class RetryMCPInterceptor:
    """Retry transient MCP failures and surface permanent failures as errors.

    ``max_attempts`` counts the initial call, so the default value of 3 means
    one initial attempt plus two retries.
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("retry delays must be non-negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    ) -> MCPToolCallResult:
        for attempt in range(1, self.max_attempts + 1):
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    return await handler(request)
            except McpError as exc:
                code = getattr(getattr(exc, "error", None), "code", None)
                log.warning(
                    "MCP error on %s (code=%s, attempt %d/%d): %s",
                    request.name,
                    code,
                    attempt,
                    self.max_attempts,
                    exc,
                )
                if code not in RETRYABLE_MCP_CODES:
                    return _mcp_error_result(request.name, exc)
            except Exception as exc:  # noqa: BLE001 - transport errors are broad
                if not _is_transient_mcp_exception(exc):
                    log.exception("Unexpected MCP failure on %s", request.name)
                    return _mcp_error_result(request.name, "unexpected tool failure")
                log.warning(
                    "Transient MCP failure on %s (attempt %d/%d): %s",
                    request.name,
                    attempt,
                    self.max_attempts,
                    exc,
                )

            if attempt < self.max_attempts:
                delay = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
                await asyncio.sleep(delay + random.uniform(0, delay * 0.25))

        return _mcp_error_result(
            request.name,
            f"failed after {self.max_attempts} attempts",
        )

def _should_retry_model_error(exc: Exception) -> bool:
    """Retry transient model failures, but not quota or auth errors."""
    text = str(exc).lower()
    non_retryable = (
        "401",
        "402",
        "429",
        "unauthorized",
        "invalid api key",
        "insufficient credits",
        "rate limit",
        "free-models-per-day",
    )
    return not any(marker in text for marker in non_retryable)

def model_failure_message(exc: Exception) -> str:
    """Turn a model-call failure into an actionable message."""
    detail = str(exc).strip() or type(exc).__name__
    lowered = detail.lower()

    if "429" in detail or "rate limit" in lowered or "free-models-per-day" in lowered:
        return (
            "OpenRouter's free-model daily limit is exhausted, so no wedding "
            "plan could be generated. The limit resets at 00:00 UTC, or you "
            "can add credits to the OpenRouter account to continue immediately.\n\n"
            f"Details: {detail}"
        )
    if "402" in detail or "insufficient credits" in lowered:
        return (
            "The OpenRouter account has no credits available for this model, "
            "so no wedding plan could be generated. Add credits at "
            "https://openrouter.ai/settings/credits or configure a free model.\n\n"
            f"Details: {detail}"
        )
    if "timeout" in lowered or "timed out" in lowered:
        return (
            "The model request timed out before a wedding plan could be generated. "
            "Try again, or set OPENROUTER_MODEL to a faster model.\n\n"
            f"Details: {detail}"
        )
    return f"Wedding planner failed: {detail}"

def _retry_middleware() -> ModelRetryMiddleware:
    """Build model retry middleware that fails fast on quota/auth errors."""
    return ModelRetryMiddleware(
        max_retries=2,
        jitter=True,
        retry_on=_should_retry_model_error,
        # Re-raise so ``main`` can print one clear, actionable error.
        on_failure="error",
    )

def build_openrouter_client() -> Any:
    """Build an OpenRouter SDK client with retries disabled.

    ``ChatOpenRouter(max_retries=0)`` still falls back to the SDK's one-hour
    default retry window, which makes failures look like a hang. Passing an
    explicit client with ``strategy="none"`` avoids that.
    """
    import openrouter  # noqa: PLC0415
    from openrouter.utils import BackoffStrategy, RetryConfig  # noqa: PLC0415

    return openrouter.OpenRouter(
        api_key=_require_env("OPENROUTER_API_KEY"),
        timeout_ms=int(os.environ.get("OPENROUTER_TIMEOUT_MS", "60000")),
        retry_config=RetryConfig(
            strategy="none",
            backoff=BackoffStrategy(
                initial_interval=0,
                max_interval=0,
                exponent=1.0,
                max_elapsed_time=0,
            ),
            retry_connection_errors=False,
        ),
    )

def build_model() -> BaseChatModel:
    """Build the shared chat model."""
    model_name = os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    model_name = model_name.removeprefix("openrouter:")
    log.info("Using OpenRouter model %s", model_name)
    if model_name == "openrouter/free":
        log.warning(
            "OpenRouter's free router may select a model without tool-calling "
            "support. Set OPENROUTER_MODEL to a pinned tool-capable model for "
            "reliable agent behavior."
        )
    return cast(
        BaseChatModel,
        init_chat_model(
            f"openrouter:{model_name}",
            temperature=0,
            client=build_openrouter_client(),
        ),
    )

def build_database() -> SQLDatabase:
    """Open the Chinook database in read-only mode."""
    from langchain_community.utilities import SQLDatabase  # sunset package; see pyproject

    db_path = Path(__file__).resolve().parent / "resources" / "Chinook.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"Chinook database not found at {db_path}")
    engine = create_engine(f"sqlite:///file:{db_path}?mode=ro&uri=true")
    return SQLDatabase(engine)

def build_mcp_client() -> MultiServerMCPClient:
    """Build the MCP client with retry and transport timeouts."""
    return MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "streamable_http",
                "url": os.environ.get("TRAVEL_MCP_URL", "https://mcp.kiwi.com"),
                "timeout": 30,
                "sse_read_timeout": 300,
            }
        },
        tool_interceptors=[RetryMCPInterceptor()],
        handle_tool_errors=True,
    )

def build_web_search() -> BaseTool:
    """Build the Tavily web-search tool."""
    _require_env("TAVILY_API_KEY")
    return TavilySearch(
        name="web_search",
        description=(
            "Search the web for current venue information. Use plain-text queries "
            "without accented or special characters."
        ),
        max_results=5,
        search_depth="basic",
        include_raw_content=False,
        handle_tool_error=True,
    )

def build_sql_tools(db: SQLDatabase) -> list[BaseTool]:
    """Build narrow, read-only SQL tools for the playlist specialist."""

    @tool
    def list_tables() -> str:
        """List the tables available in the playlist database."""
        return ", ".join(db.get_usable_table_names())

    @tool
    def get_schema(table_names: str) -> str:
        """Return schema and sample rows for comma-separated table names."""
        allowed = set(db.get_usable_table_names())
        names = [name.strip() for name in table_names.split(",") if name.strip()]
        invalid = sorted(set(names) - allowed)
        if invalid:
            raise ToolException(
                f"Unknown table(s): {', '.join(invalid)}. "
                f"Available tables: {', '.join(sorted(allowed))}"
            )
        if not names:
            raise ToolException("Provide at least one table name.")
        return db.get_table_info(names)

    @tool
    def query_playlist_db(query: str) -> str:
        """Run a read-only SELECT query and return at most 200 rows as JSON."""
        normalized = query.strip().lower()
        if not normalized.startswith(("select", "with")):
            raise ToolException("Only SELECT queries are allowed.")

        try:
            result = db.run(query, fetch="cursor")
        except Exception as exc:
            raise ToolException(f"SQL error: {exc}") from exc

        try:
            rows = result.fetchmany(MAX_SQL_ROWS + 1)
        finally:
            result.close()

        truncated = len(rows) > MAX_SQL_ROWS
        rows = rows[:MAX_SQL_ROWS]
        if not rows:
            return "No rows returned."

        payload = [dict(row._mapping) for row in rows]
        text = json.dumps(payload, default=str, ensure_ascii=False)
        if truncated:
            text += f"\n\n[Truncated to {MAX_SQL_ROWS} rows.]"
        return text

    return [list_tables, get_schema, query_playlist_db]

def build_travel_agent(
    model: BaseChatModel,
    mcp_tools: list[BaseTool],
) -> CompiledStateGraph:
    return create_agent(
        model=model,
        tools=mcp_tools,
        middleware=[_retry_middleware()],
        system_prompt=TRAVEL_PROMPT,
        name="travel_agent",
    )

def build_venue_agent(model: BaseChatModel) -> CompiledStateGraph:
    return create_agent(
        model=model,
        tools=[build_web_search()],
        middleware=[
            _retry_middleware(),
            ToolCallLimitMiddleware(
                tool_name="web_search",
                run_limit=MAX_WEB_SEARCHES,
                exit_behavior="continue",
            ),
        ],
        system_prompt=VENUE_PROMPT,
        name="venue_agent",
    )

def build_playlist_agent(model: BaseChatModel, db: SQLDatabase) -> CompiledStateGraph:
    return create_agent(
        model=model,
        tools=build_sql_tools(db),
        middleware=[_retry_middleware()],
        system_prompt=PLAYLIST_PROMPT,
        name="playlist_agent",
    )

async def _invoke_specialist(
    agent: CompiledStateGraph,
    prompt: str,
    *,
    tag: str,
) -> str:
    try:
        async with asyncio.timeout(SPECIALIST_TIMEOUT_SECONDS):
            response = await agent.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={
                    "recursion_limit": SPECIALIST_RECURSION_LIMIT,
                    "tags": ["WP", tag],
                },
            )
    except TimeoutError as exc:
        raise ToolException(
            f"{tag} timed out after {SPECIALIST_TIMEOUT_SECONDS:.0f} seconds"
        ) from exc
    return message_text(response["messages"][-1])

@tool
def update_state(
    origin: Annotated[str, Field(min_length=1)],
    destination: Annotated[str, Field(min_length=1)],
    guest_count: Annotated[int, Field(ge=1, le=10_000)],
    genre: Annotated[str, Field(min_length=1)],
    runtime: ToolRuntime[WeddingContext, WeddingState],
) -> Command:
    """Update the state once origin, destination, guest_count, and genre are known.

    Call this tool alone, without any other tool calls. It must return before the
    specialists can read the updated state.
    """
    if runtime.tool_call_id is None:
        raise ToolException("Missing tool_call_id for state update.")
    return Command(
        update={
            "origin": origin,
            "destination": destination,
            "guest_count": guest_count,
            "genre": genre,
            "messages": [
                ToolMessage(
                    content="Successfully updated state",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

async def build_coordinator(
    model: BaseChatModel,
    mcp_client: MultiServerMCPClient,
    db: SQLDatabase,
) -> CompiledStateGraph:
    """Build the coordinator and its specialist agents."""
    mcp_tools = await mcp_client.get_tools()
    travel_agent = build_travel_agent(model, mcp_tools)
    venue_agent = build_venue_agent(model)
    playlist_agent = build_playlist_agent(model, db)

    @tool
    async def search_flights(
        runtime: ToolRuntime[WeddingContext, WeddingState],
    ) -> str:
        """Travel agent searches for flights to the destination wedding location."""
        origin = runtime.state.get("origin")
        destination = runtime.state.get("destination")
        if not origin or not destination:
            raise ToolException(
                "origin and destination must be set before searching flights."
            )
        return await _invoke_specialist(
            travel_agent,
            f"Find flights from {origin} to {destination}",
            tag="travel_agent",
        )

    @tool
    async def search_venues(runtime: ToolRuntime[WeddingContext, WeddingState]) -> str:
        """Venue agent chooses the best venue for the location and capacity."""
        destination = runtime.state.get("destination")
        capacity = runtime.state.get("guest_count")
        if not destination or capacity is None or capacity < 1:
            raise ToolException(
                "destination and a positive guest_count must be set before searching venues."
            )
        return await _invoke_specialist(
            venue_agent,
            f"Find wedding venues in {destination} for {capacity} guests",
            tag="venue_agent",
        )

    @tool
    async def suggest_playlist(
        runtime: ToolRuntime[WeddingContext, WeddingState],
    ) -> str:
        """Playlist agent curates a playlist for the requested genre."""
        genre = runtime.state.get("genre")
        if not genre:
            raise ToolException("genre must be set before curating a playlist.")
        return await _invoke_specialist(
            playlist_agent,
            f"Find {genre} tracks for a wedding playlist",
            tag="playlist_agent",
        )

    return create_agent(
        model=model,
        tools=[search_flights, search_venues, suggest_playlist, update_state],
        state_schema=WeddingState,
        context_schema=WeddingContext,
        checkpointer=InMemorySaver(),
        middleware=[_retry_middleware()],
        response_format=WeddingPlan,
        system_prompt=COORDINATOR_PROMPT,
        name="wedding_coordinator",
    )

async def main() -> int:
    configure_logging()
    try:
        model = build_model()
        db = build_database()
        mcp_client = build_mcp_client()
        coordinator = await build_coordinator(model, mcp_client, db)

        response = await coordinator.ainvoke(
            {"messages": [HumanMessage(content=DEFAULT_USER_REQUEST)]},
            config={
                "tags": ["WP"],
                "recursion_limit": COORDINATOR_RECURSION_LIMIT,
                "configurable": {
                    "thread_id": os.environ.get("WEDDING_THREAD_ID", "wedding-demo")
                },
            },
            context=WeddingContext(
                user_id=os.environ.get("WEDDING_USER_ID", "demo-user"),
            ),
        )
    except Exception as exc:  # noqa: BLE001 - surface a readable failure
        print(model_failure_message(exc), file=sys.stderr)
        log.debug("Wedding planner failed", exc_info=True)
        return 1

    plan = response.get("structured_response")
    if isinstance(plan, WeddingPlan):
        print(plan.model_dump_json(indent=2))
        return 0
    if isinstance(plan, dict):
        print(json.dumps(plan, indent=2, default=str))
        return 0

    messages = response.get("messages") or []
    text = message_text(messages[-1]).strip() if messages else ""
    if text:
        print(text)
        return 0

    print(
        "The coordinator returned neither a structured plan nor any text. "
        "Re-run with LOG_LEVEL=DEBUG for details.",
        file=sys.stderr,
    )
    return 1

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
