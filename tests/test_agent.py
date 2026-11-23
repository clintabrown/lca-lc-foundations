"""Focused unit tests for the wedding planner example.

Run with:

    python -m unittest discover -s tests

These tests do not call OpenRouter, Tavily, or the remote MCP server.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from langchain.messages import AIMessage
from langchain.tools import ToolException, ToolRuntime
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langgraph.types import Command
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ErrorData, TextContent

MODULE_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "module-2"
AGENT_PATH = next(
    path
    for path in (
        MODULE_DIR / "wedding_planner.py"
    )
    if path.is_file()
)
SPEC = importlib.util.spec_from_file_location("wedding_agent", AGENT_PATH)
assert SPEC is not None and SPEC.loader is not None
agent = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = agent
SPEC.loader.exec_module(agent)


class MessageTextTests(unittest.TestCase):
    def test_string_content(self) -> None:
        message = agent.HumanMessage(content="hello")
        self.assertEqual(agent.message_text(message), "hello")

    def test_content_blocks(self) -> None:
        message = AIMessage(
            content=[
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ]
        )
        self.assertEqual(agent.message_text(message), "first\nsecond")


class SQLToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = agent.build_database()
        cls.tools = {tool.name: tool for tool in agent.build_sql_tools(cls.db)}

    def test_lists_tables(self) -> None:
        tables = self.tools["list_tables"].invoke({})
        self.assertIn("Artist", tables)
        self.assertIn("Track", tables)

    def test_schema_rejects_unknown_table(self) -> None:
        with self.assertRaises(ToolException):
            self.tools["get_schema"].invoke({"table_names": "NotATable"})

    def test_query_returns_json(self) -> None:
        result = self.tools["query_playlist_db"].invoke(
            {"query": "select ArtistId, Name from Artist limit 2"}
        )
        self.assertIn('"Name": "AC/DC"', result)

    def test_query_rejects_writes(self) -> None:
        with self.assertRaises(ToolException):
            self.tools["query_playlist_db"].invoke({"query": "delete from Artist"})


class StateToolTests(unittest.TestCase):
    def test_update_state_returns_command(self) -> None:
        runtime = ToolRuntime(
            state={"messages": []},
            context=agent.WeddingContext(),
            config={},
            stream_writer=lambda _: None,
            tool_call_id="call_1",
            store=None,
        )
        result = agent.update_state.func(
            origin="London",
            destination="Paris",
            guest_count=100,
            genre="jazz",
            runtime=runtime,
        )
        self.assertIsInstance(result, Command)
        self.assertEqual(result.update["guest_count"], 100)
        self.assertEqual(result.update["messages"][0].tool_call_id, "call_1")


class MCPInterceptorTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_transient_failures(self) -> None:
        interceptor = agent.RetryMCPInterceptor(
            max_attempts=3,
            base_delay=0.001,
            timeout_seconds=1,
        )
        request = MCPToolCallRequest(name="test_tool", args={}, server_name="test")
        attempts = 0

        async def handler(_: MCPToolCallRequest) -> CallToolResult:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("temporary")
            return CallToolResult(
                content=[TextContent(type="text", text="ok")],
                isError=False,
            )

        result = await interceptor(request, handler)
        self.assertEqual(attempts, 3)
        self.assertFalse(result.isError)

    async def test_permanent_mcp_error_is_surfaced(self) -> None:
        interceptor = agent.RetryMCPInterceptor(
            max_attempts=3,
            base_delay=0.001,
            timeout_seconds=1,
        )
        request = MCPToolCallRequest(name="test_tool", args={}, server_name="test")

        async def handler(_: MCPToolCallRequest) -> CallToolResult:
            raise McpError(ErrorData(code=-32602, message="bad arguments"))

        result = await interceptor(request, handler)
        self.assertTrue(result.isError)
        self.assertIn("bad arguments", result.content[0].text)


class ModelFailureTests(unittest.TestCase):
    def test_quota_errors_are_not_retried(self) -> None:
        error = RuntimeError("429 Rate limit exceeded: free-models-per-day")
        self.assertFalse(agent._should_retry_model_error(error))

    def test_transient_errors_are_retried(self) -> None:
        self.assertTrue(agent._should_retry_model_error(TimeoutError("timed out")))

    def test_quota_message_is_actionable(self) -> None:
        message = agent.model_failure_message(
            RuntimeError("429 free-models-per-day")
        )
        self.assertIn("free-model daily limit", message)
        self.assertIn("00:00 UTC", message)


if __name__ == "__main__":
    unittest.main()
