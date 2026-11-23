# 🔗 LangChain Academy - Introduction to LangChain

My notes and completed projects from the LangChain Academy
**Introduction to LangChain** course.

## Completed modules

### Module 1 – Foundations

[`notebooks/module-1/personal_chef.py`](notebooks/module-1/personal_chef.py)

A personal chef agent built with `create_agent`. Given the leftovers in your
kitchen, it uses a Tavily-powered `web_search` tool to find recipes and can walk
you through the instructions on request.

### Module 2 – Memory, State, and Multi-agent Systems

[`notebooks/module-2/wedding_planner.py`](notebooks/module-2/wedding_planner.py)

A multi-agent wedding planner. A coordinator collects the origin, destination,
guest count, and music genre, then delegates to three specialist agents:

- **Travel agent** – finds the best flights using MCP-provided tools
- **Venue agent** – searches for venues that match the location and capacity
- **Playlist agent** – builds a playlist from the read-only Chinook SQLite database

The example demonstrates runtime state and context, MCP tool integration with a
retry interceptor, middleware, structured responses, and a checkpointer for
conversation memory. The supporting MCP server lives in
[`notebooks/module-2/resources/2.1_mcp_server.py`](notebooks/module-2/resources/2.1_mcp_server.py).

### Module 3 – Human-in-the-loop and Dynamic Agents

[`notebooks/module-3/email_agent.py`](notebooks/module-3/email_agent.py)

An email agent that keeps users
unauthenticated until they provide valid credentials, then swaps in the inbox
and email-sending tools and prompts. Sending an email requires human approval.
