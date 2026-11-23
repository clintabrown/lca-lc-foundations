import os

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

# Free OpenRouter model that supports tool calling. Some other :free models do
# not support tools, which would leave the agent unable to call web_search.
# Override with the OPENROUTER_MODEL environment variable if needed.
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3.5-lightning:free"


@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)


system_prompt = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

model_name = os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
model_name = model_name.removeprefix("openrouter:")

agent = create_agent(
    model=init_chat_model(
        f"openrouter:{model_name}",
        temperature=0,
    ),
    tools=[web_search],
    system_prompt=system_prompt,
)
