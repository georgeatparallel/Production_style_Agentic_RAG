"""Optional anonymous Parallel Search MCP adapter for the web-search node."""

import anyio
import httpx
from anyio.from_thread import start_blocking_portal
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "Production_style_Agentic_RAG/ParallelSearch"


class ParallelSearchRun:
    """Expose the same synchronous run(query) interface as DuckDuckGoSearchRun."""

    def __init__(self, timeout=30):
        self.timeout = timeout

    def run(self, query):
        # The graph is also invoked from an async FastAPI endpoint. A separate
        # event loop lets the synchronous node use MCP in either calling context.
        with start_blocking_portal() as portal:
            return portal.call(self._search, query)

    async def _search(self, query):
        # Bound the entire handshake and search, not just individual HTTP reads.
        with anyio.fail_after(self.timeout):
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT}, timeout=self.timeout
            ) as client:
                async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(
                            "web_search",
                            {"objective": query, "search_queries": [query]},
                        )
        if result.isError:
            raise RuntimeError("Parallel search returned an error.")
        text = "\n".join(block.text for block in result.content if block.type == "text")
        if not text.strip():
            raise RuntimeError("Parallel search returned no text.")
        return text
