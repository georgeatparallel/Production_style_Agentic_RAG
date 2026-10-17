"""Exercise MCP requests and the real app graph without loading remote models."""

import asyncio
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from parallel_search import MCP_URL, USER_AGENT, ParallelSearchRun


class MCPFixture:
    def __init__(self, result=None, status=200, stall=False):
        self.result = result if result is not None else {
            "content": [{"type": "text", "text": "Python release: https://www.python.org/downloads/"}]
        }
        self.status = status
        self.stall = stall
        self.requests = []
        self.started = asyncio.Event()
        self.cancelled = False

    async def handle(self, request):
        self.requests.append(request)
        message = json.loads(request.content)
        if message["method"] == "notifications/initialized":
            return httpx.Response(202)
        if message["method"] == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif message["method"] == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        elif message["method"] == "tools/call":
            if self.stall:
                self.started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled = True
            return httpx.Response(self.status, json={"jsonrpc": "2.0", "id": message["id"],
                                                     "result": self.result})
        else:
            raise AssertionError(message)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    @contextmanager
    def installed(self):
        client = httpx.AsyncClient

        def factory(**kwargs):
            return client(transport=httpx.MockTransport(self.handle), **kwargs)

        with patch("parallel_search.httpx.AsyncClient", side_effect=factory):
            yield self


@contextmanager
def app_with_provider(provider=None):
    """Only replace unrelated PDF/model services; use actual FastAPI and LangGraph."""
    class DummyModel:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            return SimpleNamespace(content=messages[0].content)

    class DummyDuckDuckGo:
        queries = []

        def run(self, query):
            self.queries.append(query)
            return "DuckDuckGo fixture context"

    replacements = {
        "langchain_community.document_loaders": {"PyPDFLoader": DummyModel},
        "langchain_text_splitters": {"RecursiveCharacterTextSplitter": DummyModel},
        "langchain_huggingface": {name: DummyModel for name in
                                  ("HuggingFaceEmbeddings", "HuggingFaceEndpoint", "ChatHuggingFace")},
        "langchain_community.vectorstores": {"FAISS": DummyModel},
        "langchain_community.tools": {"DuckDuckGoSearchRun": DummyDuckDuckGo},
    }
    modules = {}
    for name, attributes in replacements.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        modules[name] = module
    environment = {} if provider is None else {"WEB_SEARCH_PROVIDER": provider}
    # Reload the actual config to check env/default selection without saved credentials.
    with patch.dict(os.environ, environment, clear=True), patch.dict(sys.modules, modules):
        previous = sys.modules.pop("config", None)
        try:
            spec = importlib.util.spec_from_file_location("rag_test_app", Path(__file__).parents[1] / "main.py")
            app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app)
            yield app
        finally:
            sys.modules.pop("config", None)
            if previous is not None:
                sys.modules["config"] = previous


class ParallelTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_transport_and_graph_consume_search(self):
        fixture = MCPFixture()
        with fixture.installed(), app_with_provider("parallel") as app:
            # This async endpoint invokes the synchronous graph in a running event loop.
            response = await app.ask_question(app.QueryRequest(
                question="What is the latest Python release?", thread_id="fixture"))
        self.assertIn("https://www.python.org/downloads/", response["answer"])
        self.assertEqual(response["sources"], ["Parallel"])
        self.assertEqual(response["logic_used"], "generate")
        self.assertGreaterEqual(len(fixture.requests), 3)
        for request in fixture.requests:
            self.assertEqual(str(request.url), MCP_URL)
            self.assertEqual(request.headers["user-agent"], USER_AGENT)
            self.assertNotIn("authorization", request.headers)
        calls = [json.loads(r.content) for r in fixture.requests
                 if json.loads(r.content)["method"] == "tools/call"]
        self.assertEqual(calls[0]["params"], {
            "name": "web_search", "arguments": {
                "objective": "What is the latest Python release?",
                "search_queries": ["What is the latest Python release?"]}})

    async def test_default_does_not_import_parallel(self):
        import builtins
        original = builtins.__import__

        def no_parallel(name, *args, **kwargs):
            if name == "parallel_search" or name == "mcp" or name.startswith("mcp."):
                raise AssertionError("Default path imported optional MCP dependency")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_parallel), app_with_provider() as app:
            response = await app.ask_question(app.QueryRequest(question="Python news", thread_id="default"))
            self.assertEqual(app.search_tool.queries, ["Python news current price March 2026"])
            self.assertEqual(response["sources"], ["DuckDuckGo"])
            self.assertIn("DuckDuckGo fixture context", response["answer"])

    async def test_explicit_duckduckgo_and_pdf_routing(self):
        with app_with_provider("duckduckgo") as app:
            self.assertEqual(app.search_source, "DuckDuckGo")
            self.assertEqual(app.supervisor({"question": "Summarize the uploaded PDF"}),
                             {"next_step": "vector_store"})

    async def test_invalid_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "WEB_SEARCH_PROVIDER"):
            with app_with_provider("invalid"):
                pass

    async def test_tool_and_http_errors_do_not_become_context(self):
        fixtures = [MCPFixture({"isError": True, "content": [{"type": "text", "text": "quota exceeded"}]}),
                    MCPFixture(status=503), MCPFixture({"content": []})]
        for fixture in fixtures:
            with self.subTest(status=fixture.status, result=fixture.result):
                with fixture.installed(), app_with_provider("parallel") as app:
                    response = await app.ask_question(app.QueryRequest(question="Python news", thread_id="error"))
                self.assertEqual(response["sources"], [])
                self.assertIn("Search currently unavailable.", response["answer"])
                self.assertNotIn("quota exceeded", response["answer"])

    async def test_whole_request_timeout_and_cleanup(self):
        fixture = MCPFixture(stall=True)
        with fixture.installed():
            with self.assertRaises(TimeoutError):
                await ParallelSearchRun(timeout=0.1)._search("Python release")
        self.assertTrue(fixture.cancelled)

    async def test_timeout_reaches_existing_unavailable_path(self):
        fixture = MCPFixture(stall=True)
        with fixture.installed(), app_with_provider("parallel") as app:
            app.search_tool.timeout = 0.1
            response = await app.ask_question(app.QueryRequest(question="Python news", thread_id="timeout"))
        self.assertTrue(fixture.cancelled)
        self.assertEqual(response["sources"], [])
        self.assertIn("Search currently unavailable.", response["answer"])

    async def test_cancellation_propagates_and_closes_transport(self):
        fixture = MCPFixture(stall=True)
        with fixture.installed():
            task = asyncio.create_task(ParallelSearchRun()._search("Python release"))
            await asyncio.wait_for(fixture.started.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(fixture.cancelled)


if __name__ == "__main__":
    unittest.main()
