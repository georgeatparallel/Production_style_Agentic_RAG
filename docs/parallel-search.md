# Optional Parallel web search

Use [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
to supply web context for questions routed to `web_search`. Anonymous search is
free, uses Fast mode, and needs no Parallel account or API key. The free service
has rate limits and is intended for exploration and light use.

After completing the application's existing installation and Hugging Face token
setup, install the optional provider from the repository root:

```bash
python -m pip install -r requirements-parallel.txt
```

Set this in `.env` or your deployment's environment, then restart the app:

```dotenv
WEB_SEARCH_PROVIDER=parallel
```

Start the app as usual with `python main.py`. For Docker, install the optional
requirements in your image before starting the app; the default image only
installs `requirements.txt`.

Submit a web question through the existing API:

```bash
curl -X POST http://127.0.0.1:8000/ask/ \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the latest Python release?","thread_id":"demo"}'
```

The web node sends the original question to `web_search` over Streamable HTTP at
`https://search.parallel.ai/mcp`, then passes the returned text to the existing
answer generator. The response's `sources` field contains the provider label
`["Parallel"]`; any page links are part of the search context, not separate entries
in `sources`. This provider uses search only, not the MCP server's `web_fetch` tool.
PDF routing and the Hugging Face model configuration are unchanged.

The handshake and search have a 30-second timeout. If the service returns an
error, times out, or returns no text, the graph receives the existing
`Search currently unavailable.` context with empty sources. It does not silently
switch providers. An unsupported `WEB_SEARCH_PROVIDER` value fails at startup.

To return to DuckDuckGo, remove `WEB_SEARCH_PROVIDER` or set it to `duckduckgo`
and restart. Leaving the variable unset does not load the MCP dependency.

## Provider checks

The deterministic tests use the real MCP HTTP transport with local HTTP fixtures.
They do not download embedding models or call Hugging Face:

```bash
python -m unittest discover -s tests -v
```
