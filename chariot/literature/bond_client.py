from __future__ import annotations

import asyncio
import json
import os


class BondClient:
    """Optional read-only standalone Valency Bond MCP client."""

    def __init__(self, url: str | None = None, bearer_token: str | None = None):
        self.url = url or os.getenv("BOND_MCP_URL", "https://mcp.valency.io")
        self.bearer_token = bearer_token or os.getenv("BOND_BEARER_TOKEN", "")

    @property
    def configured(self) -> bool:
        return bool(self.bearer_token)

    async def _search_async(self, query: str, limit: int = 6) -> dict:
        if not self.bearer_token:
            raise RuntimeError(
                "Standalone Bond is not authenticated. Add BOND_BEARER_TOKEN to .env."
            )

        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        async with httpx2.AsyncClient(headers=headers, timeout=30.0) as http_client:
            async with streamable_http_client(self.url, http_client=http_client) as streams:
                read_stream, write_stream = streams[0], streams[1]
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    names = [t.name for t in listed.tools]
                    preferred = next((n for n in names if "semantic_search_papers" in n), None)
                    if preferred is None:
                        preferred = next((n for n in names if "search" in n.lower() and "paper" in n.lower()), None)
                    if preferred is None:
                        raise RuntimeError(f"No paper-search tool found. Available Bond tools: {names}")

                    result = await session.call_tool(preferred, {
                        "query": query,
                        "limit": limit,
                        "include_abstract": True,
                        "sort_by": "relevance",
                    })
                    pieces = [getattr(block, "text", "") for block in getattr(result, "content", [])]
                    raw_text = "\n".join(p for p in pieces if p).strip()
                    if not raw_text:
                        return {"papers": [], "warnings": ["Bond returned no text payload."]}
                    try:
                        parsed = json.loads(raw_text)
                        return parsed if isinstance(parsed, dict) else {"papers": [], "raw": parsed}
                    except json.JSONDecodeError:
                        return {"papers": [], "raw_text": raw_text, "warnings": ["Bond payload was not valid JSON."]}

    def search(self, query: str, limit: int = 6) -> dict:
        return asyncio.run(self._search_async(query, limit))
