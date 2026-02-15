#!/usr/bin/env python3
"""
Academic Search MCP Server
==========================
MCP server that exposes academic paper search APIs
(Semantic Scholar, OpenAlex, CrossRef, arXiv) as tools for
Claude Desktop / Cowork.

Supports two transports:
  - stdio   (local):   python server.py
  - HTTP    (remote):  MCP_TRANSPORT=streamable_http python server.py

Set MCP_TRANSPORT=streamable_http for cloud deployment (Render, Railway, etc.)
"""

import asyncio
import json
import os
import sys
import time
from typing import Optional, List, Dict, Any
from enum import Enum
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.requests import Request

from pydantic import BaseModel, Field, field_validator, ConfigDict
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Import the existing academic_search module
# The user places academic_search.py in the same directory, OR sets
# ACADEMIC_SEARCH_PATH to the directory containing it.
# ---------------------------------------------------------------------------
_search_dir = os.environ.get(
    "ACADEMIC_SEARCH_PATH",
    os.path.dirname(os.path.abspath(__file__)),
)
if _search_dir not in sys.path:
    sys.path.insert(0, _search_dir)

from academic_search import (                       # noqa: E402
    search_all,
    search_semantic_scholar,
    search_openalex,
    search_crossref,
    search_arxiv,
    snowball,
    deduplicate_papers,
    papers_to_csv,
    papers_to_bibtex_file,
    papers_to_markdown_table,
)

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP("academic_search_mcp")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summary(papers: List[Dict], top_n: int = 20) -> str:
    """Return a concise markdown summary of search results."""
    lines = [f"**{len(papers)} unique papers found.**\n"]
    for i, p in enumerate(papers[:top_n], 1):
        authors = p.get("authors", [])
        first = authors[0] if authors else "Unknown"
        if len(authors) > 1:
            first += " et al."
        cites = p.get("citation_count") or "n/a"
        title = (p.get("title") or "")[:120]
        year = p.get("year", "?")
        venue = (p.get("venue") or "")[:50]
        doi = p.get("doi", "")
        lines.append(
            f"{i}. **[{cites}]** {first} ({year}) – {title}  \n"
            f"   _Venue:_ {venue}  |  _DOI:_ {doi}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------

class SearchAllInput(BaseModel):
    """Multi-source academic search across Semantic Scholar, OpenAlex, CrossRef, and arXiv."""
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(
        ...,
        description="Search query, e.g. 'generative AI enterprise implementation'",
        min_length=2,
        max_length=500,
    )
    max_results_per_source: int = Field(
        default=15,
        description="Maximum results per database (default 15, max 50)",
        ge=1, le=50,
    )
    year_from: Optional[int] = Field(
        default=None,
        description="Only return papers published from this year onwards, e.g. 2020",
    )
    year_to: Optional[int] = Field(
        default=None,
        description="Only return papers published until this year",
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description=(
            "Which databases to search. Options: 'semantic_scholar', 'openalex', "
            "'crossref', 'arxiv'. Default: all four."
        ),
    )


class SemanticScholarInput(BaseModel):
    """Search Semantic Scholar (200M+ papers)."""
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., description="Search query", min_length=2, max_length=500)
    max_results: int = Field(default=25, ge=1, le=100)
    year_range: Optional[str] = Field(
        default=None,
        description="Year filter, e.g. '2020-2025' or '2023-'",
    )
    fields_of_study: Optional[List[str]] = Field(
        default=None,
        description="e.g. ['Computer Science', 'Business']",
    )
    min_citations: int = Field(default=0, ge=0)


class OpenAlexInput(BaseModel):
    """Search OpenAlex (474M+ works, fully open, CC0)."""
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., description="Search query", min_length=2, max_length=500)
    max_results: int = Field(default=25, ge=1, le=200)
    year_from: Optional[int] = Field(default=None)
    year_to: Optional[int] = Field(default=None)
    min_citations: int = Field(default=0, ge=0)
    open_access_only: bool = Field(default=False)


class CrossRefInput(BaseModel):
    """Search CrossRef (150M+ works, best for DOIs and journal metadata)."""
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., description="Search query", min_length=2, max_length=500)
    max_results: int = Field(default=25, ge=1, le=100)
    year_from: Optional[int] = Field(default=None)
    year_to: Optional[int] = Field(default=None)
    container_title: Optional[str] = Field(
        default=None,
        description="Filter by journal name, e.g. 'MIS Quarterly'",
    )


class ArxivInput(BaseModel):
    """Search arXiv (2.4M+ preprints, CS/AI/ML focused)."""
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., description="Search query", min_length=2, max_length=500)
    max_results: int = Field(default=25, ge=1, le=100)
    sort_by: str = Field(
        default="relevance",
        description="Sort order: 'relevance', 'lastUpdatedDate', or 'submittedDate'",
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="arXiv categories, e.g. ['cs.AI', 'cs.HC', 'cs.CY']",
    )
    start_year: Optional[int] = Field(default=None)


class SnowballInput(BaseModel):
    """Forward/backward citation snowballing via Semantic Scholar."""
    model_config = ConfigDict(str_strip_whitespace=True)

    paper_id: str = Field(
        ...,
        description="DOI (e.g. '10.1234/example') or Semantic Scholar paper ID",
        min_length=3,
    )
    direction: str = Field(
        default="both",
        description="'forward' (who cites this), 'backward' (references), or 'both'",
    )
    limit: int = Field(default=20, ge=1, le=100)


class ExportInput(BaseModel):
    """Export previously returned papers to BibTeX or CSV."""
    model_config = ConfigDict(str_strip_whitespace=True)

    papers_json: str = Field(
        ...,
        description="JSON string of papers array (as returned by search tools)",
    )
    format: str = Field(
        default="bibtex",
        description="'bibtex' or 'csv'",
    )
    filename: Optional[str] = Field(
        default=None,
        description="Output filename (without path). Saved to current directory.",
    )


class MultiQueryInput(BaseModel):
    """Run multiple queries at once and deduplicate. Ideal for SLR search strategies."""
    model_config = ConfigDict(str_strip_whitespace=True)

    queries: List[str] = Field(
        ...,
        description="List of search queries to run",
        min_length=1,
        max_length=10,
    )
    max_results_per_source: int = Field(default=15, ge=1, le=50)
    year_from: Optional[int] = Field(default=None)
    year_to: Optional[int] = Field(default=None)
    sources: Optional[List[str]] = Field(default=None)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

@mcp.tool(
    name="academic_search_all",
    annotations={
        "title": "Search All Academic Databases",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def academic_search_all(params: SearchAllInput) -> str:
    """Search across Semantic Scholar, OpenAlex, CrossRef, and arXiv simultaneously.

    Deduplicates results by DOI and title similarity, then sorts by citation count.
    Returns a markdown summary with the top papers and a JSON payload for further use.

    Args:
        params (SearchAllInput): Validated search parameters.

    Returns:
        str: Markdown summary + JSON array of papers.
    """
    papers = await asyncio.to_thread(
        search_all,
        params.query,
        params.max_results_per_source,
        params.year_from,
        params.year_to,
        params.sources,
        True,  # deduplicate
    )
    summary = _summary(papers)
    payload = json.dumps(
        [_slim(p) for p in papers],
        ensure_ascii=False, indent=2,
    )
    return f"{summary}\n\n<details><summary>Full JSON ({len(papers)} papers)</summary>\n\n```json\n{payload}\n```\n</details>"


@mcp.tool(
    name="academic_search_semantic_scholar",
    annotations={
        "title": "Search Semantic Scholar",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def academic_ss(params: SemanticScholarInput) -> str:
    """Search Semantic Scholar (200M+ papers). Returns citation counts, TLDRs, and open-access links.

    Args:
        params (SemanticScholarInput): Search parameters.

    Returns:
        str: Markdown summary + JSON array.
    """
    papers = await asyncio.to_thread(
        search_semantic_scholar,
        params.query,
        params.max_results,
        params.year_range,
        params.fields_of_study,
        params.min_citations,
    )
    return _format_results(papers)


@mcp.tool(
    name="academic_search_openalex",
    annotations={
        "title": "Search OpenAlex",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def academic_oa(params: OpenAlexInput) -> str:
    """Search OpenAlex (474M+ works, broadest coverage, CC0 data). Good for German-language papers.

    Args:
        params (OpenAlexInput): Search parameters.

    Returns:
        str: Markdown summary + JSON array.
    """
    mailto = os.environ.get("OPENALEX_MAILTO", "research@example.com")
    papers = await asyncio.to_thread(
        search_openalex,
        params.query,
        params.max_results,
        params.year_from,
        params.year_to,
        None,  # source_type
        params.min_citations,
        params.open_access_only,
        mailto,
    )
    return _format_results(papers)


@mcp.tool(
    name="academic_search_crossref",
    annotations={
        "title": "Search CrossRef",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def academic_cr(params: CrossRefInput) -> str:
    """Search CrossRef (150M+ works). Best for DOI resolution and journal metadata.

    Args:
        params (CrossRefInput): Search parameters.

    Returns:
        str: Markdown summary + JSON array.
    """
    mailto = os.environ.get("CROSSREF_MAILTO", "research@example.com")
    papers = await asyncio.to_thread(
        search_crossref,
        params.query,
        params.max_results,
        params.year_from,
        params.year_to,
        params.container_title,
        mailto,
    )
    return _format_results(papers)


@mcp.tool(
    name="academic_search_arxiv",
    annotations={
        "title": "Search arXiv",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def academic_ax(params: ArxivInput) -> str:
    """Search arXiv preprint server (2.4M+ papers). Best for cutting-edge CS/AI/ML research.

    Args:
        params (ArxivInput): Search parameters.

    Returns:
        str: Markdown summary + JSON array.
    """
    papers = await asyncio.to_thread(
        search_arxiv,
        params.query,
        params.max_results,
        params.sort_by,
        params.categories,
        params.start_year,
    )
    return _format_results(papers)


@mcp.tool(
    name="academic_snowball",
    annotations={
        "title": "Citation Snowballing",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def academic_snowball(params: SnowballInput) -> str:
    """Forward and/or backward citation snowballing via Semantic Scholar.

    Forward: finds papers that cite the given paper.
    Backward: finds papers referenced by the given paper.

    Args:
        params (SnowballInput): Paper ID, direction, and limit.

    Returns:
        str: Markdown summary of citing/referenced papers + JSON.
    """
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    results = await asyncio.to_thread(
        snowball,
        params.paper_id,
        params.direction,
        params.limit,
        api_key,
    )
    parts = []
    all_papers = []
    for direction, papers in results.items():
        parts.append(f"### {direction.title()} snowballing: {len(papers)} papers\n")
        parts.append(_summary(papers, top_n=15))
        all_papers.extend(papers)

    payload = json.dumps([_slim(p) for p in all_papers], ensure_ascii=False, indent=2)
    parts.append(
        f"\n<details><summary>Full JSON ({len(all_papers)} papers)</summary>\n\n"
        f"```json\n{payload}\n```\n</details>"
    )
    return "\n\n".join(parts)


@mcp.tool(
    name="academic_multi_query",
    annotations={
        "title": "Multi-Query Search (SLR Strategy)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def academic_multi_query(params: MultiQueryInput) -> str:
    """Run multiple search queries across all databases and deduplicate.

    Ideal for systematic literature review search strategies where you need
    to combine results from several query formulations.

    Args:
        params (MultiQueryInput): List of queries + shared filters.

    Returns:
        str: Deduplicated results as markdown + JSON.
    """
    all_papers: List[Dict] = []
    query_stats: List[str] = []

    for q in params.queries:
        papers = await asyncio.to_thread(
            search_all,
            q,
            params.max_results_per_source,
            params.year_from,
            params.year_to,
            params.sources,
            False,  # don't deduplicate per query
        )
        query_stats.append(f"- `{q}` → {len(papers)} results")
        all_papers.extend(papers)
        await asyncio.sleep(1)  # rate-limit courtesy

    deduped = deduplicate_papers(all_papers)
    deduped.sort(key=lambda p: -(p.get("citation_count") or 0))

    header = (
        f"## Multi-Query Search Results\n\n"
        f"**{len(params.queries)} queries → {len(all_papers)} raw → "
        f"{len(deduped)} unique papers**\n\n"
        f"### Query breakdown\n" + "\n".join(query_stats) + "\n"
    )
    summary = _summary(deduped, top_n=25)
    payload = json.dumps([_slim(p) for p in deduped], ensure_ascii=False, indent=2)

    return (
        f"{header}\n{summary}\n\n"
        f"<details><summary>Full JSON ({len(deduped)} papers)</summary>\n\n"
        f"```json\n{payload}\n```\n</details>"
    )


@mcp.tool(
    name="academic_export",
    annotations={
        "title": "Export Papers (BibTeX / CSV)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def academic_export(params: ExportInput) -> str:
    """Export a list of papers to BibTeX (.bib) or CSV format.

    Pass the JSON array from a previous search result. Returns the file content
    as a string, and optionally saves to disk.

    Args:
        params (ExportInput): Papers JSON, format, optional filename.

    Returns:
        str: The exported content (BibTeX entries or CSV text).
    """
    try:
        papers = json.loads(params.papers_json)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON – {e}"

    if not isinstance(papers, list):
        return "Error: Expected a JSON array of paper objects."

    if params.format == "bibtex":
        from academic_search import to_bibtex
        entries = [to_bibtex(p) for p in papers]
        content = "\n\n".join(entries)
        ext = ".bib"
    elif params.format == "csv":
        import csv
        import io
        buf = io.StringIO()
        fieldnames = [
            "title", "authors", "year", "venue", "citation_count",
            "doi", "url", "abstract",
        ]
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for p in papers:
            row = dict(p)
            row["authors"] = "; ".join(p.get("authors", []))
            writer.writerow(row)
        content = buf.getvalue()
        ext = ".csv"
    else:
        return f"Error: Unknown format '{params.format}'. Use 'bibtex' or 'csv'."

    # Optionally save to file
    if params.filename:
        fname = params.filename if params.filename.endswith(ext) else params.filename + ext
        with open(fname, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Saved {len(papers)} papers to `{fname}`.\n\n```\n{content[:3000]}\n```"

    return f"```\n{content}\n```"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _slim(p: Dict) -> Dict:
    """Return a slimmed-down paper dict (drop very long abstracts for token efficiency)."""
    return {
        "title": p.get("title", ""),
        "authors": p.get("authors", [])[:5],
        "year": p.get("year"),
        "venue": p.get("venue", ""),
        "citation_count": p.get("citation_count"),
        "doi": p.get("doi", ""),
        "url": p.get("url", ""),
        "abstract": (p.get("abstract") or "")[:400],
        "source": p.get("source", ""),
    }


def _format_results(papers: List[Dict]) -> str:
    """Standard formatting for single-source search results."""
    summary = _summary(papers)
    payload = json.dumps([_slim(p) for p in papers], ensure_ascii=False, indent=2)
    return (
        f"{summary}\n\n"
        f"<details><summary>Full JSON ({len(papers)} papers)</summary>\n\n"
        f"```json\n{payload}\n```\n</details>"
    )


# ---------------------------------------------------------------------------
# Health check (for Render / cloud deployments)
# ---------------------------------------------------------------------------

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Health check endpoint for Render/cloud monitoring."""
    return JSONResponse({"status": "ok", "tools": len(mcp._tool_manager._tools)})


# ---------------------------------------------------------------------------
# Simple Bearer Token Auth Middleware
# ---------------------------------------------------------------------------
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Always allow health check
        if request.url.path == "/health":
            return await call_next(request)

        # Check for MCP_API_KEY
        expected_key = os.environ.get("MCP_API_KEY")
        if not expected_key:
            # If no key is configured, allow all (public but functional)
            return await call_next(request)

        # 1. Try Authorization header
        token = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "").strip()
        
        # 2. Try query parameter (for clients that don't support custom headers in UI)
        if not token:
            token = request.query_params.get("api_key")

        if not token or token != expected_key:
            return JSONResponse(
                {
                    "error": "Unauthorized: Missing or invalid API key. "
                             "Use 'Authorization: Bearer <key>' header or '?api_key=<key>' query parameter."
                },
                status_code=401
            )

        return await call_next(request)

# Add middleware to the streamable_http_app
app = mcp.streamable_http_app()
app.add_middleware(AuthMiddleware)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    if transport == "streamable_http":
        import uvicorn
        port = int(os.environ.get("PORT", "8000"))
        print(f"Starting Academic Search MCP (HTTP) on port {port}...", file=sys.stderr)
        # Use uvicorn to run the internal app directly, which allows middleware to work correctly
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        mcp.run()  # stdio for local use
