#!/usr/bin/env python3
"""
Academic Paper Search Tool
Searches multiple academic APIs: arXiv, Semantic Scholar, OpenAlex, CrossRef
For use with Claude Cowork Academic Research Plugin

All APIs are free and require no authentication for basic use.
"""

import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET
import csv
import time
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional


# ============================================================================
# arXiv API
# Docs: https://info.arxiv.org/help/api/user-manual.html
# No API key required. Rate limit: 1 request per 3 seconds.
# ============================================================================

def search_arxiv(query: str, max_results: int = 25, 
                 sort_by: str = "relevance",
                 categories: Optional[List[str]] = None,
                 start_year: Optional[int] = None) -> List[Dict]:
    """
    Search arXiv for papers.
    
    Args:
        query: Search query string
        max_results: Number of results (max 2000 per request)
        sort_by: "relevance", "lastUpdatedDate", or "submittedDate"
        categories: Filter by arXiv categories, e.g. ["cs.AI", "cs.HC", "econ.GN"]
        start_year: Only return papers from this year onwards
    
    Returns:
        List of paper dictionaries
    
    Relevant arXiv categories for WI/BWL:
        cs.AI  - Artificial Intelligence
        cs.HC  - Human-Computer Interaction
        cs.CY  - Computers and Society
        cs.IR  - Information Retrieval
        cs.SE  - Software Engineering
        cs.LG  - Machine Learning
        econ.GN - General Economics
        econ.EM - Econometrics
        stat.AP - Applications (Statistics)
        stat.ML - Machine Learning (Statistics)
        q-fin   - Quantitative Finance
    """
    base_url = "http://export.arxiv.org/api/query?"
    
    # Build search query
    search_parts = [f'all:"{query}"']
    if categories:
        cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
        search_parts.append(f"({cat_query})")
    
    search_query = " AND ".join(search_parts)
    
    sort_map = {
        "relevance": "relevance",
        "lastUpdatedDate": "lastUpdatedDate",
        "submittedDate": "submittedDate"
    }
    
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_map.get(sort_by, "relevance"),
        "sortOrder": "descending"
    }
    
    url = base_url + urllib.parse.urlencode(params)
    print(f"[arXiv] Searching: {query} (max {max_results} results)...")
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AcademicResearchPlugin/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            xml_data = response.read().decode("utf-8")
    except Exception as e:
        print(f"[arXiv] Error: {e}")
        return []
    
    # Parse Atom XML
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }
    
    root = ET.fromstring(xml_data)
    papers = []
    
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)
        updated_el = entry.find("atom:updated", ns)
        id_el = entry.find("atom:id", ns)
        
        # Extract authors
        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns)
            if name is not None:
                authors.append(name.text.strip())
        
        # Extract categories
        cats = []
        for cat in entry.findall("atom:category", ns):
            term = cat.get("term", "")
            if term:
                cats.append(term)
        
        # Extract PDF link
        pdf_link = ""
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_link = link.get("href", "")
        
        # Extract DOI if available
        doi = ""
        doi_el = entry.find("arxiv:doi", ns)
        if doi_el is not None:
            doi = doi_el.text.strip()
        
        # Extract journal ref
        journal_ref = ""
        jr_el = entry.find("arxiv:journal_ref", ns)
        if jr_el is not None:
            journal_ref = jr_el.text.strip()
        
        # Extract comment (often contains page count, conference info)
        comment = ""
        comment_el = entry.find("arxiv:comment", ns)
        if comment_el is not None:
            comment = comment_el.text.strip()
        
        published = published_el.text.strip() if published_el is not None else ""
        year = int(published[:4]) if published else 0
        
        # Filter by year if specified
        if start_year and year < start_year:
            continue
        
        arxiv_id = id_el.text.strip().split("/abs/")[-1] if id_el is not None else ""
        
        paper = {
            "source": "arXiv",
            "title": " ".join(title_el.text.strip().split()) if title_el is not None else "",
            "authors": authors,
            "year": year,
            "published": published,
            "abstract": " ".join(summary_el.text.strip().split()) if summary_el is not None else "",
            "arxiv_id": arxiv_id,
            "doi": doi,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": pdf_link,
            "categories": cats,
            "journal_ref": journal_ref,
            "comment": comment,
            "venue": journal_ref if journal_ref else f"arXiv:{cats[0]}" if cats else "arXiv",
            "citation_count": None  # arXiv doesn't provide this
        }
        papers.append(paper)
    
    print(f"[arXiv] Found {len(papers)} papers")
    return papers


# ============================================================================
# Semantic Scholar API
# Docs: https://api.semanticscholar.org/api-docs/
# No API key required for basic use. Rate limit: 100 req / 5 min (unauthenticated)
# ============================================================================

def search_semantic_scholar(query: str, max_results: int = 25,
                           year_range: Optional[str] = None,
                           fields_of_study: Optional[List[str]] = None,
                           min_citations: int = 0,
                           open_access_only: bool = False,
                           api_key: Optional[str] = None) -> List[Dict]:
    """
    Search Semantic Scholar for papers.
    
    Args:
        query: Search query
        max_results: Number of results (max 100 per request)
        year_range: e.g. "2020-2025" or "2023-"
        fields_of_study: e.g. ["Computer Science", "Business", "Economics"]
        min_citations: Minimum citation count filter
        open_access_only: Only return open access papers
        api_key: Optional API key for higher rate limits
    
    Returns:
        List of paper dictionaries
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    
    fields = "title,authors,year,abstract,citationCount,venue,publicationDate,externalIds,openAccessPdf,publicationTypes,journal,fieldsOfStudy,isOpenAccess,tldr"
    
    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": fields
    }
    
    if year_range:
        params["year"] = year_range
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)
    if min_citations > 0:
        params["minCitationCount"] = str(min_citations)
    if open_access_only:
        params["openAccessPdf"] = ""
    
    url = base_url + "?" + urllib.parse.urlencode(params)
    print(f"[Semantic Scholar] Searching: {query} (max {max_results} results)...")
    
    headers = {"User-Agent": "AcademicResearchPlugin/1.0"}
    if api_key:
        headers["x-api-key"] = api_key
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"[Semantic Scholar] Error: {e}")
        return []
    
    papers = []
    for item in data.get("data", []):
        authors = [a.get("name", "") for a in item.get("authors", [])]
        
        ext_ids = item.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI", "")
        arxiv_id = ext_ids.get("ArXiv", "")
        
        oa_pdf = item.get("openAccessPdf", {})
        pdf_url = oa_pdf.get("url", "") if oa_pdf else ""
        
        journal = item.get("journal", {}) or {}
        venue = item.get("venue", "") or journal.get("name", "")
        
        tldr = item.get("tldr", {})
        tldr_text = tldr.get("text", "") if tldr else ""
        
        paper = {
            "source": "Semantic Scholar",
            "title": item.get("title", ""),
            "authors": authors,
            "year": item.get("year"),
            "published": item.get("publicationDate", ""),
            "abstract": item.get("abstract", "") or "",
            "arxiv_id": arxiv_id,
            "doi": doi,
            "url": f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
            "pdf_url": pdf_url,
            "categories": item.get("fieldsOfStudy", []) or [],
            "venue": venue,
            "citation_count": item.get("citationCount", 0),
            "tldr": tldr_text,
            "is_open_access": item.get("isOpenAccess", False),
            "publication_types": item.get("publicationTypes", []) or []
        }
        papers.append(paper)
    
    print(f"[Semantic Scholar] Found {len(papers)} papers")
    return papers


def get_paper_citations(paper_id: str, limit: int = 50, api_key: Optional[str] = None) -> List[Dict]:
    """
    Get papers that cite a given paper (forward snowballing).
    
    Args:
        paper_id: Semantic Scholar paper ID, DOI, or arXiv ID
        limit: Max citations to return
        api_key: Optional API key
    """
    fields = "title,authors,year,citationCount,venue,externalIds"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields={fields}&limit={limit}"
    
    headers = {"User-Agent": "AcademicResearchPlugin/1.0"}
    if api_key:
        headers["x-api-key"] = api_key
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"[Semantic Scholar] Error fetching citations: {e}")
        return []
    
    papers = []
    for item in data.get("data", []):
        citing = item.get("citingPaper", {})
        if citing.get("title"):
            ext_ids = citing.get("externalIds", {}) or {}
            papers.append({
                "source": "Semantic Scholar (citation)",
                "title": citing.get("title", ""),
                "authors": [a.get("name", "") for a in citing.get("authors", [])],
                "year": citing.get("year"),
                "venue": citing.get("venue", ""),
                "citation_count": citing.get("citationCount", 0),
                "doi": ext_ids.get("DOI", ""),
            })
    
    return papers


def get_paper_references(paper_id: str, limit: int = 50, api_key: Optional[str] = None) -> List[Dict]:
    """
    Get papers referenced by a given paper (backward snowballing).
    """
    fields = "title,authors,year,citationCount,venue,externalIds"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields={fields}&limit={limit}"
    
    headers = {"User-Agent": "AcademicResearchPlugin/1.0"}
    if api_key:
        headers["x-api-key"] = api_key
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"[Semantic Scholar] Error fetching references: {e}")
        return []
    
    papers = []
    for item in data.get("data", []):
        cited = item.get("citedPaper", {})
        if cited.get("title"):
            ext_ids = cited.get("externalIds", {}) or {}
            papers.append({
                "source": "Semantic Scholar (reference)",
                "title": cited.get("title", ""),
                "authors": [a.get("name", "") for a in cited.get("authors", [])],
                "year": cited.get("year"),
                "venue": cited.get("venue", ""),
                "citation_count": cited.get("citationCount", 0),
                "doi": ext_ids.get("DOI", ""),
            })
    
    return papers


# ============================================================================
# OpenAlex API
# Docs: https://docs.openalex.org/
# Free, no key required. 100,000 requests/day. CC0 license.
# 474M+ works indexed (more than Scopus or Web of Science)
# ============================================================================

def search_openalex(query: str, max_results: int = 25,
                    year_from: Optional[int] = None,
                    year_to: Optional[int] = None,
                    source_type: Optional[str] = None,
                    min_citations: int = 0,
                    open_access_only: bool = False,
                    mailto: str = "research@example.com") -> List[Dict]:
    """
    Search OpenAlex for papers.
    
    Args:
        query: Search query
        max_results: Number of results (max 200 per page)
        year_from: Filter papers from this year
        year_to: Filter papers until this year
        source_type: "journal", "conference", "repository", etc.
        min_citations: Minimum citation count
        open_access_only: Only open access papers
        mailto: Your email for polite pool (faster responses)
    
    Returns:
        List of paper dictionaries
    """
    base_url = "https://api.openalex.org/works"
    
    # Build filters
    filters = []
    if year_from and year_to:
        filters.append(f"publication_year:{year_from}-{year_to}")
    elif year_from:
        filters.append(f"publication_year:{year_from}-")
    elif year_to:
        filters.append(f"publication_year:-{year_to}")
    if min_citations > 0:
        filters.append(f"cited_by_count:>{min_citations}")
    if open_access_only:
        filters.append("is_oa:true")
    if source_type:
        filters.append(f"type:{source_type}")
    
    params = {
        "search": query,
        "per_page": min(max_results, 200),
        "sort": "relevance_score:desc",
        "mailto": mailto
    }
    
    if filters:
        params["filter"] = ",".join(filters)
    
    url = base_url + "?" + urllib.parse.urlencode(params)
    print(f"[OpenAlex] Searching: {query} (max {max_results} results)...")
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AcademicResearchPlugin/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"[OpenAlex] Error: {e}")
        return []
    
    papers = []
    for item in data.get("results", []):
        # Extract authors
        authors = []
        for authorship in item.get("authorships", []):
            author = authorship.get("author", {})
            name = author.get("display_name", "")
            if name:
                authors.append(name)
        
        # Extract venue/source
        primary_loc = item.get("primary_location", {}) or {}
        source = primary_loc.get("source", {}) or {}
        venue = source.get("display_name", "")
        
        # Extract DOI
        doi = (item.get("doi", "") or "").replace("https://doi.org/", "")
        
        # Best open access URL
        best_oa = item.get("best_oa_location", {}) or {}
        pdf_url = best_oa.get("pdf_url", "") or ""
        
        # Extract concepts/topics
        topics = []
        for topic in item.get("topics", [])[:5]:
            topics.append(topic.get("display_name", ""))
        
        paper = {
            "source": "OpenAlex",
            "title": item.get("display_name", "") or item.get("title", ""),
            "authors": authors,
            "year": item.get("publication_year"),
            "published": item.get("publication_date", ""),
            "abstract": reconstruct_abstract(item.get("abstract_inverted_index", {})),
            "doi": doi,
            "url": item.get("id", "").replace("https://openalex.org/", "https://openalex.org/works/"),
            "pdf_url": pdf_url,
            "categories": topics,
            "venue": venue,
            "citation_count": item.get("cited_by_count", 0),
            "is_open_access": item.get("open_access", {}).get("is_oa", False),
            "openalex_id": item.get("id", ""),
            "type": item.get("type", ""),
            "language": item.get("language", "")
        }
        papers.append(paper)
    
    total = data.get("meta", {}).get("count", 0)
    print(f"[OpenAlex] Found {len(papers)} papers (total matching: {total})")
    return papers


def reconstruct_abstract(inverted_index: Optional[Dict]) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(word for _, word in word_positions)


# ============================================================================
# CrossRef API
# Docs: https://api.crossref.org/swagger-ui/index.html
# Free, no key required. Include mailto for polite pool.
# ============================================================================

def search_crossref(query: str, max_results: int = 25,
                    year_from: Optional[int] = None,
                    year_to: Optional[int] = None,
                    container_title: Optional[str] = None,
                    mailto: str = "research@example.com") -> List[Dict]:
    """
    Search CrossRef for papers. Best for finding DOIs and published journal articles.
    
    Args:
        query: Search query
        max_results: Number of results
        year_from: Filter by publication year
        year_to: Filter by publication year
        container_title: Filter by journal name
        mailto: Email for polite pool
    """
    base_url = "https://api.crossref.org/works"
    
    params = {
        "query": query,
        "rows": min(max_results, 100),
        "sort": "relevance",
        "order": "desc",
        "mailto": mailto
    }
    
    filters = []
    if year_from:
        filters.append(f"from-pub-date:{year_from}")
    if year_to:
        filters.append(f"until-pub-date:{year_to}")
    if container_title:
        params["query.container-title"] = container_title
    if filters:
        params["filter"] = ",".join(filters)
    
    url = base_url + "?" + urllib.parse.urlencode(params)
    print(f"[CrossRef] Searching: {query} (max {max_results} results)...")
    
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": f"AcademicResearchPlugin/1.0 (mailto:{mailto})"
        })
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"[CrossRef] Error: {e}")
        return []
    
    papers = []
    for item in data.get("message", {}).get("items", []):
        # Extract authors
        authors = []
        for author in item.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")
            if given and family:
                authors.append(f"{given} {family}")
            elif family:
                authors.append(family)
        
        # Extract year
        date_parts = item.get("published-print", item.get("published-online", {})).get("date-parts", [[]])
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        
        # Extract venue
        venue = ""
        container = item.get("container-title", [])
        if container:
            venue = container[0]
        
        doi = item.get("DOI", "")
        
        paper = {
            "source": "CrossRef",
            "title": item.get("title", [""])[0] if item.get("title") else "",
            "authors": authors,
            "year": year,
            "published": item.get("created", {}).get("date-time", ""),
            "abstract": clean_html(item.get("abstract", "")),
            "doi": doi,
            "url": f"https://doi.org/{doi}" if doi else "",
            "pdf_url": "",
            "categories": item.get("subject", []),
            "venue": venue,
            "citation_count": item.get("is-referenced-by-count", 0),
            "type": item.get("type", ""),
            "issn": item.get("ISSN", []),
            "publisher": item.get("publisher", "")
        }
        papers.append(paper)
    
    total = data.get("message", {}).get("total-results", 0)
    print(f"[CrossRef] Found {len(papers)} papers (total matching: {total})")
    return papers


def get_paper_by_doi(doi: str, mailto: str = "research@example.com") -> Optional[Dict]:
    """Get full metadata for a paper by DOI from CrossRef."""
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={mailto}"
    
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": f"AcademicResearchPlugin/1.0 (mailto:{mailto})"
        })
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("message", {})
    except:
        return None


def clean_html(text: str) -> str:
    """Remove HTML tags from text (for CrossRef abstracts)."""
    if not text:
        return ""
    import re
    return re.sub(r'<[^>]+>', '', text).strip()


# ============================================================================
# Multi-Source Search (combine all APIs)
# ============================================================================

def search_all(query: str, max_results_per_source: int = 15,
               year_from: Optional[int] = None,
               year_to: Optional[int] = None,
               sources: Optional[List[str]] = None,
               deduplicate: bool = True) -> List[Dict]:
    """
    Search across all academic APIs and combine results.
    
    Args:
        query: Search query
        max_results_per_source: Results per database
        year_from: Filter start year
        year_to: Filter end year
        sources: List of sources to search. Default: all.
                 Options: "arxiv", "semantic_scholar", "openalex", "crossref"
        deduplicate: Remove duplicate papers (matched by DOI or title similarity)
    
    Returns:
        Combined, deduplicated list of papers
    """
    if sources is None:
        sources = ["semantic_scholar", "openalex", "crossref", "arxiv"]
    
    all_papers = []
    
    # Semantic Scholar (broadest academic coverage)
    if "semantic_scholar" in sources:
        year_range = None
        if year_from and year_to:
            year_range = f"{year_from}-{year_to}"
        elif year_from:
            year_range = f"{year_from}-"
        
        ss_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        papers = search_semantic_scholar(
            query, max_results_per_source,
            year_range=year_range,
            api_key=ss_key
        )
        all_papers.extend(papers)
        time.sleep(1)
    
    # OpenAlex (474M+ works, fully open)
    if "openalex" in sources:
        mailto = os.environ.get("OPENALEX_MAILTO", "research@example.com")
        papers = search_openalex(
            query, max_results_per_source,
            year_from=year_from, year_to=year_to,
            mailto=mailto
        )
        all_papers.extend(papers)
        time.sleep(0.5)
    
    # CrossRef (best for DOIs and journal metadata)
    if "crossref" in sources:
        mailto = os.environ.get("CROSSREF_MAILTO", "research@example.com")
        papers = search_crossref(
            query, max_results_per_source,
            year_from=year_from, year_to=year_to,
            mailto=mailto
        )
        all_papers.extend(papers)
        time.sleep(0.5)
    
    # arXiv (preprints, CS/AI/ML focused)
    if "arxiv" in sources:
        papers = search_arxiv(
            query, max_results_per_source,
            start_year=year_from
        )
        all_papers.extend(papers)
        time.sleep(3)  # arXiv rate limit: 1 req / 3 sec
    
    if deduplicate:
        all_papers = deduplicate_papers(all_papers)
    
    # Sort by citation count (descending), then by year (descending)
    all_papers.sort(key=lambda p: (
        -(p.get("citation_count") or 0),
        -(p.get("year") or 0)
    ))
    
    print(f"\n{'='*60}")
    print(f"Total unique papers found: {len(all_papers)}")
    print(f"{'='*60}")
    
    return all_papers


def deduplicate_papers(papers: List[Dict]) -> List[Dict]:
    """Remove duplicate papers based on DOI or similar titles."""
    seen_dois = set()
    seen_titles = set()
    unique = []
    
    for paper in papers:
        doi = paper.get("doi", "").strip().lower()
        title = paper.get("title", "").strip().lower()
        
        # Normalize title for comparison
        title_normalized = "".join(c for c in title if c.isalnum() or c.isspace())
        title_words = set(title_normalized.split())
        
        # Skip if DOI already seen
        if doi and doi in seen_dois:
            continue
        
        # Skip if very similar title already seen
        is_duplicate = False
        for seen_title in seen_titles:
            seen_words = set(seen_title.split())
            if title_words and seen_words:
                overlap = len(title_words & seen_words) / max(len(title_words), len(seen_words))
                if overlap > 0.85:
                    is_duplicate = True
                    break
        
        if is_duplicate:
            continue
        
        if doi:
            seen_dois.add(doi)
        if title_normalized:
            seen_titles.add(title_normalized)
        unique.append(paper)
    
    print(f"[Dedup] {len(papers)} → {len(unique)} unique papers")
    return unique


# ============================================================================
# BibTeX Generation
# ============================================================================

def to_bibtex(paper: Dict) -> str:
    """Convert a paper dictionary to a BibTeX entry."""
    authors = paper.get("authors", [])
    year = paper.get("year", "")
    title = paper.get("title", "")
    
    # Generate citation key
    first_author = authors[0].split()[-1] if authors else "Unknown"
    first_author = "".join(c for c in first_author if c.isalpha())
    title_word = "".join(c for c in title.split()[0] if c.isalpha()) if title else ""
    key = f"{first_author}{year}_{title_word}"
    
    # Determine entry type
    venue = paper.get("venue", "")
    pub_type = paper.get("type", "")
    if "conference" in pub_type.lower() or "proceedings" in venue.lower():
        entry_type = "inproceedings"
        venue_field = f"  booktitle = {{{venue}}},"
    else:
        entry_type = "article"
        venue_field = f"  journal   = {{{venue}}}," if venue else ""
    
    author_str = " and ".join(authors)
    
    lines = [f"@{entry_type}{{{key},"]
    lines.append(f"  author    = {{{author_str}}},")
    lines.append(f"  title     = {{{title}}},")
    if venue_field:
        lines.append(venue_field)
    lines.append(f"  year      = {{{year}}},")
    
    doi = paper.get("doi", "")
    if doi:
        lines.append(f"  doi       = {{{doi}}},")
    
    url = paper.get("url", "")
    if url:
        lines.append(f"  url       = {{{url}}},")
    
    abstract = paper.get("abstract", "")
    if abstract and len(abstract) < 2000:
        # Escape special BibTeX characters
        abstract = abstract.replace("{", "\\{").replace("}", "\\}")
        lines.append(f"  abstract  = {{{abstract}}},")
    
    lines.append("}")
    return "\n".join(lines)


def papers_to_bibtex_file(papers: List[Dict], filepath: str):
    """Save all papers as a .bib file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"% Generated by Academic Research Plugin\n")
        f.write(f"% Date: {datetime.now().isoformat()}\n")
        f.write(f"% Papers: {len(papers)}\n\n")
        for paper in papers:
            f.write(to_bibtex(paper))
            f.write("\n\n")
    print(f"Saved {len(papers)} entries to {filepath}")


# ============================================================================
# Output Formatting
# ============================================================================

def papers_to_markdown_table(papers: List[Dict], filepath: Optional[str] = None) -> str:
    """Format papers as a markdown table."""
    lines = []
    lines.append("| # | Authors (Year) | Title | Venue | Source | Citations |")
    lines.append("|---|---------------|-------|-------|--------|-----------|")
    
    for i, p in enumerate(papers, 1):
        authors = p.get("authors", [])
        if len(authors) > 2:
            author_str = f"{authors[0]} et al."
        elif len(authors) == 2:
            author_str = f"{authors[0]} & {authors[1]}"
        elif authors:
            author_str = authors[0]
        else:
            author_str = "Unknown"
        
        year = p.get("year", "?")
        title = p.get("title", "")
        if len(title) > 80:
            title = title[:77] + "..."
        
        venue = p.get("venue", "")
        if len(venue) > 30:
            venue = venue[:27] + "..."
        
        citations = p.get("citation_count", "N/A")
        source = p.get("source", "")
        
        doi = p.get("doi", "")
        if doi:
            title_linked = f"[{title}](https://doi.org/{doi})"
        else:
            url = p.get("url", "")
            title_linked = f"[{title}]({url})" if url else title
        
        lines.append(f"| {i} | {author_str} ({year}) | {title_linked} | {venue} | {source} | {citations} |")
    
    result = "\n".join(lines)
    
    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Literature Search Results\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(result)
        print(f"Saved markdown table to {filepath}")
    
    return result


def papers_to_csv(papers: List[Dict], filepath: str):
    """Export papers to CSV for Excel/spreadsheet analysis."""
    if not papers:
        print("No papers to export")
        return
    
    fieldnames = ["title", "authors", "year", "venue", "source", "citation_count", 
                  "doi", "url", "pdf_url", "abstract", "categories", "is_open_access"]
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for p in papers:
            row = dict(p)
            row["authors"] = "; ".join(p.get("authors", []))
            row["categories"] = "; ".join(p.get("categories", []))
            writer.writerow(row)
    
    print(f"Saved {len(papers)} papers to {filepath}")


# ============================================================================
# Snowballing
# ============================================================================

def snowball(paper_id: str, direction: str = "both", 
             limit: int = 20, api_key: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    Perform citation snowballing on a paper.
    
    Args:
        paper_id: DOI (e.g. "10.1234/...") or Semantic Scholar paper ID
        direction: "forward" (who cites this), "backward" (references), or "both"
        limit: Max papers per direction
        api_key: Optional Semantic Scholar API key
    
    Returns:
        Dict with "forward" and/or "backward" lists
    """
    results = {}
    
    if direction in ("forward", "both"):
        print(f"[Snowball] Forward: Finding papers that cite {paper_id}...")
        results["forward"] = get_paper_citations(paper_id, limit, api_key)
        time.sleep(1)
    
    if direction in ("backward", "both"):
        print(f"[Snowball] Backward: Finding references of {paper_id}...")
        results["backward"] = get_paper_references(paper_id, limit, api_key)
    
    return results


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for the academic search tool."""
    if len(sys.argv) < 2:
        print("""
Academic Paper Search Tool — CLI Usage
=======================================

Search all databases:
    python academic_search.py search "digital transformation SME"

Search with filters:
    python academic_search.py search "AI adoption" --year-from 2020 --max 30

Search specific source:
    python academic_search.py arxiv "machine learning healthcare"
    python academic_search.py semantic_scholar "technology acceptance"
    python academic_search.py openalex "business model innovation"
    python academic_search.py crossref "organizational resilience"

Snowballing:
    python academic_search.py snowball "10.1234/example.doi" --direction both

Export results:
    python academic_search.py search "query" --output results.csv
    python academic_search.py search "query" --bibtex references.bib
    python academic_search.py search "query" --markdown results.md
""")
        return
    
    command = sys.argv[1]
    
    if command == "search":
        query = sys.argv[2] if len(sys.argv) > 2 else ""
        max_results = 15
        year_from = None
        year_to = None
        output_file = None
        bibtex_file = None
        md_file = None
        
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--max" and i + 1 < len(sys.argv):
                max_results = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--year-from" and i + 1 < len(sys.argv):
                year_from = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--year-to" and i + 1 < len(sys.argv):
                year_to = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                output_file = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--bibtex" and i + 1 < len(sys.argv):
                bibtex_file = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--markdown" and i + 1 < len(sys.argv):
                md_file = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        papers = search_all(query, max_results, year_from, year_to)
        
        print("\n" + papers_to_markdown_table(papers))
        
        if output_file:
            papers_to_csv(papers, output_file)
        if bibtex_file:
            papers_to_bibtex_file(papers, bibtex_file)
        if md_file:
            papers_to_markdown_table(papers, md_file)
    
    elif command in ("arxiv", "semantic_scholar", "openalex", "crossref"):
        query = sys.argv[2] if len(sys.argv) > 2 else ""
        max_results = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        
        papers = search_all(query, max_results, sources=[command])
        print("\n" + papers_to_markdown_table(papers))
    
    elif command == "snowball":
        paper_id = sys.argv[2] if len(sys.argv) > 2 else ""
        direction = "both"
        for i, arg in enumerate(sys.argv):
            if arg == "--direction" and i + 1 < len(sys.argv):
                direction = sys.argv[i + 1]
        
        results = snowball(paper_id, direction)
        for dir_name, papers in results.items():
            print(f"\n{'='*40}")
            print(f"{dir_name.upper()} SNOWBALLING: {len(papers)} papers")
            print(f"{'='*40}")
            print(papers_to_markdown_table(papers))


if __name__ == "__main__":
    main()
