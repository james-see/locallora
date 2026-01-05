#!/usr/bin/env python3
"""
Multi-source gravity research paper scraper.
Targets: arXiv, NASA ADS, Semantic Scholar
"""

import argparse
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import arxiv
import httpx

try:
    import ads
except ImportError:
    ads = None

try:
    from semanticscholar import SemanticScholar
except ImportError:
    SemanticScholar = None


@dataclass
class Paper:
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str | None
    source: str
    identifier: str


GRAVITY_QUERIES = [
    "general relativity",
    "gravitational waves",
    "quantum gravity",
    "LIGO gravitational",
    "black hole physics",
    "spacetime curvature",
    "Einstein field equations",
    "graviton",
    "loop quantum gravity",
    "string theory gravity",
]

ARXIV_CATEGORIES = ["gr-qc", "astro-ph.CO", "hep-th", "astro-ph.HE"]


def make_safe_filename(title: str, identifier: str) -> str:
    """Create safe filename from paper title and identifier."""
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)[:50]
    safe_title = safe_title.strip().replace(" ", "_")
    short_hash = hashlib.md5(identifier.encode()).hexdigest()[:8]
    return f"{safe_title}_{short_hash}.pdf"


def search_arxiv(
    query: str, max_results: int = 50, categories: list[str] | None = None
) -> list[Paper]:
    """Search arXiv for papers matching query."""
    papers = []
    cat_filter = " OR ".join(f"cat:{c}" for c in (categories or ARXIV_CATEGORIES))
    full_query = f"({query}) AND ({cat_filter})"

    client = arxiv.Client()
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    for result in client.results(search):
        papers.append(
            Paper(
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                pdf_url=result.pdf_url,
                source="arxiv",
                identifier=result.entry_id,
            )
        )
        time.sleep(0.3)  # Rate limit

    return papers


def search_ads(query: str, max_results: int = 50) -> list[Paper]:
    """Search NASA ADS for papers matching query."""
    if ads is None:
        print("Warning: ads package not installed, skipping ADS search")
        return []

    if not os.getenv("ADS_DEV_KEY"):
        print("Warning: ADS_DEV_KEY not set, skipping ADS search")
        return []

    papers = []
    try:
        results = ads.SearchQuery(
            q=query,
            fl=["title", "author", "abstract", "identifier", "bibcode"],
            rows=max_results,
            sort="relevance",
        )

        for article in results:
            # ADS doesn't always have direct PDF links, construct from bibcode
            bibcode = article.bibcode if hasattr(article, "bibcode") else None
            pdf_url = None
            if bibcode:
                # Try arXiv mirror for ADS results
                pdf_url = f"https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/ARTICLE"

            papers.append(
                Paper(
                    title=article.title[0] if article.title else "Unknown",
                    authors=article.author[:10] if article.author else [],
                    abstract=article.abstract or "",
                    pdf_url=pdf_url,
                    source="ads",
                    identifier=bibcode or str(article.id),
                )
            )
            time.sleep(0.2)

    except Exception as e:
        print(f"ADS search error: {e}")

    return papers


def search_semantic_scholar(query: str, max_results: int = 50) -> list[Paper]:
    """Search Semantic Scholar for papers matching query."""
    if SemanticScholar is None:
        print("Warning: semanticscholar package not installed, skipping S2 search")
        return []

    papers = []
    try:
        sch = SemanticScholar()
        results = sch.search_paper(
            query,
            limit=max_results,
            fields=["title", "authors", "abstract", "openAccessPdf", "paperId"],
        )

        for result in results:
            pdf_url = None
            if result.openAccessPdf:
                pdf_url = result.openAccessPdf.get("url")

            papers.append(
                Paper(
                    title=result.title or "Unknown",
                    authors=[a.name for a in (result.authors or [])],
                    abstract=result.abstract or "",
                    pdf_url=pdf_url,
                    source="semantic_scholar",
                    identifier=result.paperId,
                )
            )
            time.sleep(0.5)  # Stricter rate limit for S2

    except Exception as e:
        print(f"Semantic Scholar search error: {e}")

    return papers


def download_pdf(paper: Paper, output_dir: Path, timeout: float = 30.0) -> Path | None:
    """Download PDF for a paper."""
    if not paper.pdf_url:
        return None

    filename = make_safe_filename(paper.title, paper.identifier)
    output_path = output_dir / filename

    if output_path.exists():
        print(f"  Already exists: {filename}")
        return output_path

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(paper.pdf_url)
            resp.raise_for_status()

            # Verify it's actually a PDF
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and not resp.content[:4] == b"%PDF":
                print(f"  Not a PDF: {paper.title[:40]}...")
                return None

            output_path.write_bytes(resp.content)
            print(f"  Downloaded: {filename}")
            return output_path

    except Exception as e:
        print(f"  Download failed for {paper.title[:40]}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Scrape gravity research papers from multiple sources"
    )
    parser.add_argument("--output-dir", default="papers", help="Output directory for PDFs")
    parser.add_argument(
        "--max-per-query", type=int, default=20, help="Max papers per query per source"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["arxiv", "ads", "semantic_scholar"],
        choices=["arxiv", "ads", "semantic_scholar"],
        help="Sources to search",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        help="Custom search queries (default: built-in gravity queries)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Search only, don't download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = args.queries or GRAVITY_QUERIES
    all_papers: list[Paper] = []

    for query in queries:
        print(f"\n=== Searching: {query} ===")

        if "arxiv" in args.sources:
            print("  Searching arXiv...")
            papers = search_arxiv(query, max_results=args.max_per_query)
            print(f"  Found {len(papers)} papers on arXiv")
            all_papers.extend(papers)

        if "ads" in args.sources:
            print("  Searching NASA ADS...")
            papers = search_ads(query, max_results=args.max_per_query)
            print(f"  Found {len(papers)} papers on ADS")
            all_papers.extend(papers)

        if "semantic_scholar" in args.sources:
            print("  Searching Semantic Scholar...")
            papers = search_semantic_scholar(query, max_results=args.max_per_query)
            print(f"  Found {len(papers)} papers on Semantic Scholar")
            all_papers.extend(papers)

    # Deduplicate by title similarity (simple approach)
    seen_titles = set()
    unique_papers = []
    for p in all_papers:
        title_key = p.title.lower()[:50]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_papers.append(p)

    print(f"\n=== Total unique papers: {len(unique_papers)} ===")

    if args.dry_run:
        for p in unique_papers[:10]:
            print(f"  - {p.title[:60]}... ({p.source})")
        if len(unique_papers) > 10:
            print(f"  ... and {len(unique_papers) - 10} more")
        return

    # Download PDFs
    print(f"\n=== Downloading PDFs to {output_dir} ===")
    downloaded = 0
    for paper in unique_papers:
        result = download_pdf(paper, output_dir)
        if result:
            downloaded += 1
        time.sleep(0.5)  # Be nice to servers

    print(f"\n=== Downloaded {downloaded}/{len(unique_papers)} papers ===")


if __name__ == "__main__":
    main()
