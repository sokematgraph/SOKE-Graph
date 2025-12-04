"""
formatters.py

Utility functions for formatting data for display in Streamlit UI.

This module provides formatters for converting data into display-friendly
formats, particularly for creating clickable links in Markdown/HTML views.

Functions:
- make_url_link: Convert URL strings to clickable Markdown links
- make_doi_link: Convert DOI strings to clickable Markdown links
"""

def make_url_link(url: str) -> str:
    """Return a clickable Markdown link for a URL."""
    if isinstance(url, str) and url.strip():
        return f"[Link]({url})"
    return ""

def make_doi_link(doi: str) -> str:
    """Return a clickable Markdown link for a DOI."""
    if isinstance(doi, str) and doi.strip():
        # Add https://doi.org/ prefix if not already present
        if not doi.startswith("http"):
            return f"[DOI](https://doi.org/{doi})"
        return f"[DOI]({doi})"
    return ""
