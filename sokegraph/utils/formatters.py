# utils/formatters.py

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
