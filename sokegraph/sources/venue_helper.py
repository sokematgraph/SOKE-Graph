# venue_from_title_plus_doi.py
# Minimal change style (mirrors your abstract helper):
# - When given a URL, try to extract venue/year/DOI from THAT page first.
# - If not found, follow meta/JS/Elsevier "Linking Hub" redirects and retry.
#
# Also supports title-based flow (Crossref → OpenAlex → Europe PMC → arXiv → journal HTML).
#
# Requires: pip install requests beautifulsoup4 lxml pandas

import os, requests, re, json, html
from typing import Optional, Dict, Tuple
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# ---------- basics ----------
TIMEOUT = 25
UA = {"User-Agent": "title-venue-fetcher/1.0 (mailto:you@example.com)"}

def _norm(s: Optional[str]) -> str:
    if s is None: return ""
    return re.sub(r"\s+", " ", html.unescape(str(s))).strip()

# ---------- HTML venue/year/doi extractor ----------
def _extract_venue_year_doi_from_html(html_text: str, url: str) -> Dict:
    """
    Best-effort extraction from a publisher/article landing HTML.
    Returns dict with keys: venue, year, doi (any may be None if not found).
    """
    soup = BeautifulSoup(html_text, "lxml")
    host = (urlparse(url).hostname or "").lower()
    out = {"venue": None, "year": None, "doi": None}

    # 1) JSON-LD (ScholarlyArticle / Periodical)
    for sc in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(sc.string or "")
        except Exception:
            continue
        objs = data if isinstance(data, list) else [data]
        for obj in objs:
            if not isinstance(obj, dict): 
                continue
            t = obj.get("@type") or obj.get("type")
            if isinstance(t, list):
                t = next((x for x in t if isinstance(x, str)), None)
            t_norm = str(t).lower() if t else ""
            if t_norm in {"scholarlyarticle", "article"}:
                # DOI
                doi = obj.get("identifier") or obj.get("doi")
                if isinstance(doi, dict):
                    doi = doi.get("@value") or doi.get("value")
                if isinstance(doi, list):
                    # sometimes mixed identifiers; pick first DOI-looking
                    for v in doi:
                        if isinstance(v, str) and re.search(r"\b10\.\d{4,9}/\S+\b", v):
                            doi = v; break
                if isinstance(doi, str):
                    m = re.search(r"(10\.\d{4,9}/\S+)", doi)
                    out["doi"] = m.group(1) if m else (doi if doi.startswith("10.") else out["doi"])
                # Venue via isPartOf/periodical
                is_part = obj.get("isPartOf") or obj.get("periodical") or {}
                if isinstance(is_part, dict):
                    cand = is_part.get("name") or is_part.get("title")
                    cand = _norm(cand)
                    if cand: out["venue"] = out["venue"] or cand
                # Year via datePublished
                dp = obj.get("datePublished") or obj.get("dateCreated")
                if isinstance(dp, str) and re.match(r"\d{4}", dp):
                    try: out["year"] = int(dp[:4])
                    except Exception: pass
            elif t_norm in {"periodical", "publicationissue", "publicationvolume"}:
                name = _norm(obj.get("name") or obj.get("title"))
                if name: out["venue"] = out["venue"] or name

    # 2) Meta tags (common across publishers)
    meta_map = {
        "citation_journal_title": "venue",
        "citation_conference_title": "venue",
        "prism.publicationname": "venue",
        "dc.source": "venue", "DC.Source": "venue",
        "citation_doi": "doi",
        "dc.identifier": "doi", "DC.Identifier": "doi",
        "prism.doi": "doi",
        "prism.publicationdate": "year",
        "citation_publication_date": "year",
        "citation_date": "year",
        "dc.date": "year", "DC.Date": "year",
    }
    for meta in soup.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").lower()
        if not name: continue
        if name in meta_map and meta.get("content"):
            key = meta_map[name]
            val = _norm(meta["content"])
            if key == "doi":
                m = re.search(r"(10\.\d{4,9}/\S+)", val)
                if m: out["doi"] = out["doi"] or m.group(1)
                elif val.lower().startswith("doi:"):
                    m = re.search(r"(10\.\d{4,9}/\S+)", val[4:].strip())
                    if m: out["doi"] = out["doi"] or m.group(1)
            elif key == "year":
                m = re.search(r"(\d{4})", val)
                if m and not out["year"]:
                    try: out["year"] = int(m.group(1))
                    except Exception: pass
            else:
                if val and not out["venue"]:
                    out["venue"] = val

    # 3) ScienceDirect JSON (after Elsevier LinkingHub)
    if "sciencedirect.com" in host:
        for sc in soup.find_all("script"):
            txt = sc.string or ""
            if not txt: continue
            # Attempt to locate publication title and pii/doi objects
            m = re.search(r'"publicationTitle"\s*:\s*"(.+?)"', txt, flags=re.S|re.I)
            if m and not out["venue"]:
                out["venue"] = _norm(m.group(1))
            m2 = re.search(r'"doi"\s*:\s*"(.+?)"', txt, flags=re.S|re.I)
            if m2 and not out["doi"]:
                cand = _norm(m2.group(1))
                if re.match(r"^10\.", cand): out["doi"] = cand

    # 4) Any visible 'Journal' label block (weak heuristic)
    if not out["venue"]:
        lab = soup.find(lambda tag: tag.name in {"div","span","li"} and "journal" in _norm(tag.get_text(" ")).lower())
        if lab:
            # take nearby strong/b tag text
            t = _norm(lab.get_text(" "))
            # avoid taking a whole paragraph; prefer concise
            if 4 < len(t) < 200: out["venue"] = t

    return out

# ---------- fetch helpers ----------
def _fetch_http_only(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetch once (HTTP redirects only). No meta/JS tricks."""
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT, allow_redirects=True)
        if r.ok: return r.text, r.url
    except requests.RequestException:
        pass
    return None, None

def _fetch_follow_redirects(url: str, max_hops: int = 3) -> Tuple[Optional[str], Optional[str]]:
    """
    Follow meta refresh, JS location redirects, and Elsevier Linking Hub hidden inputs.
    Returns (html, final_url).
    """
    current = url
    for _ in range(max_hops):
        try:
            r = requests.get(current, headers=UA, timeout=TIMEOUT, allow_redirects=True)
            if not r.ok: return None, None
        except requests.RequestException:
            return None, None

        html_txt, final_url = r.text, r.url
        soup = BeautifulSoup(html_txt, "lxml")

        # <meta http-equiv="refresh" content="2; url='...'>
        meta = soup.find(lambda tag: tag.name == "meta" and any(
            (k.lower() == "http-equiv" and str(v).lower() == "refresh") for k, v in tag.attrs.items()))
        if meta and meta.get("content"):
            m = re.search(r"(?i)\d+\s*;\s*url\s*=\s*[\"']?([^\"'>;]+)", meta["content"])
            if m:
                target = requests.compat.urljoin(final_url, m.group(1).strip())
                if target != final_url:
                    current = target
                    continue

        # JS redirects
        m = re.search(r'(?i)(?:window\.)?location(?:\.href)?\s*=\s*[\'"]([^\'"]+)[\'"]', html_txt)
        if not m:
            m = re.search(r'(?i)location\.replace\(\s*[\'"]([^\'"]+)[\'"]\s*\)', html_txt)
        if m:
            target = requests.compat.urljoin(final_url, m.group(1).strip())
            if target != final_url:
                current = target
                continue

        # Elsevier Linking Hub → ScienceDirect
        host = (urlparse(final_url).hostname or "").lower()
        if "linkinghub.elsevier.com" in host:
            redirectURL = soup.find(id="redirectURL") or soup.find("input", attrs={"name":"redirectURL"})
            resultName  = soup.find(id="resultName")  or soup.find("input", attrs={"name":"resultName"})
            key         = soup.find(id="key")         or soup.find("input", attrs={"name":"key"})
            if redirectURL and resultName and key:
                path = f"/retrieve/{resultName.get('value','').strip()}?Redirect={redirectURL.get('value','').strip()}&key={key.get('value','').strip()}"
                target = requests.compat.urljoin(final_url, path)
                if target != final_url:
                    current = target
                    continue
            else:
                a = soup.find("a", href=re.compile(r"sciencedirect\.com", re.I))
                if a and a.get("href"):
                    target = requests.compat.urljoin(final_url, a["href"])
                    if target != final_url:
                        current = target
                        continue

        # No more special redirects
        return html_txt, final_url
    return None, None

# ---------- Title → APIs ----------
def crossref_title(title: str) -> Dict:
    """
    Returns {doi, venue, year} for the best Crossref match (or empties).
    """
    r = requests.get("https://api.crossref.org/works",
                     params={"query.title": title, "rows": 1},
                     headers=UA, timeout=TIMEOUT)
    if not r.ok: return {}
    items = r.json().get("message", {}).get("items", [])
    if not items: return {}
    it = items[0]
    doi = it.get("DOI")
    venue = " ".join(it.get("container-title") or []) or " ".join(it.get("short-container-title") or [])
    # year from date-parts
    def _year_from(it):
        for k in ("issued","published-print","published-online","created"):
            dp = (it.get(k) or {}).get("date-parts")
            if isinstance(dp, list) and dp and isinstance(dp[0], list) and dp[0]:
                return dp[0][0]
        return None
    year = _year_from(it)
    return {"doi": doi, "venue": venue or None, "year": year}

def openalex_title(title: str) -> Dict:
    r = requests.get("https://api.openalex.org/works",
                     params={"search": title, "per_page": 1},
                     headers=UA, timeout=TIMEOUT)
    if not r.ok: return {}
    items = r.json().get("results", [])
    if not items: return {}
    it = items[0]
    ids = it.get("ids") or {}
    doi_url = ids.get("doi")
    doi = doi_url.split("https://doi.org/")[1] if (doi_url and doi_url.startswith("https://doi.org/")) else None
    venue = (
        (((it.get("primary_location") or {}).get("source") or {}).get("display_name"))
        or ((it.get("host_venue") or {}).get("display_name"))
    )
    year = it.get("publication_year")
    return {"doi": doi, "venue": venue or None, "year": year}

def epmc_title(title: str) -> Dict:
    r = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                     params={"query": f'TITLE:"{title}"', "format": "json", "pageSize": 1},
                     headers=UA, timeout=TIMEOUT)
    if not r.ok: return {}
    res = r.json().get("resultList", {}).get("result", [])
    if not res: return {}
    it = res[0]
    venue = it.get("journalTitle")
    try: year = int(it.get("pubYear")) if it.get("pubYear") else None
    except Exception: year = None
    return {"doi": None, "venue": venue or None, "year": year}

def arxiv_title(title: str) -> Dict:
    r = requests.get("http://export.arxiv.org/api/query",
                     params={"search_query": f'ti:"{title}"', "max_results": 1},
                     headers=UA, timeout=TIMEOUT)
    if not r.ok: return {}
    soup = BeautifulSoup(r.text, "lxml-xml")
    entry = soup.find("entry")
    if not entry: return {}
    # Prefer <arxiv:journal_ref>, else "arXiv"
    jref = entry.find("arxiv:journal_ref")
    venue = jref.get_text(strip=True) if jref else "arXiv"
    published = entry.find("published")
    year = None
    if published and re.match(r"\d{4}", published.get_text(strip=True)):
        try: year = int(published.get_text(strip=True)[:4])
        except Exception: pass
    return {"doi": None, "venue": venue or None, "year": year}

# ---------- Crossref helpers ----------
def crossref_find_doi_by_title(title: str) -> Optional[str]:
    try:
        r = requests.get("https://api.crossref.org/works",
                         params={"query.title": title, "rows": 1},
                         headers=UA, timeout=TIMEOUT)
        if not r.ok: return None
        items = r.json().get("message", {}).get("items", [])
        if not items: return None
        return items[0].get("DOI")
    except requests.RequestException:
        return None

def crossref_resolve_url_from_doi(doi: str) -> Optional[str]:
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", headers=UA, timeout=TIMEOUT)
        if not r.ok: return None
        return (r.json().get("message") or {}).get("URL")
    except requests.RequestException:
        return None

def crossref_doi_metadata(doi: str) -> Dict:
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", headers=UA, timeout=TIMEOUT)
        if not r.ok: return {}
        msg = (r.json().get("message") or {})
        venue = " ".join(msg.get("container-title") or []) or " ".join(msg.get("short-container-title") or [])
        # year
        y = None
        for k in ("issued","published-print","published-online","created"):
            dp = (msg.get(k) or {}).get("date-parts")
            if isinstance(dp, list) and dp and isinstance(dp[0], list) and dp[0]:
                y = dp[0][0]; break
        url = msg.get("URL")
        return {"venue": venue or None, "year": y, "url": url}
    except requests.RequestException:
        return {}

# ---------- URL-first journal info ----------
def journal_info_by_url(url: str) -> Dict:
    """
    EASY LOGIC:
      1) Fetch the given URL and try to extract venue/year/doi.
      2) If not found, follow special redirects (meta/JS/LinkingHub) and try again.
    """
    # First: try this URL as-is
    html_txt, final_url = _fetch_http_only(url)
    if html_txt:
        out = _extract_venue_year_doi_from_html(html_txt, final_url or url)
        if any(out.values()):
            out["url"] = final_url or url
            out["source"] = "journal_html"
            return out

    # Then: special redirects
    html_txt2, final_url2 = _fetch_follow_redirects(url)
    if html_txt2:
        out2 = _extract_venue_year_doi_from_html(html_txt2, final_url2 or url)
        if any(out2.values()):
            out2["url"] = final_url2 or url
            out2["source"] = "journal_html"
            return out2

    return {"url": url, "venue": None, "year": None, "doi": None, "source": None}

def journal_info_by_title(title: str) -> Dict:
    """
    For title-only cases: resolve DOI → URL (Crossref) then use the same URL-first logic.
    """
    doi = crossref_find_doi_by_title(title)
    if not doi: return {"venue": None, "year": None, "doi": None, "url": None}
    url = crossref_resolve_url_from_doi(doi)
    if not url: return {"venue": None, "year": None, "doi": doi, "url": None}
    info = journal_info_by_url(url)
    if not info.get("doi"):
        info["doi"] = doi
    return info

# ---------- Orchestrators ----------
def venue_from_title(title: str) -> Dict:
    """
    Title-based path:
    Crossref → OpenAlex → Europe PMC → arXiv → journal HTML (URL-first, then redirects).
    Returns {title, doi, venue, year, url, source, confidence}
    """
    # 1) Crossref
    try:
        cr = crossref_title(title)
        if cr.get("venue") or cr.get("doi") or cr.get("year"):
            url = crossref_resolve_url_from_doi(cr["doi"]) if cr.get("doi") else None
            return {"title": title, "doi": cr.get("doi"), "venue": cr.get("venue"),
                    "year": cr.get("year"), "url": url, "source": "Crossref", "confidence": "source"}
    except Exception:
        pass

    # 2) OpenAlex
    try:
        oa = openalex_title(title)
        if oa.get("venue") or oa.get("doi") or oa.get("year"):
            url = crossref_resolve_url_from_doi(oa["doi"]) if oa.get("doi") else None
            return {"title": title, "doi": oa.get("doi"), "venue": oa.get("venue"),
                    "year": oa.get("year"), "url": url, "source": "OpenAlex", "confidence": "source"}
    except Exception:
        pass

    # 3) Europe PMC
    try:
        ep = epmc_title(title)
        if ep.get("venue") or ep.get("year"):
            return {"title": title, "doi": None, "venue": ep.get("venue"),
                    "year": ep.get("year"), "url": None, "source": "EuropePMC", "confidence": "source"}
    except Exception:
        pass

    # 4) arXiv
    try:
        ax = arxiv_title(title)
        if ax.get("venue") or ax.get("year"):
            return {"title": title, "doi": None, "venue": ax.get("venue"),
                    "year": ax.get("year"), "url": None, "source": "arXiv", "confidence": "source"}
    except Exception:
        pass

    # 5) Resolve to journal page and try HTML (may also surface DOI)
    try:
        info = journal_info_by_title(title)
        if info.get("venue") or info.get("year") or info.get("doi"):
            return {"title": title, **info, "source": info.get("source") or "journal_html", "confidence": "source"}
    except Exception:
        pass

    return {"title": title, "doi": None, "venue": None, "year": None, "url": None, "source": None, "confidence": None}

def venue_from_url(url: str) -> Dict:
    """
    URL-first path:
    Try to extract venue/year/doi from the given page (following redirects as needed).
    """
    info = journal_info_by_url(url)
    return info

# ---------- demo ----------
if __name__ == "__main__":
    import pandas as pd

    # DEMO 1: Title-based
    titles = [
        "Earth Abundant Catalysts for Water Electrolysis at Low Overpotentials",
        "Attention Is All You Need",
        "Deep Residual Learning for Image Recognition"
    ]
    for t in titles:
        res = venue_from_title(t)
        print(f"Title: {t}\nSource: {res.get('source')}\nDOI: {res.get('doi')}\nVenue: {res.get('venue')}\nYear: {res.get('year')}\nURL: {res.get('url')}\n")

    # DEMO 2: If you already have a landing-page URL
    # url = "https://www.sciencedirect.com/science/article/pii/S092583881934592X"
    # print(venue_from_url(url))
