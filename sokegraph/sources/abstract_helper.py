# abstract_from_title_plus_openai.py
# Minimal change: when given a URL, try to extract abstract from THAT page first.
# If not found, THEN follow meta/JS/Elsevier "Linking Hub" redirects and retry.
#
# Also supports title-based flow (EPMC → OpenAlex → Crossref → arXiv → journal HTML),
# optional OA PDF extraction, and optional OpenAI fallback.
#
# Requires: pip install requests beautifulsoup4 lxml pdfminer.six pandas

import os, requests, re, json, html, tempfile
from typing import Optional, Dict, Tuple
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text  # optional, for PDF fallback

# ---------- basics ----------
TIMEOUT = 25
UA = {"User-Agent": "title-abstract-fetcher/1.4 (mailto:you@example.com)"}
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL") or "you@example.com"  # for Unpaywall politeness




import requests
import feedparser
from bs4 import BeautifulSoup



from bs4 import BeautifulSoup

def clean_crossref_abstract(text: str) -> str:
    # Parse the XML/HTML structure
    soup = BeautifulSoup(text, "lxml")

    # Replace <jats:sub> and <jats:sup> with readable notation
    for sub in soup.find_all("jats:sub"):
        sub.replace_with(f"_{sub.get_text()}_")
    for sup in soup.find_all("jats:sup"):
        sup.replace_with(f"^{sup.get_text()}^")

    # Get plain text (BeautifulSoup automatically removes other tags like <jats:p>)
    clean_text = soup.get_text(separator=" ", strip=True)

    return clean_text

# ---------- EuropePMC ----------
def epmc_by_doi(doi: str):
    r = requests.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": f"DOI:{doi}", "format": "json", "pageSize": 1},
        timeout=15
    )
    if not r.ok:
        return None
    result = r.json().get("resultList", {}).get("result", [])
    if result:
        return result[0].get("abstractText")

# ---------- ChemRxiv ----------
def chemrxiv_by_doi(doi: str):
    r = requests.get(f"https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/doi/{doi}", timeout=15)
    if not r.ok:
        return None
    return r.json().get("abstract")


def clean_openalex_abstract(abstract_inverted_index):
    if not abstract_inverted_index:
        return None

    # Create a list long enough to hold all words in the right order
    index_word_map = {}
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            index_word_map[pos] = word

    # Sort positions and join words
    abstract_words = [index_word_map[i] for i in sorted(index_word_map)]
    return " ".join(abstract_words)


# ---------- OpenAlex ----------
def openalex_by_doi(doi: str):
    r = requests.get(f"https://api.openalex.org/works/https://doi.org/{doi}", timeout=15)
    if not r.ok:
        return None
    abstract = r.json().get("abstract_inverted_index")
    return clean_openalex_abstract(abstract)

# ---------- Crossref ----------
import requests
def crossref_by_doi(doi: str):
    r = requests.get(f"https://api.crossref.org/works/{doi}")
    if r.ok:
        data = r.json()["message"]
        abstract = clean_crossref_abstract(data.get("abstract"))
        return abstract
    return None

# ---------- arXiv ----------
def arxiv_by_doi(doi: str):
    if not doi.startswith("10.48550/arXiv."):
        return None
    arxiv_id = doi.split("arXiv.")[-1]
    feed = feedparser.parse(f"http://export.arxiv.org/api/query?id_list={arxiv_id}")
    if feed.entries:
        return feed.entries[0].summary
    return None

# ---------- Semantic Scholar ----------
def semanticscholar_by_doi(doi: str):
    r = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=title,abstract",
        timeout=15
    )
    if not r.ok:
        return None
    data = r.json()
    return data.get("abstract")


def _strip_tags(s: Optional[str]) -> Optional[str]:
    if not s: return None
    return BeautifulSoup(s, "lxml").get_text(" ", strip=True)

def _norm(s: Optional[str]) -> str:
    if s is None: return ""
    return re.sub(r"\s+", " ", html.unescape(str(s))).strip()

# ---------- HTML abstract extractor ----------
def _extract_abstract_from_html(html_text: str, url: str) -> str:
    soup = BeautifulSoup(html_text, "lxml")
    host = (urlparse(url).hostname or "").lower()

    # 1) JSON-LD
    for sc in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(sc.string or "")
        except Exception:
            continue
        objs = data if isinstance(data, list) else [data]
        for obj in objs:
            if not isinstance(obj, dict): continue
            t = obj.get("@type") or obj.get("type")
            if isinstance(t, list):
                t = next((x for x in t if isinstance(x, str)), None)
            if t and str(t).lower() in {"scholarlyarticle", "article"}:
                a = obj.get("abstract") or obj.get("description")
                if isinstance(a, dict): a = a.get("@value") or a.get("text")
                a = _norm(a)
                if len(a) > 30: return a

    # 2) ScienceDirect JSON (common after Linking Hub)
    if "sciencedirect.com" in host:
        for sc in soup.find_all("script"):
            txt = sc.string or ""
            if not txt or "abstract" not in txt.lower(): continue
            m = re.search(r'"abstracts?"\s*:\s*(\[[^\]]+\])', txt, flags=re.S|re.I)
            if m:
                try:
                    arr = json.loads(m.group(1))
                    for item in arr:
                        if isinstance(item, dict):
                            t = _norm(item.get("text") or item.get("fullText") or "")
                            if len(t) > 30: return t
                except Exception:
                    pass
            m2 = re.search(r'"abstract"\s*:\s*"(.+?)"', txt, flags=re.S|re.I)
            if m2:
                t = _norm(m2.group(1))
                if len(t) > 30: return t

    # 3) Meta tags
    for name in ["citation_abstract", "dc.description", "DC.Description", "dcterms.abstract", "prism.teaser"]:
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"): return _norm(tag["content"])
    og = soup.find("meta", attrs={"property": "og:description"})
    desc = soup.find("meta", attrs={"name": "description"})
    if og and og.get("content"): return _norm(og["content"])
    if desc and desc.get("content"): return _norm(desc["content"])

    # 4) Generic heading "Abstract" → following <p>
    for h in soup.find_all(re.compile(r"^h[1-6]$"))[:20]:
        if re.search(r"\babstract\b", _norm(h.get_text(" ")), flags=re.I):
            chunks = []
            for sib in h.find_all_next():
                if sib.name and re.match(r"^h[1-6]$", sib.name): break
                if sib.name == "p":
                    t = _norm(sib.get_text(" "))
                    if t: chunks.append(t)
                if len(" ".join(chunks)) > 4000: break
            if chunks: return _norm(" ".join(chunks))
    return ""

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

        # Elsevier Linking Hub
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
def epmc_by_title(title: str) -> Optional[str]:
    r = requests.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": f'TITLE:"{title}"', "format": "json", "pageSize": 1},
        headers=UA, timeout=TIMEOUT
    )
    if not r.ok: return None
    res = r.json().get("resultList", {}).get("result", [])
    return res[0].get("abstractText") if res else None

def openalex_by_title(title: str) -> Optional[str]:
    r = requests.get("https://api.openalex.org/works",
                     params={"search": title, "per_page": 1},
                     headers=UA, timeout=TIMEOUT)
    if not r.ok: return None
    items = r.json().get("results", [])
    if not items: return None
    inv = items[0].get("abstract_inverted_index")
    if not inv: return None
    positions = []
    for w, idxs in inv.items():
        for i in idxs: positions.append((i, w))
    if not positions: return None
    positions.sort(key=lambda x: x[0])
    words = [""] * (positions[-1][0] + 1)
    for i, w in positions: words[i] = w
    return " ".join(w for w in words if w)

def crossref_by_title(title: str) -> Optional[str]:
    r = requests.get("https://api.crossref.org/works",
                     params={"query.title": title, "rows": 1},
                     headers=UA, timeout=TIMEOUT)
    if not r.ok: return None
    items = r.json().get("message", {}).get("items", [])
    if not items: return None
    return _strip_tags(items[0].get("abstract"))

def arxiv_by_title(title: str) -> Optional[str]:
    r = requests.get("http://export.arxiv.org/api/query",
                     params={"search_query": f'ti:"{title}"', "max_results": 1},
                     headers=UA, timeout=TIMEOUT)
    if not r.ok: return None
    soup = BeautifulSoup(r.text, "lxml-xml")
    s = soup.find("summary")
    return s.get_text(" ", strip=True) if s else None


def chemrxiv_by_title(title: str) -> Optional[str]:
    r = requests.get(
        "https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/v1/articles",
        params={"title": title, "page": 1, "size": 1},
        headers=UA,
        timeout=TIMEOUT
    )
    if not r.ok:
        return None

    data = r.json()
    if "content" in data and data["content"]:
        article = data["content"][0]
        abstract = article.get("abstract")
        return abstract.strip() if abstract else None
    return None


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

# ---------- URL-first journal abstract ----------
def journal_abstract_by_url(url: str) -> Optional[str]:
    """
    EASY LOGIC:
      1) Fetch the given URL and try to extract abstract.
      2) If not found, follow special redirects (meta/JS/LinkingHub) and try again.
    """
    # First: try this URL as-is
    print(f"DEBUG: Trying journal HTML abstract extraction from URL: {url}")
    html_txt, final_url = _fetch_http_only(url)
    print(f"DEBUG: Fetched HTML length: {len(html_txt) if html_txt else 'None'}; final URL: {final_url}")

    if html_txt:
        a = _extract_abstract_from_html(html_txt, final_url or url)
        if a: return a

    # Then: special redirects
    html_txt2, final_url2 = _fetch_follow_redirects(url)
    if html_txt2:
        a2 = _extract_abstract_from_html(html_txt2, final_url2 or url)
        if a2: return a2
    return None

def journal_abstract_by_title(title: str) -> Optional[str]:
    """
    For title-only cases: resolve DOI → URL (Crossref) then use the same URL-first logic.
    """
    print(f"DEBUG: Resolving DOI for title: {title}")
    doi = crossref_find_doi_by_title(title)
    if not doi:
        print(f"DEBUG: No DOI found for title: {title}")  
        return None
    print(f"DEBUG: Found DOI: {doi}")
    url = crossref_resolve_url_from_doi(doi)
    if not url:
        print(f"DEBUG: No URL found for DOI: {doi}") 
        return None
    return journal_abstract_by_url(url)

# ---------- OA PDF (optional; unchanged) ----------
def _download_pdf_to_temp(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=UA, timeout=35, stream=True, allow_redirects=True)
        if not r.ok: return None
        ctype = r.headers.get("Content-Type","").lower()
        if ("pdf" not in ctype) and (not url.lower().endswith(".pdf")):
            return None
        fd, tmp = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, "wb") as f:
            for chunk in r.iter_content(1 << 14):
                if chunk: f.write(chunk)
        return tmp
    except requests.RequestException:
        return None

def _extract_abstract_from_pdf_text(txt: str) -> str:
    if not txt: return ""
    head = txt[:12000]
    pats = [
        r"(?is)\babstract\b[:\.\-\–\—\s]*\n?(.*?)(?=\n\s*(?:keywords?|index\s+terms?|introduction|1[\.\s]|I\.)\b)",
        r"(?is)\babstract\b[:\.\-\–\—\s]*\n?(.*?)(?=\n[A-Z][^\n]{0,60}\n)",
    ]
    for pat in pats:
        m = re.search(pat, head)
        if m:
            cand = _norm(m.group(1))
            if 30 <= len(cand) <= 4000:
                return cand
    return ""

def pdf_abstract_from_page_url(url: str) -> Optional[str]:
    """
    If a page is already fetched, look for any .pdf links on that page and extract.
    """
    html_txt, final_url = _fetch_follow_redirects(url)  # follow if needed to land on the real page
    if not html_txt: return None
    soup = BeautifulSoup(html_txt, "lxml")
    pdf_url = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            pdf_url = requests.compat.urljoin(final_url or url, href)
            break
    if not pdf_url: return None

    tmp = _download_pdf_to_temp(pdf_url)
    if not tmp: return None
    try:
        txt = extract_text(tmp)
    except Exception:
        txt = ""
    finally:
        try: os.remove(tmp)
        except Exception: pass

    return _extract_abstract_from_pdf_text(txt) or None

# ---------- OpenAI URL-based abstract retrieval ----------
def openai_fallback(url: str,
                                   api_key: Optional[str] = None,
                                   model: str = "gpt-4o-mini") -> Dict:
    """
    Given a URL to a paper, use OpenAI to extract the plain-text abstract from the page.
    Returns a dict with abstract, source, confidence, and any note on errors.
    """
    try:
        from openai import OpenAI
        

        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        if not client.api_key:
            return {"abstract": None, "source": None, "confidence": None, "note": "OPENAI_API_KEY missing"}

        system_prompt = ""

        user_prompt = f"DAN Go to the provided URL of a scientific paper : {url}. Locate the section labeled 'Abstract'. Extract the full text of the abstract only (do not include the heading, metadata, references, or any other surrounding text). Return the clean, plain-text abstract exactly as it appears on the page. "

        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=[{"type": "web_search"}]
        )

        abstract_text = resp.output_text.strip()
        if not abstract_text:
            return {"abstract": None, "source": "openai_url", "confidence": "unknown", "note": "No abstract found"}
        return {"abstract": abstract_text, "source": "openai_url", "confidence": "high"}

    except Exception as e:
        return {"abstract": None, "source": None, "confidence": None, "note": f"OpenAI error: {e}"}


# ---------- Orchestrators ----------
def abstract_from_url(url: str, try_pdf: bool = False) -> Dict:
    """
    NEW: For when you already have a URL.
    1) Try extracting on this URL as-is.
    2) If empty, follow special redirects (Linking Hub, meta/JS) and try again.
    3) Optionally try to pull a PDF from the page and extract the abstract.
    """
    a = journal_abstract_by_url(url)
    if a:
        return {"url": url, "abstract": a, "source": "journal_html"}

    if try_pdf:
        a_pdf = pdf_abstract_from_page_url(url)
        if a_pdf:
            return {"url": url, "abstract": a_pdf, "source": "pdf_extracted"}

    return {"url": url, "abstract": None, "source": None}

def abstract_from_paper(paper: Dict) -> Dict:

    abstract = fetch_abstract_from_semanticscholar(paper["url"])
    print(f"DEBUG: Fetched abstract from Semantic Scholar for paper '{paper['title']}': {abstract}")
    if abstract:
        return {"title": paper['title'], "abstract": abstract, "source": "Semantic Scholar URL Fetch"}
    else:
        if paper['doi']:
            complete_doi = f"https://doi.org/{paper['doi']}"
            abstract_html = journal_abstract_by_url(complete_doi)
            if abstract_html:
                print(f"DEBUG: Found abstract via DOI URL for paper '{paper['title']}'")
                return {"title": paper['title'], "abstract": abstract_html, "source": "DOI URL Fetch"}
        else:
            complete_doi = crossref_find_doi_by_title(paper['title'])
            if complete_doi:
                print(f"DEBUG: Found DOI '{complete_doi}' for title '{paper['title']}'")
                abstract_html = journal_abstract_by_url(complete_doi)
                if abstract_html:
                    print(f"DEBUG: Found abstract via CrossRef DOI URL for paper '{paper['title']}'")
                    return {"title": paper['title'], "abstract": abstract_html, "source": "CrossRef DOI URL Fetch"}
        
        print(f"DEBUG: Abstract after DOI fetch attempts for '{paper['title']}': {paper['abstract']}")
        
        for fn, src in (
                            (epmc_by_doi, "EuropePMC"),
                            (chemrxiv_by_doi, "ChemRxiv"),
                            (crossref_by_doi, "Crossref"),
                            (openalex_by_doi, "OpenAlex"),
                            (arxiv_by_doi, "arXiv"),
                        ):
                            try:
                                abstract_website = fn(paper["doi"])
                                print(f"DEBUG: Fetched abstract from {src} for paper '{paper['title']}': {abstract_website}")
                                if abstract_website:
                                    return {"title": paper['title'], "abstract": abstract_website, "source": f"{src} Title Fetch"}
                            except Exception:
                                print(f"DEBUG: {src} abstract fetch failed for '{paper['title']}'")
    return {"title": paper['title'], "abstract": None, "source": None}



def fetch_abstract_from_semanticscholar(url_or_id: str) -> str:
    """Fetch the abstract text directly from a Semantic Scholar webpage."""
    try:
        # Accept either the full URL or just the paper ID
        if not url_or_id.startswith("http"):
            url = f"https://www.semanticscholar.org/paper/{url_or_id}"
        else:
            url = url_or_id

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AbstractFetcher/1.0; +https://example.com)"
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # The abstract is usually in <meta name="description"> or <span data-testid="abstract">
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return meta_desc["content"].strip()

        # fallback: find the abstract container
        abstract_div = soup.find("span", {"data-testid": "abstract"})
        if abstract_div:
            return abstract_div.get_text(strip=True)

        # last resort: look for paragraph with "Abstract" label
        for h2 in soup.find_all(["h2", "strong"]):
            if "abstract" in h2.text.lower():
                nxt = h2.find_next("p")
                if nxt:
                    return nxt.get_text(strip=True)

        return ""
    except Exception as e:
        print("Error fetching abstract:", e)
        return ""





