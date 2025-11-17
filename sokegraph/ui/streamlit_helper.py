import base64
import urllib.parse
from pathlib import Path

import streamlit as st
import pandas as pd
import json

import streamlit as st
import pandas as pd
import json


def get_theme_colors():
    """Return active Streamlit theme colors or neutral fallbacks."""
    return {
        "bg": st.get_option("theme.backgroundColor") or "#FFFFFF",
        "text": st.get_option("theme.textColor") or "#000000",
        "secondary": st.get_option("theme.secondaryBackgroundColor") or "#F5F5F5",
        "primary": st.get_option("theme.primaryColor") or "#4E9A06"
    }


def show_json_file(path: str, height: int = 360):
    """Display JSON content inside a scrollable, theme-aware box."""
    import json
    import streamlit as st

    # Load JSON file
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Pretty-print JSON string
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    # Read active Streamlit theme colors
    bg = st.get_option("theme.secondaryBackgroundColor") or "#f9f9f9"
    text = st.get_option("theme.textColor") or "#202124"
    border = st.get_option("theme.primaryColor") or "#4E9A06"

    # Render in styled scrollable box
    st.markdown(f"""
    <div style="
        border: 1px solid rgba(120,120,120,0.3);
        border-radius: 8px;
        padding: 10px 12px;
        background: {bg};
        color: {text};
        font-family: ui-monospace, monospace;
        font-size: 13px;
        white-space: pre-wrap;
        overflow: auto;
        max-height: {height}px;
        line-height: 1.4;
    ">
    {json_str}
    </div>
    """, unsafe_allow_html=True)




def reduce_columns(df: pd.DataFrame, type_df) -> pd.DataFrame:
    df_display = df.copy()
    if type_df == "retrieved":
        columns_to_remove = [
            "paper_id", "doi", "bibtex"
        ]
        return df_display.drop(columns=[col for col in columns_to_remove if col in df_display.columns])
    elif type_df == "results":
        columns_to_remove = [
            "paper_id", "doi", "bibtex", "Shared Pairs", "Relevance Level", "Scoring"
        ]
        df_display = df_display.drop(columns=[col for col in columns_to_remove if col in df_display.columns])
        col = df_display.pop("Relevant Keyword Score")
        df_display.insert(2, col.name, col)
        col = df_display.pop("Pair Count")
        df_display.insert(1, col.name, col)
        return df_display
    else:
        raise ValueError(f"Unknown type_df: {type_df}")
    


def _show_dataframe_papers(df: pd.DataFrame, header: str, type_df:str , page_size: int = 10):
    df_display = reduce_columns(df, type_df)


    # Short, clean hyperlinks
    if "doi" in df_display.columns:
        def fmt_doi(x):
            if isinstance(x, str) and x.strip():
                x = x.strip()
                href = f"https://doi.org/{x}" if not x.startswith("http") else x
                return f'<a href="{href}" target="_blank" rel="noopener">DOI</a>'
            return ""
        df_display["doi"] = df_display["doi"].apply(fmt_doi)

    if "url" in df_display.columns:
        df_display["url"] = df_display["url"].apply(
            lambda x: f'<a href="{x.strip()}" target="_blank" rel="noopener">Link</a>'
            if isinstance(x, str) and x.strip() and x.startswith("http") else ""
        )

    # Add Order column (1..N) – rendered client-side without scrollbars
    df_display.insert(0, "Index", range(1, len(df_display) + 1))

    st.subheader(header)

    data = df_display.to_dict(orient="records")
    cols = list(df_display.columns)

    # Columns we expect to be long; clamp them to 2 lines (click to expand)
    long_cols = {"title", "abstract", "authors", "venue"}
    theme = get_theme_colors()
    html = f"""
<div id="papers-wrap">
  <div id="meta" style="margin: 6px 0; font-size: 13px;"></div>
  <div class="table-container">
    <table id="papers-table">
      <thead></thead>
      <tbody></tbody>
    </table>
  </div>
  <div id="pager" class="pager"></div>
</div>

<style>
  .table-container {{
    overflow-x: auto;
  }}

  #papers-table {{
    border-collapse: collapse;
    width: 100%;
    table-layout: fixed;
    font-size: 14px;
    background-color: {theme["secondary"]};
    color: {theme["text"]};
  }}

  #papers-table th, #papers-table td {{
    border: 1px solid rgba(120,120,120,0.25);
    padding: 8px 10px;
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: anywhere;
    white-space: normal;
  }}

  #papers-table th {{
    background-color: rgba(120,120,120,0.20);
    color: {theme["text"]};
    font-weight: 600;
  }}

  /* Gentle zebra striping for readability */
  #papers-table tbody tr:nth-child(even) {{
    background-color: rgba(120,120,120,0.07);
  }}

  /* Hover highlight */
  #papers-table tbody tr:hover td {{
    background-color: rgba(120,120,120,0.15);
  }}

    /* Pagination styling */
  .pager {{
    margin-top: 8px;
    text-align: center;
  }}

  /* Pagination styling */
  .pager {{
    margin-top: 10px;
    text-align: center;
  }}

  .pager a, .pager span {{
    margin: 0 6px;
    font-size: 15px;
    text-decoration: none;
    padding: 3px 8px;
    border-radius: 6px;
    transition: all 0.2s ease-in-out;
    color: {theme["text"]};
  }}

  /* Make all text fully opaque by default for readability */
  .pager a, .pager span {{
    opacity: 0.85;
  }}

  /* Hover and active states */
  .pager a:hover {{
    background-color: rgba(120,120,120,0.15);
    color: {theme["primary"]};
    opacity: 1;
  }}

  /* Current (active) page number */
  .pager .current {{
    font-weight: 700;
    color: {theme["primary"]};
    border-bottom: 3px solid {theme["primary"]};
    border-radius: 2px;
    padding-bottom: 2px;
    opacity: 1;
  }}

  /* Disabled buttons (Prev/Next when unavailable) */
  .pager span:not(.current):not([data-p]) {{
    color: rgba(150,150,150,0.6);
    cursor: default;
  }}



  /* Links */
  a {{
    color: {theme["primary"]};
    text-decoration: none;
  }}
  a:hover {{
    text-decoration: underline;
  }}

  /* Column widths remain the same */
  #papers-table th:nth-child(1), #papers-table td:nth-child(1) {{
    width: 64px;
    text-align: right;
  }}
  #papers-table th, #papers-table td {{
    max-width: 520px;
  }}

  /* Clamp long text to two lines */
  .clamp {{
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }}
  .clamp.expand {{
    display: block;
    -webkit-line-clamp: unset;
  }}
</style>


<script>
  const ROWS = {json.dumps(data)};
  const COLS = {json.dumps(cols)};
  const PAGE_SIZE = {int(page_size)};
  const LONG_COLS = new Set({json.dumps(list(long_cols))});

  let currentPage = 1;

  const thead = document.createElement("thead");
  const tbody = document.createElement("tbody");
  const table = document.getElementById("papers-table");
  table.appendChild(thead);
  table.appendChild(tbody);

  const meta  = document.getElementById("meta");
  const pager = document.getElementById("pager");

  // Render header
  thead.innerHTML = "<tr>" + COLS.map(c => `<th>${{c}}</th>`).join("") + "</tr>";

  function cellHTML(colName, val) {{
    const raw = (val === null || val === undefined) ? "" : String(val);

    // Allow our link HTML to pass through
    if (raw.startsWith("<a ")) {{
      return raw;
    }}

    // For long columns, clamp to two lines and put full text in title tooltip
    const escaped = raw
      .replaceAll("&","&amp;")
      .replaceAll("<","&lt;")
      .replaceAll(">","&gt;");

    if (LONG_COLS.has(colName)) {{
      return `<div class="clamp" title="${{escaped}}">${{escaped}}</div>`;
    }}
    return escaped;
  }}

  function renderPage(p) {{
    const total = ROWS.length;
    const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    currentPage = Math.min(Math.max(1, p), pages);

    const start = (currentPage - 1) * PAGE_SIZE;
    const end = Math.min(start + PAGE_SIZE, total);
    const slice = ROWS.slice(start, end);

    // Rows
    tbody.innerHTML = slice.map(row => {{
      return "<tr>" + COLS.map(c => `<td>${{cellHTML(c, row[c])}}</td>`).join("") + "</tr>";
    }}).join("");

    // Click to expand clamped cells (toggle)
    tbody.querySelectorAll(".clamp").forEach(el => {{
      el.addEventListener("click", () => el.classList.toggle("expand"));
    }});

    // Meta text
    meta.textContent = `Showing papers ${{start+1}} to ${{end}} of ${{total}}`;

    // Pager (windowed)
    const windowSize = 5;
    let left = Math.max(1, currentPage - windowSize);
    let right = Math.min(pages, currentPage + windowSize);

    const parts = [];
    if (currentPage > 1) parts.push(`<a data-p="${{currentPage-1}}">« Prev</a>`); else parts.push(`<span>« Prev</span>`);

    if (left > 1) {{
      parts.push(`<a data-p="1">1</a>`);
      if (left > 2) parts.push(`<span>…</span>`);
    }}

    for (let i = left; i <= right; i++) {{
      if (i === currentPage) parts.push(`<span class="current">${{i}}</span>`);
      else parts.push(`<a data-p="${{i}}">${{i}}</a>`);
    }}

    if (right < pages) {{
      if (right < pages - 1) parts.push(`<span>…</span>`);
      parts.push(`<a data-p="${{pages}}">${{pages}}</a>`);
    }}

    if (currentPage < pages) parts.push(`<a data-p="${{currentPage+1}}">Next »</a>`); else parts.push(`<span>Next »</span>`);

    pager.innerHTML = parts.join(" ");

    // No-reload page changes
    pager.querySelectorAll("a[data-p]").forEach(a => {{
      a.addEventListener("click", e => {{
        const p = parseInt(e.target.getAttribute("data-p"));
        renderPage(p);
        // keep URL in sync without reload
        if (history && history.replaceState) {{
          const url = new URL(window.location);
          url.searchParams.set("page", p);
          history.replaceState(null, "", url.toString());
        }}
      }});
    }});
  }}

  renderPage(1);
</script>
"""
    st.components.v1.html(html, height=560, scrolling=True)






def _make_download_link(label: str, data_bytes: bytes, file_name: str, mime: str):
    """Embed *data_bytes* in a Base64 data-URI and show a clickable link."""
    b64 = base64.b64encode(data_bytes).decode()
    data_uri = f"data:{mime};base64,{urllib.parse.quote(b64)}"
    st.markdown(
        f'<a href="{data_uri}" download="{file_name}">📥 {label}</a>',
        unsafe_allow_html=True,
    )

def show_ranked_results_with_links(output_paths_all: dict, caption: str = "Ranked Papers"):
    """Display top rows of CSV and show download links for all formats."""
    # Show top rows from CSV
    csv_path = output_paths_all.get("csv")
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        _show_dataframe_papers(df, caption)

    # Create download links for each format
    st.markdown("### Download Results")
    for fmt, path in output_paths_all.items():
        if Path(path).exists():
            mime = {
                "csv": "text/csv",
                "json": "application/json",
                "jsonl": "application/json",
                "parquet": "application/octet-stream",
                "ttl": "text/turtle",
                "nt": "application/n-triples",
                "jsonld": "application/ld+json",
                "graphml": "application/xml"
            }.get(fmt, "application/octet-stream")
            with open(path, "rb") as f:
                data = f.read()
            label = f"Download {fmt.upper()}"
            file_name = Path(path).name
            _make_download_link(label, data, file_name, mime)
