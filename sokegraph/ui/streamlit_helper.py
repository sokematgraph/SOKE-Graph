"""
streamlit_helper.py

UI helper functions for the SOKEGraph Streamlit application.

This module provides utilities for rendering data in Streamlit, including:
- Paginated dataframe display with dark/light theme support
- JSON file visualization
- Excel export with formatted columns
- Click-to-expand text fields for long content

Key Functions:
- show_json_file: Display JSON files with theme-aware formatting
- reduce_columns: Filter and reorder dataframe columns for display
- _show_dataframe_papers: Render paginated table with theme detection
"""
import base64
import urllib.parse
from pathlib import Path

import streamlit as st
import pandas as pd
import json

import streamlit as st
import pandas as pd
import json


def show_json_file(path: str, height: int = 360):
    """Display JSON file content with theme-aware formatting.
    
    Renders JSON files in Streamlit using st.json for collapsible
    tree view, with fallback to code block if needed.

    Parameters
    ----------
    path : str
        Path to JSON file to display
    height : int, optional
        Display height in pixels (default: 360)
        Note: Currently unused, kept for backward compatibility
    """
    # Load JSON file
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use st.json for theme-aware display
    try:
        st.json(data, expanded=False)
    except Exception:
        # Fallback to code with theme-aware styling
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        st.code(json_str, language="json")



def reduce_columns(df: pd.DataFrame, type_df) -> pd.DataFrame:
    """Filter and reorder dataframe columns for display.
    
    Removes internal columns (paper_id, doi, bibtex) and reorders
    columns for better readability in the UI.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with paper data
    type_df : str
        Type of dataframe: "retrieved" or "results"
        - "retrieved": papers from source APIs
        - "results": ranked papers with scores
        
    Returns
    -------
    pd.DataFrame
        Dataframe with filtered and reordered columns
        
    Raises
    ------
    ValueError
        If type_df is not "retrieved" or "results"
    """
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

    # Detect Streamlit's theme by checking the background color of the main container
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
  /* Default light theme styles */
  #papers-wrap {{
    color: #262626;
    background-color: transparent;
  }}
  
  .table-container {{
    overflow-x: auto;
    background-color: transparent;
  }}
  
  #papers-table {{
    border-collapse: collapse; 
    width: 100%;
    table-layout: fixed;
    font-size: 14px;
    background-color: #FFFFFF;
    color: #262626;
  }}
  
  #papers-table th, #papers-table td {{
    padding: 8px 10px;
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: anywhere;
    white-space: normal;
    border: 1px solid #D3D3D3;
    background-color: #FFFFFF;
    color: #262626;
  }}
  
  #papers-table th {{
    font-weight: 600;
    background-color: #F0F2F6 !important;
  }}

  #papers-table tbody tr:nth-child(even) td {{
    background: #F9F9F9 !important;
  }}

  /* Compact widths for small columns */
  #papers-table th:nth-child(1), #papers-table td:nth-child(1) {{ width: 64px; text-align: right; }}
  #papers-table th, #papers-table td {{ max-width: 520px; }}

  /* Pagination */
  .pager {{
    color: #262626;
  }}
  .pager a, .pager span {{
    margin: 0 4px; text-decoration: none; font-size: 14px; cursor: pointer;
    color: #262626;
  }}
  .pager a:hover {{
    color: #0066CC;
  }}
  .pager .current {{
    font-weight: 700; border-bottom: 2px solid currentColor; cursor: default;
  }}

  /* Clamping for long-text cells; click toggles expansion */
  .clamp {{
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    cursor: pointer;
  }}
  .clamp.expand {{
    display: block;
    -webkit-line-clamp: unset;
  }}

  a {{ 
    color: #0066CC;
    text-decoration: underline; 
  }}
  
  /* Meta text (showing papers X to Y of Z) */
  #meta {{
    color: #262626;
  }}
  
  /* Dark theme overrides - will be applied by JavaScript if dark theme detected */
  body.dark-theme #papers-wrap,
  .dark-theme #papers-wrap {{
    color: #FAFAFA;
  }}
  
  body.dark-theme #meta,
  .dark-theme #meta {{
    color: #FAFAFA;
  }}
  
  body.dark-theme #papers-table,
  .dark-theme #papers-table {{
    background-color: #0E1117;
    color: #FAFAFA;
  }}
  
  body.dark-theme #papers-table th,
  body.dark-theme #papers-table td,
  .dark-theme #papers-table th,
  .dark-theme #papers-table td {{
    border: 1px solid #444444;
    background-color: #0E1117;
    color: #FAFAFA;
  }}
  
  body.dark-theme #papers-table th,
  .dark-theme #papers-table th {{
    background-color: #262730 !important;
  }}
  
  body.dark-theme #papers-table tbody tr:nth-child(even) td,
  .dark-theme #papers-table tbody tr:nth-child(even) td {{
    background: rgba(255,255,255,0.03) !important;
  }}
  
  body.dark-theme .pager,
  body.dark-theme .pager a,
  body.dark-theme .pager span,
  .dark-theme .pager,
  .dark-theme .pager a,
  .dark-theme .pager span {{
    color: #FAFAFA;
  }}
  
  body.dark-theme .pager a:hover,
  .dark-theme .pager a:hover {{
    color: #4A9EFF;
  }}
  
  body.dark-theme a,
  .dark-theme a {{
    color: #4A9EFF;
  }}
</style>

<script>
  // Detect Streamlit theme by checking background color
  function detectAndApplyTheme() {{
    const stApp = window.parent.document.querySelector('.stApp');
    if (stApp) {{
      const bgColor = window.getComputedStyle(stApp).backgroundColor;
      // Parse RGB values
      const rgb = bgColor.match(/\d+/g);
      if (rgb) {{
        const brightness = (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
        const isDark = brightness < 128;
        
        const wrapper = document.getElementById('papers-wrap');
        if (wrapper) {{
          if (isDark) {{
            wrapper.classList.add('dark-theme');
          }} else {{
            wrapper.classList.remove('dark-theme');
          }}
        }}
      }}
    }}
  }}
  
  // Run on load and periodically check for theme changes
  detectAndApplyTheme();
  setInterval(detectAndApplyTheme, 500);

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
