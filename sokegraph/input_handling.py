
from typing import List, Dict
from semanticscholar import SemanticScholar
from sokegraph.util.logger import LOG
from time import sleep
from sokegraph.functions import *
import pandas as pd
import os
from PyPDF2 import PdfReader
import zipfile
import zipfile
import os
import requests
from sokegraph.utils import check_file




def get_papers(params):
    if not(params.pdfs_file=="" or params.number_papers==""):
        LOG.error("please enter number of papers to get from web, or a path for a zip file containing paper pdfs for ranking!")
        return
    if params.pdfs_file!="" and params.number_papers!="" :
        LOG.error("please just enter one of number of papers to get from web and a path for a zip file containing paper pdfs for ranking!")
        return
    
    if params.number_papers!="":
        num_papers = int(params.number_papers)
        if params.paper_query_file=="":
            LOG.error("when you enter number of papers to get from web, you should enter keyword for finiding relevent papers!")
            return
        else:
            search_queries_list = load_queries_list(params.paper_query_file)
            papers = get_unique_papers_from_sch(num_papers, search_queries_list, params.output_dir)
    else:
        params.pdfs_file = check_file(params.pdfs_file)
        papers = extract_paper_info_from_zip(params.pdfs_file, params.output_dir)

    return papers
            
def unzip_papers(zip_path, output_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(f"{output_dir}/extracted_pdfs")
    return [os.path.join(f"{output_dir}/extracted_pdfs", f) for f in os.listdir(f"{output_dir}/extracted_pdfs") if f.endswith(".pdf")]


def extract_title_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        metadata = reader.metadata or {}
        title = metadata.get("/Title", "Unknown Title")
        return title
    except Exception as e:
        print(f"⚠️ Error extracting metadata from {pdf_path}: {e}")
        return ""


def query_semantic_scholar(title):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&limit=1&fields=title,abstract,authors,year,venue,url,externalIds"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["data"]:
            paper = data["data"][0]
            return {
                "paper_id": paper.get("paperId"),
                "title": paper.get("title"),
                "abstract": paper.get("abstract", ""),
                "authors": ", ".join([a["name"] for a in paper.get("authors", [])]),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "url": paper.get("url"),
                "doi": paper.get("externalIds", {}).get("DOI", "")
            }
    return None


def extract_paper_info_from_zip(zip_path, output_dir):
    pdf_paths = unzip_papers(zip_path, output_dir)
    papers = []
    for pdf in pdf_paths:
        print(f"Processing {pdf}")
        title = extract_title_from_pdf(pdf)
        if title:
            info = query_semantic_scholar(title)
            if info:
                papers.append(info)
    export_semantic_scholar_metadata_to_excel(papers, output_dir)
    return papers




##### user enter number of papers to get from semantic scholar for ranking
def get_unique_papers_from_sch(max_total: int, search_queries_list, output_dir) -> List[Dict]:
    """
    Search Semantic Scholar for research papers matching a list of queries,
    and return a deduplicated list of paper metadata.

    Args:
        max_total (int): The maximum number of papers to retrieve in total.
        search_queries_list (List[str]): A list of search query strings.

    Returns:
        List[Dict]: A list of unique papers, where each paper is represented as a dictionary
                    containing keys: paper_id, title, abstract, authors, year, venue, url, and doi.
    """
    #search_queries_list = load_queries_list(paper_query_file_path)


    # Initialize the Semantic Scholar API
    sch = SemanticScholar()
    
    # List to store all retrieved paper metadata
    semantic_scholar_papers = []

    # Loop through each query in the list
    for query in search_queries_list:
        # Stop fetching if we've reached the maximum allowed number of papers
        if len(semantic_scholar_papers) >= max_total:
            break
        LOG.info(f"🔍 Searching Semantic Scholar for: {query}")    
        #print(f"🔍 Searching Semantic Scholar for: {query}")
        
        try:
            # Fetch up to 100 papers for the current query
            results = sch.search_paper(query, limit=100)

            # Process each paper
            for paper in results:
                if len(semantic_scholar_papers) >= max_total:
                    break

                semantic_scholar_papers.append({
                    "paper_id": paper.paperId,
                    "title": paper.title,
                    "abstract": paper.abstract or "",
                    "authors": ", ".join([a["name"] for a in paper.authors]) if paper.authors else "",
                    "year": paper.year,
                    "venue": paper.venue,
                    "url": paper.url,
                    "doi": paper.externalIds.get("DOI") if paper.externalIds else ""
                })

            # Be polite to the API :)
            sleep(1)

        except Exception as e:
            LOG.error(f"❌ Error searching '{query}': {e}")
            #print(f"❌ Error searching '{query}': {e}")
            continue

    # Remove duplicate papers based on their paper ID
    unique_papers = {p['paper_id']: p for p in semantic_scholar_papers}
    semantic_scholar_papers = list(unique_papers.values())

    LOG.info(f"✅ Retrieved {len(semantic_scholar_papers)} unique papers.")
    #print(f"✅ Retrieved {len(semantic_scholar_papers)} unique papers.")

    LOG.info(f"✅ Organized {len(semantic_scholar_papers)} papers with title-based IDs.")
    #print(f"✅ Organized {len(text_data)} papers with title-based IDs.")

    export_semantic_scholar_metadata_to_excel(semantic_scholar_papers, output_dir)
    return semantic_scholar_papers

# ✅ Save Semantic Scholar metadata to Excel
def export_semantic_scholar_metadata_to_excel(paper_list, output_dir):
    if not paper_list:
        LOG.error("⚠️ No papers to export.")
        #print("⚠️ No papers to export.")
        return

    df = pd.DataFrame(paper_list)

    # Show preview
    LOG.info("📄 Exporting the following columns:", list(df.columns))
    LOG.info(f"✅ Total papers: {len(df)}")
    #print("📄 Exporting the following columns:", list(df.columns))
    #print(f"✅ Total papers: {len(df)}")

    # Save to Excel
    df.to_excel(f"{output_dir}/semantic_scholar_papers.xlsx", index=False)
    LOG.info(f"✅ Saved to '{output_dir}/semantic_scholar_papers.xlsx'")
    #print(f"✅ Saved to '{output_dir}/semantic_scholar_papers.xlsx'")


def load_paper_data(papers) -> tuple[dict, dict, dict]:
    """
    Retrieve and organize research paper data up to a specified maximum number.

    Args:
        max_paper (int): The maximum number of unique papers to fetch.

    Returns:
        tuple:
            - text_data (dict): Maps a sanitized paper ID to its abstract text (empty string if missing).
            - title_map (dict): Maps the sanitized paper ID to the paper's title (empty string if missing).
            - abstract_map (dict): Maps the sanitized paper ID to the paper's abstract (empty string if missing).

    Description:
        This function fetches unique papers from a scholarly source, sanitizes their titles
        to create safe IDs, and organizes the abstracts and titles into dictionaries keyed
        by these safe IDs. If a paper's title or abstract is missing, an empty string is used instead.
    """
    text_data, title_map, abstract_map = {}, {}, {}
    #papers = get_unique_papers_from_sch(max_paper, paper_query_file_path, output_dir)
    for paper in papers:
        safe_id = safe_title(paper['title'] or paper['paper_id'])
        text_data[safe_id] = paper['abstract'] or ""
        title_map[safe_id] = paper['title'] or ""
        abstract_map[safe_id] = paper['abstract'] or ""
    
    #import pdb
    #pdb.set_trace()

    return text_data, title_map, abstract_map

def load_api_keys(API_file_path: str) -> List[str]:
    """
    Load API keys from a text file, one key per line.

    Args:
        API_file_path (str): Path to the file containing API keys.

    Returns:
        List[str]: A list of non-empty, stripped API keys.
    """
    # Open the file and read all non-empty lines, stripping whitespace
    with open(API_file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_queries_list(paper_query_file_path):
    with open(paper_query_file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]