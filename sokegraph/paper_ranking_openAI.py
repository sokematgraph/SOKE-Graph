from sokegraph.util.logger import LOG
from collections import defaultdict
from openai import OpenAI
from sokegraph.functions import *
import json
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Set
import pandas as pd
import json
from collections import defaultdict
from itertools import combinations
from sokegraph.util.logger import LOG

import re



def ranking_papers(
    API_keys,
    ontology_extractions,
    title_map,
    abstract_map,
    keyword_query,
    output_dir
):
    category_to_papers = defaultdict(lambda: defaultdict(int))   # {(layer, category): {paper_id: count}}
    paper_keyword_map = defaultdict(lambda: defaultdict(set))    # {(layer, category): {paper_id: set(keywords)}}
    kw_lookup = {}  # Keyword to (layer, category) lookup for quick reference

    for layer, categories in ontology_extractions.items():
        for category, items in categories.items():
            for item in items:
                paper_id = item["paper_id"]
                for kw in item["keywords"]:
                    kw_lower = kw.lower()
                    kw_lookup[kw_lower] = (layer, category)
                    paper_keyword_map[(layer, category)][paper_id].add(kw_lower)
                    category_to_papers[(layer, category)][paper_id] += 1
    
    client = OpenAI(api_key=get_next_api_key(API_keys))
    ranked, low_sorted, pair_file_path = find_common_papers(
        user_query=keyword_query,
        ontology_extractions=ontology_extractions,
        category_to_papers=category_to_papers,
        paper_keyword_map=paper_keyword_map,
        title_map=title_map,
        client=client,
        kw_lookup=kw_lookup,
        abstract_map=abstract_map,
        output_dir = output_dir,
        threshold=1.5

    )
    final_ranked_shared = rank_shared_papers_by_pairs_and_mentions(ranked, low_sorted, pair_file_path, output_dir)


#### user_query = keywords_query_list (str)
def find_common_papers(
    user_query: str,
    ontology_extractions: dict,
    category_to_papers: dict,
    paper_keyword_map: dict,
    title_map: dict,
    client,
    kw_lookup: dict,
    abstract_map: dict,
    output_dir,
    threshold: float = 1.5
) -> tuple:
    """
    Main pipeline to process a user query, classify it into ontology categories,
    filter papers dominated by opposite keywords, rank relevant papers, and
    analyze overlaps between categories.

    Args:
        user_query (str): The search query input by the user.
        ontology_extractions (dict): Ontology categories extracted for classification.
        category_to_papers (dict): Mapping from (layer, category) to papers and their counts.
        paper_keyword_map (dict): Keywords per paper, per category.
        title_map (dict): Paper titles keyed by paper ID.
        client: Client instance for LLM calls.
        kw_lookup (dict): Keyword lookup dictionary.
        abstract_map (dict): Paper abstracts keyed by paper ID.
        threshold (float): Threshold for filtering opposite keyword dominance (default 1.5).

    Returns:
        tuple: 
            - ranked (list): List of tuples (paper_id, score) for ranked papers.
            - low_sorted (list): List of filtered low relevance papers sorted by threshold exceed count.
    """
    #@Sana : the code didnt use threshold. what is that?

    LOG.info(f"🔍 User query: {user_query}")
    #print(f"🔍 User query: {user_query}")

    # Step 1: Classify the user query to ontology categories with fallback
    categories = classify_query_with_fallback(user_query, ontology_extractions, client, kw_lookup)
    if not categories:
        LOG.error("⚠️ No valid categories found.")
        #print("⚠️ No valid categories found.")
        return
    
    LOG.info(f"\n✅ Categories Used: {len(categories)}")
    #print(f"\n✅ Categories Used: {len(categories)}")
    #
    for i, (layer, cat, kw) in enumerate(categories, 1):
        print(f"  {i}. '{kw}' → {layer} / {cat}")

    # Extract and lowercase keywords for consistent matching
    query_keywords = [kw.lower() for _, _, kw in categories]

    # Step 2: Fetch opposites and synonyms for keywords via LLM
    opposites = get_opposites_via_llm(query_keywords, client)
    synonyms = get_synonyms_via_llm(query_keywords, client)

    # Build expanded keywords dictionary including synonyms
    expanded_keywords = {kw: [kw] + synonyms.get(kw, []) for kw in query_keywords}

    print(f"\n🧪 Opposites used for filtering:")
    for qk in query_keywords:
        print(f"  - '{qk}': {opposites.get(qk, [])}")

    # Step 3: Aggregate paper hits and keyword frequencies per category
    per_cat_hits = {}
    total_scores = defaultdict(int)
    paper_keyword_frequencies = defaultdict(lambda: defaultdict(int))
    filtered_out = {}

    for (layer, cat), hits in category_to_papers.items():
        if (layer, cat) not in [(l, c) for l, c, _ in categories]:
            continue

        per_cat_hits[(layer, cat)] = hits

        for pid, count in hits.items():
            total_scores[pid] += count
            for kw in paper_keyword_map[(layer, cat)][pid]:
                paper_keyword_frequencies[pid][kw] += 1

    # Step 4: Filter out papers dominated by opposite keywords
    all_paper_ids = list(title_map.keys())
    ranked = []

    for pid in all_paper_ids:
        score = total_scores.get(pid, 0)
        if not is_dominated_by_opposites(
            pid, title_map, abstract_map, query_keywords, expanded_keywords, opposites, filtered_out
        ):
            ranked.append((pid, score))

    ranked_paper_ids = {pid for pid, _ in ranked}

    # Step 5: Display report for filtered out papers, if any
    if filtered_out:
        summary_df = summarize_filtered_papers_with_opposites(
            filtered_out, query_keywords, opposites, paper_keyword_frequencies, title_map, abstract_map
        )
        try:
            from IPython.display import display
            display(summary_df)
        except ImportError:
            pass

        summary_df.to_csv(f"{output_dir}/filtered_paper_breakdown.csv", index=False)

        LOG.info(f"\n🚫 Filtered out {len(filtered_out)} papers due to dominance of opposite keywords.")
        #print(f"\n🚫 Filtered out {len(filtered_out)} papers due to dominance of opposite keywords.")

    # Step 6: Separate and sort low relevance papers (title fail)
    low_relevance = [(pid, info.get("title_exceeded", 0)) for pid, info in filtered_out.items() if info["reason"] == "title_fail"]
    low_sorted = sorted(low_relevance, key=lambda x: x[1])

    # Step 7: Print ranked paper results
    LOG.info("\n📚 Ranked Papers:")
    LOG.info("\n🟢🟡 High & Moderate Relevance:")
    #print("\n📚 Ranked Papers:")
    #print("\n🟢🟡 High & Moderate Relevance:")
    for i, (pid, score) in enumerate(sorted(ranked, key=lambda x: -x[1]), 1):
        print(f"  {i}. {pid} (mentions={score})")

    LOG.info("\n🔴 Low Relevance (sorted by number of over-threshold keywords):")
    #print("\n🔴 Low Relevance (sorted by number of over-threshold keywords):")
    for i, (pid, count) in enumerate(low_sorted, 1):
        print(f"  {i}. {pid} (keywords over threshold: {count})")

    # Step 8: Filter hits to only ranked papers for overlap analysis
    filtered_hits = {
        k: {pid: v for pid, v in d.items() if pid in ranked_paper_ids}
        for k, d in per_cat_hits.items()
    }

    
    if len(per_cat_hits) < 2:
        LOG.info("\n⚠️ Overlap analysis needs 2+ categories.")
        #print("\n⚠️ Overlap analysis needs 2+ categories.")
        return ranked, low_sorted, ""

    # Step 9: Rank papers by number of category pairs they appear in
    ranked_by_pair_overlap, total_possible_pairs = rank_by_pair_overlap_filtered(filtered_hits, ranked_paper_ids)

    LOG.info(f"\n🏆 Full Ranking by Number of Category Pairs Shared (out of {total_possible_pairs} pairs):")
    #print(f"\n🏆 Full Ranking by Number of Category Pairs Shared (out of {total_possible_pairs} pairs):")
    for i, (pid, pair_set) in enumerate(ranked_by_pair_overlap, 1):
        pair_count = len(pair_set)
        print(f"  {i}. {pid} → shared in {pair_count}/{total_possible_pairs} pair(s)")
        print(f"     ⤷ Pairs: {sorted(pair_set)}")

    if not ranked_by_pair_overlap:
        LOG.info("⚠️ No multi-category-pair overlaps found.")
        print("⚠️ No multi-category-pair overlaps found.")

    # Export overlap ranking to CSV
    df_pairs = pd.DataFrame([
        {
            "Paper ID": pid,
            "Pair Count": len(pair_set),
            "Shared Pairs": ", ".join(sorted(pair_set))
        }
        for pid, pair_set in ranked_by_pair_overlap
    ])

    df_pairs_file_path = f"{output_dir}/shared_pair_ranked_papers.csv"

    df_pairs.to_csv(df_pairs_file_path, index=False)
    LOG.info("✅ Exported shared papers to 'shared_pair_ranked_papers.csv'")
    #print("✅ Exported shared papers to 'shared_pair_ranked_papers.csv'")

    # Step 10: Show category overlap statistics
    LOG.info("\n🔄 Overlaps Between Categories:")
    #print("\n🔄 Overlaps Between Categories:")
    for (l1, c1), (l2, c2) in combinations(per_cat_hits.keys(), 2):
        pids1 = set(paper_keyword_map[(l1, c1)].keys())
        pids2 = set(paper_keyword_map[(l2, c2)].keys())
        shared = pids1 & pids2
        filtered_shared = shared & ranked_paper_ids

        total_mentions = sum(
            len(paper_keyword_map[(l1, c1)][pid]) + len(paper_keyword_map[(l2, c2)][pid])
            for pid in filtered_shared
        )
        LOG.info(f"  • '{c1}' ↔ '{c2}': {len(filtered_shared)} shared papers, {total_mentions} keyword mentions in common")
        #print(f"  • '{c1}' ↔ '{c2}': {len(filtered_shared)} shared papers, {total_mentions} keyword mentions in common")

    #import pdb
    #pdb.set_trace()
    return ranked, low_sorted, df_pairs_file_path


def rank_shared_papers_by_pairs_and_mentions(
    ranked: list,
    low_sorted: list,
    pair_file: str,
    output_dir
) -> pd.DataFrame:
    """
    Ranks shared papers by pair count and keyword relevance.

    This function combines the paper relevance scores (from keyword matching) with pair frequency data,
    then ranks them based on both metrics.

    Args:
        ranked (list): List of tuples (paper_id, keyword_score) for highly relevant papers.
        low_sorted (list): List of tuples (paper_id, count) for less relevant papers.
        pair_file (str): Path to CSV file containing 'Paper ID' and 'Pair Count' columns.

    Returns:
        pd.DataFrame: Sorted DataFrame with relevance, keyword scores, and pair counts.
    """

    # Step 1: Load paper pair data
    try:
        df_pairs = pd.read_csv(pair_file)
    except FileNotFoundError:
        LOG.error(f"❌ File not found: {pair_file}")
        #print(f"❌ File not found: {pair_file}")
        return pd.DataFrame()

    # Step 2: Build a mapping of paper_id to (relevance_level, score)
    relevance_scores = {paper_id: ("high", score) for paper_id, score in ranked}
    for paper_id, _ in low_sorted:
        if paper_id not in relevance_scores:
            relevance_scores[paper_id] = ("low", 0)

    # Step 3: Annotate DataFrame with relevance level and keyword match score
    df_pairs["Relevance Level"] = df_pairs["Paper ID"].map(lambda x: relevance_scores.get(x, ("unknown", 0))[0])
    df_pairs["Relevant Keyword Score"] = df_pairs["Paper ID"].map(lambda x: relevance_scores.get(x, ("unknown", 0))[1])

    # Step 4: Sort by pair count (descending), then by keyword score (descending)
    df_pairs = df_pairs.sort_values(
        by=["Pair Count", "Relevant Keyword Score"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Step 5: Print ranked list
    LOG.info("\n🏆 Shared Papers Ranked by Pair Count → Mentions:")
    #print("\n🏆 Shared Papers Ranked by Pair Count → Mentions:")
    for _, row in df_pairs.iterrows():
        LOG.info(f"{row['Paper ID']} | Pairs: {row['Pair Count']} | Mentions: {row['Relevant Keyword Score']} | Relevance: {row['Relevance Level']}")
        #print(f"{row['Paper ID']} | Pairs: {row['Pair Count']} | Mentions: {row['Relevant Keyword Score']} | Relevance: {row['Relevance Level']}")

    # Step 6: Save subsets by relevance level
    for level in ["high", "low", "unknown"]:
        subset = df_pairs[df_pairs["Relevance Level"] == level]
        if not subset.empty:
            filename = f"{output_dir}/shared_ranked_by_pairs_then_mentions_{level}.csv"
            subset.to_csv(filename, index=False)
            LOG.info(f"✅ Saved: {filename}")
            #print(f"✅ Saved: {filename}")

    #import pdb
    #pdb.set_trace()
    return df_pairs


def classify_query_with_fallback(user_query: str, ontology: dict, client, kw_lookup: dict) -> list:
    """
    Classify a user query into ontology categories using OpenAI, with fallback logic.

    Parameters:
    - user_query (str): The natural language query input from the user.
    - ontology (dict): A nested dictionary representing the ontology structure.
    - client: An OpenAI API client instance for calling GPT-4o.
    - kw_lookup (dict): A dictionary mapping keywords to (layer, category) for fallback classification.

    Returns:
    - categories (list): A list of tuples (layer, category, keyword) representing matched ontology classifications.
    """

    # Tokenize the query: extract all alphanumeric tokens longer than one character
    tokens = [t for t in re.findall(r"\w+", user_query.lower()) if len(t) > 1]

    # Build a simplified summary of the ontology for inclusion in the prompt
    ontology_summary = {layer: list(cats.keys()) for layer, cats in ontology.items()}

    # Create the prompt to send to OpenAI's chat model
    prompt = f'''
        You are a structured classification AI for materials science.
        Ontology:
        {json.dumps(ontology_summary, indent=2)}

        Given the user query: "{user_query}"

        Extract each meaningful keyword (ignore stopwords like 'and', 'for', etc).
        Assign each keyword to the most relevant category and layer from the ontology.
        Return only the results in valid JSON format:

        [
        {{
            "keyword": "cheap",
            "category": "Earth-Abundant Elements",
            "layer": "Elemental Composition"
        }},
        ...
        ]
    '''

    # Call OpenAI's GPT model to classify keywords
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    # Extract and clean the model's output
    result = response.choices[0].message.content.strip()
    result = result.replace("```json", "").replace("```", "")

    # Parse the model's JSON output
    try:
        model_output = json.loads(result)
    except json.JSONDecodeError:
        print("⚠️ OpenAI output couldn't be parsed. Raw:\n", result)
        model_output = []

    categories = []        # Final list of classified keywords
    found_set = set()      # Tracks (layer, category) to avoid duplicates
    model_keywords = set() # Tracks which keywords were already classified by the model

    # Process the model's output
    for item in model_output:
        kw = item["keyword"].lower()
        model_keywords.add(kw)
        if item["layer"] != "Unknown" and item["category"] != "Unknown":
            if (item["layer"], item["category"]) not in found_set:
                categories.append((item["layer"], item["category"], kw))
                found_set.add((item["layer"], item["category"]))
                print(f"✅ OpenAI: '{kw}' → {item['layer']} / {item['category']}")

    # Fallback for unclassified keywords using the local lookup
    for tok in tokens:
        if tok not in model_keywords:
            if tok in kw_lookup:
                layer, cat = kw_lookup[tok]
                if (layer, cat) not in found_set:
                    categories.append((layer, cat, tok))
                    found_set.add((layer, cat))
                    print(f"🛟 Fallback: '{tok}' → {layer} / {cat}")
            else:
                print(f"❌ No match for: '{tok}'")

    return categories


def get_opposites_via_llm(keywords: List[str], client: Any) -> Dict[str, List[str]]:
    """
    Uses an LLM (e.g., OpenAI GPT-4o) to generate opposing or contradictory keywords
    for a given list of scientific keywords, specifically in the context of
    electrocatalysis, water splitting, or electrochemical environments.

    Parameters:
    - keywords (List[str]): List of scientific keywords (e.g., "acidic", "HER").
    - client (Any): OpenAI client instance used to send the prompt.

    Returns:
    - Dict[str, List[str]]: A dictionary where each keyword maps to a list of opposing terms.
    """
    
    # Construct the prompt for the language model
    prompt = f"""
        You are a materials science assistant.

        For each of the following scientific keywords, return a list of their opposite or contradictory concepts 
        in the context of electrocatalysis, water splitting, or electrochemical environments.

        Keywords: {keywords}

        Respond only in JSON format like this:
        {{
          "acidic": ["alkaline", "basic"],
          "HER": ["OER", "oxygen evolution"],
          ...
        }}
    """

    # Send prompt to the LLM
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # Zero temperature for deterministic output
    )

    # Clean the response content
    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    # Attempt to parse the response as JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("⚠️ Could not parse opposite keyword JSON. Raw output:\n", content)
        return {}


def get_synonyms_via_llm(keywords: List[str], client: Any) -> Dict[str, List[str]]:
    """
    Uses an LLM (e.g., OpenAI GPT-4o) to generate synonyms or semantically similar terms
    for a given list of scientific keywords. This is specific to language used in 
    electrocatalysis, water splitting, or electrochemical contexts.

    Parameters:
    - keywords (List[str]): A list of scientific terms (e.g., "HER", "acidic").
    - client (Any): The OpenAI API client used to submit the query.

    Returns:
    - Dict[str, List[str]]: A dictionary mapping each input keyword to a list of synonyms.
    """

    # Create a clear and structured prompt for the model
    prompt = f"""
        You are a materials science assistant.

        For each of the following scientific keywords, return a list of their synonyms or semantically 
        similar expressions used in electrocatalysis, water splitting, or electrochemistry papers.

        Keywords: {keywords}

        Respond only in JSON format like this:
        {{
          "acidic": ["low pH", "acid media"],
          "HER": ["hydrogen evolution", "H2 generation"],
          ...
        }}
    """

    # Make the API call to the OpenAI chat completion model
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # Deterministic output
    )

    # Clean up the raw model response
    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    # Attempt to parse the response as JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("⚠️ Could not parse synonym keyword JSON. Raw output:\n", content)
        return {}
    


def is_dominated_by_opposites(
    paper_id: str,
    title_map: Dict[str, str],
    abstract_map: Dict[str, str],
    query_keywords: List[str],
    expanded_keywords: Dict[str, List[str]],
    opposites: Dict[str, List[str]],
    filtered_out: Dict[str, Dict]
) -> bool:
    """
    Determines whether a paper should be filtered out based on whether opposing keywords
    dominate relevant ones in the title and abstract.

    Parameters:
    - paper_id: Unique identifier of the paper.
    - title_map: Mapping from paper ID to title text.
    - abstract_map: Mapping from paper ID to abstract text.
    - query_keywords: List of original query keywords.
    - expanded_keywords: Dictionary mapping each keyword to its expanded synonym list.
    - opposites: Dictionary mapping each keyword to its list of opposing terms.
    - filtered_out: Dictionary to record reasons for filtering papers.

    Returns:
    - True if the paper is filtered out (dominated by opposites), otherwise False.
    """

    # Normalize and prepare title and abstract (excluding references section)
    title_text = (title_map.get(paper_id, "") or "").lower()
    full_abstract = (abstract_map.get(paper_id, "") or "").lower()
    abstract_text = re.split(r'(references|refs?\.?:)', full_abstract)[0]

    title_flags = {}
    title_over_threshold = 0

    # Analyze keyword relevance and opposition in title
    for qk in query_keywords:
        relevant_terms = expanded_keywords.get(qk, [qk])
        title_rel = sum(len(re.findall(rf'\b{re.escape(term)}\b', title_text)) for term in relevant_terms)
        title_opp = sum(len(re.findall(rf'\b{re.escape(opp)}\b', title_text)) for opp in opposites.get(qk, []))

        if title_rel == 0 and title_opp == 0:
            title_flags[qk] = "missing"
        else:
            ratio = float('inf') if title_rel == 0 else title_opp / title_rel
            if ratio > 1:
                title_flags[qk] = "fail"
                title_over_threshold += 1
            elif ratio in [0, 1]:
                title_flags[qk] = "pass"
            else:
                title_flags[qk] = "other"

    # 🔴 Case 1: Low relevance - Opposing terms dominate in title
    if "fail" in title_flags.values():
        filtered_out[paper_id] = {
            "reason": "title_fail",
            "title_flags": title_flags,
            "title_exceeded": title_over_threshold,
            "threshold": 1.0,
        }
        return True

    # 🟡 Case 2: Moderate relevance - Mixed signal in title, check abstract
    if "pass" in title_flags.values() and "missing" in title_flags.values():
        abstract_ratios = []

        for qk in query_keywords:
            relevant_terms = expanded_keywords.get(qk, [qk])
            abs_rel = sum(len(re.findall(rf'\b{re.escape(term)}\b', abstract_text)) for term in relevant_terms)
            abs_opp = sum(len(re.findall(rf'\b{re.escape(opp)}\b', abstract_text)) for opp in opposites.get(qk, []))
            ratio = float('inf') if abs_rel == 0 else abs_opp / abs_rel
            abstract_ratios.append(ratio)

        if all(r <= 1.5 for r in abstract_ratios):
            return False  # Considered high relevance despite title mix
        else:
            filtered_out[paper_id] = {
                "reason": "moderate_abstract_check",
                "abstract_ratios": abstract_ratios,
                "title_exceeded": 0,
                "threshold": 1.5,
            }
            return False  # Not excluded, but marked for review

    # 🟢 Case 3: High relevance - All keywords pass in title
    if all(flag == "pass" for flag in title_flags.values()):
        return False

    # Default fallback: Keep the paper
    return False

def summarize_filtered_papers_with_opposites(
    filtered_out: Dict[str, Dict[str, Any]],
    query_keywords: List[str],
    opposites: Dict[str, List[str]],
    paper_keyword_frequencies: Dict[str, Dict[str, int]],
    title_map: Dict[str, str],
    abstract_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Summarizes why certain papers were filtered out based on presence of opposing keywords.

    Parameters:
    - filtered_out (dict): Dictionary of paper IDs mapped to filtering metadata (e.g., threshold).
    - query_keywords (list): List of relevant query keywords used for matching.
    - opposites (dict): Dictionary mapping each query keyword to its list of opposing keywords.
    - paper_keyword_frequencies (dict): Not used in the current implementation but may be needed for extensions.
    - title_map (dict): Maps paper IDs to their title texts.
    - abstract_map (dict): Maps paper IDs to their abstract texts.

    Returns:
    - pd.DataFrame: A summary table showing keyword matches in title/abstract and reasons for filtering.
    """

    rows = []

    # Loop through each filtered-out paper
    for pid, info in filtered_out.items():
        title_text = (title_map.get(pid, "") or "").lower()
        abstract_text = (abstract_map.get(pid, "") or "").lower()

        # Evaluate each keyword from the user query
        for qk in query_keywords:
            matched_opp_keywords = []

            # --- Title Matching ---
            # Count exact matches of the query keyword in title
            title_rel = len(re.findall(rf'\b{re.escape(qk)}\b', title_text))
            title_opp = 0

            # Count opposing keywords in title
            for opp in opposites.get(qk, []):
                count = len(re.findall(rf'\b{re.escape(opp)}\b', title_text))
                title_opp += count
                if count > 0:
                    matched_opp_keywords.append(opp)

            # --- Abstract Matching ---
            # Count query keyword matches in abstract
            abs_rel = len(re.findall(rf'\b{re.escape(qk)}\b', abstract_text))
            abs_opp = 0

            # Count opposing keywords in abstract
            for opp in opposites.get(qk, []):
                count = len(re.findall(rf'\b{re.escape(opp)}\b', abstract_text))
                abs_opp += count
                if count > 0 and opp not in matched_opp_keywords:
                    matched_opp_keywords.append(opp)

            # Total counts
            total_rel = title_rel + abs_rel
            total_opp = title_opp + abs_opp

            # Compute ratio of opposing to relevant keywords
            ratio = round(total_opp / total_rel, 2) if total_rel else float('inf')

            # Determine filtering status
            status = "Filtered" if ratio > info["threshold"] else "Kept"

            # Collect results
            rows.append({
                "Paper ID": pid,
                "Query Keyword": qk,
                "Title Relevant Count": title_rel,
                "Title Opposing Count": title_opp,
                "Abstract Relevant Count": abs_rel,
                "Abstract Opposing Count": abs_opp,
                "Total Relevant Count": total_rel,
                "Total Opposing Count": total_opp,
                "Matched Opposing Keywords": ", ".join(sorted(set(matched_opp_keywords))),
                "Ratio": ratio,
                "Status": status
            })

    return pd.DataFrame(rows)

def rank_by_pair_overlap_filtered(
    per_cat_hits: Dict[Tuple[str, str], Dict[str, float]],
    ranked_paper_ids: Set[str]
) -> Tuple[List[Tuple[str, Set[str]]], int]:
    """
    Ranks papers based on how many unique category-pair overlaps they appear in,
    considering only a filtered set of paper IDs.

    Parameters:
    - per_cat_hits: A dictionary where each key is a (layer, category) tuple and
      the value is another dict mapping paper IDs to their relevance scores.
    - ranked_paper_ids: A set of paper IDs that have passed some prior filtering.

    Returns:
    - ranked_by_pair_overlap: A list of tuples (paper ID, set of category pair labels),
      sorted by the number of unique category-pair overlaps in descending order.
    - total_possible_pairs: The total number of unique category-category pairs.
    """

    pair_paper_map = defaultdict(set)  # Maps each paper ID to the set of category pairs it overlaps in
    category_pairs = list(combinations(per_cat_hits.keys(), 2))  # All unique pairs of (layer, category)
    total_possible_pairs = len(category_pairs)

    for (cat1, cat2) in category_pairs:
        papers1 = set(per_cat_hits[cat1].keys())  # All paper IDs under category 1
        papers2 = set(per_cat_hits[cat2].keys())  # All paper IDs under category 2
        shared_papers = papers1 & papers2         # Papers common to both categories
        filtered_shared = shared_papers & ranked_paper_ids  # Only consider filtered/ranked paper IDs

        pair_label = f"{cat1[1]} ↔ {cat2[1]}"  # Use only category names for display

        for pid in filtered_shared:
            pair_paper_map[pid].add(pair_label)  # Record that this paper overlaps this pair

    # Rank papers by number of unique overlapping category pairs (descending)
    ranked_by_pair_overlap = sorted(pair_paper_map.items(), key=lambda x: len(x[1]), reverse=True)

    return ranked_by_pair_overlap, total_possible_pairs