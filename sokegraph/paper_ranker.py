"""
paper_ranker.py

Implements :class:`PaperRanker`, which scores a list of research papers
for relevance to a user keyword query using:

1. An ontology of layers → categories → keywords
2. An :class:`~sokegraph.ai_agent.AIAgent` to classify the query and expand
   keywords (synonyms / opposites)
3. Keyword‑mention counts in titles & abstracts
4. Pair‑overlap scores across different categories

CSV reports are exported to `output_dir`.
"""

import json
import re
import pandas as pd
from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Any, Tuple, Set

from sokegraph.util.logger import LOG
from sokegraph.functions import load_keyword, safe_title


from sokegraph.ai_agent import AIAgent

class PaperRanker:
    def __init__(self, ai_tool: AIAgent, papers, ontology_path, keyword_path, output_dir: str):
        """
        Initialize the PaperRanker.

        Parameters
        ----------
        ai_tool : AIAgent
            AI agent for classification and synonym/opposite expansion.
        papers : list[dict]
            List of paper metadata (each must contain 'title', 'abstract', 'paper_id').
        ontology_path : str
            Path to the ontology JSON file.
        keyword_path : str
            Path to the user keyword query text file.
        output_dir : str
            Directory where output CSVs will be saved.
        """
        self.ai_tool = ai_tool
        self.papers = papers
        self.ontology_path = ontology_path
        self.ontology = None  # Will be set by _load_ontology
        self._load_ontology()
        self.output_dir = output_dir
        self.keyword_query = load_keyword(keyword_path)



    def _load_ontology(self):
        """Load ontology JSON from self.ontology_path into self.ontology."""
        try:
            with open(self.ontology_path, "r", encoding="utf-8") as f:
                self.ontology = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load ontology from '{self.ontology_path}': {e}")


    def rank_papers(self) -> List[Tuple[str, float]]:
        """
        Ranks papers based on their overlap with ontology-extracted keywords.

        Returns:
            A list of (paper_id, score) tuples, ranked by relevance.
        """

        # Extract mappings from paper ID to title and abstract
        title_map = self._get_title_map()
        abstract_map = self._get_abstract_map()

        # Dictionary to count how many times each paper is linked to a category (by keyword)
        category_to_papers = defaultdict(lambda: defaultdict(int))

        # Maps each (layer, category) to a set of keywords found per paper
        paper_keyword_map = defaultdict(lambda: defaultdict(set))

        # Lookup dictionary to map keyword → (layer, category) for quick reverse lookup
        kw_lookup = {}

        # Iterate over all layers and categories in the ontology
        for layer, categories in self.ontology.items():
            for category, items in categories.items():
                for item in items:
                    paper_id = item["paper_id"]
                    for kw in item["keywords"]:
                        kw_lower = kw.lower()  # Normalize keyword to lowercase
                        kw_lookup[kw_lower] = (layer, category)

                        # Track which keywords are linked to this paper in this category
                        paper_keyword_map[(layer, category)][paper_id].add(kw_lower)

                        # Increment count of how many times this paper maps to this category
                        category_to_papers[(layer, category)][paper_id] += 1

        # Find candidate papers that match ontology categories and keywords
        ranked, low_sorted, pair_file_path = self.find_common_papers(
            category_to_papers,
            paper_keyword_map,
            title_map,
            abstract_map,
            kw_lookup,
        )

        # Rank those candidate papers by their pairwise shared categories and keyword mentions
        return self.rank_shared_papers_by_pairs_and_mentions(
            ranked, low_sorted, pair_file_path, self.output_dir
        )


    def find_common_papers(
        self,
        category_to_papers: dict,
        paper_keyword_map: dict,
        title_map: dict,
        kw_lookup: dict,
        abstract_map: dict,
        threshold: float = 1.5,
    ) -> tuple:
        """
        Finds papers matching the user's query based on keyword overlap, filters them using opposites,
        and ranks them by category overlap and keyword relevance.

        Returns:
            - ranked: list of (paper_id, score) with high or moderate relevance
            - low_sorted: list of (paper_id, count) with low relevance (e.g., title fails)
            - df_pairs_file_path: path to saved CSV of shared category pair overlaps
        """

        # Step 1: Log and process the user's keyword query
        LOG.info(f"🔍 User query: {self.keyword_query}")

        # Step 2: Classify the query into categories using keyword-to-ontology matching
        categories = self.classify_query_with_fallback(self.keyword_query, kw_lookup)
        if not categories:
            LOG.error("⚠️ No valid categories found.")
            return [], [], ""

        # Step 3: Print the matching categories
        LOG.info(f"\n✅ Categories Used: {len(categories)}")
        for i, (layer, cat, kw) in enumerate(categories, 1):
            print(f"  {i}. '{kw}' → {layer} / {cat}")

        # Step 4: Normalize keywords and expand with opposites and synonyms
        query_keywords = [kw.lower() for _, _, kw in categories]
        opposites = self.ai_tool.get_opposites(query_keywords)
        synonyms = self.ai_tool.get_synonyms(query_keywords)
        expanded_keywords = {kw: [kw] + synonyms.get(kw, []) for kw in query_keywords}

        print(f"\n🧪 Opposites used for filtering:")
        for qk in query_keywords:
            print(f"  - '{qk}': {opposites.get(qk, [])}")

        # Step 5: Gather keyword hits per category, and compute relevance scores
        per_cat_hits = {}
        total_scores = defaultdict(int)
        paper_keyword_frequencies = defaultdict(lambda: defaultdict(int))
        filtered_out = {}

        for (layer, cat), hits in category_to_papers.items():
            if (layer, cat) not in [(l, c) for l, c, _ in categories]:
                continue  # Skip irrelevant categories

            per_cat_hits[(layer, cat)] = hits
            for pid, count in hits.items():
                total_scores[pid] += count
                for kw in paper_keyword_map[(layer, cat)][pid]:
                    paper_keyword_frequencies[pid][kw] += 1

        # Step 6: Score and filter papers
        all_paper_ids = list(title_map.keys())
        ranked = []

        for pid in all_paper_ids:
            score = total_scores.get(pid, 0)
            if not self.is_dominated_by_opposites(
                pid, title_map, abstract_map, query_keywords, expanded_keywords, opposites, filtered_out
            ):
                ranked.append((pid, score))

        ranked_paper_ids = {pid for pid, _ in ranked}

        # Step 7: Summarize and export filtered-out papers
        if filtered_out:
            summary_df = PaperRanker.summarize_filtered_papers_with_opposites(
                filtered_out, query_keywords, opposites,
                paper_keyword_frequencies, title_map, abstract_map
            )
            try:
                from IPython.display import display
                display(summary_df)
            except ImportError:
                pass

            summary_df.to_csv(f"{self.output_dir}/filtered_paper_breakdown.csv", index=False)
            LOG.info(f"\n🚫 Filtered out {len(filtered_out)} papers due to dominance of opposite keywords.")

        # Step 8: Handle low-relevance papers (e.g., those filtered by title)
        low_relevance = [
            (pid, info.get("title_exceeded", 0))
            for pid, info in filtered_out.items() if info["reason"] == "title_fail"
        ]
        low_sorted = sorted(low_relevance, key=lambda x: x[1])

        # Step 9: Print ranked papers
        LOG.info("\n📚 Ranked Papers:")
        LOG.info("\n🟢🟡 High & Moderate Relevance:")
        for i, (pid, score) in enumerate(sorted(ranked, key=lambda x: -x[1]), 1):
            print(f"  {i}. {pid} (mentions={score})")

        LOG.info("\n🔴 Low Relevance (sorted by number of over-threshold keywords):")
        for i, (pid, count) in enumerate(low_sorted, 1):
            print(f"  {i}. {pid} (keywords over threshold: {count})")

        # Step 10: Filter hits for papers that survived opposite keyword filtering
        filtered_hits = {
            k: {pid: v for pid, v in d.items() if pid in ranked_paper_ids}
            for k, d in per_cat_hits.items()
        }

        # Step 11: If too few categories, skip pairwise ranking
        if len(per_cat_hits) < 2:
            LOG.info("\n⚠️ Overlap analysis needs 2+ categories.")
            return ranked, low_sorted, ""

        # Step 12: Rank papers based on how many category pairs they are shared in
        ranked_by_pair_overlap, total_possible_pairs = PaperRanker.rank_by_pair_overlap_filtered(
            filtered_hits, ranked_paper_ids
        )

        LOG.info(f"\n🏆 Full Ranking by Number of Category Pairs Shared (out of {total_possible_pairs} pairs):")
        for i, (pid, pair_set) in enumerate(ranked_by_pair_overlap, 1):
            print(f"  {i}. {pid} → shared in {len(pair_set)}/{total_possible_pairs} pair(s)")
            print(f"     ⤷ Pairs: {sorted(pair_set)}")

        if not ranked_by_pair_overlap:
            LOG.info("⚠️ No multi-category-pair overlaps found.")

        # Step 13: Export overlap summary to CSV
        df_pairs = pd.DataFrame([
            {
                "Paper ID": pid,
                "Pair Count": len(pair_set),
                "Shared Pairs": ", ".join(sorted(pair_set))
            }
            for pid, pair_set in ranked_by_pair_overlap
        ])
        df_pairs_file_path = f"{self.output_dir}/shared_pair_ranked_papers.csv"
        df_pairs.to_csv(df_pairs_file_path, index=False)
        LOG.info("✅ Exported shared papers to 'shared_pair_ranked_papers.csv'")

        # Step 14: Print detailed overlap between categories
        LOG.info("\n🔄 Overlaps Between Categories:")
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

        # Step 15: Return ranked papers, low-relevance list, and path to pairwise results CSV
        return ranked, low_sorted, df_pairs_file_path


    def classify_query_with_fallback(self, user_query: str, kw_lookup: dict) -> list:
        """
        Attempts to classify user query keywords into ontology categories using an AI tool,
        and falls back to keyword lookup if the AI classification fails or misses keywords.

        Args:
            user_query (str): User's free-text query.
            kw_lookup (dict): Fallback dictionary mapping keyword → (layer, category).

        Returns:
            List of tuples: (layer, category, keyword)
        """
        import re

        # Step 1: Tokenize query into words (ignore single letters)
        tokens = [t for t in re.findall(r"\w+", user_query.lower()) if len(t) > 1]

        # Step 2: Create a summary of the ontology layers and their categories (used in the prompt)
        ontology_summary = {layer: list(cats.keys()) for layer, cats in self.ontology.items()}

        # Step 3: Build the AI prompt for structured classification
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

        # Step 4: Send prompt to the AI agent and clean the response
        response = self.ai_tool.ask(prompt)
        response = response.strip().replace("```json", "").replace("```", "")
        print("response : ", response)

        # Step 5: Attempt to parse the AI response as JSON
        try:
            model_output = json.loads(response)
        except json.JSONDecodeError:
            print("⚠️ AI output couldn't be parsed. Raw:\n", response)
            model_output = []

        # Step 6: Parse valid entries from the AI output
        categories = []
        found_set = set()        # To avoid duplicates by (layer, category)
        model_keywords = set()   # Track which keywords were handled by the model

        for item in model_output:
            kw = item["keyword"].lower()
            model_keywords.add(kw)

            # Accept only known classifications
            if item["layer"] != "Unknown" and item["category"] != "Unknown":
                if (item["layer"], item["category"]) not in found_set:
                    categories.append((item["layer"], item["category"], kw))
                    found_set.add((item["layer"], item["category"]))
                    print(f"✅ OpenAI: '{kw}' → {item['layer']} / {item['category']}")

        # Step 7: Fallback logic — try to classify remaining tokens manually using kw_lookup
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

        # Step 8: Return the final list of classified (layer, category, keyword) entries
        return categories



    def is_dominated_by_opposites(
        self,
        paper_id: str,
        title_map: Dict[str, str],
        abstract_map: Dict[str, str],
        query_keywords: List[str],
        expanded_keywords: Dict[str, List[str]],
        opposites: Dict[str, List[str]],
        filtered_out: Dict[str, Dict]
    ) -> bool:
        """
        Decide whether a paper should be filtered out because “opposite” terms
        outweigh relevant terms in its title (and sometimes abstract).

        Returns
        -------
        bool
            True  → paper is dominated by opposites and should be excluded.
            False → paper passes the opposite‑keyword check.
        """

        # ------------------------------------------------------------------ #
        # 1.  Pre‑processing: normalize title and abstract to lowercase       #
        # ------------------------------------------------------------------ #
        title_text = (title_map.get(paper_id, "") or "").lower()
        full_abstract = (abstract_map.get(paper_id, "") or "").lower()

        # Strip everything after the references section to avoid noise
        abstract_text = re.split(r'(references|refs?\.?:)', full_abstract)[0]

        # Per‑keyword status flags for the title
        title_flags: Dict[str, str] = {}
        title_over_threshold = 0  # Count keywords where opposites > relevant

        # ------------------------------------------------------------------ #
        # 2.  Scan title for relevant vs. opposite keywords                   #
        # ------------------------------------------------------------------ #
        for qk in query_keywords:
            relevant_terms = expanded_keywords.get(qk, [qk])
            opp_terms = opposites.get(qk, [])

            # Count exact‑word matches for relevant and opposite terms
            title_rel = sum(
                len(re.findall(rf'\b{re.escape(term)}\b', title_text))
                for term in relevant_terms
            )
            title_opp = sum(
                len(re.findall(rf'\b{re.escape(opp)}\b', title_text))
                for opp in opp_terms
            )

            # Flag logic: missing / pass / fail / other
            if title_rel == 0 and title_opp == 0:
                title_flags[qk] = "missing"
            else:
                ratio = float('inf') if title_rel == 0 else title_opp / title_rel
                if ratio > 1:
                    title_flags[qk] = "fail"
                    title_over_threshold += 1
                elif ratio in (0, 1):
                    title_flags[qk] = "pass"
                else:
                    title_flags[qk] = "other"

        # ------------------------------------------------------------------ #
        # 3.  Hard filter: opposites clearly dominate in the title           #
        # ------------------------------------------------------------------ #
        if "fail" in title_flags.values():
            filtered_out[paper_id] = {
                "reason": "title_fail",
                "title_flags": title_flags,
                "title_exceeded": title_over_threshold,
                "threshold": 1.0,
            }
            return True  # Paper rejected

        # ------------------------------------------------------------------ #
        # 4.  Mixed case: some keywords pass, others missing → inspect abs.  #
        # ------------------------------------------------------------------ #
        if "pass" in title_flags.values() and "missing" in title_flags.values():
            abstract_ratios = []
            for qk in query_keywords:
                relevant_terms = expanded_keywords.get(qk, [qk])
                opp_terms = opposites.get(qk, [])

                abs_rel = sum(
                    len(re.findall(rf'\b{re.escape(term)}\b', abstract_text))
                    for term in relevant_terms
                )
                abs_opp = sum(
                    len(re.findall(rf'\b{re.escape(opp)}\b', abstract_text))
                    for opp in opp_terms
                )
                ratio = float('inf') if abs_rel == 0 else abs_opp / abs_rel
                abstract_ratios.append(ratio)

            # If opposites are moderate (≤ 1.5× relevant) we keep the paper
            if all(r <= 1.5 for r in abstract_ratios):
                return False
            else:
                # Otherwise mark for informational filtering but don’t hard‑fail
                filtered_out[paper_id] = {
                    "reason": "moderate_abstract_check",
                    "abstract_ratios": abstract_ratios,
                    "title_exceeded": 0,
                    "threshold": 1.5,
                }
                return False  # Still keep the paper for ranking

        # ------------------------------------------------------------------ #
        # 5.  If every keyword passes in the title, the paper is good        #
        # ------------------------------------------------------------------ #
        if all(flag == "pass" for flag in title_flags.values()):
            return False

        # ------------------------------------------------------------------ #
        # 6.  Default: do not filter out                                     #
        # ------------------------------------------------------------------ #
        return False

    
    @staticmethod
    def summarize_filtered_papers_with_opposites(
        filtered_out: Dict[str, Dict[str, Any]],
        query_keywords: List[str],
        opposites: Dict[str, List[str]],
        paper_keyword_frequencies: Dict[str, Dict[str, int]],
        title_map: Dict[str, str],
        abstract_map: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Build a DataFrame summarising *why* each paper was flagged as dominated
        by opposite keywords.

        For every paper in ``filtered_out`` and for every query keyword:

        1. Count occurrences of the query keyword and its opposites
        separately in the **title** and **abstract**.
        2. Compute total relevant vs. total opposite counts and
        their ratio.
        3. Label the row as "Filtered" if the ratio exceeded the threshold
        stored in ``filtered_out[paper_id]['threshold']``.

        The resulting table is useful for debugging keyword filters and is
        exported to CSV by the caller.

        Returns
        -------
        pd.DataFrame
            One row per (paper, query_keyword) combination with detailed counts.
        """

        rows = []  # Collect one dict per row for DataFrame construction

        # -------------------------------------------------------------- #
        # Iterate over every paper that was filtered out                 #
        # -------------------------------------------------------------- #
        for pid, info in filtered_out.items():
            # Retrieve and normalise text
            title_text = (title_map.get(pid, "") or "").lower()
            abstract_text = (abstract_map.get(pid, "") or "").lower()

            # ---------------------------------------------------------- #
            # For each query keyword, count relevant vs. opposite terms  #
            # ---------------------------------------------------------- #
            for qk in query_keywords:
                matched_opp_keywords = []  # Track which opposites were found

                # ---- Title counts ----
                title_rel = len(re.findall(rf'\b{re.escape(qk)}\b', title_text))
                title_opp = 0
                for opp in opposites.get(qk, []):
                    count = len(re.findall(rf'\b{re.escape(opp)}\b', title_text))
                    title_opp += count
                    if count > 0:
                        matched_opp_keywords.append(opp)

                # ---- Abstract counts ----
                abs_rel = len(re.findall(rf'\b{re.escape(qk)}\b', abstract_text))
                abs_opp = 0
                for opp in opposites.get(qk, []):
                    count = len(re.findall(rf'\b{re.escape(opp)}\b', abstract_text))
                    abs_opp += count
                    # Avoid double‑adding a keyword already found in title
                    if count > 0 and opp not in matched_opp_keywords:
                        matched_opp_keywords.append(opp)

                # ---- Aggregate counts & ratio ----
                total_rel = title_rel + abs_rel
                total_opp = title_opp + abs_opp
                ratio = round(total_opp / total_rel, 2) if total_rel else float("inf")
                status = "Filtered" if ratio > info["threshold"] else "Kept"

                # ---- Append row ----
                rows.append(
                    {
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
                        "Status": status,
                    }
                )

        # Convert list-of-dicts to a DataFrame for convenient analysis / export
        return pd.DataFrame(rows)

    @staticmethod
    def rank_by_pair_overlap_filtered(
        per_cat_hits: Dict[Tuple[str, str], Dict[str, float]],
        ranked_paper_ids: Set[str]
    ) -> Tuple[List[Tuple[str, Set[str]]], int]:
        """
        Ranks papers based on how many unique category-pair overlaps they appear in,
        considering only a filtered set of paper IDs.

        Parameters
        ----------
        per_cat_hits : dict
            Maps each (layer, category) to a dictionary of {paper_id: score} for that category.
        ranked_paper_ids : set
            Set of paper IDs that passed earlier relevance filtering and are eligible for ranking.

        Returns
        -------
        ranked_by_pair_overlap : list of tuples
            Each tuple contains (paper_id, set of category pair labels where it appears).
            Sorted in descending order of how many pairs the paper is shared across.
        total_possible_pairs : int
            The number of unique category pairs considered.
        """

        pair_paper_map = defaultdict(set)  # paper_id → set of category-pair labels where it appears
        category_pairs = list(combinations(per_cat_hits.keys(), 2))  # all unique pairs of categories
        total_possible_pairs = len(category_pairs)

        # Iterate through each unique pair of categories
        for (cat1, cat2) in category_pairs:
            # Get the set of paper IDs in each category
            papers1 = set(per_cat_hits[cat1].keys())
            papers2 = set(per_cat_hits[cat2].keys())

            # Find papers that appear in both categories
            shared_papers = papers1 & papers2

            # Only keep papers that are in the ranked set
            filtered_shared = shared_papers & ranked_paper_ids

            # Create a human-readable label for the category pair (e.g., "Conductivity ↔ Stability")
            pair_label = f"{cat1[1]} ↔ {cat2[1]}"

            # For each shared paper, record this category-pair label
            for pid in filtered_shared:
                pair_paper_map[pid].add(pair_label)

        # Rank papers by how many unique category-pairs they appear in
        ranked_by_pair_overlap = sorted(
            pair_paper_map.items(),
            key=lambda x: len(x[1]),  # sort by number of pairs
            reverse=True
        )

        return ranked_by_pair_overlap, total_possible_pairs


    @staticmethod
    def rank_shared_papers_by_pairs_and_mentions(
        ranked: List[Tuple[str, float]],
        low_sorted: List[Tuple[str, float]],
        pair_file: str,
        output_dir: str
    ) -> pd.DataFrame:
        """
        Ranks shared papers by both category pair count and keyword relevance.

        This method merges data from two dimensions:
        - How many category pairs a paper appears in (interdisciplinary relevance)
        - How many keyword mentions it had (topical relevance)

        It then ranks papers with the aim of identifying those that are not only
        broadly relevant across topics but also rich in relevant keyword usage.

        Parameters
        ----------
        ranked : list of tuples
            List of (paper_id, keyword_score) for papers deemed highly or moderately relevant.
        low_sorted : list of tuples
            List of (paper_id, count) for low-relevance papers.
        pair_file : str
            Path to a CSV file with "Paper ID" and "Pair Count" columns.
        output_dir : str
            Directory to save ranked subsets of the shared papers.

        Returns
        -------
        pd.DataFrame
            A DataFrame ranked by pair count and keyword score, with relevance labels.
        """

        # Step 1: Load the CSV file that contains how many category-pairs each paper appears in
        try:
            df_pairs = pd.read_csv(pair_file)
        except FileNotFoundError:
            LOG.error(f"❌ File not found: {pair_file}")
            return pd.DataFrame()

        # Step 2: Build a lookup mapping from paper ID to (relevance level, keyword score)
        relevance_scores = {paper_id: ("high", score) for paper_id, score in ranked}
        for paper_id, _ in low_sorted:
            # Mark papers not already ranked as "low" relevance with score 0
            if paper_id not in relevance_scores:
                relevance_scores[paper_id] = ("low", 0)

        # Step 3: Add two new columns to df_pairs
        # - "Relevance Level": either "high", "low", or "unknown"
        # - "Relevant Keyword Score": keyword score (number of mentions, etc.)
        df_pairs["Relevance Level"] = df_pairs["Paper ID"].map(lambda x: relevance_scores.get(x, ("unknown", 0))[0])
        df_pairs["Relevant Keyword Score"] = df_pairs["Paper ID"].map(lambda x: relevance_scores.get(x, ("unknown", 0))[1])

        # Step 4: Sort the DataFrame first by pair count, then by keyword score (both descending)
        df_pairs = df_pairs.sort_values(
            by=["Pair Count", "Relevant Keyword Score"],
            ascending=[False, False]
        ).reset_index(drop=True)

        # Step 5: Display ranked results
        LOG.info("\n🏆 Shared Papers Ranked by Pair Count → Mentions:")
        for _, row in df_pairs.iterrows():
            LOG.info(f"{row['Paper ID']} | Pairs: {row['Pair Count']} | Mentions: {row['Relevant Keyword Score']} | Relevance: {row['Relevance Level']}")

        output_path_files = {}
        # Step 6: Save separate CSVs for each relevance level
        for level in ["high", "low", "unknown"]:
            subset = df_pairs[df_pairs["Relevance Level"] == level]
            if not subset.empty:
                filename = f"{output_dir}/shared_ranked_by_pairs_then_mentions_{level}.csv"
                output_path_files[level] = filename
                subset.to_csv(filename, index=False)
                LOG.info(f"✅ Saved: {filename}")

        return output_path_files

    

    def _get_title_map(self):
        """
        Creates a mapping from a safe, normalized paper title (or paper ID fallback)
        to the paper's abstract text.

        This is used for quick lookup of abstracts based on a consistent paper ID
        that can be used as a dictionary key.

        Returns
        -------
        dict
            Dictionary where key = safe title or paper ID, value = abstract string.
        """
        title_map = {}
        for paper in self.papers:
            # Use the title if available; otherwise fall back to paper_id
            # Apply `safe_title()` to normalize the string into a consistent key
            safe_id = safe_title(paper['title'] or paper['paper_id'])

            # Store the abstract (or empty string if missing) as the value
            title_map[safe_id] = paper['abstract'] or ""
        return title_map


    def _get_abstract_map(self):
        """
        Creates a mapping from a safe, normalized paper title (or paper ID fallback)
        to the paper's abstract text.

        Note: This method is functionally identical to `_get_title_map()` and may be
        redundant unless used for semantic distinction elsewhere.

        Returns
        -------
        dict
            Dictionary where key = safe title or paper ID, value = abstract string.
        """
        abstract_map = {}
        for paper in self.papers:
            # Normalize title or use paper ID, just like in _get_title_map
            safe_id = safe_title(paper['title'] or paper['paper_id'])

            # Assign abstract text (or an empty string if not available)
            abstract_map[safe_id] = paper['abstract'] or ""
        return abstract_map
