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
        self.ai_tool = ai_tool
        self.papers = papers
        self.ontology_path = ontology_path
        self.ontology = None
        self._load_ontology()
        self.output_dir = output_dir
        self.keyword_query = load_keyword(keyword_path)


    def _load_ontology(self):
        with open(self.ontology_path, 'r', encoding='utf-8') as f:
            self.ontology = json.load(f)

    def rank_papers(self,
                    #title_map: Dict[str, str],
                    #abstract_map: Dict[str, str],
                    #keyword_query: str
                    ) -> List[Tuple[str, float]]:

        title_map = self._get_title_map()
        abstract_map = self._get_abstract_map()

        category_to_papers = defaultdict(lambda: defaultdict(int))
        paper_keyword_map = defaultdict(lambda: defaultdict(set))
        kw_lookup = {}

        print(self.ontology)

        for layer, categories in self.ontology.items():
            for category, items in categories.items():
                for item in items:
                    paper_id = item["paper_id"]
                    for kw in item["keywords"]:
                        kw_lower = kw.lower()
                        kw_lookup[kw_lower] = (layer, category)
                        paper_keyword_map[(layer, category)][paper_id].add(kw_lower)
                        category_to_papers[(layer, category)][paper_id] += 1
                        print(f"kw : {kw_lookup}")

        ranked, low_sorted, pair_file_path = self.find_common_papers(
            #keyword_query,
            #ontology_extractions,
            category_to_papers,
            paper_keyword_map,
            title_map,
            abstract_map,
            kw_lookup,
            #output_dir
        )

        return self.rank_shared_papers_by_pairs_and_mentions(ranked, low_sorted, pair_file_path, self.output_dir)

    def find_common_papers(
        self,
        #user_query: str,
        #ontology_extractions: dict,
        category_to_papers: dict,
        paper_keyword_map: dict,
        title_map: dict,
        kw_lookup: dict,
        abstract_map: dict,
        #output_dir: str,
        threshold: float = 1.5,
    ) -> tuple:

        LOG.info(f"🔍 User query: {self.keyword_query}")

        categories = self.classify_query_with_fallback(self.keyword_query,
                                                        #ontology_extractions, 
                                                        kw_lookup)
        if not categories:
            LOG.error("⚠️ No valid categories found.")
            return [], [], ""

        LOG.info(f"\n✅ Categories Used: {len(categories)}")
        for i, (layer, cat, kw) in enumerate(categories, 1):
            print(f"  {i}. '{kw}' → {layer} / {cat}")

        query_keywords = [kw.lower() for _, _, kw in categories]

        opposites = self.get_opposites(query_keywords, self.ai_tool)
        synonyms = self.get_synonyms(query_keywords, self.ai_tool)
        expanded_keywords = {kw: [kw] + synonyms.get(kw, []) for kw in query_keywords}

        print(f"\n🧪 Opposites used for filtering:")
        for qk in query_keywords:
            print(f"  - '{qk}': {opposites.get(qk, [])}")

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

        all_paper_ids = list(title_map.keys())
        ranked = []

        for pid in all_paper_ids:
            score = total_scores.get(pid, 0)
            if not self.is_dominated_by_opposites(
                pid, title_map, abstract_map, query_keywords, expanded_keywords, opposites, filtered_out
            ):
                ranked.append((pid, score))

        ranked_paper_ids = {pid for pid, _ in ranked}

        if filtered_out:
            summary_df = PaperRanker.summarize_filtered_papers_with_opposites(
                filtered_out, query_keywords, opposites, paper_keyword_frequencies, title_map, abstract_map
            )
            try:
                from IPython.display import display
                display(summary_df)
            except ImportError:
                pass

            summary_df.to_csv(f"{self.output_dir}/filtered_paper_breakdown.csv", index=False)
            LOG.info(f"\n🚫 Filtered out {len(filtered_out)} papers due to dominance of opposite keywords.")

        low_relevance = [
            (pid, info.get("title_exceeded", 0))
            for pid, info in filtered_out.items() if info["reason"] == "title_fail"
        ]
        low_sorted = sorted(low_relevance, key=lambda x: x[1])

        LOG.info("\n📚 Ranked Papers:")
        LOG.info("\n🟢🟡 High & Moderate Relevance:")
        for i, (pid, score) in enumerate(sorted(ranked, key=lambda x: -x[1]), 1):
            print(f"  {i}. {pid} (mentions={score})")

        LOG.info("\n🔴 Low Relevance (sorted by number of over-threshold keywords):")
        for i, (pid, count) in enumerate(low_sorted, 1):
            print(f"  {i}. {pid} (keywords over threshold: {count})")

        filtered_hits = {
            k: {pid: v for pid, v in d.items() if pid in ranked_paper_ids}
            for k, d in per_cat_hits.items()
        }

        if len(per_cat_hits) < 2:
            LOG.info("\n⚠️ Overlap analysis needs 2+ categories.")
            return ranked, low_sorted, ""

        ranked_by_pair_overlap, total_possible_pairs = PaperRanker.rank_by_pair_overlap_filtered(filtered_hits, ranked_paper_ids)

        LOG.info(f"\n🏆 Full Ranking by Number of Category Pairs Shared (out of {total_possible_pairs} pairs):")
        for i, (pid, pair_set) in enumerate(ranked_by_pair_overlap, 1):
            print(f"  {i}. {pid} → shared in {len(pair_set)}/{total_possible_pairs} pair(s)")
            print(f"     ⤷ Pairs: {sorted(pair_set)}")

        if not ranked_by_pair_overlap:
            LOG.info("⚠️ No multi-category-pair overlaps found.")

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

        return ranked, low_sorted, df_pairs_file_path

    def classify_query_with_fallback(self, user_query: str, kw_lookup: dict) -> list:
        import re

        tokens = [t for t in re.findall(r"\w+", user_query.lower()) if len(t) > 1]
        ontology_summary = {layer: list(cats.keys()) for layer, cats in self.ontology.items()}

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

        response = self.ai_tool.ask(prompt)

        try:
            model_output = json.loads(response)
        except json.JSONDecodeError:
            print("⚠️ OpenAI output couldn't be parsed. Raw:\n", response)
            model_output = []

        categories = []
        found_set = set()
        model_keywords = set()

        for item in model_output:
            kw = item["keyword"].lower()
            model_keywords.add(kw)
            if item["layer"] != "Unknown" and item["category"] != "Unknown":
                if (item["layer"], item["category"]) not in found_set:
                    categories.append((item["layer"], item["category"], kw))
                    found_set.add((item["layer"], item["category"]))
                    print(f"✅ OpenAI: '{kw}' → {item['layer']} / {item['category']}")

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

        import re

        # Get and normalize title and abstract
        title_text = (title_map.get(paper_id, "") or "").lower()
        full_abstract = (abstract_map.get(paper_id, "") or "").lower()
        abstract_text = re.split(r'(references|refs?\.?:)', full_abstract)[0]

        title_flags = {}
        title_over_threshold = 0

        for qk in query_keywords:
            relevant_terms = expanded_keywords.get(qk, [qk])
            opp_terms = opposites.get(qk, [])

            title_rel = sum(len(re.findall(rf'\b{re.escape(term)}\b', title_text)) for term in relevant_terms)
            title_opp = sum(len(re.findall(rf'\b{re.escape(opp)}\b', title_text)) for opp in opp_terms)

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

        # Case 1: Opposites dominate in the title
        if "fail" in title_flags.values():
            filtered_out[paper_id] = {
                "reason": "title_fail",
                "title_flags": title_flags,
                "title_exceeded": title_over_threshold,
                "threshold": 1.0,
            }
            return True

        # Case 2: Mixed signal - check abstract
        if "pass" in title_flags.values() and "missing" in title_flags.values():
            abstract_ratios = []
            for qk in query_keywords:
                relevant_terms = expanded_keywords.get(qk, [qk])
                opp_terms = opposites.get(qk, [])

                abs_rel = sum(len(re.findall(rf'\b{re.escape(term)}\b', abstract_text)) for term in relevant_terms)
                abs_opp = sum(len(re.findall(rf'\b{re.escape(opp)}\b', abstract_text)) for opp in opp_terms)

                ratio = float('inf') if abs_rel == 0 else abs_opp / abs_rel
                abstract_ratios.append(ratio)

            if all(r <= 1.5 for r in abstract_ratios):
                return False
            else:
                filtered_out[paper_id] = {
                    "reason": "moderate_abstract_check",
                    "abstract_ratios": abstract_ratios,
                    "title_exceeded": 0,
                    "threshold": 1.5,
                }
                return False

        # Case 3: All keywords pass
        if all(flag == "pass" for flag in title_flags.values()):
            return False

        # Fallback case: Not filtered
        return False
    

    @staticmethod
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
        """
        rows = []

        for pid, info in filtered_out.items():
            title_text = (title_map.get(pid, "") or "").lower()
            abstract_text = (abstract_map.get(pid, "") or "").lower()

            for qk in query_keywords:
                matched_opp_keywords = []

                title_rel = len(re.findall(rf'\b{re.escape(qk)}\b', title_text))
                title_opp = 0
                for opp in opposites.get(qk, []):
                    count = len(re.findall(rf'\b{re.escape(opp)}\b', title_text))
                    title_opp += count
                    if count > 0:
                        matched_opp_keywords.append(opp)

                abs_rel = len(re.findall(rf'\b{re.escape(qk)}\b', abstract_text))
                abs_opp = 0
                for opp in opposites.get(qk, []):
                    count = len(re.findall(rf'\b{re.escape(opp)}\b', abstract_text))
                    abs_opp += count
                    if count > 0 and opp not in matched_opp_keywords:
                        matched_opp_keywords.append(opp)

                total_rel = title_rel + abs_rel
                total_opp = title_opp + abs_opp
                ratio = round(total_opp / total_rel, 2) if total_rel else float('inf')
                status = "Filtered" if ratio > info["threshold"] else "Kept"

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
    
    @staticmethod
    def rank_by_pair_overlap_filtered(
        per_cat_hits: Dict[Tuple[str, str], Dict[str, float]],
        ranked_paper_ids: Set[str]
    ) -> Tuple[List[Tuple[str, Set[str]]], int]:
        """
        Ranks papers based on how many unique category-pair overlaps they appear in,
        considering only a filtered set of paper IDs.
        """
        pair_paper_map = defaultdict(set)
        category_pairs = list(combinations(per_cat_hits.keys(), 2))
        total_possible_pairs = len(category_pairs)

        for (cat1, cat2) in category_pairs:
            papers1 = set(per_cat_hits[cat1].keys())
            papers2 = set(per_cat_hits[cat2].keys())
            shared_papers = papers1 & papers2
            filtered_shared = shared_papers & ranked_paper_ids

            pair_label = f"{cat1[1]} ↔ {cat2[1]}"

            for pid in filtered_shared:
                pair_paper_map[pid].add(pair_label)

        ranked_by_pair_overlap = sorted(
            pair_paper_map.items(),
            key=lambda x: len(x[1]),
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
    



    def _get_title_map(self):
        title_map = {}
        for paper in self.papers:
            safe_id = safe_title(paper['title'] or paper['paper_id'])
            title_map[safe_id] = paper['abstract'] or ""
        return title_map
    

    def _get_abstract_map(self):
        abstract_map = {}
        for paper in self.papers:
            safe_id = safe_title(paper['title'] or paper['paper_id'])
            abstract_map[safe_id] = paper['abstract'] or ""
        return abstract_map