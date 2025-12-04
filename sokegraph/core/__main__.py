# sokegraph/main.py
import sys
import datetime
import argparse
from sokegraph.util.logger import create_logger, get_logger
from sokegraph.full_pipeline import full_pipeline_main
from sokegraph.__init__ import __version__
from sokegraph.utils import check_file, validate_range
from pathlib import Path


def main():
    run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    parser = argparse.ArgumentParser(
        description="ranking graphs",
        prog="sokegraph",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("-n", "--number_papers", help="the number of papers", default="")
    parser.add_argument("-pdfs", "--pdfs_file", help="path for a zip file contains papers for ranking", default="", type=str)
    parser.add_argument("-api_journal", "--api_key_journal_api", help="path for a file contains API key of Journal API", default="", type=check_file)
    parser.add_argument("-pq", "--paper_query_file", help="path for a file contains query for paper finiding", default="", type=check_file)
    parser.add_argument("-ky", "--keyword_query_file", help="path for a file contains keywords for paper ranking", type=check_file)
    parser.add_argument("-ont", "--ontology_file", help="path for a file contains keywords for paper ranking", default="", type=check_file)
    parser.add_argument("-foi", "--field_of_interest", help="field of interest", default="material sciences")
    parser.add_argument("-API", "--API_keys", help="path for a file contains API keys", type=check_file)
    parser.add_argument("-ckg", "--credentials_for_knowledge_graph", help="path for a file contains crendetials", type=check_file)
    parser.add_argument("-mkg", "--model_knowledge_graph", help="Which knowledge graph", default="neo4j")
    parser.add_argument("-AI", "--AI", help="Which AI", default="openAI")
    parser.add_argument(
        "--verbose",
        default=False,
        action='store_true',
        help='Provide verbose debugging output when logging, and keep intermediate files',
    )
    parser.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Force overwrite any previous files/output directories",
    )
    parser.add_argument("-o", "--output_dir", help="output directory", default=Path(f"sarand_results_{run_time}"),)

    # Parse arguments
    args = parser.parse_args()
    # Get the logger
    log = get_logger()

    # logging file
    open(log_file, 'w').close()

    # execute main workflow
    full_pipeline_main(args)


if __name__ == "__main__":
    main()