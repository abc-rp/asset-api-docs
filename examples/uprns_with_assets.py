import logging
import os
import re

from rdflib.plugins.stores.sparqlstore import SPARQLStore
from rdflib.query import ResultRow

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---------------------------------------------------------

# SPARQL endpoint
DB_URL = "http://ec2-18-175-116-201.eu-west-2.compute.amazonaws.com:3030/didtriplestore/query"
endpoint = SPARQLStore(query_endpoint=DB_URL, returnFormat="json")


# SPARQL query: return both the UPRN value and the asset URL
QUERY = f"""
PREFIX dob:   <https://w3id.org/dob/voc#>
PREFIX so:    <http://schema.org/>
PREFIX sosa:  <http://www.w3.org/ns/sosa/>
PREFIX prov:  <http://www.w3.org/ns/prov#>

SELECT DISTINCT ?uprnValue
WHERE {{
  # 1) Grab any resource (?res) carrying a contentUrl
  ?res
    so:contentUrl ?contentUrl .

 # 2) Traverse backwards through derivation chain
  ?res
    (
      ^sosa:hasResult
      | ^prov:generated / prov:used
    )*
    / sosa:hasFeatureOfInterest
    / so:identifier
    ?idNode .

  # 3) Require that the identifier is a dob:UPRNValue
  ?idNode a dob:UPRNValue ;
          so:value ?uprnValue .

}}
"""


# Your API key from environment
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError(
        "API_KEY environment variable is not set. Please set it to your API key."
    )


# --- Helper to download a single asset into a given folder ----------------


def main():
    # Run the SPARQL query
    results = endpoint.query(QUERY)

    output_file = "uprns_with_assets.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for row in results:
            if not isinstance(row, ResultRow):
                continue

            uprn_val = str(row["uprnValue"]).strip()
            if not uprn_val:
                continue
            f.write(f"{uprn_val}\n")
    logging.info(f"Wrote UPRNs with assets to {output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {e}")
