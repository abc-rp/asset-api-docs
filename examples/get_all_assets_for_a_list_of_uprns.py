import os
import re

import httpx
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from rdflib.query import ResultRow

# --- Configuration ---------------------------------------------------------

# SPARQL endpoint
DB_URL = "http://100.64.153.8:3030/didtriplestore/query"
endpoint = SPARQLStore(query_endpoint=DB_URL, returnFormat="json")

# Base download directory
here = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(here, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Comma-separated list of UPRNs (note the space after each comma)
UPRNs = "200003455212, 5045394"

# SPARQL query: return both the UPRN value and the asset URL
QUERY = f"""
PREFIX dob:   <https://w3id.org/dob/voc#>
PREFIX so:    <http://schema.org/>
PREFIX sosa:  <http://www.w3.org/ns/sosa/>
PREFIX prov:  <http://www.w3.org/ns/prov#>

SELECT DISTINCT ?uprnValue ?contentUrl
WHERE {{
  # 1) Grab any resource (?res) carrying a contentUrl
  ?res
    so:contentUrl ?contentUrl .

  # 2) Crawl back through:
  #      – sosa:hasResult from an Observation
  #      – prov:generated / prov:used chains from Processing → DerivedResult → Result
  #    (including any number of chained DerivedResults)
  ?res
    (
      ^sosa:hasResult
      | ^prov:generated / prov:used
    )*
    / sosa:hasFeatureOfInterest
    / so:identifier
    / so:value
    ?uprnValue .

  # 3) Restrict to only the UPRNs you care about
  FILTER (?uprnValue IN ({UPRNs}))
}}
"""


# Your API key from environment
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError(
        "API_KEY environment variable is not set. Please set it to your API key."
    )


# --- Helper to download a single asset into a given folder ----------------


def download_asset(url: str, save_dir: str) -> None:
    try:
        resp = httpx.get(url, headers={"x-api-key": API_KEY})
        resp.raise_for_status()

        # Derive filename from Content-Disposition or fallback to URL basename
        cd_header = resp.headers.get("Content-Disposition", "")
        m = re.search(r'filename="([^"]+)"', cd_header)
        filename = m.group(1) if m else os.path.basename(url)

        # Ensure the target folder exists
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        # Write out the file
        with open(save_path, "wb") as f:
            f.write(resp.content)

        print(f"✔ Saved {url} → {save_path}")
    except Exception as e:
        print(f"✖ Failed to download {url}: {e}")


# --- Main execution --------------------------------------------------------


def main():
    # Run the SPARQL query
    results = endpoint.query(QUERY)

    # Iterate and dispatch each download into its UPRN folder
    for row in results:
        if not isinstance(row, ResultRow):
            continue

        uprn_val = str(row["uprnValue"])
        content_url = str(row["contentUrl"])
        uprn_folder = os.path.join(DOWNLOAD_DIR, uprn_val)

        print(f"⤷ Downloading {content_url} into {uprn_folder}/ …")
        download_asset(content_url, uprn_folder)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
