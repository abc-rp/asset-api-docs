import logging
import os
import re

import httpx
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

# Base download directory
here = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(here, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Query parameters
UPRN = "5045394"

# SPARQL query: return all asset URLs for the given UPRN
QUERY = f"""
PREFIX dob:   <https://w3id.org/dob/voc#>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX sosa:  <http://www.w3.org/ns/sosa/>
PREFIX so:    <http://schema.org/>
PREFIX owl:   <http://www.w3.org/2002/07/owl#>

SELECT DISTINCT ?contentUrl
WHERE {{
  ?result a sosa:Result ;
          so:contentUrl ?contentUrl .
  ?observation a sosa:Observation ;
               sosa:hasResult ?result ;
               sosa:hasFeatureOfInterest ?foi .
  ?foi a sosa:FeatureOfInterest ;
       so:identifier ?uprn .
  ?uprn a dob:UPRNValue ;
        so:value ?uprnValue .
  FILTER(str(?uprnValue) = "{UPRN}")
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

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)

        with open(path, "wb") as f:
            f.write(resp.content)

        logging.info(f"✔ Saved {url} → {path}")
    except Exception as e:
        logging.error(f"✖ Failed to download {url}: {e}")


# --- Main execution --------------------------------------------------------


def main():
    # Run the SPARQL query
    results = endpoint.query(QUERY)

    # Download each asset into the base download directory
    for row in results:
        if not isinstance(row, ResultRow):
            continue

        content_url = str(row["contentUrl"])
        logging.info(f"⤷ Downloading {content_url} …")
        download_asset(content_url, DOWNLOAD_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {e}")
