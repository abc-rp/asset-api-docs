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
SENSOR = "bess:OusterLidarSensor"

# The available sensors are:
# - bess:PhidgetHumiditySensor
# - bess:PhidgetTemperatureSensor
# - bess:OusterLidarSensor
# - bess:FlirOryxCamera
# - bess:FlirA70Camera

# SPARQL query: return the asset URL for the given UPRN and sensor
QUERY = f"""
PREFIX dob:   <https://w3id.org/dob/voc#>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX sosa:  <http://www.w3.org/ns/sosa/>
PREFIX bess:  <https://w3id.org/bess/voc#>
PREFIX so:    <http://schema.org/>
PREFIX prov:  <http://www.w3.org/ns/prov#>
PREFIX owl:   <http://www.w3.org/2002/07/owl#>

SELECT DISTINCT ?uprnValue ?contentUrl
WHERE {{
  # 1) Grab any resource carrying a contentUrl
  ?res
    so:contentUrl  ?contentUrl .

  # 2) Crawl back through either:
  #    - Observation → sosa:hasResult
  #    - Processing → DerivedResult → (prov:generated / prov:used)
  #    (any number of times)
  ?res
    (
      ^sosa:hasResult
      | ^prov:generated / prov:used
    )*
    ?obs .

  # 3) Now we’re at the Observation; pull out sensor & UPRN
  ?obs a sosa:Observation ;
       sosa:madeBySensor      ?sensor ;
       sosa:hasFeatureOfInterest/so:identifier/so:value  ?uprnValue .

  # 4) Filter on specific sensor type and UPRN
  ?sensor a {SENSOR} .
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
        cd = resp.headers.get("Content-Disposition", "")
        m = re.search(r'filename="([^"]+)"', cd)
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

    # Dispatch each download into the UPRN folder
    for row in results:
        if not isinstance(row, ResultRow):
            continue

        uprn_val = str(row["uprnValue"])
        content_url = str(row["contentUrl"])
        target_dir = os.path.join(DOWNLOAD_DIR, uprn_val)

        logging.info(f"⤷ Downloading {content_url} into {target_dir}/ …")
        download_asset(content_url, target_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {e}")
