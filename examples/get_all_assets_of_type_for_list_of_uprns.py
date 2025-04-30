import os
import re

import httpx
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from rdflib.query import ResultRow

# --- Configuration ---------------------------------------------------------

# Your Fuseki/SPARQL endpoint (keep “/query” or switch to “/sparql” as needed)
DB_URL = "http://ec2-18-175-116-201.eu-west-2.compute.amazonaws.com:3030/didtriplestore/query"
endpoint = SPARQLStore(query_endpoint=DB_URL, returnFormat="json")

# Base download directory
here = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(here, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Assets can be of the following types:
# - Merged lidar point clouds: https://w3id.org/dob/id/lidar-pointcloud-merged
# - Lidar range panorama images: https://w3id.org/dob/id/lidar-range-pano
# - Lidar reflectance for panorama: https://w3id.org/dob/id/lidar-reflectance-pano
# - Temperature in celsius: https://w3id.org/dob/id/celsius-temperature (no contentUrl)
# - Lidar signal intensity for panoramas: https://w3id.org/dob/id/lidar-signal-pano
# - Lidar Near Infrared for panoramas: https://w3id.org/dob/id/lidar-nearir-pano
# - Relative humidity: https://w3id.org/dob/id/relative-humidity (no contentUrl)
# - Pointcloud frame: https://w3id.org/dob/id/lidar-pointcloud-frame
# - IR false colour: https://w3id.org/dob/id/ir-false-color-image
# - IR temperature array: https://w3id.org/dob/id/ir-temperature-array
# - IR counts: https://w3id.org/dob/id/ir-count-image
# - RBG image: https://w3id.org/dob/id/rgb-image

# Which UPRNs and which enums (types) to pull
UPRNs = "200003455212, 5045394"
TYPES = "did:rgb-image, did:lidar-pointcloud-merged, did:ir-temperature-array"

# --- SPARQL: find any resource with your enum & contentUrl, then crawl back to UPRN ---
QUERY = f"""
PREFIX dob:  <https://w3id.org/dob/voc#>
PREFIX sosa: <http://www.w3.org/ns/sosa/>
PREFIX so:   <http://schema.org/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX did:  <https://w3id.org/dob/id/>

SELECT DISTINCT ?uprnValue ?contentUrl
WHERE {{
  # 1) Pick up any resource carrying the enum & contentUrl
  ?res
    dob:typeQualifier  ?enum ;
    so:contentUrl      ?contentUrl .
  FILTER(?enum IN ({TYPES}))

  # 2) Crawl back arbitrarily through DerivedResult→Processing→Result→Observation
  #    (and even chained DerivedResults) to get the UPRN literal
  ?res
    (
    ^prov:generated   /   prov:used
    | ^sosa:hasResult
    )*
    / sosa:hasFeatureOfInterest
    / so:identifier
    / so:value
    ?uprnValue .

  # 3) Only the UPRNs you care about
  FILTER(?uprnValue IN ({UPRNs}))
}}
"""

# Your API key from environment
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set.")

# --- Download helper -------------------------------------------------------


def download_asset(url: str, save_dir: str) -> None:
    try:
        resp = httpx.get(url, headers={"x-api-key": API_KEY}, timeout=60)
        resp.raise_for_status()

        # Derive filename
        cd = resp.headers.get("Content-Disposition", "")
        m = re.search(r'filename="([^"]+)"', cd)
        fn = m.group(1) if m else os.path.basename(url)

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, fn)
        with open(path, "wb") as f:
            f.write(resp.content)

        print(f"✔ {url} → {path}")
    except Exception as e:
        print(f"✖ Failed {url}: {e}")


# --- Main ------------------------------------------------------------------


def main():
    results = endpoint.query(QUERY)

    for row in results:
        if not isinstance(row, ResultRow):
            continue

        uprn = str(row["uprnValue"])
        url = str(row["contentUrl"])
        folder = os.path.join(DOWNLOAD_DIR, uprn)

        print(f"Downloading into {folder}/  ←  {url}")
        download_asset(url, folder)


if __name__ == "__main__":
    main()
