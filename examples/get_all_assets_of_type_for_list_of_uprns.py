import os
import httpx
import re
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from rdflib.query import ResultRow

# --- Configuration ---------------------------------------------------------

DB_URL = "http://100.64.153.8:3030/mytriplestore/query"
endpoint = SPARQLStore(query_endpoint=DB_URL, returnFormat="json")

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
# If using prefixes you can write the above IRIs as dob:ENTITY
# e.g. dob:lidar-pointcloud-merged


# Comma-separated UPRNs (space after comma)
UPRNs  = "200003455212, 5045394"
# Comma-separated types to include
TYPES  = "did:rgb-image, did:lidar-pointcloud-merged, did:ir-temperature-array "

# --- SPARQL: get both direct & derived results, filtered by TYPE & UPRN ---
QUERY = f"""
PREFIX dob:  <https://w3id.org/dob/voc#>
PREFIX sosa: <http://www.w3.org/ns/sosa/>
PREFIX so:   <http://schema.org/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX did:  <https://w3id.org/dob/id/>

SELECT DISTINCT ?uprnValue ?contentUrl
WHERE {{
  # Find observations → results → featureOfInterest → UPRNValue
  ?observation a sosa:Observation ;
               sosa:hasResult          ?result ;
               sosa:hasFeatureOfInterest ?foi .
  ?foi so:identifier ?uprn .
  ?uprn a dob:UPRNValue ;
        so:value     ?uprnValue .
  FILTER(?uprnValue IN ({UPRNs}))

  # Two ways to get contentUrl, union’d:
  {{
    ?result dob:typeQualifier ?enum ;
            so:contentUrl    ?contentUrl .
    FILTER(?enum IN ({TYPES}))
  }}
  UNION
  {{
    ?proc a dob:Processing ;
          prov:used       ?result ;
          prov:generated  ?derived .
    ?derived a dob:DerivedResult ;
             dob:typeQualifier ?enum ;
             so:contentUrl    ?contentUrl .
    FILTER(?enum IN ({TYPES}))
  }}
}}
"""

API_KEY = os.getenv("DID_API_KEY")
if not API_KEY:
    raise ValueError("DID_API_KEY environment variable is not set.")


# --- Download helper -------------------------------------------------------

def download_asset(url: str, save_dir: str) -> None:
    try:
        resp = httpx.get(url, headers={"x-api-key": API_KEY})
        resp.raise_for_status()

        # Filename from header or URL
        cd = resp.headers.get("Content-Disposition", "")
        m  = re.search(r'filename="([^"]+)"', cd)
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
        url  = str(row["contentUrl"])
        folder = os.path.join(DOWNLOAD_DIR, uprn)

        print(f"Downloading into {folder}/  ←  {url}")
        download_asset(url, folder)


if __name__ == "__main__":
    main()
