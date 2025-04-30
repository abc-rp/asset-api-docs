import os
import httpx
import re
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from rdflib.query import ResultRow

# Connect to the SPARQL endpoint (ensuring all the .ttl files are loaded into the triplestore)
DB_URL = "http://localhost:3030/dob-subset-14-04-25/query"
endpoint = SPARQLStore(query_endpoint=DB_URL, returnFormat="json")

# Directory to save downloaded assets
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

# Comma separated list of UPRNs (you must leave a space after the comma)
UPRNs = "200003455212, 5045394"
# Types of assets to filter on
TYPES = "did:rgb-image, did:lidar-pointcloud-merged"
# Define the query
QUERY = f"""
PREFIX dob: <https://w3id.org/dob/voc#>
PREFIX sosa: <http://www.w3.org/ns/sosa/>
PREFIX so: <http://schema.org/>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX did: <https://w3id.org/dob/id/>

SELECT DISTINCT ?contentUrl
WHERE {{
  ?result a sosa:Result ;
            so:contentUrl ?contentUrl ;
            dob:typeQualifier ?enum .
  FILTER(?enum IN ({TYPES}))
  ?observation a sosa:Observation ;
            sosa:hasResult ?result ;
            sosa:hasFeatureOfInterest ?foi .
  ?foi a sosa:FeatureOfInterest ;
            so:identifier ?uprn .
   ?uprn a dob:UPRNValue ;
           so:value ?uprnValue .
  FILTER(?uprnValue IN ({UPRNs}))
}}"""


API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError(
        "API_KEY environment variable is not set. Please set it to your API key."
    )

# Run the query
results = endpoint.query(QUERY)


# A simple synchronous fetch function
def download_asset(url):
    # You can parallelise this with asyncio but be careful with the number of concurrent requests
    # so that you stay within the rate limits
    try:
        response = httpx.get(url, headers={"x-api-key": API_KEY})
        response.raise_for_status()  # Raise an error for bad responses

        # Try to get the filename from the content-disposition header
        content_disposition = response.headers.get("Content-Disposition")
        filename = None
        if content_disposition:
            # Example: content-disposition: inline; filename="somefile.webp"
            match = re.search(r'filename="(?P<filename>[^"]+)"', content_disposition)
            if match:
                filename = match.group("filename")

        # Use the last part of the URL as a fallback filename
        if not filename:
            filename = url.split("/")[-1]

        save_path = os.path.join(DOWNLOAD_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        print(f"Failed to download {url}: {e}")


try:
    for row in results:
        if not isinstance(row, ResultRow):
            continue
        url = row["contentUrl"]
        print(f"Downloading {url}...")
        download_asset(url)
except Exception as e:
    print(f"Query error: {e}")
