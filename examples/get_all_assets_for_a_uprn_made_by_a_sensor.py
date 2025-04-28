import os
import httpx
import re
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from rdflib.query import ResultRow

# NOTE: THIS QUERY WON'T WORK IF THE BESS SENSOR STACK IS NOT IN THE GRAPH
# https://github.com/abc-rp/bess_rdf/blob/feature/data_model/sensor_stack/6284_156243.ttl

# Connect to the SPARQL endpoint (ensuring all the .ttl files are loaded into the triplestore)
DB_URL = "http://localhost:3030/dob-subset-14-04-25/query"
endpoint = SPARQLStore(query_endpoint=DB_URL, returnFormat="json")

# Directory to save downloaded assets
here = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(here, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)



# Query params
UPRN = "5045394"
# The available sensor labels you can filter on are defined in the ontology:
# https://github.com/abc-rp/bess/blob/main/voc/index.ttl
SENSOR = "bess:OusterLidarSensor"
# Define the query
QUERY = f"""
# Get all assets (dob:Result) for a specific UPRN (dob:UPRNValue) made by the Lidar (bess:OusterLidarSensor)
PREFIX dob: <https://w3id.org/dob/voc#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX sosa: <http://www.w3.org/ns/sosa/>
PREFIX bess: <https://w3id.org/bess/voc#>
PREFIX so: <http://schema.org/>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT DISTINCT ?contentUrl
WHERE {{
  ?result a sosa:Result ;
            so:contentUrl ?contentUrl .
  ?observation a sosa:Observation ;
            sosa:hasResult ?result ;
            sosa:madeBySensor ?sensor;
            sosa:hasFeatureOfInterest ?foi .
  ?sensor a {SENSOR} .
  ?foi a sosa:FeatureOfInterest ;
            so:identifier ?uprn.
   ?uprn a dob:UPRNValue ;
           so:value ?uprnValue .
  FILTER(?uprnValue = {UPRN})
}}
"""


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
