from rdflib import Graph
import os
import httpx
import re

# Path to your directory of .ttl files
TTL_DIR = "examples/turtles/"
# Directory to save downloaded assets
DOWNLOAD_DIR = "examples/downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Query params
UPRN = "5045394"
SENSOR_LABEL = "Ouster Lidar Sensor"
# Define the query
QUERY = f"""
# Get all assets (dob:Result) for a specific UPRN (dob:UPRNValue) made by the sensor (sosa:Sensor) with label "Ouster Lidar Sensor"
PREFIX dob: <https://w3id.org/dob/voc#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX sosa: <http://www.w3.org/ns/sosa/>
PREFIX so: <http://schema.org/>
PREFIX bot: <https://w3id.org/bot#>
PREFIX prov: <http://www.w3.org/ns/prov#>

SELECT DISTINCT ?contentUrl
WHERE {{
  ?sensor a rdfs:Class ;
            rdfs:subClassOf sosa:Sensor ;
            rdfs:label ?sensorLabel .
  FILTER(?sensorLabel = "{SENSOR_LABEL}"@en)
  ?observation a sosa:Observation ;
               sosa:hasResult ?result ;
               sosa:madeBySensor ?sensor .
  ?result a dob:Result ;
            sosa:hasFeatureOfInterest ?zone ;
            so:contentUrl ?contentUrl ;
            prov:wasInformedBy ?observation .
  ?zone a bot:Zone, sosa:FeatureOfInterest ;
        so:identifier ?uprn .
  ?uprn a dob:UPRNValue ;
        so:value ?uprnValue .
  FILTER(?uprnValue = {UPRN})
}}"""

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError(
        "API_KEY environment variable is not set. Please set it to your API key."
    )

# Create a new graph (using Oxigraph as a store to make things fast)
g = Graph(store="Oxigraph", bind_namespaces="core")

# Iterate over files in the directory to load the turtle files
for filename in os.listdir(TTL_DIR):
    if filename.endswith(".ttl"):
        file_path = os.path.join(TTL_DIR, filename)
        print(f"Loading {file_path}...")
        g.parse(file_path, format="ox-ttl")

print(f"Graph loaded with {len(g)} triples.")

# Run the query
results = g.query(QUERY)


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
        url = row["contentUrl"]
        print(f"Downloading {url}...")
        download_asset(url)
except Exception as e:
    print(f"Query error: {e}")
