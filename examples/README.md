# Query Assist Scripts

WARNING: This project is very much in its testing phase and things may not work as expected, or at all.

This directory contains a few Python scripts that will load and provision the graph database with provided turtle (.ttl) files and execute queries against it. Some will also then use results from queries to download assets from the API.

  * `query_assist.py`: The core script for building queries using command-line arguments.
  * `nl_query_assist.py`: A user-friendly interface that uses an LLM to translate natural language into commands for `query_assist.py`.

## Setup

### 1\. Create a virtual environment

```bash
cd /path/to/your/scripts
python3 -m venv venv
```

### 2\. Activate environment

**macOS/Linux**

```bash
source venv/bin/activate
```

**Windows (CMD)**

```bash
venv\Scripts\activate.bat
```

### 3\. Install dependencies

```bash
pip install -r requirements.txt
```

### 4\. Set your API keys

You will need an API key to download assets. If you plan to use OpenAI with the Natural Language script, you will also need an OpenAI API key.

**Temporary (session only)**

```bash
export API_KEY="your_asset_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

**Permanent (automatic when Virtualenv is activated)**

You can add the export lines to the `venv/bin/activate` script (for macOS/Linux) or `venv\Scripts\activate.bat` (for Windows) to set these variables automatically every time you activate your virtual environment.

## Running Scripts

There are two ways to query the system: the direct method using `query_assist.py` for precise control, and the natural language method using `nl_query_assist.py` for ease of use.

-----

## `query_assist.py` (Direct Method)

This is the core utility for programmatic use. It allows you to build a query by combining identifiers, geographies, property filters, and asset types. All conditions are combined with AND logic.

The primary way to use the script is by providing a JSON configuration file via `--config`. Alternatively, you can build a query using the following granular flags:

  * **`--identifier <TYPE> <VALUES...>`**: Specify properties by a known ID. Can be used multiple times.
  * **`--geography <TYPE> <VALUES...>`**: Specify a geographical area. Can be used multiple times.
  * **`--filter <TYPE> <VALUES...>`**: Add property or statistical data filters. Can be used multiple times.
  * **`--sensor <IRI>`**: Filter for assets created by a specific sensor.
  * **`--types <IRI,IRI...>`**: Filter for specific asset types.
  * **`--ontop-url <URL>`**: The Ontop SPARQL endpoint URL for the LBSM data. **Required** if using `toid`, geography, or filter arguments.
  * **`--db-url <URL>`**: Base graph SPARQL endpoint URL
  * **`--download-dir <DOWNLOAD_DIR>`**: Base directory for downloads (default: ./downloads)
  * **`--api-key-env <API_KEY_ENV>`**: Environment variable name for the API key

#### Supported Identifiers 

  - `uprn`: The Unqiue Property Reference Number assigned to every building in the UK;
  - `ods`: A unique identifier for organizations within the NHS and health and social care sectors in the UK;
  - `toid`: A unique identifier assigned by the Ordnance Survey (OS) to every topographical feature in Great Britain.

#### Supported Geographies

  - `oa`: Output Areas (OAs), the smallest standard building blocks used by the ONS for Census data.
  - `lsoa`: Lower layer Super Output Areas (LSOAs) are made of groups of OAs, and are used by the ONS for Census data.
  - `ward`: Electoral wards are the spatial units used to elect local government councillors in metropolitan and non-metropolitan districts, unitary authorities and the London boroughs in England.
  - `london-borough`: An administratve area in Greater London.
  - `postcode`: An alphanumeric code assigned by Royal Mail as part of a postal address.

#### Supported Filters

##### Categorical Filters

  - `property-type`: flat, house, park-home-caravan
  - `built-form`: detached, semi-detached, end-terrace, mid-terrace
  - `accommodation-type`: flat, semi-detached-house, detached-house, end-terraced-house, mid-terraced-house, park-home-caravan
  - `tenure`: owner-occupied, social-housing, privately-rented
  - `epc-rating`: AB, C, D, E, FG
  - `potential-epc-rating`: AB, C, D, E, FG
  - `construction-age-band`: pre-1900, 1900-1929, 1930-1949, 1950-1966, 1967-1982, 1983-1995, 1996-2011, 2012-onwards
  - `building-use`: residential-only, mixed-use
  - `main-heating-system`: boiler, room-storage-heaters, heat-pump, communal, none, other
  - `main-fuel-type`: mains-gas, electricity, no-heating-system, other
  - `wall-type`: cavity, solid, other
  - `roof-type`: pitched, flat, room-in-roof, another-dwelling-above
  - `glazing-type`: single-partial, secondary, double-triple
  - `loac-supergroup`: A, B, C, D, E, F, G
  - `loac-group`: A1, A2, A3, B1, B2, C1, C2, D1, D2, D3, E1, E2, F1, F2, G1, G2
  - `listed-building-grade`: I, II, IIStar, Unknown

Categorical filters can take any number of values from their allowed categories (as described above). For example, the following argument,

> `--filter epc-rating C D`

filters properties with an EPC rating of C or D.

##### Boolean Filters

  - `wall-insulation`
  - `roof-insulation`
  - `in-conservation-area`

If filtering for buildings in conservation areas, the argument would looks like:

> `--filter in-conservation-area true`

##### Numeric Filters

  - `heat-risk-quintile`: INTEGER from 1 to 5
  - `imd19-income-decile`: INTEGER from 1 to 10
  - `imd19-national-decile`: INTEGER from 1 to 10
  - `epc-score`: INTEGER from 1 to 100
  - `potential-epc-score`: INTEGER from 1 to 100
  - `floor count`: INTEGER
  - `basement-floor-count`: INTEGER
  - `number-of-habitable-rooms`: INTEGER
  - `easting`: INTEGER
  - `northing`: INTEGER
  - `total-floor-area`: INTEGER
  - `energy-consumption`: INTEGER
  - `solar-pv-area`: INTEGER
  - `average-roof-tilt`: INTEGER
  - `solar-pv-potential`: DECIMAL
  - `fuel-poverty`: DECIMAL

Numeric filters have two formats. If requesting a specific value, for example EPC Score 85, the argument can be entered as follows:

> `--filter epc-score 85`

If requesting a range of values, for example a total floor area between 100 and 200 metres squared, the argument can be entered as follows:

> `--filter total-floor-area 100 200`

### Supported Sensors and Asset Types

#### Supported sensors

- `bess:PhidgetHumiditySensor`
- `bess:PhidgetTemperatureSensor`
- `bess:OusterLidarSensor`
- `bess:FlirOryxCamera`
- `bess:FlirA70Camera`

#### Supported asset types

- `did:lidar-pointcloud-merged`: **Merged lidar point clouds**
- `did:lidar-pointcloud-frame`: **Pointcloud frame**
- `did:lidar-range-pano`: **Lidar range panorama images**
- `did:lidar-reflectance-pano`: **Lidar reflectance for panorama**
- `did:lidar-signal-pano`: **Lidar signal intensity for panoramas**
- `did:lidar-nearir-pano`: **Lidar Near Infrared for panoramas**
- `did:celsius-temperature`: **Temperature in celsius** (no contentUrl)
- `did:relative-humidity`: **Relative humidity** (no contentUrl)
- `did:ir-false-color-image`: **IR false colour**
- `did:ir-temperature-array`: **IR temperature array**
- `did:ir-count-image`: **IR counts**
- `did:rgb-image`: **RGB image**

### `query_assist.py` Usage Examples

```bash
# Get assets for a single UPRN
python3 query_assist.py --identifier uprn 100023334911

# Get assets for multiple UPRNs from a CSV file
python3 query_assist.py --identifier uprn path/to/uprns.csv

# Get assets for all properties in an Output Area
python3 query_assist.py --geography output-area E00004550 --ontop-url <URL>

# Get assets for houses with owner-occupied tenure in a specific Ward
python3 query_assist.py \
  --geography ward E05013561 \
  --filter tenure owner-occupied \
  --filter property-type house \
  --ontop-url <URL>

# Get assets with an EPC score between 50 and 75 in a specific London Borough, and filter by asset type
python3 query_assist.py \
  --geography london-borough E09000030 \
  --filter epc-score 50 75 \
  --types did:rgb-image \
  --ontop-url <URL>
```

-----

## `nl_query_assist.py` (Natural Language Method)

This script provides an interface for natural language querying. It translates plain English queries into `query_assist.py` commands (via the --config commmand) and then executes it.

### LLM Setup

This script can use a local Ollama instance or the remote OpenAI API.

  * **Ollama (Default):**
    1.  Ensure your Ollama server is running (e.g., in a Docker container).
    2.  Pull a model suitable for following instructions, e.g., `ollama pull llama3:8b-instruct`.
  * **OpenAI:**
    1.  Make sure your `OPENAI_API_KEY` environment variable is set.
    2.  Use the `--api openai` flag when running the script.

### Key Arguments

  * `--api <provider>`: Choose the API provider: `ollama` (default) or `openai`.
  * `--model-id <name>`: Specify the Ollama model to use (e.g., `llama3:8b-instruct`).
  * `--openai-model <name>`: Specify the OpenAI model to use (e.g., `gpt-4o-mini`).
  * `--force-llm`: Skip the fast heuristic parser and send the query directly to the LLM. Useful for very complex queries.
  * `--dry-run`: Show the command that would be run without actually executing it.
  * `--base-url`: Base URL of the Ollama server

### `nl_query_assist.py` Usage Examples

```bash
# Start the interactive script (defaults to Ollama)
python3 nl_query_assist.py --base-url http://ollama:11434

# Start the interactive script with OpenAI
python3 nl_query_assist.py --api openai
```

The script runs in an interactive loop. Just type your request and press Enter.

**Example Queries:**

> `get me assets for uprn 100023334911`

> `show me rgb images for flats in camden`

> `find assets for houses in islington with an epc score between 50 and 75`

> `get merged lidar point clouds for properties built before 1930 without wall insulation`

> `get assets for the uprns listed in ./data/my_uprns.csv`

-----

# Additional Data Information

## RGB

sRGB images are provided in the API at a resolution optimised for computer vision tasks. Vehicles and humans are masked out using an automated process, if a user finds an unmasked person or vehicle (most critically the number plate), please report it to [xRI](mailto:info@xri.online).


## IR

The outermost regions of the IR images and temperature arrays have been masked out, this is due to hot edges due to the IR detector heating itself up during operation. In the temperature arrays the masked areas are NaN elements in the compressed numpy array.

Additionally when working with the IR data there are some assumptions to note about the way in which radiometic temperature pixels themselves are calculated.

- The formula requires the distance of each pixel from the detector, currently this is a sensible hard coded value, we are in the process of calculating these distances from the lidar.
- Building materials tend to be in a narrow range of emissivities $\epsilon\in [0.85,0.93]$, we currently hard code a single sensible value for emissivity but are developing methods for estimating building materials dynamically.
- During the day, we are in a reflectance dominated regime due to the influence of the sun, radiometric temperatures calculated in this regime are not reliable. Thermal data is provided for the night hours only (1 hour after sunset to 1 hour before sunrise).
- The sky is an object outside the scope of the radiometric temperature calculation, this is a low reflectance, low emissivity regime that our radiometric temperature calculations cannot say anything meaningful about.

## LiDAR

We have four 360 degree grey scale panoramas these are:

- Near-infrared (NIR) capturing light in the near-infrared spectrum (just beyond visible light). NIR is often used to assess vegetation health, surface properties, and for capturing detailed textures in low-light conditions.

- The range modality provides the distance from the LiDAR sensor to objects in the environment. Each pixel in this image represents a distance measurement in millimeters, creating a depth map of the scene.

- The reflectivity image captures the intensity of the LiDAR signal that bounces back to the sensor. Reflectivity depends on the surface material and angle of incidence, making it useful for distinguishing between materials or identifying road markings, signs, and other objects.

- The signal strength or return signal intensity measures the quality of the LiDAR return. Stronger signals usually indicate clearer, more reliable measurements. It can also reflect surface properties and environmental conditions.

We also have two pointcloud types one is a single frame that is closest to orthogonal to the UPRN, the other is a dense, orchstrated pointcloud created by merging many pointcloud frames on either side of the most orthogonal frame using the [Iterative Closes Point (ICP) registration algorithm](http://ki-www.cvl.iis.u-tokyo.ac.jp/class2013/2013w/paper/correspondingAndRegistration/03_Levoy.pdf).

ICP registration can also fail completely resulting in dense but unaligned pointclouds. The single centre frame is provided as a failback pointcloud in the event of an unusable merged pointcloud.
