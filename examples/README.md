
# Python examples

This directory contains a few Python scripts that will load and provision the graph database with provided turtle (.ttl) files and execute queries against it. Some will also then use results from queries to download assets from the API.

## Setup

### 1. Create a virtual environment

```bash
cd examples
python3 -m venv venv
```

### 2. Activate and install requirements

**macOS/Linux**
```bash
source venv/bin/activate
```

**Windows (CMD)**

```bash
venv\Scripts\activate.bat
```

**Windows (PowerShell)**

```powershell
venv\Scripts\Activate.ps1
```

Install dependencies.

```bash
pip install -r requirements.txt
```

### 3. Set your API key

**Temporary (session only)**

```bash
export API_KEY="your_api_key"
```

**Permanent (automatic when Virtualenv is activated)**

If you want this variable to be automatically set every time you activate your virtual environment, add the export line to the activate script inside your virtual environment. For example (macOS/Linux):

Open `venv/bin/activate` and add the environment variable near the bottom (but before any final `unset` lines if present).

## Running scripts

Once your environment is set up then you can run any of the `.py` files in the `/examples` directory. **Before you run a script** verify that the constants in the file are set correctly to match your local environment.

### Scripts

Granular scripts to improve legibility when viewing the SPARQL queries:
- `get_all_assets_for_a_list_of_uprns.py`
- `get_all_assets_for_a_uprn.py`
- `get_all_assets_for_a_uprn_made_by_a_sensor.py`
- `get_all_assets_of_type_for_list_of_uprns.py`

Unified script:
- `query_assist.py`

This unified script replaces and extends the above utilities by allowing you to:

- Specify one or more UPRNs via **`--uprn`** (space- or comma-separated), or provide a CSV file path (column `uprn`) to `--uprn`.
- Specify one or more ODS codes via **`--ods`** (space- or comma-separated), or provide a CSV file path (column `ods`) to `--ods` for ODS→UPRN mapping.
- Specify one or more output-area IRIs or codes via **`--output-area`**/`--oa` (space- or comma-separated), or provide a CSV file path (column `output_area`) to list UPRNs by output area.
- Filter by **sensor** type (`--sensor`, e.g. `bess:OusterLidarSensor`).
- Filter by **asset type** (`--types`, e.g. `did:rgb-image,did:lidar-pointcloud-merged`).
- Override the **SPARQL endpoint** (`--db-url`).
- Change the **download directory** (`--download-dir`).
- Use a custom **API key** environment variable (`--api-key-env`).

#### Supported sensors

- `bess:PhidgetHumiditySensor`
- `bess:PhidgetTemperatureSensor`
- `bess:OusterLidarSensor`
- `bess:FlirOryxCamera`
- `bess:FlirA70Camera`

#### Supported asset types

- **Merged lidar point clouds**: `did:lidar-pointcloud-merged`
- **Pointcloud frame**: `did:lidar-pointcloud-frame`
- **Lidar range panorama images**: `did:lidar-range-pano`
- **Lidar reflectance for panorama**: `did:lidar-reflectance-pano`
- **Lidar signal intensity for panoramas**: `did:lidar-signal-pano`
- **Lidar Near Infrared for panoramas**: `did:lidar-nearir-pano`
- **Temperature in celsius** (no contentUrl): `did:celsius-temperature`
- **Relative humidity** (no contentUrl): `did:relative-humidity`
- **IR false colour**: `did:ir-false-color-image`
- **IR temperature array**: `did:ir-temperature-array`
- **IR counts**: `did:ir-count-image`
- **RGB image**: `did:rgb-image`

Pointclouds are brotli compressed .pcd files. These can be decompressed using the Brotli CLI tool

```bash
brew install brotli
```

Or using the `br_decompress.py` script.

#### `query_assist.py` Usage

```bash
# Single UPRN
python3 query_assist.py --uprn 5045394

# Multiple UPRNs (space-separated)
python3 query_assist.py --uprn 200003455212 5045394

# Multiple UPRNs (comma-separated)
python3 query_assist.py --uprn 200003455212, 5045394, 45127845

# CSV-only for UPRNs
python3 query_assist.py --uprn path/to/uprns.csv

# ODS→UPRN mapping with recommendation code A (accepted) I (intervention recommended)
python3 query_assist.py --ods G85013

# Output-area mode (single code)
python3 query_assist.py --output-area E00004550

# Output-area mode (multiple codes)
python3 query_assist.py --output-area E00004550 E00032882 E00063193 E00047411

# CSV-only for output-area
python3 query_assist.py --output-area path/to/areas.csv

# Sensor filter
python3 query_assist.py --uprn 5045394 --sensor bess:OusterLidarSensor

# Type filter
python3 query_assist.py --uprn 5045394 --types did:rgb-image,did:lidar-pointcloud-merged

# Custom SPARQL endpoint
python3 query_assist.py --uprn 200003455212 --db-url http://myhost:3030/mytriplestore/query

# Custom download directory
python3 query_assist.py --uprn 5045394 --download-dir /data/assets

# Custom API key env var
export MY_KEY="..."
python3 query_assist.py --uprn 5045394 --api-key-env MY_KEY

# A Few options at once
export MY_KEY="..."
python3 query_assist.py \
  --uprn 200003455212,5045394 \
  --sensor bess:OusterLidarSensor \
  --types did:lidar-pointcloud-merged \
  --db-url http://myhost:3030/mytriplestore/query \
  --download-dir /mnt/data/downloads \
  --api-key-env MY_KEY
```

Run `python3 query_assist.py -h` to see the full list of command-line options and examples.


# Additional Data Information

## RGB

sRGB images are provided at a resolution optimised for computer vision tasks. Vehicles and humans are masked out, if a user finds an unmasked person or vehicle (most critically the number plate), please report it to xRI.


## IR

The outermost regions of the IR images and temperature arrays have been masked out, this is due to hot edges due to the IR detector heating itself up during operation. In the temperature arrays the masked areas are NaN elements in the compressed numpy array.

Additionally when working with the IR data there are some assumptions to note about the way in which radiometic temperature pixels themselves are calculated.

- The formula requires the distance of each pixel from the detector, currently this is a sensible hard coded value, we are in the process of calculating these distances from the lidar.
- Building materials tend to be in a narrow range of emissivities $\epsilon\in [0.85,0.93]$, we currently hard code a single sensible value for emissivity but are developing methods for estimating building materials dynamically.
- The formula only governs a certain region of materials in a number of variables. This means that temperature arrays calculated for buildings during the night are valid sources of data for understanding temperature.
- During the day, we are in a reflectance dominated regime due to the influence of the sun, radiometric temperatures calculated in this regime are not reliable.
- The sky is an object outside the scope of the radiometric temperature calculation, this is a low reflectance, low emissivity regime that our radiometric temperature calculations cannot say anything meaningful about.

## LiDAR

We have four 360 degree grey scale panormas these are:

- Near-infrared (NIR) capturing light in the near-infrared spectrum (just beyond visible light). NIR is often used to assess vegetation health, surface properties, and for capturing detailed textures in low-light conditions.

- The range modality provides the distance from the LiDAR sensor to objects in the environment. Each pixel in this image represents a distance measurement in millimeters, creating a depth map of the scene.

- The reflectivity image captures the intensity of the LiDAR signal that bounces back to the sensor. Reflectivity depends on the surface material and angle of incidence, making it useful for distinguishing between materials or identifying road markings, signs, and other objects.

- The signal strength or return signal intensity measures the quality of the LiDAR return. Stronger signals usually indicate clearer, more reliable measurements. It can also reflect surface properties and environmental conditions.

We also have two pointclouds one is a single frame that is closest to orthogonal to the UPRN, the other is a dense, orchstrated pointcloud created by merging many pointcloud frames on either side of the most orthogonal frame using the [Iterative Closes Point (ICP) registration algorithm](http://ki-www.cvl.iis.u-tokyo.ac.jp/class2013/2013w/paper/correspondingAndRegistration/03_Levoy.pdf).

ICP registration can also fail completely resulting in dense but unaligned pointclouds. A single centre frame has been provided as a failback pointcloud in the event of an unusable merged pointcloud.
