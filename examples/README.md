# Python examples 

This directory contains a few Python scripts that will load and provision the graph database with provided turtle (.ttl) files and execute queries against it. Some will also then use results from queries to download assets from the API.

## Setup

### 1. Create a virtual environment

```
cd examples
python3 -m venv venv
```

### 2. Activate and install requirements

**macOS/Linux**
```
source venv/bin/activate
```

**Windows (CMD)**
```
venv\Scripts\activate.bat
```

**Windows (PowerShell)**
```
venv\Scripts\Activate.ps1
```

Install dependencies.
```
pip install -r requirements.txt
```

### 3. Set your API key

**Temporary (session only)**
```
export API_KEY="your_api_key"
```

**Permanent (automatic when Virtualenv is activated)**

If you want this variable to be automatically set every time you activate your virtual environment, add the export line to the activate script inside your virtual environment. e.g. for macOS/Linux

Open `venv/bin/activate` and add the environment variable near the bottom (but before the final `unset` lines if present).

## Running scripts

Once your environment is set up then you can run any of the `.py` files in the `/examples` directory. **Before you run a script** verify that the constants in the file are set correctly to match your local environment.

### Scripts

Granular scripts to improve legibility when viewing the SPARQL queries:
- `get_all_assets_for_a_list_of_uprns.py`
- `get_all_assets_for_a_uprn.py`
- `get_all_assets_for_a_uprn_made_by_a_sensor.py`
- `get_all_assets_of_type_for_list_of_uprns.py`

Unified script:
- `download_assets.py`

This unified script replaces and extends the above utilities by allowing you to:

- Specify a **single** UPRN (`--uprn`) or a **list** of UPRNs (`--uprns`).
- Provide a **CSV file** with a column named `uprn` (`--csv`).
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
- **Lidar range panorama images**: `did:lidar-range-pano`
- **Lidar reflectance for panorama**: `did:lidar-reflectance-pano`
- **Temperature in celsius** (no contentUrl): `did:celsius-temperature`
- **Lidar signal intensity for panoramas**: `did:lidar-signal-pano`
- **Lidar Near Infrared for panoramas**: `did:lidar-nearir-pano`
- **Relative humidity** (no contentUrl): `did:relative-humidity`
- **Pointcloud frame**: `did:lidar-pointcloud-frame`
- **IR false colour**: `did:ir-false-color-image`
- **IR temperature array**: `did:ir-temperature-array`
- **IR counts**: `did:ir-count-image`
- **RGB image**: `did:rgb-image`

#### Usage

```bash
./download_assets.py --uprn 5045394

./download_assets.py --uprns 200003455212,5045394

./download_assets.py --csv path/to/uprns.csv --types did:rgb-image

./download_assets.py --uprn 5045394 \
  --sensor bess:OusterLidarSensor \
  --types did:lidar-pointcloud-merged \
  --db-url http://myhost:3030/mytriplestore/query \
  --download-dir /data/assets \
  --api-key-env MY_KEY
```

Run `./download_assets.py -h` to see the full list of command-line options and examples.
