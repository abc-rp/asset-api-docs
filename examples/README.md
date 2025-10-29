# Python examples

This directory contains Python scripts for interacting with the DID triplestore (SPARQL endpoint) and the Asset API:

1. `query_assist.py` – Structured CLI for downloading assets, listing UPRNs by output area, or mapping ODS→UPRN.
2. `nl_query_assist.py` – Natural language (NL) wrapper that plans and executes one- or two-stage workflows using `query_assist.py` underneath.

The scripts can:

- Discover UPRNs by output area codes.
- Map NHS ODS codes to UPRNs (with recommendation codes).
- Download assets for specified UPRNs with optional sensor and asset-type filters.
- Accept CSV file inputs for batch operations.
- Use NL instructions (via `nl_query_assist.py`) to infer plans automatically.

## Setup

### 1. Create a virtual environment

```bash
cd examples
python3 -m venv venv
```

### 2. Activate environment

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

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

**Temporary (session only)**
```bash
export API_KEY="your_api_key"
```

## Running the structured CLI (`query_assist.py`)

`query_assist.py` supports three modes (mutually exclusive per invocation):

1. Asset download: `--uprn` (one/many or CSV path with column `uprn`)
2. Output area → UPRN listing: `--output-area` / `--oa` (codes or CSV path column `output_area`)
3. ODS → UPRN mapping: `--ods` (codes or CSV path column `ods`)

Optional filters / overrides:
- `--sensor bess:OusterLidarSensor` (or other supported sensor IRI)
- `--types did:rgb-image,did:lidar-pointcloud-merged` (comma separated IRIs)
- `--db-url http://host:3030/didtriplestore/query` (override SPARQL endpoint)
- `--download-dir /path/to/downloads` (default `./downloads`)
- `--api-key-env MY_KEY` (environment var containing API key; default `API_KEY`)

Example usages:

```bash
# Single UPRN asset download
python3 query_assist.py --uprn 100023334911

# Multiple UPRNs (space separated)
python3 query_assist.py --uprn 100023334911 100023268138

# Multiple UPRNs (comma separated in one argument)
python3 query_assist.py --uprn 100023334911,100023268138,46251044

# CSV of UPRNs
python3 query_assist.py --uprn path/to/uprns.csv

# ODS→UPRN mapping
python3 query_assist.py --ods G85013

# Output areas → UPRN listing (mixed raw codes)
python3 query_assist.py --output-area E00004550 E00032882 E00063193 E00047411

# CSV of output areas
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

# Multiple options combined
export MY_KEY="..."
python3 query_assist.py \
  --uprn 200003455212,5045394 \
  --sensor bess:OusterLidarSensor \
  --types did:lidar-pointcloud-merged \
  --db-url http://myhost:3030/mytriplestore/query \
  --download-dir /mnt/data/downloads \
  --api-key-env MY_KEY
```

Run `python3 query_assist.py -h` for full help text.

## Natural language workflow (`nl_query_assist.py`)

`nl_query_assist.py` lets you describe tasks conversationally; it plans steps (e.g. output-area lookup → asset download) and calls `query_assist.py` accordingly.

Prerequisite: An [Ollama](https://ollama.com/) server must be running locally (or remotely) with the desired model already pulled. Set the server URL via `export OLLAMA_HOST=http://host:port` (defaults to `http://localhost:11434`). Pull a model first, e.g.:

```bash
ollama pull gpt-oss:20b
```

If you use a different model tag, pass it with `--model-id`.

Key flags:
- `--once "your NL request"` run a single NL instruction and exit.
- `--dry-run` plan and show commands without executing downloads.
- `--plan-only` output the inferred plan (JSON-like) and exit.
- `--model-id gpt-oss:20b` choose Ollama model (set `OLLAMA_HOST` to change server URL).
- Decoding knobs: `--temperature`, `--top-p`, `--num-predict`, `--num-ctx`, `--keep-alive`, `--no-force-json`.

Interactive session:
```bash
python3 nl_query_assist.py
> download merged lidar point clouds and rgb images for UPRNs 5045394 and 200003455212 into /tmp/assets
```

Single command:
```bash
python3 nl_query_assist.py --once "list UPRNs in output areas E00004550 and E00032882 then download rgb images"
```

Dry run:
```bash
python3 nl_query_assist.py --dry-run --once "download point clouds for ODS G85013"
```

Verbose (show planning internals):
```bash
python3 nl_query_assist.py -vv --once "rgb images for UPRNs in areas E00004550,E00032882"
```

## Supported sensors

- `bess:PhidgetHumiditySensor`
- `bess:PhidgetTemperatureSensor`
- `bess:OusterLidarSensor`
- `bess:FlirOryxCamera`
- `bess:FlirA70Camera`

## Supported asset types

- Merged lidar point clouds: `did:lidar-pointcloud-merged`
- Pointcloud frame: `did:lidar-pointcloud-frame`
- Lidar range panorama images: `did:lidar-range-pano`
- Lidar reflectance panorama images: `did:lidar-reflectance-pano`
- Lidar signal intensity panorama images: `did:lidar-signal-pano`
- Lidar Near Infrared panorama images: `did:lidar-nearir-pano`
- Temperature in celsius (no contentUrl): `did:celsius-temperature`
- Relative humidity (no contentUrl): `did:relative-humidity`
- IR false colour images: `did:ir-false-color-image`
- IR temperature arrays: `did:ir-temperature-array`
- IR counts images: `did:ir-count-image`
- RGB images: `did:rgb-image`

Point clouds are now provided as LAZ (.laz) compressed files. Most point cloud processing tools (e.g. PDAL, CloudCompare, Potree converters) handle `.laz` directly—no manual decompression step is required.

## Additional Data Information

### RGB

sRGB images are optimised for computer vision tasks. Vehicles and humans are masked out automatically. Please report any unmasked person or vehicle (especially number plates) to [xRI](mailto:info@xri.online).

### IR

Edge regions are masked due to sensor heating. In temperature arrays masked regions are NaN. Radiometric assumptions:

1. Pixel distance is currently a sensible hard-coded value (dynamic derivation from LiDAR is in progress).
2. Emissivity is assumed constant (typical building materials fall in $\epsilon \in [0.85, 0.93]$; dynamic estimation is in development).
3. Daytime data is reflectance dominated; radiometric temperatures are only provided for night hours (1h after sunset to 1h before sunrise).
4. Sky regions fall outside reliable radiometric interpretation and are excluded.

### LiDAR

Four 360° grayscale panoramas are provided:

- Near-infrared (NIR): captures near-infrared spectrum for vegetation and surface texture analysis.
- Range: distance (mm) from sensor to objects (depth map).
- Reflectance: intensity of returned signal (material/angle dependent).
- Signal strength: quality of LiDAR return (helps assess reliability & environmental conditions).

Point cloud modalities:

- Merged point cloud: dense, registered aggregate from multiple frames using Iterative Closest Point (ICP).
- Single frame point cloud: most orthogonal frame (fallback if ICP merge is unusable).

ICP can fail, producing dense but misaligned merged clouds; use the single frame as a fallback.

## Troubleshooting

- Missing downloads? Ensure the API key environment variable (`API_KEY` or your override) is exported in the same shell session.
- Empty CSV outputs: Verify the codes (UPRN / ODS / Output area) exist in the triplestore and that `--db-url` is correct.
- Slow queries: Consider filtering with `--types` and/or `--sensor` to reduce result size.
- NL planning returns "No actionable plan": add explicit codes (e.g. UPRNs) or clarify intent ("download rgb images" vs. "rgb").

## License

See the root `LICENSE` file for details.
