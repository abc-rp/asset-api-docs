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