#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import textwrap
import re

import json
import requests
import tempfile
import shlex
import subprocess
import sys

from thefuzz import process
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import Any, List, Optional, Union
from heuristic_parser import HeuristicParser

SYSTEM_ROUTER_PROMPT = """You are a rigorous function-call router for a Python CLI named query_assist.py.

There is one supported command called download_assets that downloads assets based on a number of possible filters. It is populated as follows:

   Optional:   identifier (types: uprn, ods, toid. May be a string CSV path like ["uprn.csv"] OR array of strings like ["5045394","200003455212"] OR a mixture of the two like ["uprn.csv", "6384826492"]. If the user provides a file path ending in `.csv`, you MUST treat the file path string itself as a value. Do NOT try to invent the contents of the file.)
               geography (types: output-area, lower-layer-super-output-area, ward, london-borough, postcode, natural-name; may be a string CSV path like ["output-area.csv"] OR array of strings like ["E0001234", "E0007628"] OR a mixture of the two such as ["output-area.csv", "E0001234"]. If the user does not supply a code or csv file path, but rather a common or human readable name for a geographical area, please place that name into the natural-name section e.g. ["camden", "tower hamlets"]. If the user provides a file path ending in `.csv`, you MUST treat the file path string itself as a value. Do NOT try to invent the contents of the file.)
               filter (types: see below for list of types; the datatypes are also specified below)
               sensor (string, e.g., "bess:OusterLidarSensor")
               types  (array of strings; each a type IRI, e.g., ["did:rgb-image","did:lidar-pointcloud-merged"])
               download_dir (string path)
               api_key_env  (string, name of env var with API key)
               db_url       (string URL to SPARQL endpoint)
    Other:     ontop-url (required IF user specifies toid, geography filters, or property filters)

Schema (MUST output exactly one JSON object with these keys as needed):
{
  "identifier": {
    "uprn":           string[] | null,
    "ods":            string[] | null,
    "toid":           string[] | null
  },
  "geography": {
    "output-area":    string[] | null,
    "lower-layer-super-output-area": string[] | null,
    "ward":           string[] | null,
    "london-borough": string[] | null,
    "postcode":      string[] | null,
    "natural-name": string[] | null
  },
  
  "filter": {
    "property-type":  string[] | null,
    "built-form":     string[] | null,
    "accommodation-type": string[] | null,
    "tenure":         string[] | null,
    "epc-rating":   string[] | null,
    "potential-epc-rating": string[] | null,
    "construction-age-band": string[] | null,
    "building-use": string[] | null,
    "main-heating-system": string[] | null,
    "main-fuel-type": string[] | null,
    "wall-type": string[] | null,
    "roof-type": string[] | null,
    "glazing-type": string[] | null,
    "loac-group": string[] | null,
    "loac-supergroup": string[] | null,
    "listed-building-grade": string[] | null,
    "heat-risk-quintile": An array of INTEGERS taking values from 1 to 5 (inclusive) | null,
    "imd19-income-decile": An array of INTEGERS taking values from 1 to 10 (inclusive) | null,
    "imd19-national-decile": An array of INTEGERS taking values from 1 to 10 (inclusive) | null,
    "wall-insulation": boolean as string inside array (e.g. ["true"]) | null,   
    "roof-insulation": boolean as string inside array (e.g. ["true"]) | null,
    "in-conservation-area": boolean as string inside array (e.g. ["true"]) | null,
    "epc-score":  array of one OR two INTEGERS | null,
    "potential-epc-score": array of one OR two INTEGERS | null,
    "floor-count": array of one OR two INTEGERS | null,
    "basement-floor-count": array of one OR two INTEGERS | null,
    "number-of-habitable-rooms": array of one OR two INTEGERS | null,
    "easting": array of one OR two INTEGERS | null,
    "northing": array of one OR two INTEGERS | null,
    "total-floor-area": array of one or two INTEGERS | null,
    "energy-consumption": array of one or two INTEGERS | null,
    "solar-pv-area": array of one or two INTEGERS | null,
    "solar-pv-potential": array of one or two DECIMALS | null,
    "average-roof-tilt": array of one or two INTEGERS | null
    "fuel-poverty": array of one or two DECIMALS | null
  },
  "sensor":          string | string[] | null,
  "types":           string[] | null,
  "download-dir":    string | null,
  "api-key-env":     string | null,
  "db-url":          string | null,
  "ontop-url":       string | null
}


Constraints:
- Return ONLY the JSON object. No prose, no markdown.
- If the user request implies some identifiers but does not give a type, please deduce the best fit based on the structure of the identifier:
  - UPRNs are numeric, at least 6 digits, e.g., "5045394" or "200003455212"
  - ODS codes start with a letter followed by 5 digits, e.g., "A12345"
  - TOIDs are numeric, at least 10 digits e.g., "1000006033182"
  - Postcodes are alphanumeric, e.g., "SW1A 1AA"
  - Output areas start with "E" followed by 8 digits, e.g., "E0001234"
  - Lower Layer Super Output Areas start with "E" followed by 9 digits, e.g., "E00001234"
  - Wards start with "E" followed by 7 digits, e.g., "E0501234"
  - London Boroughs start with "E09" followed by 4 digits, e.g., "E09000001" 
- Some filter types only accept a limited set of values. Please try to map user input to these values (case-sensitive):
  - property-type: flat, house, park-home-caravan
  - built-form: detached, semi-detached, end-terrace, mid-terrace
  - accommodation-type: flat, semi-detached-house, detached-house, end-terraced-house, mid-terraced-house, park-home-caravan
  - tenure: owner-occupied, social-housing, privately-rented
  - epc-rating: AB, C, D, E, FG
  - potential-epc-rating: AB, C, D, E, FG
  - construction-age-band: pre-1900, 1900-1929, 1930-1949, 1950-1966, 1967-1982, 1983-1995, 1996-2011, 2012-onwards
  - building-use: residential-only, mixed-use
  - main-heating-system: boiler, room-storage-heaters, heat-pump, communal, none, other
  - main-fuel-type: mains-gas, electricity, no-heating-system, other
  - wall-type: cavity, solid, other
  - roof-type: pitched, flat, room-in-roof, another-dwelling-above
  - glazing-type: single-partial, secondary, double-triple
  - loac-supergroup: A, B, C, D, E, F, G
  - loac-group: A1, A2, A3, B1, B2, C1, C2, D1, D2, D3, E1, E2, F1, F2, G1, G2
  - listed-building-grade: I, II, IIStar, Unknown
- If the user request implies sensor types or similar, map them to the supported IRIs if possible:
  - lidar -> bess:OusterLidarSensor
  - ir-camera -> bess:FlirOryxCamera
  - rgb-camera -> bess:FlirA70Camera
  - ins -> bess:LordMicrostrainINSGQ7
  - temperature -> bess:PhidgetTemperatureSensor
  - humidity -> bess:PhidgetHumiditySensor
- If the user request implies asset types, map them to the supported IRIs if possible:
  - RGB image -> "did:rgb-image"
  - merged lidar point cloud -> "did:lidar-pointcloud-merged"
  - lidar range panorama -> "did:lidar-range-pano"
  - lidar reflectance panorama -> "did:lidar-reflectance-pano"
  - lidar signal panorama -> "did:lidar-signal-pano"
  - lidar near-infrared panorama -> "did:lidar-nearir-pano"
  - IR false color -> "did:ir-false-color-image"
  - IR temperature array -> "did:ir-temperature-array"
  - IR counts -> "did:ir-count-image"
  - temperature (no contentUrl) -> "did:celsius-temperature"
  - relative humidity (no contentUrl) -> "did:relative-humidity"
- UPRNs are the UK OS Unique Property Reference Numbers. Queries may call them buildings or other built environment associated words.
- Output areas may be called OAs and lower layer super output areas may be called LSOAs. Wards may be called electoral wards or just wards. London boroughs may be called boroughs or London boroughs, sometimes administrative areas.
- ODS codes are unique identifiers for UK NHS buildings, hence words like medical, practice, hospital, etc... may be used.
- Prefer being decisive. When in doubt, infer sensible defaults.
"""

# Define data schema
class IdentifierModel(BaseModel):
    uprn: Optional[List[str]] = None
    ods: Optional[List[str]] = None
    toid: Optional[List[str]] = None

class GeographyModel(BaseModel):
    output_area: Optional[List[str]] = Field(None, alias="output-area")
    lower_layer_super_output_area: Optional[List[str]] = Field(None, alias="lower-layer-super-output-area")
    ward: Optional[List[str]] = None
    london_borough: Optional[List[str]] = Field(None, alias="london-borough")
    postcode: Optional[List[str]] = None
    natural_name: Optional[List[str]] = Field(None, alias="natural-name")

class FilterModel(BaseModel):
    property_type: Optional[List[str]] = Field(None, alias="property-type")
    built_form: Optional[List[str]] = Field(None, alias="built-form")
    accommodation_type: Optional[List[str]] = Field(None, alias="accommodation-type")
    tenure: Optional[List[str]] = None
    epc_rating: Optional[List[str]] = Field(None, alias="epc-rating")
    potential_epc_rating: Optional[List[str]] = Field(None, alias="potential-epc-rating")
    construction_age_band: Optional[List[str]] = Field(None, alias="construction-age-band")
    building_use: Optional[List[str]] = Field(None, alias="building-use")
    main_heating_system: Optional[List[str]] = Field(None, alias="main-heating-system")
    main_fuel_type: Optional[List[str]] = Field(None, alias="main-fuel-type")
    glazing_type: Optional[List[str]] = Field(None, alias="glazing-type")
    wall_type: Optional[List[str]] = Field(None, alias="wall-type")
    roof_type: Optional[List[str]] = Field(None, alias="roof-type")
    loac_group: Optional[List[str]] = Field(None, alias="loac-group")
    loac_supergroup: Optional[List[str]] = Field(None, alias="loac-supergroup")
    listed_building_grade: Optional[List[str]] = Field(None, alias="listed-building-grade")
    heat_risk_quintile: Optional[List[int]] = Field(None, alias="heat-risk-quintile")
    imd19_income_decile: Optional[List[int]] = Field(None, alias="imd19-income-decile")
    imd19_national_decile: Optional[List[int]] = Field(None, alias="imd19-national-decile")
    roof_insulation: Optional[List[str]] = Field(None, alias="roof-insulation")
    in_conservation_area: Optional[List[str]] = Field(None, alias="in-conservation-area")
    energy_consumption: Optional[List[Union[float, int]]] = Field(None, alias="energy-consumption")
    solar_pv_area: Optional[List[Union[float, int]]] = Field(None, alias="solar-pv-area")
    solar_pv_potential: Optional[List[Union[float, int]]] = Field(None, alias="solar-pv-potential")
    average_roof_tilt: Optional[List[Union[float, int]]] = Field(None, alias="average-roof-tilt")
    total_floor_area: Optional[List[Union[float, int]]] = Field(None, alias="total-floor-area")
    epc_score: Optional[List[int]] = Field(None, alias="epc-score")
    potential_epc_score: Optional[List[int]] = Field(None, alias="potential-epc-score")
    floor_count: Optional[List[int]] = Field(None, alias="floor-count")
    basement_floor_count: Optional[List[int]] = Field(None, alias="basement-floor-count")
    number_of_habitable_rooms: Optional[List[int]] = Field(None, alias="number-of-habitable-rooms")
    easting: Optional[List[int]] = None
    northing: Optional[List[int]] = None
    fuel_poverty: Optional[List[Union[float, int]]] = Field(None, alias="fuel-poverty")
    wall_insulation: Optional[List[str]] = Field(None, alias="wall-insulation")

class QueryAssistConfig(BaseModel):
    identifier: Optional[IdentifierModel] = None
    geography: Optional[GeographyModel] = None
    filter: Optional[FilterModel] = None
    sensor: Optional[Union[str, List[str]]] = None
    types: Optional[List[str]] = None
    download_dir: Optional[str] = Field(None, alias="download-dir")
    api_key_env: Optional[str] = Field(None, alias="api-key-env")
    db_url: Optional[str] = Field(None, alias="db-url")
    ontop_url: Optional[str] = Field(None, alias="ontop-url")

def _render_box(title: str, body: str) -> str:
    term_width = shutil.get_terminal_size(fallback=(100, 24)).columns
    max_width = max(60, min(term_width - 2, 100))
    wrap_width = max_width - 4
    body_lines = []
    for para in body.splitlines():
        if not para.strip():
            body_lines.append("")
        else:
            body_lines.extend(textwrap.wrap(para, width=wrap_width))
    title = title.strip()
    title_line = f" {title} "
    top = "┌" + "─" * (max_width - 2) + "┐"
    sep = "├" + "─" * (max_width - 2) + "┤"
    bot = "└" + "─" * (max_width - 2) + "┘"
    if len(title_line) <= (max_width - 2):
        left = (max_width - 2 - len(title_line)) // 2
        right = max_width - 2 - len(title_line) - left
        top = "┌" + "─" * left + title_line + "─" * right + "┐"
    content = "\n".join("│ " + line.ljust(max_width - 4) + " │" for line in body_lines)
    return "\n".join([top, sep, content, bot])


def _extract_first_json(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, c in enumerate(text[start:], start=start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None
    return None

def generate_config_from_nl(
    nl: str, 
    args: argparse.Namespace, 
    llm_opts: dict
) -> dict | None:
    """
    Uses an LLM to transform a natural language query into a JSON config object.
    """
    messages = [
        {"role": "system", "content": SYSTEM_ROUTER_PROMPT},
        {"role": "user", "content": nl},
    ]
    try:
        if args.api == "openai":
            logging.info(f"Sending request to OpenAI model: {args.openai_model}")
            resp = openai_chat(
                model=args.openai_model,
                messages=messages,
                **llm_opts
            )
        else: 
            logging.info(f"Sending request to Ollama model: {args.model_id}")
            resp = ollama_chat(
                base_url=args.base_url,
                model=args.model_id,
                messages=messages,
                **llm_opts
            )
        content = resp.get("message", {}).get("content", "")
        if not content:
            return None
        
        config_obj = _extract_first_json(content)
        if not isinstance(config_obj, dict):
            logging.error(f"LLM did not return a valid JSON object. Response:\n{content}")
            return None
        
        try:
            # Validate the raw dictionary against Pydantic schema
            valid_config = QueryAssistConfig.model_validate(config_obj)
            
            validated_dict = valid_config.model_dump(by_alias=True, exclude_none=True)
            
            logging.info(f"LLM generated valid config:\n{json.dumps(validated_dict, indent=2)}")
            return validated_dict
        
        except ValidationError as e:
            logging.error(f"LLM JSON is malformed. Validation failed:\n{e}")
            return None
            
    except Exception as e:
        logging.error(f"Error calling LLM for config generation: {e}")
        return None

def ollama_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    top_p: float = 0.95,
    num_predict: int = 512,
    num_ctx: int | None = None,
    keep_alive: str | None = None,
    force_json: bool = True,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(num_predict),
        },
    }
    if num_ctx is not None:
        payload["options"]["num_ctx"] = int(num_ctx)
    if keep_alive:
        payload["keep_alive"] = str(keep_alive)
    if force_json:
        payload["format"] = "json"
    r = requests.post(url, json=payload, timeout=(5.0, timeout_s))
    r.raise_for_status()
    return r.json()
    
def openai_chat(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    top_p: float = 0.95,
    force_json: bool = True,
    **kwargs # To catch unused ollama-specific args
) -> dict:
    """
    Sends a chat request to the OpenAI API and returns the response
    in a format consistent with ollama_chat.
    """
    try:
        # The client automatically finds the OPENAI_API_KEY from your environment
        client = OpenAI()
        
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        # OpenAI"s method for forcing JSON output
        if force_json:
            request_params["response_format"] = {"type": "json_object"}

        completion = client.chat.completions.create(**request_params)
        
        content = completion.choices[0].message.content

        # IMPORTANT: We wrap the response to match the structure of ollama_chat
        # so the rest of our code doesn"t need to change.
        return {
            "message": {
                "content": content
            }
        }
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return {"message": {"content": ""}} # Return empty content on error

def load_gazetteer(path: str = "gazetteer.json") -> dict:
    """Loads the name-to-code mapping file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Gazetteer file not found at {path}. Geographical name matching will be disabled.")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Gazetteer file at {path} is not valid JSON.")
        return {}
    
def process_geographical_names(config: dict, gazetteer: dict, score_cutoff: int = 85) -> dict:
    """
    Post-processes the LLM-generated config by resolving a list of mixed
    geographical names into their appropriate ONS codes and categories.

    It reads from config["geography"]["natural-name"], finds the best match for each
    name across all types in the gazetteer, and then populates the config with
    categorized codes, e.g., config["geography"]["london-borough"] = ["E09000007"].
    """
    if not gazetteer:
        logging.warning("Gazetteer not loaded. Skipping geographical name resolution.")
        return config

    geography_section = config.get("geography", {})
    if not isinstance(geography_section, dict):
        return config
    
    natural_names = geography_section.get("natural-name")
    if not isinstance(natural_names, list) or not natural_names:
        return config

    # This mapping is used to convert the gazetteer's internal keys
    # to the final keys expected in the config.
    relabelled_geo = {
        "administrative-area": "london-borough",
        "ward": "ward",
    }
    
    resolved_geographies = {}

    # Iterate through each name provided by the LLM.
    for name in natural_names:
        if not isinstance(name, str) or not name.strip():
            continue

        clean_name = name.strip().lower()
        
        best_overall_score = -1
        best_overall_code = None
        best_overall_type = None
        best_canonical_name = None

        # For each name, search across all geography types in the gazetteer.
        for gazetteer_type, name_to_code_map in gazetteer.items():
            if not isinstance(name_to_code_map, dict): continue

            choices = list(name_to_code_map.keys())
            if not choices: continue

            match, score = process.extractOne(clean_name, choices)

            if score > best_overall_score:
                best_overall_score = score
                best_canonical_name = match
                best_overall_code = name_to_code_map[match]
                best_overall_type = relabelled_geo.get(gazetteer_type, gazetteer_type)

        # After checking all categories, decide if the best match is good enough.
        if best_overall_score >= score_cutoff:
            logging.info(
                f"Resolved '{name}' -> '{best_canonical_name}' (type: {best_overall_type}) with score {best_overall_score}."
            )
            # Add the resolved code to the correct list in our results dictionary.
            resolved_geographies.setdefault(best_overall_type, []).append(best_overall_code)
        else:
            logging.warning(
                f"Could not find a confident match for '{name}'. Best attempt was '{best_canonical_name}' "
                f"(score: {best_overall_score}, cutoff: {score_cutoff}). Ignoring."
            )
            
    # Update the original config
    if "natural-name" in geography_section:
        del geography_section["natural-name"]

    geography_section.update(resolved_geographies)

    return config


def find_best_match(query: str, choices: list[str], score_cutoff: int = 85) -> str | None:
    """
    Finds the best match for a query string from a list of choices.

    Args:
        query: The string to match (e.g., "tower hamlets").
        choices: A list of canonical names to match against.
        score_cutoff: The minimum similarity score (0-100) to consider a match valid.

    Returns:
        The best matching choice, or None if no match exceeds the cutoff score.
    """
    if not query or not choices:
        return None
    
    # process.extractOne returns a tuple of (best_match, score)
    best_match, score = process.extractOne(query, choices)

    if score >= score_cutoff:
        print(f"Matched \"{query}\" to \"{best_match}\" with score {score}")
        return best_match
    else:
        print(f"No confident match for \"{query}\". Best attempt \"{best_match}\" had score {score} (cutoff: {score_cutoff}).")
        return None

def run_query_assist_with_config(
    config: dict, py_exe: str, qa_path: str, dry_run: bool
) -> tuple[int, str]:
    """
    Executes the new query_assist.py by writing the config to a temporary
    JSON file and passing its path to the --config argument.
    """
    # Use a temporary file that is automatically deleted on exit
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_f:
        json.dump(config, temp_f, indent=2)
        temp_fname = temp_f.name

    argv = [py_exe, qa_path, "--config", temp_fname]
    printable = " ".join(shlex.quote(x) for x in argv)
    logging.info("Generated Config File: %s", temp_fname)
    logging.info("Command: %s", printable)

    if dry_run:
        with open(temp_fname, "r") as f:
            content = f.read()
            print(f"\n--- [dry-run] Config File Content ({temp_fname}) ---\n{content}\n----------------------------------------------------")
        os.remove(temp_fname)
        return 0, f"[dry-run] {printable}\n"

    # Execute the process
    p = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    captured_lines: list[str] = []
    try:
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            captured_lines.append(line)
    finally:
        rc = p.wait()
        os.remove(temp_fname) # Always clean up the temp file

    return rc, "".join(captured_lines)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="LangGraph NL workflow for query_assist.py (one- and two-stage)"
    )
    ap.add_argument(
        "--api", 
        default="ollama", 
        choices=["ollama", "openai"], 
        help="The LLM model to use."
    )
    ap.add_argument(
        "--force-llm",
        action="store_true",
        help="Force using the LLM even if heuristics succeed.",
    )
    ap.add_argument("--model-id", default="gpt-oss:20b", help="Ollama model name/tag")
    ap.add_argument(
        "--openai-model", 
        default="gpt-4o-mini", 
        help="OpenAI model name (e.g., gpt-4o-mini, gpt-4-turbo)."
    )
    ap.add_argument(
        "--query-assist-path",
        default=os.path.join(os.path.dirname(__file__), "query_assist.py"),
        help="Path to query_assist.py",
    )
    ap.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Base URL of the Ollama server (or set OLLAMA_HOST)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan & print commands but do not execute",
    )
    ap.add_argument("--once", "-q", help="Run a single NL query and exit")

    # Decoding/runtime knobs
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-predict", type=int, default=512)
    ap.add_argument("--num-ctx", type=int, default=None)
    ap.add_argument("--keep-alive", default=None)
    ap.add_argument("--no-force-json", action="store_true")

    # Logging controls
    ap.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v=info, -vv=debug"
    )
    return ap.parse_args()

def clean_empty_nodes(d: Any) -> Any:
    """
    Recursively removes keys from a dictionary if their value is None, [], or {}.
    """
    if isinstance(d, dict):
        # Create a new dict to avoid issues with changing dict size during iteration
        return {
            k: clean_empty_nodes(v)
            for k, v in d.items()
            if v is not None and v != [] and v != {}
        }
    return d

def is_config_sufficient_for_download(config: dict) -> bool:
    """
    Validates that the generated config is runnable.
    """
    if not config:
        return False
    # A query is sufficient if it has at least one identifier or geography constraint.
    # Otherwise, it might try to download assets for the entire database.
    has_identifier = "identifier" in config and config["identifier"]
    has_geography = "geography" in config and config["geography"]

    if not (has_identifier or has_geography):
        logging.error("Validation failed: The query is too broad. Please specify a geography or identifier constraint, e.g., UPRN, ODS code, output area, ward, etc.")
        return False

    return True

def main() -> None:
    args = parse_args() 

    # Configure logging
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    if level <= logging.INFO:
        body = (
            "• Parses natural language into a JSON configuration.\n"
            "• Executes query_assist.py with the generated config.\n"
            "• Supports geographical name resolution, CSV inputs, and filters."
        )
        print(_render_box(f"Query Assist NL Interface — {args.api}", body))

    def run_once(nl: str, args: argparse.Namespace):
        gazetteer = load_gazetteer("gazetteer.json")
        use_llm = False
        config = None

        print("Interpreting your request...")

        # 1. Try heuristic parsing first
        if not args.force_llm:
            parser = HeuristicParser(gazetteer)
            config = parser.parse(nl)

            if config is None:
                logging.info("Heuristic check failed, falling back to LLM for complex query.")
                use_llm = True
        else: 
            use_llm = True
            logging.info("--force-llm flag detected. Skipping heuristic parser.")
        
        # 2. Generate the config from Natural Language
        llm_options = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
            "keep_alive": args.keep_alive,
            "force_json": not args.no_force_json,
        }
        if config is None:
            config = generate_config_from_nl(
                nl,
                args,
                llm_options
            )

        if not config:
            logging.error("Could not generate a valid configuration from your query.")
            return 1
        
        # 3. Post-process the config if an LLM was used to resolve geographical names
        if use_llm:
            print("Resolving geographical names...")
            processed_config = process_geographical_names(config, gazetteer)
            logging.info(f"Processed config:\n{json.dumps(processed_config, indent=2)}")

        else:
            processed_config = config
            logging.info(f"Heuristically generated config:\n{json.dumps(processed_config, indent=2)}")

        # 4. Validate + clean the config
        if not is_config_sufficient_for_download(processed_config):
            logging.error("Aborting due to insufficient configuration.")
            return 1 
        
        processed_config = clean_empty_nodes(processed_config)
        logging.info(f"Cleaned config:\n{json.dumps(processed_config, indent=2)}")

        # 5. Execute the script with the generated config
        rc, output = run_query_assist_with_config(
            processed_config,
            py_exe=sys.executable,
            qa_path=args.query_assist_path,
            dry_run=args.dry_run,
        )

        if rc != 0:
            logging.warning(f"Workflow exited with non-zero return code: {rc}")
        
        return rc

    # Interactive loop or single-shot execution
    if args.once:
        run_once(args.once, args)
    else:
        print("Natural Language Interface for Query Assist. Type \"exit\" or Ctrl-D to quit.")
        while True:
            try:
                nl_query = input("> ")
                if nl_query.strip().lower() in {"exit", "quit"}:
                    break
                if nl_query.strip():
                    run_once(nl_query, args)
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break

if __name__ == "__main__":
    main()

