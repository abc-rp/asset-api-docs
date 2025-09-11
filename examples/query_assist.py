#!/usr/bin/env python3
import argparse
import csv
import logging
import os
import re
import json
from datetime import datetime

import httpx
from rdflib.plugins.stores.sparqlstore import SPARQLStore

from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args():
    """Parses command-line arguments."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", action='append', help="Path to JSON config file")
    config_args, remaining_argv = pre_parser.parse_known_args()

    defaults = {}
    if config_args.config:
        for conf_fname in config_args.config:
            with open(conf_fname, "r") as f:
                try:
                    conf_data = json.load(f)
                    defaults.update(conf_data)
                except Exception as e:
                    raise RuntimeError(f"Failed to load config file '{conf_fname}': {e}")

    parser = argparse.ArgumentParser(
        description="Download assets from DID triplestore with optional federation across LBSM Ontop endpoint.",
        parents=[pre_parser],  
    )
    parser.set_defaults(**defaults)

    parser.add_argument(
        "--identifier",
        "--id",
        "--i",
        nargs="+",
        action="append",
        metavar=("TYPE", "VALUES"),
        help="Specify identifier type and values, e.g. --identifier uprn 123 456",
    )
    parser.add_argument(
        "--geography",
        "--geo",
        "--g",
        nargs="+",
        action="append",
        metavar=("TYPE", "VALUES"),
        help="Specify geography type and code, e.g. --geography output-area E00032882",
    )
    parser.add_argument(
        "--filter",
        "--f",
        nargs="+",
        action="append",
        metavar=("TYPE", "VALUES"),
        help="Specify additional filter conditions, e.g. --filter tenure owner-occupied or --filter epc-rating D",
    )
    parser.add_argument(
        "--sensor", 
        "--s",
        help="Sensor IRI, e.g. bess:OusterLidarSensor"
    )
    parser.add_argument(
        "--types", "--t", help="Comma-separated list of asset type IRIs, e.g. did:rgb-image"
    )
    parser.add_argument(
        "--db-url",
        default="http://ec2-18-175-116-201.eu-west-2.compute.amazonaws.com:3030/didtriplestore/query",
        help="Base graph SPARQL endpoint URL",
    )
    parser.add_argument(
        "--download-dir",
        help="Base directory for downloads (default: ./downloads)",
    )
    parser.add_argument(
        "--api-key-env",
        default="API_KEY",
        help="Environment variable name for the API key",
    )
    parser.add_argument(
        "--ontop-url",
        help="Ontop SPARQL endpoint URL",
        default="localhost:8080/sparql"
    )
    args = parser.parse_args(remaining_argv)
    return args, parser

def load_column_from_csv(path, column):
    """Loads a single column from a CSV file."""
    values = []
    with open(path, newline="") as cf:
        reader = csv.DictReader(cf)
        if column not in reader.fieldnames:
            raise RuntimeError(f"CSV {path!r} missing required '{column}' column")
        for row in reader:
            v = row[column].strip()
            if v:
                values.append(v)
    return values


def asset_subdir(enum_iri: str) -> str:
    """
    Convert an asset type IRI such as 'did:lidar-pointcloud-merged'
    into a safe sub-directory name, e.g. 'lidar-pointcloud-merged'.
    """
    name = enum_iri.split("/")[-1]
    if name.startswith("did:"):
        name = name[len("did:") :]
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)

def kebab_to_camel(kebab_str: str) -> str:
    parts = kebab_str.strip().split('-')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

def kebab_to_pascal(kebab_str: str) -> str:
    parts = kebab_str.strip().split('-')
    return ''.join(word.capitalize() for word in parts)

def kebab_to_snake(kebab_str: str) -> str:
    return kebab_str.strip().replace('-', '_')

def sensor_query(sensor):
    query = []
    if sensor is list:
        query.extend[
            f"  ?obs sosa:madeBySensor ?sensor .",
            "  VALUES ?sensor {"
        ]
        for s in sensor:
            cleaned_sensor = sensor_cleaned(s)
            if cleaned_sensor is None:
                raise ValueError(f"Unknown sensor type: {s!r}")
            query.append(f"    {cleaned_sensor}")
        query.append("  }")
    else:
        cleaned_sensor = sensor_cleaned(sensor)
        if cleaned_sensor is None:
            raise ValueError(f"Unknown sensor type: {sensor!r}")
        query.append (f"  ?obs sosa:madeBySensor {sensor_cleaned(sensor)} .")
    return query    

def identifier_query_type_1(id_map):
    query = [
        f"  ?foi so:identifier ?identifierValue1 .",
        f"  ?identifierValue1 a ?id_type_1 ; so:value ?id_value_1 .",
        "  VALUES (?id_type_1 ?id_value_1) {",
    ]
    for id_type, id_list in id_map.items():
        if id_type not in ['uprn', 'ods']:
            continue
        rdf_class = f"dob:{id_type.upper()}Value"
        for val in id_list:
            query.append(f"    ({rdf_class} {val})")
    query.append("  }")
    return query

def identifier_query_type_2(id_map):
    query = [
        f"  ?foi adms:identifier ?identifierValue2 .",
        f"  ?identifierValue2 a ?id_type_2 ; skos:notation ?id_value_2 .",
        "   VALUES (?id_type_2 ?id_value_2) {",
    ]
    for id_type, id_list in id_map.items():
        if id_type not in ["toid"]:
            continue
        rdf_class = f"dob:{id_type.upper()}Value"
        for val in id_list:
            query.append(f"     ({rdf_class} \"{val}\")")
    query.append("  }")
    return query

def geography_query(geo_map):
    query = [
        f"  ?foi geo:sfWithin ?geoArea .",
        f"  ?geoArea a ?geo_type ; adms:identifier / skos:notation ?geo_code .",
    ]
    query.append("  VALUES (?geo_type ?geo_code) {")
    for geo_type, code_list in geo_map.items():
        rdf_class = f"dob:{kebab_to_pascal(geo_type)}"
        for code in code_list:
            query.append(f'    ({rdf_class} \"{code}\")')
    query.append("  }")
    return query

def filter_query(filter_map):

    def filter_geo_query(query, rdf_property, filter_type):
        query.extend([
            f"  ?foi geo:sfWithin ?geoArea .",
            f"  ?result_{filter_type} a dob:Result ; dop:describes ?geoArea ; {rdf_property} ?{filter_type}_value .",
        ])
    
    def filter_zone_query(query, rdf_property, filter_type):
        query.extend([
            f"  ?result_{filter_type} a dob:Result ; dop:describes ?foi .",
            f"  ?result_{filter_type} {rdf_property} ?{filter_type}_value .",
        ])

    query = []

    type_1_filters = {
        "property-type", 
        "built-form",
        "accommodation-type",
        "building-use",
        "tenure",
        "epc-rating",
        "potential-epc-rating",
        "roof-type",
        "wall-type",
        "glazing-type",
        "listed-building-grade",
        "main-heating-system",
        "main-fuel-type",
        "construction-age-band"
    }
    type_2_filters = {
        "wall-insulation",
        "roof-insulation",
        "in-conservation-area",
    }
    type_3_filters = {
        "imd19-national-decile",
        "imd19-income-decile",
        "fuel-poverty",
        "heat-risk-quintile"
    }
    type_4_filters = {
        "loac-group",
        "loac-supergroup"
    }
    type_5_filters = {
        "easting",
        "northing",
        "epc-score",
        "potential-epc-score",
        "total-floor-area",
        "floor-count",
        "basement-floor-count",
        "energy-consumption",
        "solar-pv-area",
        "solar-pv-potential",
        "average-roof-tilt"
    }

    for filter_type, filter_values in filter_map.items():
        rdf_property = f"dop:{filter_property(filter_type)}"
        alt_filter = kebab_to_snake(filter_type)

        if filter_type in type_1_filters:
            filter_zone_query(query, rdf_property, alt_filter)
            query.append( f"   VALUES (?{alt_filter}_value) {{")
            for val in filter_values:
                if filter_type in {"property-type", "built-form", "accommodation-type"}:
                    rdf_class = f"dop:accommodationType-{kebab_to_camel(val)}"
                elif filter_type in {"construction-age-band", "epc-rating", "listed-building-grade"}:
                    rdf_class = f"dop:{kebab_to_camel(filter_type)}-{val}"
                elif filter_type == "potential-epc-rating":
                    rdf_class = f"dop:epcRating-{val}"
                elif filter_type in {"main-heating-system", "main-fuel-type"}:
                    rdf_class = f"dop:{kebab_to_camel(filter_type[5:])}-{kebab_to_camel(val)}"
                else:
                    rdf_class = f"dob:{kebab_to_camel(filter_type)}-{kebab_to_camel(val)}"
                query.append(f"     ({rdf_class})")
            query.append("   }")

        elif filter_type in type_2_filters:
            filter_zone_query(query, rdf_property, alt_filter)
            query.append(f"   FILTER(?{alt_filter}_value = {filter_values[0]})",)

        elif filter_type in type_3_filters:
            filter_geo_query(query, rdf_property, alt_filter)
            if len(filter_values) == 1:
                query.append(f'  FILTER(?{alt_filter}_value = {filter_values[0]})')
            elif len(filter_values) == 2:
                query.append(f'  FILTER(?{alt_filter}_value >= {min(filter_values)} && ?{filter_type[:3]}_value <= {max(filter_values)})')

        elif filter_type in type_4_filters:
            filter_geo_query(query, rdf_property, alt_filter)
            query.append(f"   VALUES (?{alt_filter}_value) {{")
            for val in filter_values:
                rdf_class = f"dob:loacGroup-{val}"
                query.append(f"     ({rdf_class})")
            query.append("   }")

        elif filter_type in type_5_filters:
            filter_zone_query(query, rdf_property, alt_filter)
            if len(filter_values) == 1:
                query.append(f'  FILTER(?{alt_filter}_value = {filter_values[0]})')
            elif len(filter_values) == 2:
                query.append(f'  FILTER(?{alt_filter}_value >= {min(filter_values)} && ?{alt_filter}_value <= {max(filter_values)})')
        
        elif filter_type == "number-of-habitable-rooms":
            filter_zone_query(query, rdf_property, alt_filter)
            query.append(f"   VALUES (?{alt_filter}_value) {{") 
            if len(filter_values) == 1:
                if filter_values[0] == '1 or 2':
                    query.append(f"   (\"up to 2\")")
                elif filter_values[0] > 2 and filter_values[0] < 7:
                    query.append(f"   (\"{filter_values[0]}\")")
                elif filter_values[0] == '7' or '8':
                    query.append(f"   (\"7 or 8\")")
                elif filter_values[0] > '8':
                    query.append(f"   (\"9 or more\")")

    return query

def filter_property(filter_type):
    type_1 = {
        "property-type", 
        "built-form",
        "accommodation-type",
        "building-use",
        "tenure",
        "epc-rating",
        "potential-epc-rating",
        "epc-score",
        "potential-epc-score",
        "roof-type",
        "wall-type",
        "glazing-type",
        "wall-insulation",
        "roof-insulation",
        "main-heating-system",
        "main-fuel-type",
    }
    if filter_type in {"easting", "northing"}:
        return "bng-" + filter_type
    if filter_type in type_1:
        return kebab_to_camel("has-" + filter_type)
    else:
        return kebab_to_camel(filter_type)

def geography_cleaned(geo_type):
    geo_map = {
        'oa': 'output-area',
        'output-area': 'output-area',
        'lsoa': 'lower-layer-super-output-area',
        'lower-layer-super-output-area': 'lower-layer-super-output-area',
        'ward': 'ward',
        'postcode': 'postcode-unit-area',
        'administrative-area': 'london-borough',
        'london-borough': 'london-borough',
        'lb': 'london-borough',
    }
    return geo_map.get(geo_type.lower())

def sensor_cleaned(sensor):
    sensor_map = {
        "temperature": "bess:PhidgetTemperatureSensor",
        "temperature-sensor": "bess:PhidgetTemperatureSensor",
        "phidget-temperature-sensor": "bess:PhidgetTemperatureSensor",
        "bess:PhidgetTemperatureSensor": "bess:PhidgetTemperatureSensor",
        "humidity": "bess:PhidgetHumiditySensor",
        "humidity-sensor": "bess:PhidgetHumiditySensor",
        "relative-humidity": "bess:PhidgetHumiditySensor",
        "relative-humidity-sensor": "bess:PhidgetHumiditySensor",
        "phidget-humidity-sensor": "bess:PhidgetHumiditySensor",
        "bess:PhidgetHumiditySensor": "bess:PhidgetHumiditySensor",
        "lidar": "bess:OusterLidarSensor",
        "lidar-sensor": "bess:OusterLidarSensor",
        "ouster-lidar-sensor": "bess:OusterLidarSensor",
        "bess:OusterLidarSensor": "bess:OusterLidarSensor",
        "rgb": "bess:FlirA70Camera",
        "rgb-camera": "bess:FlirA70Camera",
        "flir-a70-camera": "bess:FlirA70Camera",
        "bess:FlirA70Camera": "bess:FlirA70Camera",
        "ir" : "bess:FlirOryxCamera",
        "ir-camera": "bess:FlirOryxCamera",
        "ir-sensor": "bess:FlirOryxCamera",
        "flir-oryx-camera": "bess:FlirOryxCamera",
        "bess:FlirOryxCamera": "bess:FlirOryxCamera",
        "ins" : "bess:LordMicrostrainINSGQ7",
        "ins-sensor": "bess:LordMicrostrainINSGQ7",
        "lord-microstrain-ins-gq7": "bess:LordMicrostrainINSGQ7",
        "bess:LordMicrostrainINSGQ7": "bess:LordMicrostrainINSGQ7",
    }
    return sensor_map.get(sensor.lower())

def build_identifier_map(args, parser):
    identifier_map = defaultdict(list)

    if isinstance(args.identifier, dict):
        return args.identifier 

    for block in args.identifier:
        if len(block) < 2:
            parser.error("--identifier requires a key followed by at least one value")
        key, *values = block
        if key not in ["uprn", "ods", "toid"]:
            parser.error(f"Unknown identifier type: {key!r} (supported: uprn, ods, toid)")

        expanded_values = []
        for val in values:
            if os.path.isfile(val) and val.endswith(".csv"):
                expanded_values.extend(load_column_from_csv(val, key))
            else:
                expanded_values.append(val.strip())

        identifier_map[key].extend(expanded_values)

    return dict(identifier_map)

def build_geography_map(args, parser):
    geography_map = defaultdict(list)

    if isinstance(args.geography, dict):
        return args.geography 

    for block in args.geography:
        if len(block) < 2:
            parser.error("--geography requires a key followed by at least one value")
        key, *values = block
        cleaned_key = geography_cleaned(key)
        if cleaned_key is None:
            parser.error(f"Unknown geography type: {key!r}")

        expanded_values = []
        for val in values:
            if os.path.isfile(val) and val.lower().endswith(".csv"):
                expanded_values.extend(load_column_from_csv(val, key))
            else:
                expanded_values.append(val.strip())

        geography_map[cleaned_key].extend(expanded_values)

    return dict(geography_map)

def build_filter_map(args, parser):
    filter_map = defaultdict(list)

    if isinstance(args.filter, dict):
        return args.filter

    for block in args.filter:
        if len(block) < 2:
            parser.error("--filter requires a key followed by at least one value")
        key, *values = block

        expanded_values = []
        for val in values:
            if os.path.isfile(val) and val.lower().endswith(".csv"):
                expanded_values.extend(load_column_from_csv(val, key))
            else:
                expanded_values.append(val.strip())

        filter_map[key].extend(expanded_values)

    filter_map_validation(filter_map, parser)

    return dict(filter_map)

def filter_map_validation(filter_map, parser):

    def validate_numeric(filter_type, values, integer_only=False):
        if len(values) not in (1, 2):
            parser.error(f"Filter type {filter_type!r} must have either one (exact match) or two (range) values, got {len(values)}")
        for v in values:
            try:
                float(v)
                if integer_only:
                    int(v)
            except ValueError:
                expected = "integer" if integer_only else "numeric"
                parser.error(f"Filter type {filter_type!r} requires {expected} values, got {v!r}")

    def validate_boolean(filter_type, values):
        for v in values:
            if v.lower() not in {"true", "false"}:
                parser.error(f"Filter type {filter_type!r} requires boolean values (true/false), got {v!r}")

    def validate_enum(filter_type, values, valid_set):
        for v in values:
            if v.lower() not in valid_set and v.upper() not in valid_set:
                parser.error(f"Filter type {filter_type!r} has invalid value {v!r}, must be one of {valid_set}")

    validators = {

        **{k: lambda ft, vs: validate_numeric(ft, vs) for k in {
            "solar-pv-potential", "fuel-poverty"
        }},
        **{k: lambda ft, vs: validate_numeric(ft, vs, integer_only=True) for k in {
            "total-floor-area", "energy-consumption", "solar-pv-area",
            "floor-count", "basement-floor-count", "imd19-national-decile",
            "imd19-income-decile", "heat-risk-quintile", "average-roof-tilt",
            "number-of-habitable-rooms", "easting", "northing",
            "epc-score", "potential-epc-score"
        }},

        **{k: validate_boolean for k in {
            "in-conservation-area", "wall-insulation", "roof-insulation"
        }},

        "property-type": lambda ft, vs: validate_enum(ft, vs, {
            "house", "flat", "park-home-caravan"
        }),
        "built-form": lambda ft, vs: validate_enum(ft, vs, {
            "detached", "semi-detached", "mid-terrace", "end-terrace"
        }),
        "accommodation-type": lambda ft, vs: validate_enum(ft, vs, {
            "detached-house", "semi-detached-house", "mid-terraced-house",
            "end-terraced-house", "park-home-caravan", "flat"
        }),
        "tenure": lambda ft, vs: validate_enum(ft, vs, {
            "owner-occupied", "privately-rented", "social-housing"
        }),
        "epc-rating": lambda ft, vs: validate_enum(ft, vs, {
            "AB", "C", "D", "E", "FG"
        }),
        "potential-epc-rating": lambda ft, vs: validate_enum(ft, vs, {
            "AB", "C", "D", "E", "FG"
        }),
        "building-use": lambda ft, vs: validate_enum(ft, vs, {
            "residential-only", "mixed-use"
        }),
        "construction-age-band": lambda ft, vs: validate_enum(ft, vs, {
            "pre-1900", "1900-1929", "1930-1949", "1950-1966",
            "1967-1982", "1983-1995", "1996-2011", "2012-onwards"
        }),
        "wall-type": lambda ft, vs: validate_enum(ft, vs, {
            "cavity", "solid", "other"
        }),
        "roof-type": lambda ft, vs: validate_enum(ft, vs, {
            "pitched", "flat", "room-in-roof", "another-dwelling-above"
        }),
        "glazing-type": lambda ft, vs: validate_enum(ft, vs, {
            "single-partial", "secondary", "double-triple"
        }),
        "main-heating-system": lambda ft, vs: validate_enum(ft, vs, {
            "boiler", "room-storage-heaters", "heat-pump", "communal", "none", "other"
        }),
        "main-fuel-type": lambda ft, vs: validate_enum(ft, vs, {
            "mains-gas", "electricity", "no-heating-system", "other"
        }),
        "loac-group": lambda ft, vs: validate_enum(ft, vs, {
            "A1", "A2", "A3", "B1", "B2", "C1", "C2", "D1",
            "D2", "D3", "E1", "E2", "F1", "F2", "G1", "G2"
        }),
        "loac-supergroup": lambda ft, vs: validate_enum(ft, vs, {
            "A", "B", "C", "D", "E", "F", "G"
        }),
        "listed-building-grade": lambda ft, vs: validate_enum(ft, vs, {
            "I", "IIStar", "II", "Unknown"
        }),
    }

    for filter_type, values in filter_map.items():
        if not values:
            parser.error(f"Filter type {filter_type!r} has no values specified")
        validator = validators.get(filter_type)
        if validator:
            validator(filter_type, values)
        else:
            parser.error(f"Unsupported filter type: {filter_type!r}")

def select_vars(args):
    select_vars = ["?contentUrl", "?enum", "?phenomenonTime", "?uprn"]

    return select_vars

def build_asset_query(args, parser):
    """Builds the SPARQL query to fetch asset data including phenomenon times."""
    prefixes = """
    PREFIX did:   <https://w3id.org/dob/id/>
    PREFIX dob:   <https://w3id.org/dob/voc#>
    PREFIX so:    <http://schema.org/>
    PREFIX sosa:  <http://www.w3.org/ns/sosa/>
    PREFIX prov:  <http://www.w3.org/ns/prov#>
    PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
    PREFIX dop:   <https://w3id.org/dob/voc/prop#>
    PREFIX adms:  <http://www.w3.org/ns/adms#>
    PREFIX geo:   <http://www.opengis.net/ont/geosparql#>
    PREFIX skos:  <http://www.w3.org/2004/02/skos/core#>
    """

    select = "SELECT DISTINCT " + " ".join(select_vars(args))

    where = [
        "  ?res so:contentUrl ?contentUrl .",
        "  ?res dob:typeQualifier ?enum .",
        "  ?res ( ^sosa:hasResult | ^prov:generated / prov:used )* ?obs .",
        "  ?obs a sosa:Observation ;",
        "       sosa:phenomenonTime ?phenomenonTime ;",
        "       sosa:hasFeatureOfInterest ?foi .",
    ]

    if args.sensor:
        where.extend(sensor_query(args.sensor))

    id_map = build_identifier_map(args, parser) if args.identifier else {}
    geo_map = build_geography_map(args, parser) if args.geography else {}
    filter_map = build_filter_map(args, parser) if args.filter else {}

    ontop_required = bool(geo_map or filter_map or 'toid' in id_map)

    if ontop_required and not args.ontop_url:
        parser.error(
            "Queries using --geography, --filter, or --identifier toid "
            "require the --ontop-url to be specified."
        )

    if 'uprn' in id_map or 'ods' in id_map:
        non_ontop_ids = {
            k: v for k, v in id_map.items() if k in ['uprn', 'ods']
        }
        where.extend(identifier_query_type_1(non_ontop_ids))

    if ontop_required:
        where.append(f"SERVICE <{args.ontop_url}> {{")

        if 'toid' in id_map:
            toid_id = {'toid': id_map['toid']}
            where.extend(identifier_query_type_2(toid_id))

        if geo_map:
            where.extend(geography_query(geo_map))

        if filter_map:
            where.extend(filter_query(filter_map))

        where.append("}")

    if args.types:
        if isinstance(args.types, str):
            args.types = re.split(r'[, ]+', args.types)
        quoted_types = ", ".join(f"{t.strip()}" for t in args.types if t.strip())
        where.append(f"  FILTER(?enum IN ({quoted_types}))")

    where.extend([
        "  ?foi so:identifier ?uprnValue .",
        "  ?uprnValue a dob:UPRNValue ; so:value ?uprn .",
    ])

    select += "\n"

    return prefixes + select + "WHERE {\n" + "\n".join(where) + "\n}"

def download_asset(url: str, save_dir: str, api_key: str):
    """Downloads a single asset from a URL to a specified directory."""
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.get(url, headers={"x-api-key": api_key})
            resp.raise_for_status()

            # Determine filename from Content-Disposition or URL
            cd = resp.headers.get("Content-Disposition", "")
            m = re.search(r'filename="([^"]+)"', cd)
            fn = m.group(1) if m else os.path.basename(url)

            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, fn)

            with open(path, "wb") as f:
                f.write(resp.content)
            logging.info(f"✔ Saved {url} → {path}")

    except httpx.HTTPStatusError as e:
        logging.error(
            f"✖ HTTP error downloading {url}: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        logging.error(f"✖ Failed to download {url}: {e}")

def main():
    """Main execution function."""
    args, parser = parse_args()
    download_base = args.download_dir or os.path.join(os.getcwd(), "downloads")
    os.makedirs(download_base, exist_ok=True)

    # --- ASSET-DOWNLOAD MODE ---
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        logging.error(
            f"API key environment variable {args.api_key_env!r} is not set."
        )
        return            

    q = build_asset_query(args, parser)
    logging.info("SPARQL query for assets:\n%s", q)
    store = SPARQLStore(query_endpoint=args.db_url, returnFormat="json")
    res = store.query(q)

    for row in res:
        try:
            uprn = str(row["uprn"])
            url = str(row["contentUrl"])
            enum_iri = str(row["enum"])

            phenomenon_time_obj = row["phenomenonTime"].value
            if isinstance(phenomenon_time_obj, datetime):
                date_str = phenomenon_time_obj.strftime("%Y-%m-%d")
            else:
                date_str = str(phenomenon_time_obj).split("T")[0]

            asset_type_subdir = asset_subdir(enum_iri)

            tgt_dir = os.path.join(
                download_base, uprn, date_str, asset_type_subdir
            )

            logging.info(f"⤷ Queuing download for {url} into {tgt_dir}/")
            download_asset(url, tgt_dir, api_key)

        except KeyError as e:
            logging.error(
                f"✖ Query result row was missing expected key: {e}. Row: {row}"
            )
        except Exception as e:
            logging.error(
                f"✖ An unexpected error occurred while processing row {row}: {e}"
            )


if __name__ == "__main__":
    main()
