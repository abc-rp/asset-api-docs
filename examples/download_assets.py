#!/usr/bin/env python3
import os
import re
import argparse
import csv

import httpx
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from rdflib.query import ResultRow

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download assets from a DID triplestore based on UPRNs, sensors, types, or CSV input."
    )
    parser.add_argument(
        "--uprn",
        nargs="+",
        help=(
            "One or more UPRNs. "
            "Pass them space-separated or comma-separated, e.g.: "
            "--uprn 12345 67890  or  --uprn 12345,67890"
        ),
    )
    parser.add_argument(
        "--csv",
        help="Path to a CSV file containing a column 'uprn' with UPRN values",
    )
    parser.add_argument(
        "--sensor",
        help="Sensor IRI (prefixed or full) to filter by sensor type, e.g. bess:OusterLidarSensor",
    )
    parser.add_argument(
        "--types",
        help="Comma-separated list of asset type IRIs to filter by, e.g. did:rgb-image,did:lidar-pointcloud-merged",
    )
    parser.add_argument(
        "--db-url",
        default="http://100.64.153.8:3030/didtriplestore/query",
        help="SPARQL endpoint URL",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Base directory to store downloads (default: ./downloads)",
    )
    parser.add_argument(
        "--api-key-env",
        default="API_KEY",
        help="Environment variable name for the API key",
    )
    return parser.parse_args()

def load_uprns_from_csv(path):
    uprns = []
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if 'uprn' not in reader.fieldnames:
            raise RuntimeError(f"CSV file {path!r} missing required 'uprn' column")
        for row in reader:
            val = row['uprn'].strip()
            if val:
                uprns.append(val)
    return uprns

def build_query(uprn_list, args):
    prefixes = """
PREFIX dob:   <https://w3id.org/dob/voc#>
PREFIX so:    <http://schema.org/>
PREFIX sosa:  <http://www.w3.org/ns/sosa/>
PREFIX prov:  <http://www.w3.org/ns/prov#>
PREFIX bess:  <https://w3id.org/bess/voc#>
"""
    select = "SELECT DISTINCT ?uprnValue ?contentUrl\n"
    where_clauses = ["  ?res so:contentUrl ?contentUrl ."]

    if args.types:
        where_clauses.append("  ?res dob:typeQualifier ?enum .")
    if args.sensor:
        where_clauses += [
            "  ?res ( ^sosa:hasResult | ^prov:generated / prov:used )* ?obs .",
            f"  ?obs a {args.sensor} .",
            "  ?obs sosa:hasFeatureOfInterest/so:identifier/so:value ?uprnValue .",
        ]
    else:
        where_clauses.append(
            "  ?res ( ^sosa:hasResult | ^prov:generated / prov:used )*"
            " / sosa:hasFeatureOfInterest/so:identifier/so:value ?uprnValue ."
        )

    quoted = ", ".join(f'"{u}"' for u in uprn_list)
    where_clauses.append(f"  FILTER(str(?uprnValue) IN ({quoted}))")

    if args.types:
        where_clauses.append(f"  FILTER(?enum IN ({args.types}))")

    where = "WHERE {\n" + "\n".join(where_clauses) + "\n}\n"
    return prefixes + select + where

def download_asset(url: str, save_dir: str, api_key: str):
    try:
        resp = httpx.get(url, headers={"x-api-key": api_key}, timeout=120)
        resp.raise_for_status()
        cd = resp.headers.get("Content-Disposition", "")
        m = re.search(r'filename="([^"]+)"', cd)
        filename = m.group(1) if m else os.path.basename(url)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        with open(path, "wb") as f:
            f.write(resp.content)
        print(f"✔ Saved {url} → {path}")
    except Exception as e:
        print(f"✖ Failed to download {url}: {e}")

def main():
    args = parse_args()

    # Gather UPRNs from --uprn and/or --csv
    uprn_list = []
    if args.uprn:
        for entry in args.uprn:
            uprn_list.extend(u.strip() for u in entry.split(",") if u.strip())
    if args.csv:
        uprn_list.extend(load_uprns_from_csv(args.csv))

    # Ensure we have at least one UPRN
    uprn_list = list(dict.fromkeys(uprn_list))
    if not uprn_list:
        raise RuntimeError("Must provide at least one UPRN via --uprn or --csv")

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is not set")

    download_base = args.download_dir or os.path.join(os.getcwd(), "downloads")
    store = SPARQLStore(query_endpoint=args.db_url, returnFormat="json")
    query = build_query(uprn_list, args)

    print("SPARQL query:\n", query)
    results = store.query(query)

    for row in results:
        if not isinstance(row, ResultRow):
            continue
        uprn_val = str(row["uprnValue"])
        url = str(row["contentUrl"])
        target_dir = os.path.join(download_base, uprn_val)
        print(f"⤷ Downloading {url} into {target_dir}/")
        download_asset(url, target_dir, api_key)

if __name__ == "__main__":
    main()
