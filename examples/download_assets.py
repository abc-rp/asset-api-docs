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
        description="Download assets or list UPRNs by output area from a DID triplestore."
    )
    parser.add_argument(
        "--uprn",
        nargs="+",
        help=(
            "One or more UPRNs (space- or comma-separated), e.g.: "
            "--uprn 12345 67890  or  --uprn 12345,67890"
        ),
    )
    parser.add_argument(
        "--csv",
        help="Path to a CSV file containing a column 'uprn' with UPRN values",
    )
    parser.add_argument(
        "--sensor",
        help="Sensor IRI to filter by sensor type, e.g. bess:OusterLidarSensor",
    )
    parser.add_argument(
        "--types",
        help="Comma-separated list of asset type IRIs, e.g. did:rgb-image,did:lidar-pointcloud-merged",
    )
    parser.add_argument(
        "--output-area", "--oa",
        dest="output_area",
        nargs="+",
        help=(
            "One or more output area IRIs (space- or comma-separated), "
            "e.g. sid:E00032882 or sid:E00032882,sid:E00063193"
        ),
    )
    parser.add_argument(
        "--db-url",
        default="http://100.64.153.8:3030/didtriplestore/query",
        help="SPARQL endpoint URL",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Base directory for downloads (default: ./downloads)",
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
            raise RuntimeError(f"CSV {path!r} missing required 'uprn' column")
        for row in reader:
            val = row['uprn'].strip()
            if val:
                uprns.append(val)
    return uprns

def build_asset_query(uprn_list, args):
    prefixes = """
PREFIX dob:   <https://w3id.org/dob/voc#>
PREFIX so:    <http://schema.org/>
PREFIX sosa:  <http://www.w3.org/ns/sosa/>
PREFIX prov:  <http://www.w3.org/ns/prov#>
PREFIX bess:  <https://w3id.org/bess/voc#>
"""
    select = "SELECT DISTINCT ?uprnValue ?contentUrl\n"
    where = ["  ?res so:contentUrl ?contentUrl ."]
    if args.types:
        where.append("  ?res dob:typeQualifier ?enum .")
    if args.sensor:
        where += [
            "  ?res ( ^sosa:hasResult | ^prov:generated / prov:used )* ?obs .",
            f"  ?obs a {args.sensor} .",
            "  ?obs sosa:hasFeatureOfInterest/so:identifier/so:value ?uprnValue .",
        ]
    else:
        where.append(
            "  ?res ( ^sosa:hasResult | ^prov:generated / prov:used )*"
            " / sosa:hasFeatureOfInterest/so:identifier/so:value ?uprnValue ."
        )
    quoted = ", ".join(f'"{u}"' for u in uprn_list)
    where.append(f"  FILTER(str(?uprnValue) IN ({quoted}))")
    if args.types:
        where.append(f"  FILTER(?enum IN ({args.types}))")
    return prefixes + select + "WHERE {\n" + "\n".join(where) + "\n}\n"

def build_output_area_query(area_list):
    prefixes = """
PREFIX spr:   <http://statistics.data.gov.uk/def/spatialrelations/>
PREFIX so:    <http://schema.org/>
PREFIX sosa:  <http://www.w3.org/ns/sosa/>
PREFIX bot:   <https://w3id.org/bot/voc#>
PREFIX sid:   <http://statistics.data.gov.uk/id/statistical-geography/> 
"""
    select = "SELECT DISTINCT ?outputArea ?uprnValue\n"
    where = [
        "  VALUES ?outputArea { " + " ".join(area_list) + " } .",
        "  ?zone spr:within ?outputArea .",
        "  ?zone so:identifier/so:value ?uprnValue ."
    ]
    return prefixes + select + "WHERE {\n" + "\n".join(where) + "\n}\n"

def download_asset(url: str, save_dir: str, api_key: str):
    try:
        resp = httpx.get(url, headers={"x-api-key": api_key}, timeout=120)
        resp.raise_for_status()
        cd = resp.headers.get("Content-Disposition", "")
        m = re.search(r'filename="([^"]+)"', cd)
        fn = m.group(1) if m else os.path.basename(url)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, fn)
        with open(path, "wb") as f:
            f.write(resp.content)
        print(f"✔ Saved {url} → {path}")
    except Exception as e:
        print(f"✖ Failed to download {url}: {e}")

def local_name(iri):
    # If it's a full IRI, take the segment after the last slash
    if iri.startswith("http://") or iri.startswith("https://"):
        return iri.rstrip("/").split("/")[-1]
    # If it's a prefixed name, take the part after the colon
    if ":" in iri:
        return iri.split(":", 1)[1]
    return iri

def main():
    args = parse_args()
    download_base = args.download_dir or os.path.join(os.getcwd(), "downloads")
    os.makedirs(download_base, exist_ok=True)

    # --- OUTPUT-AREA MODE ---
    if args.output_area:
        areas = []
        for entry in args.output_area:
            parts = [a.strip() for a in entry.split(",") if a.strip()]
            areas.extend(parts)
        # Only wrap true IRIs in <>
        formatted = [
            f"<{a}>" if a.startswith("http://") or a.startswith("https://") else a
            for a in areas
        ]

        store = SPARQLStore(query_endpoint=args.db_url, returnFormat="json")
        q = build_output_area_query(formatted)
        print("SPARQL query for output areas:\n", q)
        res = store.query(q)

        grouping = {}
        for row in res:
            oa = str(row["outputArea"])
            uv = str(row["uprnValue"])
            grouping.setdefault(oa, []).append(uv)

        # Write one CSV per output area, named by local name only
        for oa, uprns in grouping.items():
            name = local_name(oa)
            out_csv = os.path.join(download_base, f"{name}.csv")
            with open(out_csv, "w", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow(["uprn"])
                for u in uprns:
                    writer.writerow([u])
            print(f"✔ Saved CSV for {oa} → {out_csv}")

    # --- ASSET-DOWNLOAD MODE ---
    uprn_list = []
    if args.uprn:
        for e in args.uprn:
            uprn_list.extend(u.strip() for u in e.split(",") if u.strip())
    if args.csv:
        uprn_list.extend(load_uprns_from_csv(args.csv))
    uprn_list = list(dict.fromkeys(uprn_list))

    if uprn_list:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"Env var {args.api_key_env!r} is not set")

        store = SPARQLStore(query_endpoint=args.db_url, returnFormat="json")
        q = build_asset_query(uprn_list, args)
        print("SPARQL query for assets:\n", q)
        res = store.query(q)

        for row in res:
            uprn_val = str(row["uprnValue"])
            url      = str(row["contentUrl"])
            tgt_dir  = os.path.join(download_base, uprn_val)
            print(f"⤷ Downloading {url} into {tgt_dir}/")
            download_asset(url, tgt_dir, api_key)

if __name__ == "__main__":
    main()
