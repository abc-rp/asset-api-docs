#!/usr/bin/env python3
import argparse
import csv
import os
import re

import httpx
from rdflib.plugins.stores.sparqlstore import SPARQLStore


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download assets, list UPRNs by output area, or map ODS→UPRN from a DID triplestore."
    )
    parser.add_argument(
        "--uprn",
        nargs="+",
        help="One or more UPRNs or a CSV path (column 'uprn'), e.g. --uprn 200003455212 5045394 or --uprn uprns.csv",
    )
    parser.add_argument(
        "--ods",
        nargs="+",
        help="One or more ODS codes or a CSV path (column 'ods'), e.g. --ods 00LAA or --ods ods.csv",
    )
    parser.add_argument("--sensor", help="Sensor IRI, e.g. bess:OusterLidarSensor")
    parser.add_argument(
        "--types", help="Comma-separated list of asset type IRIs, e.g. did:rgb-image"
    )
    parser.add_argument(
        "--output-area",
        "--oa",
        dest="output_area",
        nargs="+",
        help="One or more output-area IRIs or a CSV path (column 'output_area'), e.g. --output-area E00032882 or --output-area areas.csv",
    )
    parser.add_argument(
        "--db-url",
        default="http://ec2-18-175-116-201.eu-west-2.compute.amazonaws.com:3030/didtriplestore/query",
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


def load_column_from_csv(path, column):
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


def build_asset_query(uprn_list, args):
    prefixes = """
PREFIX did:   <https://w3id.org/dob/id/>
PREFIX dob:   <https://w3id.org/dob/voc#>
PREFIX so:    <http://schema.org/>
PREFIX sosa:  <http://www.w3.org/ns/sosa/>
PREFIX prov:  <http://www.w3.org/ns/prov#>
PREFIX bess:  <https://w3id.org/bess/voc#>

"""
    select = "SELECT DISTINCT ?uprnValue ?contentUrl\n"

    where = [
        # 1) grab any resource with a contentUrl
        "  ?res so:contentUrl ?contentUrl .",
        # optional enum filter on the resource
    ]

    if args.types:
        where.append("  ?res dob:typeQualifier ?enum .")

    # 2) walk back through any number of sosa:hasResult or prov steps
    where.append("  ?res ( ^sosa:hasResult | ^prov:generated / prov:used )* ?obs .")

    # 3) assert it's an Observation and pull sensor & UPRNValue via explicit triples
    where.extend(
        [
            "  ?obs a sosa:Observation ;",
            "       sosa:madeBySensor ?sensor ;",
            "       sosa:hasFeatureOfInterest ?foi .",
            "  ?foi so:identifier ?uprnRes .",
            "  ?uprnRes a dob:UPRNValue ;",
            "           so:value ?uprnValue .",
        ]
    )

    # 4) restrict to the desired sensor type
    if args.sensor:
        where.append(f"  ?sensor a {args.sensor} .")

    # 5) filter on your list of UPRNs
    quoted = ", ".join(f'"{u}"' for u in uprn_list)
    where.append(f"  FILTER(str(?uprnValue) IN ({quoted}))")

    # 6) optionally filter the enum values
    if args.types:
        where.append(f"  FILTER(?enum IN ({args.types}))")

    # assemble full query
    query = prefixes + select + "WHERE {\n" + "\n".join(where) + "\n}"
    return query


def build_output_area_query(area_list):
    prefixes = """
PREFIX spr: <http://statistics.data.gov.uk/def/spatialrelations/>
PREFIX so:  <http://schema.org/>
PREFIX dob: <https://w3id.org/dob/voc#>
PREFIX sid: <http://statistics.data.gov.uk/id/statistical-geography/>
"""
    select = "SELECT DISTINCT ?outputArea ?uprnValue\n"
    where = [
        "  VALUES ?outputArea { " + " ".join(area_list) + " } .",
        "  ?zone spr:within ?outputArea .",
        "  ?zone so:identifier ?ident .",
        "  ?ident a dob:UPRNValue ; so:value ?uprnValue .",
    ]
    return prefixes + select + "WHERE {\n" + "\n".join(where) + "\n}\n"


def build_ods_to_uprn_query(ods_list):
    prefixes = """
PREFIX dob: <https://w3id.org/dob/voc#>
PREFIX so:  <http://schema.org/>
"""
    select = "SELECT DISTINCT ?odsValue ?uprnValue\n"
    values = " ".join(f'"{o}"' for o in ods_list)
    where = [
        f"  VALUES ?odsValue {{ {values} }} .",
        "  ?zone so:identifier ?identODS .",
        "  ?identODS a dob:ODSValue ; so:value ?odsValue .",
        "  ?zone so:identifier ?identUPRN .",
        "  ?identUPRN a dob:UPRNValue ; so:value ?uprnValue .",
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


def main():
    args = parse_args()
    download_base = args.download_dir or os.path.join(os.getcwd(), "downloads")
    os.makedirs(download_base, exist_ok=True)

    # --- ODS→UPRN mapping mode ---
    if args.ods:
        ods_list = []
        for entry in args.ods:
            if os.path.isfile(entry) and entry.lower().endswith(".csv"):
                ods_list.extend(load_column_from_csv(entry, "ods"))
            else:
                ods_list.extend(o.strip() for o in entry.split(",") if o.strip())
        ods_list = list(dict.fromkeys(ods_list))

        store = SPARQLStore(query_endpoint=args.db_url, returnFormat="json")
        q = build_ods_to_uprn_query(ods_list)
        print("SPARQL query for ODS→UPRN mapping:\n", q)
        res = store.query(q)

        out_csv = os.path.join(download_base, "ods_to_uprn.csv")
        with open(out_csv, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["ods", "uprn"])
            for row in res:
                writer.writerow([row["odsValue"], row["uprnValue"]])
        print(f"✔ Saved ODS→UPRN CSV → {out_csv}")

    # --- OUTPUT-AREA MODE ---
    if args.output_area:
        areas = []
        for entry in args.output_area:
            if os.path.isfile(entry) and entry.lower().endswith(".csv"):
                areas.extend(load_column_from_csv(entry, "output_area"))
            else:
                areas.extend(a.strip() for a in entry.split(",") if a.strip())

        standardized = []
        for a in areas:
            if ":" not in a:
                standardized.append(f"sid:{a}")
            else:
                standardized.append(a)

        formatted = []
        for a in standardized:
            if a.startswith("http://") or a.startswith("https://"):
                formatted.append(f"<{a}>")
            else:
                formatted.append(a)

        store = SPARQLStore(query_endpoint=args.db_url, returnFormat="json")
        q = build_output_area_query(formatted)
        print("SPARQL query for output areas:\n", q)
        res = store.query(q)

        grouping = {}
        for row in res:
            grouping.setdefault(row["outputArea"], []).append(row["uprnValue"])

        for oa, uprns in grouping.items():
            name = oa.split("/")[-1]
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
        for entry in args.uprn:
            if os.path.isfile(entry) and entry.lower().endswith(".csv"):
                uprn_list.extend(load_column_from_csv(entry, "uprn"))
            else:
                uprn_list.extend(u.strip() for u in entry.split(",") if u.strip())
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
            url = str(row["contentUrl"])
            tgt_dir = os.path.join(download_base, uprn_val)
            print(f"⤷ Downloading {url} into {tgt_dir}/")
            download_asset(url, tgt_dir, api_key)


if __name__ == "__main__":
    main()
