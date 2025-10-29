#!/usr/bin/env python3
import argparse
import csv
import logging
import os
import re
from datetime import datetime

import httpx
from rdflib.plugins.stores.sparqlstore import SPARQLStore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args():
    """Parses command-line arguments."""
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
        default="http://ec2-3-10-233-191.eu-west-2.compute.amazonaws.com:3030/mytriplestore/query",
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


def build_asset_query(uprn_list, args):
    """Builds the SPARQL query to fetch asset data including phenomenon times."""
    prefixes = """
    PREFIX did:   <https://w3id.org/dob/id/>
    PREFIX dob:   <https://w3id.org/dob/voc#>
    PREFIX so:    <http://schema.org/>
    PREFIX sosa:  <http://www.w3.org/ns/sosa/>
    PREFIX prov:  <http://www.w3.org/ns/prov#>
    PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
    """
    select = "SELECT DISTINCT ?uprnValue ?contentUrl ?enum ?phenomenonTime\n"
    where = [
        "  ?res so:contentUrl ?contentUrl .",
        "  ?res dob:typeQualifier ?enum .",
        "  ?res ( ^sosa:hasResult | ^prov:generated / prov:used )* ?obs .",
        "  ?obs a sosa:Observation ;",
        "       sosa:phenomenonTime ?phenomenonTime ;",
        "       sosa:hasFeatureOfInterest ?foi .",
        "  ?foi so:identifier ?uprnRes .",
        "  ?uprnRes a dob:UPRNValue ; so:value ?uprnValue .",
    ]
    if args.sensor:
        where.append(f"  ?obs sosa:madeBySensor {args.sensor} .")

    quoted_uprns = ", ".join(f'"{u}"' for u in uprn_list)
    where.append(f"  FILTER(str(?uprnValue) IN ({quoted_uprns}))")
    if args.types:
        quoted_types = ", ".join(f"<{t.strip()}>" for t in args.types.split(","))
        where.append(f"  FILTER(?enum IN ({quoted_types}))")

    return prefixes + select + "WHERE {\n" + "\n".join(where) + "\n}"


def build_output_area_query(area_list):
    """Builds the SPARQL query to fetch UPRNs within given output areas."""
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
    query = prefixes + select + "WHERE {\n" + "\n".join(where) + "\n}\n"
    return query


def build_ods_to_uprn_query(ods_list):
    """Builds the SPARQL query to map ODS codes to UPRNs."""
    prefixes = """
PREFIX dob: <https://w3id.org/dob/voc#>
PREFIX so:  <http://schema.org/>
"""
    select = "SELECT DISTINCT ?odsValue ?uprnValue ?recCodeAddress\n"
    values = " ".join(f'"{o}"' for o in ods_list)
    where = [
        f"  VALUES ?odsValue {{ {values} }} .",
        "  ?zone so:identifier ?identODS .",
        "  ?identODS a dob:ODSValue ; so:value ?odsValue ;",
        "            dob:recommendationCodeAddress ?recCodeAddress .",
        "  ?zone so:identifier ?identUPRN .",
        "  ?identUPRN a dob:UPRNValue ; so:value ?uprnValue .",
    ]
    query = prefixes + select + "WHERE {\n" + "\n".join(where) + "\n}\n"
    return query


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
        ods_list = sorted(list(dict.fromkeys(ods_list)))

        store = SPARQLStore(query_endpoint=args.db_url, returnFormat="json")
        q = build_ods_to_uprn_query(ods_list)
        logging.info("SPARQL query for ODS→UPRN mapping:\n%s", q)
        res = store.query(q)

        out_csv = os.path.join(download_base, "ods_to_uprn.csv")
        with open(out_csv, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["ods", "uprn", "recommendationCodeAddress"])
            for row in res:
                writer.writerow(
                    [row["odsValue"], row["uprnValue"], row.get("recCodeAddress")]
                )
        logging.info(f"✔ Saved ODS→UPRN CSV → {out_csv}")
        return

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
        logging.info("SPARQL query for output areas:\n%s", q)
        res = store.query(q)

        grouping = {}
        for row in res:
            grouping.setdefault(str(row["outputArea"]), []).append(
                str(row["uprnValue"])
            )

        for oa, uprns in grouping.items():
            name = oa.split("/")[-1]
            out_csv = os.path.join(download_base, f"{name}.csv")
            with open(out_csv, "w", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow(["uprn"])
                for u in sorted(uprns):
                    writer.writerow([u])
            logging.info(f"✔ Saved CSV for {oa} → {out_csv}")

    # --- ASSET-DOWNLOAD MODE ---
    uprn_list = []
    if args.uprn:
        for entry in args.uprn:
            if os.path.isfile(entry) and entry.lower().endswith(".csv"):
                uprn_list.extend(load_column_from_csv(entry, "uprn"))
            else:
                uprn_list.extend(u.strip() for u in entry.split(",") if u.strip())
    uprn_list = sorted(list(dict.fromkeys(uprn_list)))  # Sort for consistent query

    if uprn_list:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            logging.error(
                f"API key environment variable {args.api_key_env!r} is not set."
            )
            return

        store = SPARQLStore(query_endpoint=args.db_url, returnFormat="json")
        q = build_asset_query(uprn_list, args)
        logging.info("SPARQL query for assets:\n%s", q)
        res = store.query(q)

        for row in res:
            try:
                uprn_val = str(row["uprnValue"])
                url = str(row["contentUrl"])
                enum_iri = str(row["enum"])

                phenomenon_time_obj = row["phenomenonTime"].value
                if isinstance(phenomenon_time_obj, datetime):
                    date_str = phenomenon_time_obj.strftime("%Y-%m-%d")
                else:
                    date_str = str(phenomenon_time_obj).split("T")[0]

                asset_type_subdir = asset_subdir(enum_iri)

                tgt_dir = os.path.join(
                    download_base, uprn_val, date_str, asset_type_subdir
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
