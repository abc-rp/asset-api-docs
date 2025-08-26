#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from typing import Any

import requests

# -------------------------------
# System & Summary Prompts
# -------------------------------

SYSTEM_ROUTER_PROMPT = """You are a rigorous function-call router for a Python CLI named query_assist.py.

Supported commands and how to populate them:

1) download_assets
   Required:   uprn  (string CSV path OR array of strings like ["5045394","200003455212"])
   Optional:   sensor (string, e.g., "bess:OusterLidarSensor")
               types  (array of strings; each a type IRI, e.g., ["did:rgb-image","did:lidar-pointcloud-merged"])
               download_dir (string path)
               api_key_env  (string, name of env var with API key)
               db_url       (string URL to SPARQL endpoint)

2) ods_to_uprn
   Required:   ods   (string CSV path OR array of strings like ["G85013","Q12345"])

3) uprns_by_output_area
   Required:   output_area (string CSV path OR array of strings, e.g., ["E00004550","E00032882"])

Schema (MUST output exactly one JSON object with these keys as needed):
{
  "command": "download_assets" | "ods_to_uprn" | "uprns_by_output_area",
  "uprn":        string | string[] | null,
  "ods":         string | string[] | null,
  "output_area": string | string[] | null,
  "sensor":      string | null,
  "types":       string[] | null,
  "download_dir": string | null,
  "api_key_env":  string | null,
  "db_url":       string | null
}

Constraints:
- Return ONLY the JSON object. No prose, no markdown.
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
- Prefer being decisive. When in doubt, infer sensible defaults.
"""

SUMMARY_SYSTEM_PROMPT = """You transform a raw natural-language user request about assets / UPRNs / ODS / output areas into a concise structured summary.

Return ONLY a single JSON object with this exact schema (no prose):
{
    "bullets": string[]  // 2-8 short bullet points capturing intent & extracted fields
    ,"router_text": string // ONE concise imperative sentence for a routing model to decide the command
    ,"extracted": {       // best-effort extraction; omit keys you cannot infer
             "uprn": string[] | null,
             "ods": string[] | null,
             "output_area": string[] | null,
             "sensor": string | null,
             "types": string[] | null,
             "download_dir": string | null,
             "api_key_env": string | null,
             "db_url": string | null
    }
}

Guidance:
- Normalize UPRNs to digit strings.
- Keep ordering as given when sensible.
- For types, map descriptive phrases to IRIs per provided mapping when obvious.
- router_text should be minimal but sufficient (e.g., "Download merged lidar point cloud for UPRN 5045394").
- If purely informational greeting with no actionable command, set bullets to ["No actionable request"], router_text="no-op" and extracted={}.
"""

FEW_SHOTS: list[tuple[str, str]] = [
    (
        "Download the merged lidar point cloud for UPRN 5045394 into /data/assets. "
        "Use MY_KEY as the env var for the API key.",
        '{"command":"download_assets","uprn":["5045394"],"sensor":null,'
        '"types":["did:lidar-pointcloud-merged"],"download_dir":"/data/assets",'
        '"api_key_env":"MY_KEY","db_url":null,"ods":null,"output_area":null}',
    ),
    (
        "Map ODS G85013 to UPRNs.",
        '{"command":"ods_to_uprn","ods":["G85013"],"uprn":null,"output_area":null,'
        '"sensor":null,"types":null,"download_dir":null,"api_key_env":null,"db_url":null}',
    ),
    (
        "List all UPRNs in output areas E00004550 and E00032882.",
        '{"command":"uprns_by_output_area","output_area":["E00004550","E00032882"],'
        '"uprn":null,"ods":null,"sensor":null,"types":null,"download_dir":null,'
        '"api_key_env":null,"db_url":null}',
    ),
    (
        "Get RGB images and merged lidar for UPRNs 5045394 and 200003455212 to /mnt/dl (API key var KEY2).",
        '{"command":"download_assets","uprn":["5045394","200003455212"],"sensor":null,'
        '"types":["did:rgb-image","did:lidar-pointcloud-merged"],"download_dir":"/mnt/dl",'
        '"api_key_env":"KEY2","db_url":null,"ods":null,"output_area":null}',
    ),
    (
        "output areas E00004550 E00032882 E00063193 list uprns",
        '{"command":"uprns_by_output_area","output_area":["E00004550","E00032882","E00063193"],'
        '"uprn":null,"ods":null,"sensor":null,"types":null,"download_dir":null,'
        '"api_key_env":null,"db_url":null}',
    ),
    (
        "ODS codes G85013 Q12345 map to uprns",
        '{"command":"ods_to_uprn","ods":["G85013","Q12345"],"uprn":null,"output_area":null,'
        '"sensor":null,"types":null,"download_dir":null,"api_key_env":null,"db_url":null}',
    ),
    (
        "5045394 merged lidar pointcloud now",
        '{"command":"download_assets","uprn":["5045394"],"sensor":null,'
        '"types":["did:lidar-pointcloud-merged"],"download_dir":null,'
        '"api_key_env":null,"db_url":null,"ods":null,"output_area":null}',
    ),
    (
        "Download lidar range and reflectance panoramas for UPRN 5045394 sensor bess:OusterLidarSensor",
        '{"command":"download_assets","uprn":["5045394"],"sensor":"bess:OusterLidarSensor",'
        '"types":["did:lidar-range-pano","did:lidar-reflectance-pano"],"download_dir":null,'
        '"api_key_env":null,"db_url":null,"ods":null,"output_area":null}',
    ),
    (
        "Give me temperature and humidity for UPRN 200003455212",
        '{"command":"download_assets","uprn":["200003455212"],"sensor":null,'
        '"types":["did:celsius-temperature","did:relative-humidity"],"download_dir":null,'
        '"api_key_env":null,"db_url":null,"ods":null,"output_area":null}',
    ),
    (
        "Fetch IR false color and temperature array for 5045394",
        '{"command":"download_assets","uprn":["5045394"],"sensor":null,'
        '"types":["did:ir-false-color-image","did:ir-temperature-array"],"download_dir":null,'
        '"api_key_env":null,"db_url":null,"ods":null,"output_area":null}',
    ),
    (
        "List UPRNs in output area E00004550 (single)",
        '{"command":"uprns_by_output_area","output_area":["E00004550"],"uprn":null,'
        '"ods":null,"sensor":null,"types":null,"download_dir":null,'
        '"api_key_env":null,"db_url":null}',
    ),
    (
        "Map ODS G85013 and G85014 with endpoint override http://myhost:3030/ds/query",
        '{"command":"ods_to_uprn","ods":["G85013","G85014"],"uprn":null,"output_area":null,'
        '"sensor":null,"types":null,"download_dir":null,"api_key_env":null,'
        '"db_url":"http://myhost:3030/ds/query"}',
    ),
    (
        "Download rgb image for 5045394 to /tmp/dl using key var APIKEY",
        '{"command":"download_assets","uprn":["5045394"],"sensor":null,'
        '"types":["did:rgb-image"],"download_dir":"/tmp/dl","api_key_env":"APIKEY",'
        '"db_url":null,"ods":null,"output_area":null}',
    ),
    (
        "Get point cloud frame for UPRN 5045394",
        '{"command":"download_assets","uprn":["5045394"],"sensor":null,'
        '"types":["did:lidar-pointcloud-frame"],"download_dir":null,"api_key_env":null,'
        '"db_url":null,"ods":null,"output_area":null}',
    ),
]

# -------------------------------
# Helpers
# -------------------------------

log = logging.getLogger("nl_query_cli")


def extract_assistant_text_from_ollama(resp: dict[str, Any]) -> str:
    """Extract assistant text from Ollama /api/chat or /api/generate."""
    msg = resp.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
    if isinstance(resp.get("response"), str) and resp["response"].strip():
        return resp["response"]
    return json.dumps(resp, ensure_ascii=False)


def slice_first_json_object(text: str) -> str:
    """Extract the first top-level JSON object {...} from text."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found.")
    depth = 0
    for i, c in enumerate(text[start:], start=start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Unbalanced braces; JSON object not closed.")


def ensure_list_or_path(v: None | str | list[str]) -> list[str]:
    """Convert (None | str | list) into a list of CLI tokens."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    return [s]


def _find_csv_paths(text: str) -> list[str]:
    """Return a list of CSV-like path tokens found in text."""
    # Accept absolute, relative, and bare filenames ending with .csv
    return re.findall(r'(?:(?:[A-Za-z]:)?[^\s"\'<>|]+\.csv)\b', text)


def build_argv(spec: dict[str, Any], py: str, qa_path: str) -> list[str]:
    """Map the JSON spec to query_assist.py argv."""
    cmd = [py, qa_path]
    command = spec.get("command")

    db_url = spec.get("db_url")
    download_dir = spec.get("download_dir")
    api_key_env = spec.get("api_key_env")
    sensor = spec.get("sensor")
    types = spec.get("types")

    if command == "download_assets":
        uprn = ensure_list_or_path(spec.get("uprn"))
        if not uprn:
            raise ValueError("download_assets requires 'uprn'.")
        cmd += ["--uprn"] + uprn
        if sensor:
            cmd += ["--sensor", str(sensor)]
        if types:
            cmd += ["--types", ",".join([str(t) for t in types])]
    elif command == "ods_to_uprn":
        ods = ensure_list_or_path(spec.get("ods"))
        if not ods:
            raise ValueError("ods_to_uprn requires 'ods'.")
        cmd += ["--ods"] + ods
    elif command == "uprns_by_output_area":
        oa = ensure_list_or_path(spec.get("output_area"))
        if not oa:
            raise ValueError("uprns_by_output_area requires 'output_area'.")
        cmd += ["--output-area"] + oa
    else:
        raise ValueError(f"Unsupported command: {command!r}")

    if db_url:
        cmd += ["--db-url", str(db_url)]
    if download_dir:
        cmd += ["--download-dir", str(download_dir)]
    if api_key_env:
        cmd += ["--api-key-env", str(api_key_env)]

    return cmd


def heuristic_parse(nl: str) -> dict[str, Any] | None:
    """
    Heuristic parse for common patterns with CSV precedence:
      - CSV + 'uprn' → download_assets (uprn=<csv>)
      - CSV + 'output area' → uprns_by_output_area (output_area=<csv>)
      - Output areas: codes E########
      - ODS: tokens like A12345
      - UPRN asset downloads: ≥6-digit tokens + keywords
    """
    text = nl.strip()
    if not text:
        return None
    lowered = text.lower()

    csv_paths = _find_csv_paths(text)
    output_area_codes = re.findall(r"\bE\d{8}\b", text)
    ods_codes = re.findall(r"\b[A-Z]\d{5}\b", text)
    uprn_candidates = [t for t in re.findall(r"\b\d{6,}\b", text)]

    # Asset type hints
    wants_merged = bool(
        re.search(r"merged\s+(lidar\s+)?point\s*clouds?", lowered)
        or "merged lidar" in lowered
    )
    wants_rgb = ("rgb" in lowered) and ("image" in lowered or "images" in lowered)

    # Endpoint override
    endpoint_match = re.search(r"(https?://[\w\.-:%/]+)", text)
    endpoint_url = endpoint_match.group(1) if endpoint_match else None

    # --- CSV precedence ---
    if csv_paths:
        # If the user mentions UPRN(s), prefer treating the CSV as a UPRN list
        if "uprn" in lowered:
            types_list = []
            if wants_merged:
                types_list.append("did:lidar-pointcloud-merged")
            if wants_rgb:
                types_list.append("did:rgb-image")
            return {
                "command": "download_assets",
                "uprn": csv_paths,
                "sensor": None,
                "types": types_list or None,
                "download_dir": None,
                "api_key_env": None,
                "db_url": endpoint_url,
                "ods": None,
                "output_area": None,
            }
        # Else, if they mention output areas explicitly, treat CSV as an OA list
        if (
            "output area" in lowered
            or "output areas" in lowered
            or "oa" in lowered.split()
        ):
            return {
                "command": "uprns_by_output_area",
                "output_area": csv_paths,
                "uprn": None,
                "ods": None,
                "sensor": None,
                "types": None,
                "download_dir": None,
                "api_key_env": None,
                "db_url": None,
            }
        # If ambiguous: assume UPRN list (safer/more common in this CLI)
        return {
            "command": "download_assets",
            "uprn": csv_paths,
            "sensor": None,
            "types": (
                ["did:lidar-pointcloud-merged"]
                if wants_merged
                else (["did:rgb-image"] if wants_rgb else None)
            ),
            "download_dir": None,
            "api_key_env": None,
            "db_url": endpoint_url,
            "ods": None,
            "output_area": None,
        }

    # --- Pure OA codes ---
    if output_area_codes and (
        "output area" in lowered
        or "output areas" in lowered
        or len(output_area_codes) > 1
    ):
        return {
            "command": "uprns_by_output_area",
            "output_area": output_area_codes,
            "uprn": None,
            "ods": None,
            "sensor": None,
            "types": None,
            "download_dir": None,
            "api_key_env": None,
            "db_url": None,
        }

    # --- ODS codes (only if no explicit UPRN numbers present) ---
    if ods_codes and ("ods" in lowered or not uprn_candidates):
        return {
            "command": "ods_to_uprn",
            "ods": ods_codes,
            "uprn": None,
            "output_area": None,
            "sensor": None,
            "types": None,
            "download_dir": None,
            "api_key_env": None,
            "db_url": None,
        }

    # --- UPRN download with optional types ---
    asset_keywords = {
        "download",
        "get",
        "asset",
        "assets",
        "lidar",
        "image",
        "images",
        "point",
        "pointcloud",
        "point cloud",
    }
    if uprn_candidates and any(k in lowered for k in asset_keywords):
        types_list = []
        if wants_merged:
            types_list.append("did:lidar-pointcloud-merged")
        if wants_rgb:
            types_list.append("did:rgb-image")
        return {
            "command": "download_assets",
            "uprn": uprn_candidates,
            "sensor": None,
            "types": types_list or None,
            "download_dir": None,
            "api_key_env": None,
            "db_url": endpoint_url,
            "ods": None,
            "output_area": None,
        }

    return None


def ollama_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    top_p: float = 0.95,
    num_predict: int = 256,
    num_ctx: int | None = None,
    keep_alive: str | None = None,
    request_timeout_s: float = 120.0,
    force_json: bool = True,
) -> dict[str, Any]:
    """Call Ollama's /api/chat and return parsed JSON."""
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

    resp = requests.post(url, json=payload, timeout=(5.0, request_timeout_s))
    resp.raise_for_status()
    return resp.json()


# -------------------------------
# Verbose-mode intro helpers
# -------------------------------


def _fetch_model_intro(base_url: str, model_id: str) -> str:
    """
    Ask the routing model (plain text) to describe its function briefly.
    Safe: if the request fails, returns a static fallback.
    """
    try:
        resp = ollama_chat(
            base_url=base_url,
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_ROUTER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Briefly describe your function in 4–7 short bullet points without JSON. "
                        "Focus on how natural-language input is turned into a structured command "
                        "for query_assist.py and what arguments you can infer."
                    ),
                },
            ],
            force_json=False,
            temperature=0.0,
            top_p=0.95,
            num_predict=200,
        )
        text = extract_assistant_text_from_ollama(resp).strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                obj = json.loads(text)
                text = obj.get("description") or obj.get("text") or text
            except Exception:
                pass
        return text
    except Exception as e:
        logging.getLogger("nl_query_cli").debug("Intro fetch failed: %s", e)
        return (
            "- Routes natural-language queries to one of three commands: "
            "download_assets, ods_to_uprn, uprns_by_output_area.\n"
            "- Extracts UPRNs/ODS/Output Areas plus optional sensor, types, "
            "download_dir, api_key_env, db_url.\n"
            "- Maps asset phrases to canonical IRIs (e.g., merged lidar → did:lidar-pointcloud-merged).\n"
            "- Builds argv for query_assist.py and executes it (unless --dry-run).\n"
            "- Uses heuristics, few-shots, and optional summarization for robustness."
        )


def _render_box(title: str, body: str) -> str:
    """Render a Unicode box with a title and wrapped body."""
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


def _print_intro_banner(base_url: str, model_id: str) -> None:
    intro = _fetch_model_intro(base_url, model_id)
    banner = _render_box(f"query_assist.py Router — {model_id}", intro)
    print(banner)
    print()


# -------------------------------
# Core turn runner
# -------------------------------


def run_once(
    base_url: str,
    model_id: str,
    nl: str,
    qa_path: str,
    py_exe: str,
    dry_run: bool,
    temperature: float,
    top_p: float,
    num_predict: int,
    num_ctx: int | None,
    keep_alive: str | None,
    force_json: bool,
    debug_model: bool,
    summarize: bool,
    summary_model: str,
    summary_temperature: float,
    show_summary: bool,
) -> int:
    """
    One-shot turn: summarize (optional) → route spec → build argv → (dry) run query_assist.py.
    Only DEBUG shows JSON specs and raw model content.
    """
    original_nl = nl
    # log.info("Request: %s", original_nl)

    # --- Summarization (independent of routing) ---
    summary_router_text = None
    summary_obj: dict[str, Any] | None = None
    if summarize:
        try:
            sum_resp = ollama_chat(
                base_url=base_url,
                model=summary_model,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": nl},
                ],
                temperature=summary_temperature,
                top_p=top_p,
                num_predict=256,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                force_json=True,
            )
            if debug_model:
                log.debug(
                    "Summarizer raw JSON response:\n%s",
                    json.dumps(sum_resp, indent=2, ensure_ascii=False),
                )
            sum_content = extract_assistant_text_from_ollama(sum_resp)
            log.debug("Summarizer extracted content: %r", sum_content)

            if sum_content.strip():
                try:
                    summary_obj = json.loads(sum_content)
                except Exception:
                    try:
                        blob = slice_first_json_object(sum_content)
                        summary_obj = json.loads(blob)
                    except Exception:
                        summary_obj = None

            if not summary_obj:
                # Heuristic fallback summary aligned with heuristic_parse
                lowered = nl.lower()
                fallback_bullets: list[str] = []
                extracted: dict[str, Any] = {}

                csv_paths = _find_csv_paths(nl)
                oa_fb = re.findall(r"\bE\d{8}\b", nl)
                ods_fb = re.findall(r"\b[A-Z]\d{5}\b", nl)
                uprn_fb = re.findall(r"\b\d{6,}\b", nl)

                wants_merged = bool(
                    re.search(r"merged\s+(lidar\s+)?point\s*clouds?", lowered)
                    or "merged lidar" in lowered
                )
                wants_rgb = ("rgb" in lowered) and (
                    "image" in lowered or "images" in lowered
                )

                if csv_paths:
                    if "uprn" in lowered:
                        extracted["uprn"] = csv_paths
                        fallback_bullets.append(f"UPRNs CSV: {', '.join(csv_paths)}")
                    elif (
                        ("output area" in lowered)
                        or ("output areas" in lowered)
                        or ("oa" in lowered.split())
                    ):
                        extracted["output_area"] = csv_paths
                        fallback_bullets.append(
                            f"Output-area CSV: {', '.join(csv_paths)}"
                        )
                    else:
                        extracted["uprn"] = csv_paths
                        fallback_bullets.append(
                            f"UPRNs CSV (assumed): {', '.join(csv_paths)}"
                        )

                if not csv_paths:
                    if oa_fb:
                        extracted["output_area"] = oa_fb
                        fallback_bullets.append(f"Output areas: {', '.join(oa_fb)}")
                    if ods_fb and not uprn_fb:
                        extracted["ods"] = ods_fb
                        fallback_bullets.append(f"ODS: {', '.join(ods_fb)}")
                    if uprn_fb:
                        extracted["uprn"] = uprn_fb
                        fallback_bullets.append(f"UPRNs: {', '.join(uprn_fb)}")

                if wants_merged:
                    extracted["types"] = list(
                        set(
                            (extracted.get("types") or [])
                            + ["did:lidar-pointcloud-merged"]
                        )
                    )
                    fallback_bullets.append("Type: merged lidar pointcloud")
                if wants_rgb:
                    extracted["types"] = list(
                        set((extracted.get("types") or []) + ["did:rgb-image"])
                    )
                    fallback_bullets.append("Type: RGB image")

                url_fb = re.search(r"(https?://[\w\.-:%/]+)", nl)
                if url_fb:
                    extracted["db_url"] = url_fb.group(1)
                    fallback_bullets.append(f"Endpoint: {url_fb.group(1)}")

                if "uprn" in extracted:
                    router_text = (
                        f"Download assets for UPRN(s) {', '.join(extracted['uprn'])}"
                    )
                elif "output_area" in extracted:
                    router_text = f"List UPRNs in output areas {', '.join(extracted['output_area'])}"
                elif "ods" in extracted:
                    router_text = f"Map ODS {', '.join(extracted['ods'])} to UPRNs"
                else:
                    router_text = "no-op"
                    if not fallback_bullets:
                        fallback_bullets.append("No actionable request")

                summary_obj = {
                    "bullets": fallback_bullets,
                    "router_text": router_text,
                    "extracted": extracted,
                }

            bullets = summary_obj.get("bullets") or []
            summary_router_text = summary_obj.get("router_text") or None
            if show_summary and bullets and bullets != ["No actionable request"]:
                log.info("Summary: %s", " | ".join(bullets))

        except Exception as e:
            if show_summary:
                log.info("Summary step failed: %s (continuing with original input)", e)

    candidate_text = summary_router_text or nl

    # --- Try summary-extracted direct spec first ---
    if summary_obj and isinstance(summary_obj.get("extracted"), dict):
        ex = summary_obj["extracted"]
        inferred_command = None
        if ex.get("ods"):
            inferred_command = "ods_to_uprn"
        if ex.get("output_area"):
            inferred_command = "uprns_by_output_area"
        if ex.get("uprn"):
            inferred_command = "download_assets"

        if inferred_command:
            spec_direct = {
                "command": inferred_command,
                "uprn": ex.get("uprn"),
                "ods": ex.get("ods"),
                "output_area": ex.get("output_area"),
                "sensor": ex.get("sensor"),
                "types": ex.get("types"),
                "download_dir": ex.get("download_dir"),
                "api_key_env": ex.get("api_key_env"),
                "db_url": ex.get("db_url"),
            }
            if spec_direct.get("types") is None and "merged" in candidate_text.lower():
                spec_direct["types"] = ["did:lidar-pointcloud-merged"]

            try:
                argv = build_argv(spec_direct, py_exe, qa_path)
                log.debug(
                    "Router JSON (summary extracted):\n%s",
                    json.dumps(spec_direct, indent=2),
                )
                log.info("Command: %s", " ".join([shlex.quote(x) for x in argv]))
                if dry_run:
                    return 0
                proc = subprocess.run(argv)
                return proc.returncode
            except Exception:
                pass  # fall through

    # --- Heuristic fast path ---
    heuristic_spec = heuristic_parse(candidate_text)
    if heuristic_spec:
        log.debug("Heuristic spec: %s", json.dumps(heuristic_spec))
        spec = heuristic_spec
        argv = build_argv(spec, py_exe, qa_path)
        log.info("Command: %s", " ".join([shlex.quote(x) for x in argv]))
        if dry_run:
            return 0
        proc = subprocess.run(argv)
        return proc.returncode

    # --- Model routing ---
    messages = [{"role": "system", "content": SYSTEM_ROUTER_PROMPT}]
    for u, a in FEW_SHOTS:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": candidate_text})

    resp = ollama_chat(
        base_url=base_url,
        model=model_id,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        num_predict=num_predict,
        num_ctx=num_ctx,
        keep_alive=keep_alive,
        force_json=force_json,
    )
    if debug_model:
        log.debug(
            "Primary model raw JSON:\n%s",
            json.dumps(resp, indent=2, ensure_ascii=False),
        )

    content = extract_assistant_text_from_ollama(resp)
    log.debug("Primary model extracted content: %r", content)

    spec = None
    if content.strip():
        try:
            spec = json.loads(content)
        except Exception:
            try:
                blob = slice_first_json_object(content)
                spec = json.loads(blob)
            except Exception:
                spec = None

    if spec is None and force_json:
        log.debug("Empty/invalid JSON content; retrying without format=json...")
        resp2 = ollama_chat(
            base_url=base_url,
            model=model_id,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
            force_json=False,
        )
        if debug_model:
            log.debug(
                "Secondary model raw JSON:\n%s",
                json.dumps(resp2, indent=2, ensure_ascii=False),
            )
        content2 = extract_assistant_text_from_ollama(resp2)
        log.debug("Secondary model extracted content: %r", content2)
        try:
            spec = json.loads(content2)
        except Exception:
            try:
                blob = slice_first_json_object(content2)
                spec = json.loads(blob)
            except Exception:
                spec = None

    if spec is None:
        heuristic_spec = heuristic_parse(nl)
        if heuristic_spec:
            log.debug(
                "Fallback to heuristic after model failure: %s",
                json.dumps(heuristic_spec),
            )
            spec = heuristic_spec

    if not spec or spec.get("command") not in {
        "download_assets",
        "ods_to_uprn",
        "uprns_by_output_area",
    }:
        log.warning(
            "No actionable command inferred.\n"
            "Try examples like:\n"
            "  • 'Download the merged lidar point cloud for UPRN 5045394'\n"
            "  • 'Map ODS G85013 to UPRNs'\n"
            "  • 'List all UPRNs in output areas E00004550 and E00032882'"
        )
        log.debug("Raw model content:\n%s", content.strip())
        return 0

    argv = build_argv(spec, py_exe, qa_path)
    log.debug("Router JSON:\n%s", json.dumps(spec, indent=2))
    log.info("Command: %s", " ".join([shlex.quote(x) for x in argv]))

    if dry_run:
        return 0

    proc = subprocess.run(argv)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(
        description="Natural-language interface for query_assist.py using Ollama"
    )
    ap.add_argument("--model-id", default="gpt-oss:20b", help="Ollama model name/tag")
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
        help="Only log the derived command; do not execute",
    )
    ap.add_argument(
        "--once", "-q", help="Run a single NL query and exit (non-interactive)"
    )

    # Decoding / runtime knobs
    ap.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    ap.add_argument(
        "--top-p", type=float, default=0.95, help="Nucleus sampling probability"
    )
    ap.add_argument(
        "--num-predict", type=int, default=256, help="Max new tokens to generate"
    )
    ap.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Context window size hint (model-dependent)",
    )
    ap.add_argument(
        "--keep-alive", default=None, help="Ollama keep-alive (e.g., '5m', '30m', '0')"
    )
    ap.add_argument(
        "--no-force-json",
        action="store_true",
        help="Do not set format='json' in the chat call",
    )

    # Logging controls
    ap.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity ( -v = info, -vv = debug )",
    )
    ap.add_argument(
        "--debug-model",
        action="store_true",
        help="Force DEBUG and include raw model responses",
    )

    # Summarization controls
    ap.add_argument(
        "--no-summarize",
        action="store_true",
        help="Disable preliminary summarization step",
    )
    ap.add_argument(
        "--summary-model",
        default=None,
        help="Model ID for summarization (defaults to routing model)",
    )
    ap.add_argument(
        "--summary-temperature",
        type=float,
        default=0.0,
        help="Temperature for summarization model",
    )
    ap.add_argument(
        "--hide-summary", action="store_true", help="Do not log summarization bullets"
    )

    args = ap.parse_args()

    # Configure logging
    if args.debug_model or args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Verbose-mode introductory banner
    if level <= logging.INFO:
        _print_intro_banner(args.base_url, args.model_id)

    py_exe = sys.executable
    qa_path = args.query_assist_path

    try:
        if args.once:
            rc = run_once(
                base_url=args.base_url,
                model_id=args.model_id,
                nl=args.once,
                qa_path=qa_path,
                py_exe=py_exe,
                dry_run=args.dry_run,
                temperature=args.temperature,
                top_p=args.top_p,
                num_predict=args.num_predict,
                num_ctx=args.num_ctx,
                keep_alive=args.keep_alive,
                force_json=(not args.no_force_json),
                debug_model=args.debug_model,
                summarize=(not args.no_summarize),
                summary_model=(args.summary_model or args.model_id),
                summary_temperature=args.summary_temperature,
                show_summary=not args.hide_summary,
            )
            sys.exit(rc)

        if level <= logging.INFO:
            print("NL router for query_assist.py. Type 'exit' or Ctrl-D to quit.")
        while True:
            try:
                nl = input("> ").strip()
            except EOFError:
                break
            if not nl:
                continue
            if nl.lower() in {"exit", "quit"}:
                break
            rc = run_once(
                base_url=args.base_url,
                model_id=args.model_id,
                nl=nl,
                qa_path=qa_path,
                py_exe=py_exe,
                dry_run=args.dry_run,
                temperature=args.temperature,
                top_p=args.top_p,
                num_predict=args.num_predict,
                num_ctx=args.num_ctx,
                keep_alive=args.keep_alive,
                force_json=(not args.no_force_json),
                debug_model=args.debug_model,
                summarize=(not args.no_summarize),
                summary_model=(args.summary_model or args.model_id),
                summary_temperature=args.summary_temperature,
                show_summary=not args.hide_summary,
            )
            if rc != 0:
                log.warning("Subprocess exited with code %d", rc)
    except KeyboardInterrupt:
        # Cleanly handle Ctrl-C in REPL
        print()
        log.info("Interrupted.")
        sys.exit(130)
    except Exception as e:
        log.error("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
