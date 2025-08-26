#!/usr/bin/env python3
"""
nl_query_workflow_cli.py

A LangGraph-powered superset of nl_query_cli.py that can execute multi-stage
workflows against query_assist.py. Examples it can handle in one NL turn:

- "Get all point clouds in output area E00004550"  -->
    Step 1: uprns_by_output_area -> CSV(s)
    Step 2: download_assets --uprn <CSV(s)> --types did:lidar-pointcloud-merged,did:lidar-pointcloud-frame

- "For ODS G85013, download RGB and merged lidar to /data" -->
    Step 1: ods_to_uprn -> /.../downloads/ods_to_uprn.csv
    Step 2: download_assets --uprn ods_to_uprn.csv --types did:rgb-image,did:lidar-pointcloud-merged --download-dir /data

It preserves the CLI surface and routing behavior of nl_query_cli.py, but adds:
- Planner (heuristics + optional LLM) that compiles an ordered step list
- LangGraph execution loop with checkpoint-able state, retries, and artifact passing
- Robust parsing of query_assist.py logs to discover produced CSV artifacts
- Deterministic "dry-run" plan prints for auditability
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from typing import Any, Literal, TypedDict

# --- Third-party ---
# pip install langgraph[all] requests
import requests
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# ======================================================================================
# Carry forward the routing prompts / type mappings / helpers from nl_query_cli.py
# (kept in-sync conceptually; this file does not import the other to stay standalone).
# ======================================================================================

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

TYPE_ALIASES = {
    # canonical mappings used throughout your stack
    "rgb": "did:rgb-image",
    "rgb image": "did:rgb-image",
    "merged lidar": "did:lidar-pointcloud-merged",
    "merged lidar point cloud": "did:lidar-pointcloud-merged",
    "lidar point cloud": "did:lidar-pointcloud-frame",
    "point cloud": "did:lidar-pointcloud-frame",
    "point clouds": None,  # expands to both merged + frame
    "lidar range panorama": "did:lidar-range-pano",
    "lidar reflectance panorama": "did:lidar-reflectance-pano",
    "lidar signal panorama": "did:lidar-signal-pano",
    "lidar nearir panorama": "did:lidar-nearir-pano",
    "ir false color": "did:ir-false-color-image",
    "ir temperature array": "did:ir-temperature-array",
    "ir counts": "did:ir-count-image",
    "temperature": "did:celsius-temperature",
    "relative humidity": "did:relative-humidity",
}

POINTCLOUD_BOTH = ["did:lidar-pointcloud-merged", "did:lidar-pointcloud-frame"]

# ----------------------------------------------------------------------------------
# High-level planner system prompt (multi-step). Explains assets & synonyms so the
# LLM can infer intent even with loose language ("building" -> UPRN, etc.).
# ----------------------------------------------------------------------------------
PLAN_SYSTEM_PROMPT = """
You are a planning assistant that converts a natural language request about
retrieving built environment asset data into an ordered execution plan for
query_assist.py.

Available low-level commands (same as router):
    1. uprns_by_output_area – given output area code(s) yield UPRN CSV(s)
    2. ods_to_uprn         – given ODS clinical practice code(s) yield UPRN CSV
    3. download_assets     – given UPRN(s) (list or CSV path) download assets

Important asset type IRIs (use when mentioned or implied):
    did:rgb-image, did:lidar-pointcloud-merged, did:lidar-pointcloud-frame,
    did:lidar-range-pano, did:lidar-reflectance-pano, did:lidar-signal-pano,
    did:lidar-nearir-pano, did:ir-false-color-image, did:ir-temperature-array,
    did:ir-count-image, did:celsius-temperature, did:relative-humidity

Synonyms / interpretation guidance:
    "building", "buildings", "property", "properties" -> treat as UPRN(s)
    "practice", "gp practice" -> ODS code
    "thermal array", "temperature array", "thermal sensor" -> did:ir-temperature-array
    "thermal image" -> did:ir-false-color-image (unless array explicitly stated)
    "point clouds" (plural, no qualifier) -> both merged + frame
    "merged point cloud" -> did:lidar-pointcloud-merged
    "point cloud frame" / "single frame" -> did:lidar-pointcloud-frame
    If user asks for "temperature and humidity" -> did:celsius-temperature + did:relative-humidity

Planning rules:
    - If an intermediate mapping (ODS or output area) is needed to reach UPRNs, plan that first, then a download_assets step referencing previous CSV output (use uprn_from_previous_csvs=true instead of explicit uprn list).
    - If the user directly supplies UPRNs (numbers) OR a CSV path that obviously contains UPRNs, a single download_assets step may suffice.
    - Always be decisive; include only the steps required.

Return JSON ONLY, schema:
{
    "steps": [
         {
             "command": "uprns_by_output_area"|"ods_to_uprn"|"download_assets",
             "output_area": string|[string]|null,
             "ods": string|[string]|null,
             "uprn": string|[string]|null,
             "types": [string]|null,
             "sensor": string|null,
             "download_dir": string|null,
             "api_key_env": string|null,
             "db_url": string|null,
             "uprn_from_previous_csvs": true|false|null
         }, ...
    ]
}

Notes:
    - Omit keys or set null when not applicable.
    - Use uprn_from_previous_csvs=true only on a download_assets step that should read the CSV(s) produced by previous mapping step(s).
    - Do NOT include explanatory prose.
"""


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


def _find_csvs_emitted(stream_text: str) -> list[str]:
    """
    Parse query_assist.py logs to discover created CSVs.
    It prints e.g.:
      "✔ Saved CSV for {oa} → {path}"
      "✔ Saved ODS→UPRN CSV → {path}"
    Accept both unicode arrows and ASCII '->'.
    """
    csvs = []
    # Unicode arrow / ASCII arrow variants, greedy path match until whitespace end
    patterns = [
        r"Saved CSV for .*? → ([^\s]+\.csv)",
        r"Saved CSV for .*? -> ([^\s]+\.csv)",
        r"Saved ODS.?UPRN CSV .*? → ([^\s]+\.csv)",
        r"Saved ODS.?UPRN CSV .*? -> ([^\s]+\.csv)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, stream_text):
            csvs.append(m.group(1))
    # Deduplicate preserving order
    seen = set()
    out = []
    for p in csvs:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _ensure_list_or_path(v: None | str | list[str]) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    # Post-augmentation: if user asked for assets but only got a mapping step, add download step
    lowered = state.nl.lower()
    if len(plan) == 1 and ("asset" in lowered or "all the" in lowered):
        first_cmd = plan[0].get("command")
        if first_cmd in {"ods_to_uprn", "uprns_by_output_area"}:
            # Only add if no existing download step follows
            plan.append(
                {
                    "command": "download_assets",
                    "uprn_from_previous_csvs": True,  # consume prior CSV
                    "types": None,  # all asset types
                    "download_dir": plan[0].get("download_dir")
                    or defaults.get("download_dir"),
                    "api_key_env": defaults.get("api_key_env"),
                    "db_url": plan[0].get("db_url") or defaults.get("db_url"),
                }
            )
    state.plan = plan
    return [s] if s else []


def _build_argv(spec: dict[str, Any], py: str, qa_path: str) -> list[str]:
    cmd = [py, qa_path]
    command = spec.get("command")
    if command == "download_assets":
        uprn = _ensure_list_or_path(spec.get("uprn"))
        if not uprn:
            raise ValueError("download_assets requires 'uprn'.")
        cmd += ["--uprn"] + uprn
        if spec.get("sensor"):
            cmd += ["--sensor", str(spec["sensor"])]
        if spec.get("types"):
            cmd += ["--types", ",".join(spec["types"])]
    elif command == "ods_to_uprn":
        ods = _ensure_list_or_path(spec.get("ods"))
        if not ods:
            raise ValueError("ods_to_uprn requires 'ods'.")
        cmd += ["--ods"] + ods
    elif command == "uprns_by_output_area":
        oa = _ensure_list_or_path(spec.get("output_area"))
        if not oa:
            raise ValueError("uprns_by_output_area requires 'output_area'.")
        cmd += ["--output-area"] + oa
    else:
        raise ValueError(f"Unsupported command: {command!r}")

    if spec.get("db_url"):
        cmd += ["--db-url", str(spec["db_url"])]
    if spec.get("download_dir"):
        cmd += ["--download-dir", str(spec["download_dir"])]
    if spec.get("api_key_env"):
        cmd += ["--api-key-env", str(spec["api_key_env"])]
    return cmd


def _map_types_from_text(lowered: str) -> list[str] | None:
    # Broad heuristics for types based on NL
    wants_pointclouds = re.search(r"\bpoint\s*clouds?\b", lowered) is not None
    wants_merged = "merged lidar" in lowered or re.search(
        r"merged\s+lidar\s+point\s*cloud", lowered
    )
    wants_frame = "pointcloud frame" in lowered or "single frame" in lowered

    types = []
    if wants_pointclouds:
        types.extend(POINTCLOUD_BOTH)
    if wants_merged:
        types.append("did:lidar-pointcloud-merged")
    if wants_frame:
        types.append("did:lidar-pointcloud-frame")

    if "rgb" in lowered and "image" in lowered:
        types.append("did:rgb-image")
    if "range panorama" in lowered:
        types.append("did:lidar-range-pano")
    if "reflectance panorama" in lowered:
        types.append("did:lidar-reflectance-pano")
    if "signal panorama" in lowered:
        types.append("did:lidar-signal-pano")
    if "nearir" in lowered or "near-infrared" in lowered:
        types.append("did:lidar-nearir-pano")
    if "ir false" in lowered:
        types.append("did:ir-false-color-image")
    if "ir temperature array" in lowered:
        types.append("did:ir-temperature-array")
    # Additional thermal/temperature array synonyms
    if (
        re.search(r"thermal\s+arrays?", lowered)
        or re.search(r"temperature\s+arrays?", lowered)
        or "thermal array" in lowered
    ):
        types.append("did:ir-temperature-array")
    if re.search(r"thermal\s+images?", lowered):
        # Map generic thermal image request to false-color IR image if not already specified
        types.append("did:ir-false-color-image")

    if not types:
        return None
    # dedupe preserve order
    out, seen = [], set()
    for t in types:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


# ======================================================================================
# Planning + LangGraph state
# ======================================================================================


class StepSpec(TypedDict, total=False):
    command: Literal["download_assets", "ods_to_uprn", "uprns_by_output_area"]
    uprn: list[str] | str | None
    ods: list[str] | str | None
    output_area: list[str] | str | None
    sensor: str | None
    types: list[str] | None
    download_dir: str | None
    api_key_env: str | None
    db_url: str | None
    # internal: indicates this step will take CSVs from previous steps
    uprn_from_previous_csvs: bool


@dataclasses.dataclass
class WFState:
    nl: str
    plan: list[StepSpec]
    current: int = 0
    artifacts: dict[str, Any] = dataclasses.field(
        default_factory=dict
    )  # e.g., {"csvs": [...]}
    log: list[str] = dataclasses.field(default_factory=list)
    dry_run: bool = False
    py_exe: str = sys.executable
    qa_path: str = os.path.join(os.path.dirname(__file__), "query_assist.py")
    base_url: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model_id: str = "gpt-oss:20b"
    temperature: float = 0.0
    top_p: float = 0.95
    num_predict: int = 256
    num_ctx: int | None = None
    keep_alive: str | None = None
    force_json: bool = True
    verbose_level: int = logging.INFO
    max_steps: int = 8


def _coerce_wfstate(obj: Any) -> WFState:
    """Ensure we have a WFState instance (LangGraph may return a plain dict)."""
    if isinstance(obj, WFState):
        return obj
    if isinstance(obj, dict):
        # Build kwargs respecting dataclass fields
        kwargs = {}
        for f in dataclasses.fields(WFState):
            if f.name in obj:
                kwargs[f.name] = obj[f.name]
            else:
                if f.default is not dataclasses.MISSING:  # type: ignore[attr-defined]
                    kwargs[f.name] = f.default  # type: ignore[assignment]
                elif getattr(f, "default_factory", dataclasses.MISSING) is not dataclasses.MISSING:  # type: ignore[attr-defined]
                    kwargs[f.name] = f.default_factory()  # type: ignore[call-arg]
                else:
                    kwargs[f.name] = None
        return WFState(**kwargs)  # type: ignore[arg-type]
    raise TypeError(f"Cannot coerce state of type {type(obj)} to WFState")


# ======================================================================================
# Heuristic Planner (covers the common multi-step cases deterministically)
# ======================================================================================


def heuristic_plan(nl: str, defaults: dict[str, Any]) -> list[StepSpec] | None:
    """
    Returns a list of StepSpec if it can deduce a plan without the LLM.
    Covers:
      - output area + asset types -> [uprns_by_output_area, download_assets]
      - ods + asset types -> [ods_to_uprn, download_assets]
      - simple one-shot commands (download_assets / ods_to_uprn / uprns_by_output_area)
    """
    text = nl.strip()
    if not text:
        return None
    lowered = text.lower()

    csv_paths = re.findall(r'(?:(?:[A-Za-z]:)?[^\s"\'<>|]+\.csv)\b', text)
    oa_codes = re.findall(r"\bE\d{8}\b", text)
    ods_codes = re.findall(r"\b[A-Z]\d{5}\b", text)
    uprns = re.findall(r"\b\d{6,}\b", text)
    endpoint_match = re.search(r"(https?://[\w\.-:%/]+)", text)
    endpoint_url = endpoint_match.group(1) if endpoint_match else None

    # Extract common options if present
    download_dir = None
    m = re.search(r"(?: to | into )\s+(/[^ ]+)", lowered)
    if m:
        download_dir = m.group(1)

    # Types
    types = _map_types_from_text(lowered)

    # If user provided a CSV and mentioned UPRNs -> treat as download step
    if csv_paths and ("uprn" in lowered or "uprns" in lowered):
        return [
            {
                "command": "download_assets",
                "uprn": csv_paths if len(csv_paths) > 1 else csv_paths[0],
                "types": types,
                "download_dir": download_dir or defaults.get("download_dir"),
                "api_key_env": defaults.get("api_key_env"),
                "db_url": endpoint_url or defaults.get("db_url"),
            }
        ]

    # output area + types (or "point clouds")
    if (oa_codes or "output area" in lowered or "output areas" in lowered) and (
        types or "point cloud" in lowered
    ):
        oa_list = oa_codes if oa_codes else []
        # allow CSV of output areas too
        if csv_paths and not ("uprn" in lowered):
            oa_list = csv_paths if oa_list == [] else oa_list + csv_paths

        return [
            {
                "command": "uprns_by_output_area",
                "output_area": oa_list
                if oa_list
                else (csv_paths[0] if csv_paths else None),
                "download_dir": defaults.get("download_dir"),
                "db_url": defaults.get("db_url"),
            },
            {
                "command": "download_assets",
                "uprn_from_previous_csvs": True,  # take outputs of step 1
                "types": types
                or POINTCLOUD_BOTH,  # default to both if only said "point clouds"
                "download_dir": download_dir or defaults.get("download_dir"),
                "api_key_env": defaults.get("api_key_env"),
                "db_url": defaults.get("db_url"),
            },
        ]

    # ods + types
    if (ods_codes or "ods" in lowered) and (types is not None):
        ods_list = ods_codes if ods_codes else (csv_paths if csv_paths else None)
        return [
            {
                "command": "ods_to_uprn",
                "ods": ods_list,
                "download_dir": defaults.get("download_dir"),
                "db_url": defaults.get("db_url"),
            },
            {
                "command": "download_assets",
                "uprn_from_previous_csvs": True,
                "types": types,
                "download_dir": download_dir or defaults.get("download_dir"),
                "api_key_env": defaults.get("api_key_env"),
                "db_url": defaults.get("db_url"),
            },
        ]

    # plain output area listing (no types mentioned)
    if oa_codes or ("output area" in lowered or "output areas" in lowered):
        return [
            {
                "command": "uprns_by_output_area",
                "output_area": oa_codes
                if oa_codes
                else (csv_paths[0] if csv_paths else None),
                "download_dir": defaults.get("download_dir"),
                "db_url": defaults.get("db_url"),
            }
        ]

    # plain ods mapping (no types)
    if ods_codes or "ods" in lowered:
        return [
            {
                "command": "ods_to_uprn",
                "ods": ods_codes
                if ods_codes
                else (csv_paths[0] if csv_paths else None),
                "download_dir": defaults.get("download_dir"),
                "db_url": defaults.get("db_url"),
            }
        ]

    # direct UPRN download
    if uprns and (
        "download" in lowered
        or "assets" in lowered
        or "point" in lowered
        or "rgb" in lowered
        or "image" in lowered
        or "lidar" in lowered
    ):
        return [
            {
                "command": "download_assets",
                "uprn": uprns,
                "types": types,
                "download_dir": download_dir or defaults.get("download_dir"),
                "api_key_env": defaults.get("api_key_env"),
                "db_url": endpoint_url or defaults.get("db_url"),
            }
        ]

    return None


# ======================================================================================
# LLM multi-step planner (primary) – falls back to heuristics if invalid/empty.
# ======================================================================================


def llm_plan(
    nl: str,
    defaults: dict[str, Any],
    base_url: str,
    model_id: str,
    temperature: float,
    top_p: float,
    num_predict: int,
    num_ctx: int | None,
    keep_alive: str | None,
    force_json: bool,
) -> list[StepSpec] | None:
    messages = [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": nl},
    ]
    try:
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
        content = None
        if isinstance(resp.get("message"), dict):
            content = resp["message"].get("content")
        if not content and isinstance(resp.get("response"), str):
            content = resp.get("response")
        if not content:
            return None
        plan_obj = None
        try:
            plan_obj = json.loads(content)
        except Exception:
            plan_obj = _extract_first_json(content)
        if not plan_obj or not isinstance(plan_obj, dict):
            return None
        steps_raw = plan_obj.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            return None
        steps: list[StepSpec] = []
        for s in steps_raw:
            if not isinstance(s, dict):
                continue
            cmd = s.get("command")
            if cmd not in {"download_assets", "ods_to_uprn", "uprns_by_output_area"}:
                continue
            step: StepSpec = {
                "command": cmd,  # type: ignore[assignment]
            }
            for key in [
                "uprn",
                "ods",
                "output_area",
                "types",
                "sensor",
                "download_dir",
                "api_key_env",
                "db_url",
            ]:
                if key in s:
                    step[key] = s[key]  # type: ignore[index, assignment]
            if s.get("uprn_from_previous_csvs"):
                step["uprn_from_previous_csvs"] = True  # type: ignore[index]
            # Inject defaults where appropriate
            if "api_key_env" not in step and defaults.get("api_key_env"):
                step["api_key_env"] = defaults["api_key_env"]  # type: ignore[index]
            if "download_dir" not in step and defaults.get("download_dir"):
                step["download_dir"] = defaults["download_dir"]  # type: ignore[index]
            steps.append(step)
        return steps or None
    except Exception:
        return None


# ======================================================================================
# Optional: LLM Planner fallback (uses your Ollama server same as nl_query_cli.py)
# Produces a single-step spec; we then upgrade it into a plan if the NL implies multi.
# ======================================================================================


def ollama_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    top_p: float = 0.95,
    num_predict: int = 256,
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


def llm_route_to_spec(nl: str, base_url: str, model_id: str, **opts) -> StepSpec | None:
    messages = [
        {"role": "system", "content": SYSTEM_ROUTER_PROMPT},
        {"role": "user", "content": nl},
    ]
    resp = ollama_chat(base_url, model_id, messages, **opts)
    content = None
    if isinstance(resp.get("message"), dict):
        content = resp["message"].get("content")
    if not content and isinstance(resp.get("response"), str):
        content = resp["response"]
    if not content:
        return None
    obj = _extract_first_json(content) or None
    if not obj or "command" not in obj:
        return None
    # Normalize to StepSpec
    step: StepSpec = {
        "command": obj["command"],
        "uprn": obj.get("uprn"),
        "ods": obj.get("ods"),
        "output_area": obj.get("output_area"),
        "sensor": obj.get("sensor"),
        "types": obj.get("types"),
        "download_dir": obj.get("download_dir"),
        "api_key_env": obj.get("api_key_env"),
        "db_url": obj.get("db_url"),
    }
    return step


def upgrade_single_spec_to_plan(
    nl: str, spec: StepSpec, defaults: dict[str, Any]
) -> list[StepSpec]:
    """
    If NL implies multi-stage (e.g., output area + assets, or ODS + assets),
    expand the single routed spec into a 2-step plan. Otherwise return [spec].
    """
    lowered = nl.lower()
    types = spec.get("types") or _map_types_from_text(lowered)
    # output area -> assets
    if spec.get("command") == "uprns_by_output_area" and (
        types or "point cloud" in lowered or "download" in lowered
    ):
        return [
            spec,
            {
                "command": "download_assets",
                "uprn_from_previous_csvs": True,
                "types": types or POINTCLOUD_BOTH,
                "download_dir": spec.get("download_dir")
                or defaults.get("download_dir"),
                "api_key_env": spec.get("api_key_env") or defaults.get("api_key_env"),
                "db_url": spec.get("db_url") or defaults.get("db_url"),
            },
        ]
    # ODS -> assets
    if spec.get("command") == "ods_to_uprn" and (types is not None):
        return [
            spec,
            {
                "command": "download_assets",
                "uprn_from_previous_csvs": True,
                "types": types,
                "download_dir": spec.get("download_dir")
                or defaults.get("download_dir"),
                "api_key_env": spec.get("api_key_env") or defaults.get("api_key_env"),
                "db_url": spec.get("db_url") or defaults.get("db_url"),
            },
        ]
    return [spec]


# ======================================================================================
# Execution helpers
# ======================================================================================


def run_query_assist_step(
    step: StepSpec,
    py_exe: str,
    qa_path: str,
    dry_run: bool,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Execute a single step via subprocess, stream logs, and return (rc, captured_text)."""
    argv = _build_argv(step, py_exe, qa_path)
    printable = " ".join(shlex.quote(x) for x in argv)
    logging.info("Command: %s", printable)
    if dry_run:
        return 0, f"[dry-run] {printable}\n"

    p = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
    )
    captured_lines = []
    try:
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            captured_lines.append(line)
    finally:
        rc = p.wait()
    return rc, "".join(captured_lines)


def materialize_previous_uprn_csvs(state: WFState, step_idx: int) -> list[str]:
    """
    Return list of CSV paths produced by earlier steps.
    Uses parsed logs; if nothing parsed but we are in ODS->UPRN flow, default
    to <download_dir or CWD>/downloads/ods_to_uprn.csv (query_assist behavior).
    """
    from_logs = state.artifacts.get("csvs", [])
    if from_logs:
        return from_logs

    # fallback heuristic for ODS mapping
    dl_base = state.plan[0].get("download_dir") or os.path.join(
        os.getcwd(), "downloads"
    )
    candidate = os.path.join(dl_base, "ods_to_uprn.csv")
    if os.path.isfile(candidate):
        return [candidate]
    return []


# ======================================================================================
# LangGraph nodes
# ======================================================================================


def node_plan(state: WFState) -> WFState:
    defaults = {
        "download_dir": None,  # leave None -> query_assist default ./downloads
        "api_key_env": "API_KEY",  # aligns with query_assist default
        "db_url": None,  # use query_assist default unless user overrides
    }
    # Primary: LLM multi-step planner
    plan = llm_plan(
        state.nl,
        defaults,
        base_url=state.base_url,
        model_id=state.model_id,
        temperature=state.temperature,
        top_p=state.top_p,
        num_predict=state.num_predict,
        num_ctx=state.num_ctx,
        keep_alive=state.keep_alive,
        force_json=state.force_json,
    )
    # Fallback: legacy heuristics
    if not plan:
        plan = heuristic_plan(state.nl, defaults)
    if not plan:
        state.plan = []
        state.log.append("No actionable plan could be inferred.")
        return state
    state.plan = plan
    return state


def node_execute(state: WFState) -> WFState:
    if state.current >= len(state.plan):
        return state  # nothing to do
    step = state.plan[state.current]

    # If this step takes UPRNs from previous CSVs, resolve them now
    if step.get("uprn_from_previous_csvs"):
        csvs = materialize_previous_uprn_csvs(state, state.current)
        if not csvs:
            state.log.append("No CSVs found from previous step(s).")
            # Fail this step
            state.current = len(state.plan)
            return state
        step = {**step}
        step.pop("uprn_from_previous_csvs", None)
        step["uprn"] = csvs

    # Execute
    rc, captured = run_query_assist_step(
        step, state.py_exe, state.qa_path, state.dry_run
    )
    state.log.append(captured)

    # Parse any emitted CSV artifacts for downstream use
    newly_found = _find_csvs_emitted(captured)
    if newly_found:
        extant = state.artifacts.get("csvs", [])
        state.artifacts["csvs"] = list(dict.fromkeys(extant + newly_found))

    # Advance or stop on error
    if rc != 0:
        state.log.append(f"Step {state.current} returned non-zero exit {rc}.")
        state.current = len(state.plan)  # abort
    else:
        state.current += 1
    return state


def node_check_done(state: WFState) -> str:
    if state.current >= len(state.plan):
        return END
    if state.current >= state.max_steps:
        state.log.append(f"Aborting: exceeded max_steps={state.max_steps}")
        return END
    return "execute"


# ======================================================================================
# CLI / main
# ======================================================================================


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="LangGraph NL workflow CLI for query_assist.py (multi-stage capable)"
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
        "--dry-run", action="store_true", help="Plan/print but do not execute"
    )
    ap.add_argument("--once", "-q", help="Run a single NL query and exit")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--num-ctx", type=int, default=None)
    ap.add_argument("--keep-alive", default=None)
    ap.add_argument("--no-force-json", action="store_true")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v=info, -vv=debug"
    )
    ap.add_argument(
        "--plan-only", action="store_true", help="Only show the compiled plan and exit"
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Logging level
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Intro banner
    if level <= logging.INFO:
        body = (
            "- Parses NL to a multi-step plan (heuristics + optional LLM).\n"
            "- Executes steps via LangGraph with artifact passing.\n"
            "- Detects CSVs emitted by query_assist.py and feeds them forward.\n"
            "- Supports dry-run and plan-only modes for auditability."
        )
        print(_render_box(f"LangGraph NL Workflow — {args.model_id}", body))

    # Build the LangGraph
    builder = StateGraph(WFState)
    builder.add_node("plan", node_plan)
    builder.add_node("execute", node_execute)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan", lambda s: "execute" if s.plan else END, {"execute": "execute", END: END}
    )
    builder.add_conditional_edges(
        "execute", node_check_done, {"execute": "execute", END: END}
    )
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    # One-shot or REPL
    def run_once(nl: str) -> int:
        st = WFState(
            nl=nl,
            plan=[],
            dry_run=bool(args.dry_run),
            qa_path=args.query_assist_path,
            base_url=args.base_url,
            model_id=args.model_id,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
            num_ctx=args.num_ctx,
            keep_alive=args.keep_alive,
            force_json=(not args.no_force_json),
            verbose_level=level,
            max_steps=args.max_steps,
        )
        # PLAN
        st = _coerce_wfstate(
            graph.invoke(
                st, config={"configurable": {"thread_id": f"tid-{time.time_ns()}"}}
            )
        )
        st = node_plan(st)
        if not st.plan:
            logging.warning("No actionable plan produced for: %r", nl)
            if st.log:
                print("\n".join(st.log))
            return 0

        # Print plan only at INFO or DEBUG verbosity
        if level <= logging.INFO:
            print("Plan:")
            for i, step in enumerate(st.plan):
                step_disp = {
                    k: v for k, v in step.items() if k != "uprn_from_previous_csvs"
                }
                print(f"  {i+1}. {json.dumps(step_disp, ensure_ascii=False)}")
            print()
        if args.plan_only:
            return 0

        # EXECUTE
        while st.current < len(st.plan) and st.current < st.max_steps:
            status = node_check_done(st)
            if status == END:
                break
            st = node_execute(st)
            if st.current >= len(st.plan):
                # Re-check termination
                if node_check_done(st) == END:
                    break
        # Final logs (already streamed, but attach any notes)
        if st.log:
            trailing = [
                line
                for line in st.log
                if "[dry-run]" in line
                or "No CSVs found" in line
                or "non-zero exit" in line
            ]
            if trailing:
                print("\n".join(trailing))
        return 0

    try:
        if args.once:
            sys.exit(run_once(args.once))
        if level <= logging.INFO:
            print(
                "LangGraph NL workflow for query_assist.py. Type 'exit' or Ctrl-D to quit."
            )
        while True:
            try:
                nl = input("> ").strip()
            except EOFError:
                break
            if not nl:
                continue
            if nl.lower() in {"exit", "quit"}:
                break
            rc = run_once(nl)
            if rc != 0:
                logging.warning("Workflow exited with code %d", rc)
    except KeyboardInterrupt:
        print()
        logging.info("Interrupted.")
        sys.exit(130)
    except Exception as e:
        logging.error("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
