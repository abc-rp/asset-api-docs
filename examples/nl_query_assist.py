#!/usr/bin/env python3

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

import requests
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

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
  - UPRNs are the UK OS Unique Property Reference Numbers. Queries may call them buildings or other built environment associated words.
  - Output areas may be called OAs.
  - ODS codes are unique identifiers for UK NHS buildings, hence words like medical, practice, hospital, etc... may be used.
- Prefer being decisive. When in doubt, infer sensible defaults.
"""

TYPE_ALIASES = {
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


def _find_csvs_emitted(stream_text: str) -> list[str]:
    """Parse query_assist.py logs to discover created CSVs."""
    csvs: list[str] = []
    patterns = [
        r"Saved CSV for .*? → ([^\s]+\.csv)",
        r"Saved CSV for .*? -> ([^\s]+\.csv)",
        r"Saved ODS.?UPRN CSV .*? → ([^\s]+\.csv)",
        r"Saved ODS.?UPRN CSV .*? -> ([^\s]+\.csv)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, stream_text):
            csvs.append(m.group(1))
    seen: set[str] = set()
    out: list[str] = []
    for p in csvs:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _ensure_list_or_path(v: None | str | list[str]) -> list[str]:
    """Convert (None | str | list[str]) to a flat argv-ready list."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    s = str(v).strip()
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
    wants_pointclouds = re.search(r"\bpoint\s*clouds?\b", lowered) is not None
    wants_merged = "merged lidar" in lowered or re.search(
        r"merged\s+lidar\s+point\s*cloud", lowered
    )
    wants_frame = "pointcloud frame" in lowered or "single frame" in lowered

    types: list[str] = []
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
    if (
        "ir temperature array" in lowered
        or re.search(r"thermal\s+arrays?", lowered)
        or re.search(r"temperature\s+arrays?", lowered)
    ):
        types.append("did:ir-temperature-array")
    if re.search(r"thermal\s+images?", lowered):
        types.append("did:ir-false-color-image")

    if not types:
        return None

    out: list[str] = []
    seen: set[str] = set()
    for t in types:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


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
    uprn_from_previous_csvs: bool


@dataclasses.dataclass
class WFState:
    nl: str
    plan: list[StepSpec]
    current: int = 0
    artifacts: dict[str, Any] = dataclasses.field(default_factory=dict)
    log: list[str] = dataclasses.field(default_factory=list)
    actions: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    dry_run: bool = False
    plan_only: bool = False
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


def heuristic_plan(nl: str, defaults: dict[str, Any]) -> list[StepSpec] | None:
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

    download_dir = None
    m = re.search(r"(?: to | into )\s+(/[^ ]+)", lowered)
    if m:
        download_dir = m.group(1)

    types = _map_types_from_text(lowered)

    if csv_paths and ("uprn" in lowered or "uprns" in lowered):
        step: dict[str, Any] = {
            "command": "download_assets",
            "uprn": csv_paths if len(csv_paths) > 1 else csv_paths[0],
            "types": types,
            "download_dir": download_dir or defaults.get("download_dir"),
            "db_url": endpoint_url or defaults.get("db_url"),
        }
        return [step]

    if (oa_codes or "output area" in lowered or "output areas" in lowered) and (
        types
        or "point cloud" in lowered
        or "assets" in lowered
        or "download" in lowered
    ):
        oa_list: list[str] = []
        if oa_codes:
            oa_list.extend(oa_codes)
        if csv_paths and ("uprn" not in lowered):
            oa_list.extend(csv_paths)
        if not oa_list:
            return None
        second: dict[str, Any] = {
            "command": "download_assets",
            "uprn_from_previous_csvs": True,
            "download_dir": download_dir or defaults.get("download_dir"),
            "db_url": defaults.get("db_url"),
        }
        if types:
            second["types"] = types
        return [
            {
                "command": "uprns_by_output_area",
                "output_area": oa_list
                if oa_list
                else (csv_paths[0] if csv_paths else None),
                "download_dir": defaults.get("download_dir"),
                "db_url": defaults.get("db_url"),
            },
            second,
        ]

    if (ods_codes or "ods" in lowered) and (types is not None):
        ods_list: list[str] = []
        if ods_codes:
            ods_list.extend(ods_codes)
        if csv_paths:
            ods_list.extend(csv_paths)
        if not ods_list:
            return None
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
                "db_url": defaults.get("db_url"),
            },
        ]

    if oa_codes or ("output area" in lowered or "output areas" in lowered):
        oa_list: list[str] = []
        if oa_codes:
            oa_list.extend(oa_codes)
        if csv_paths and ("uprn" not in lowered):
            oa_list.extend(csv_paths)
        if not oa_list:
            return None
        return [
            {
                "command": "uprns_by_output_area",
                "output_area": oa_list if len(oa_list) > 1 else oa_list[0],
                "download_dir": defaults.get("download_dir"),
                "db_url": defaults.get("db_url"),
            }
        ]

    if ods_codes or "ods" in lowered:
        ods_list: list[str] = []
        if ods_codes:
            ods_list.extend(ods_codes)
        if csv_paths:
            ods_list.extend(csv_paths)
        if not ods_list:
            return None
        return [
            {
                "command": "ods_to_uprn",
                "ods": ods_list if len(ods_list) > 1 else ods_list[0],
                "download_dir": defaults.get("download_dir"),
                "db_url": defaults.get("db_url"),
            }
        ]

    if uprns and (
        "download" in lowered
        or "assets" in lowered
        or "point" in lowered
        or "rgb" in lowered
        or "image" in lowered
        or "lidar" in lowered
    ):
        step: dict[str, Any] = {
            "command": "download_assets",
            "uprn": uprns,
            "types": types,
            "download_dir": download_dir or defaults.get("download_dir"),
            "db_url": endpoint_url or defaults.get("db_url"),
        }
        return [step]

    return None


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
        {
            "role": "system",
            "content": (
                "You are a planning assistant that compiles an ordered plan for query_assist.py.\n"
                "Return JSON with a 'steps' array; use 'uprn_from_previous_csvs': true when a download_assets step should consume prior CSVs.\n"
                "Types to use when implied: did:rgb-image, did:lidar-pointcloud-merged, did:lidar-pointcloud-frame, did:lidar-range-pano, did:lidar-reflectance-pano, did:lidar-signal-pano, did:lidar-nearir-pano, did:ir-false-color-image, did:ir-temperature-array, did:ir-count-image, did:celsius-temperature, did:relative-humidity."
            ),
        },
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
            content = resp["response"]
        if not content:
            return None
        plan_obj = None
        try:
            plan_obj = json.loads(content)
        except Exception:
            plan_obj = _extract_first_json(content)
        if not isinstance(plan_obj, dict):
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
            step: StepSpec = {"command": cmd}
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
                    step[key] = s[key]
            if s.get("uprn_from_previous_csvs"):
                step["uprn_from_previous_csvs"] = True
            if "download_dir" not in step and defaults.get("download_dir"):
                step["download_dir"] = defaults["download_dir"]
            steps.append(step)
        return steps or None
    except Exception:
        return None


def llm_route_to_spec(nl: str, base_url: str, model_id: str, **opts) -> StepSpec | None:
    messages = [
        {"role": "system", "content": SYSTEM_ROUTER_PROMPT},
        {"role": "user", "content": nl},
    ]
    try:
        resp = ollama_chat(base_url, model_id, messages, **opts)
    except Exception:
        return None
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
    lowered = nl.lower()
    types = spec.get("types") or _map_types_from_text(lowered)
    if spec.get("command") == "uprns_by_output_area" and (
        types
        or "point cloud" in lowered
        or "download" in lowered
        or "assets" in lowered
    ):
        second: dict[str, Any] = {
            "command": "download_assets",
            "uprn_from_previous_csvs": True,
            "download_dir": spec.get("download_dir"),
            "db_url": spec.get("db_url"),
        }
        if types:
            second["types"] = types
        return [spec, second]
    if spec.get("command") == "ods_to_uprn" and (types is not None):
        return [
            spec,
            {
                "command": "download_assets",
                "uprn_from_previous_csvs": True,
                "types": types,
                "download_dir": spec.get("download_dir"),
                "api_key_env": spec.get("api_key_env"),
                "db_url": spec.get("db_url"),
            },
        ]
    return [spec]


def run_query_assist_step(
    step: StepSpec, py_exe: str, qa_path: str, dry_run: bool
) -> tuple[int, str]:
    argv = _build_argv(step, py_exe, qa_path)
    printable = " ".join(shlex.quote(x) for x in argv)
    logging.info("Command: %s", printable)
    if dry_run:
        return 0, f"[dry-run] {printable}\n"

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
    return rc, "".join(captured_lines)


def materialize_previous_uprn_csvs(state: WFState) -> list[str]:
    from_logs = state.artifacts.get("csvs", [])
    if from_logs:
        return from_logs
    dl_base = state.plan[0].get("download_dir") or os.path.join(
        os.getcwd(), "downloads"
    )
    candidate = os.path.join(dl_base, "ods_to_uprn.csv")
    if os.path.isfile(candidate):
        return [candidate]
    return []


def node_plan(state: WFState) -> WFState:
    defaults = {
        "download_dir": None,
        "api_key_env": "API_KEY",
        "db_url": None,
    }
    # 1) LLM multi-step plan
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
    # 2) Heuristic plan
    if not plan:
        plan = heuristic_plan(state.nl, defaults)
    # 3) LLM single-step → upgrade
    if not plan:
        spec = llm_route_to_spec(
            state.nl,
            base_url=state.base_url,
            model_id=state.model_id,
            temperature=state.temperature,
            top_p=state.top_p,
            num_predict=state.num_predict,
            num_ctx=state.num_ctx,
            keep_alive=state.keep_alive,
            force_json=state.force_json,
        )
        if spec:
            plan = upgrade_single_spec_to_plan(state.nl, spec, defaults)

    # Validate required args; if invalid (e.g., output_area missing), try fallback routes
    def _valid(p: list[StepSpec]) -> bool:
        for st in p:
            cmd = st.get("command")
            if cmd == "uprns_by_output_area" and not _ensure_list_or_path(
                st.get("output_area")
            ):
                return False
            if cmd == "ods_to_uprn" and not _ensure_list_or_path(st.get("ods")):
                return False
            if cmd == "download_assets":
                if st.get("uprn_from_previous_csvs"):
                    continue
                if not _ensure_list_or_path(st.get("uprn")):
                    return False
        return True

    if plan and not _valid(plan):
        plan = heuristic_plan(state.nl, defaults)
    if plan and not _valid(plan):
        spec = llm_route_to_spec(
            state.nl,
            state.base_url,
            state.model_id,
            temperature=state.temperature,
            top_p=state.top_p,
            num_predict=state.num_predict,
            num_ctx=state.num_ctx,
            keep_alive=state.keep_alive,
            force_json=state.force_json,
        )
        if spec:
            plan = upgrade_single_spec_to_plan(state.nl, spec, defaults)

    state.plan = plan or []

    if state.plan and state.verbose_level <= logging.INFO:
        print("Plan:")
        for i, step in enumerate(state.plan):
            show = {k: v for k, v in step.items() if k != "uprn_from_previous_csvs"}
            print(f"  {i+1}. {json.dumps(show, ensure_ascii=False)}")
        print()
    if not state.plan:
        state.log.append("No actionable plan could be inferred.")
    return state


def node_execute(state: WFState) -> WFState:
    if state.current >= len(state.plan):
        return state
    step = state.plan[state.current]

    if step.get("uprn_from_previous_csvs"):
        csvs = materialize_previous_uprn_csvs(state)
        if not csvs:
            state.log.append("No CSVs found from previous step(s).")
            state.current = len(state.plan)
            return state
        step = dict(step)
        step.pop("uprn_from_previous_csvs", None)
        step["uprn"] = csvs

    rc, captured = run_query_assist_step(
        step, state.py_exe, state.qa_path, state.dry_run
    )
    state.log.append(captured)

    newly_found = _find_csvs_emitted(captured)
    if newly_found:
        extant = state.artifacts.get("csvs", [])
        state.artifacts["csvs"] = list(dict.fromkeys(extant + newly_found))

    try:
        argv_for_record = _build_argv(step, state.py_exe, state.qa_path)
    except Exception:
        argv_for_record = []
    state.actions.append(
        {
            "index": state.current + 1,
            "command": step.get("command"),
            "argv": argv_for_record,
            "rc": rc,
            "emitted_csvs": newly_found,
        }
    )

    if rc != 0:
        state.log.append(f"Step {state.current} returned non-zero exit {rc}.")
        state.current = len(state.plan)
    else:
        state.current += 1
    return state


def after_plan(state: WFState) -> str:
    if not state.plan:
        return END
    if state.plan_only:
        return END
    return "execute"


def check_done(state: WFState) -> str:
    if state.current >= len(state.plan):
        return END
    if state.current >= state.max_steps:
        state.log.append(f"Aborting: exceeded max_steps={state.max_steps}")
        return END
    return "execute"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="LangGraph NL workflow for query_assist.py (one- and two-stage)"
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
        help="Plan & print commands but do not execute",
    )
    ap.add_argument(
        "--plan-only", action="store_true", help="Only compile/print the plan and exit"
    )
    ap.add_argument("--once", "-q", help="Run a single NL query and exit")

    # Decoding/runtime knobs
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--num-ctx", type=int, default=None)
    ap.add_argument("--keep-alive", default=None)
    ap.add_argument("--no-force-json", action="store_true")
    ap.add_argument("--max-steps", type=int, default=8)

    # Logging controls
    ap.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v=info, -vv=debug"
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Logging level
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    if level <= logging.INFO:
        body = (
            "• Parses natural language into a multi-stage plan to execute SPARQL and retrieve assets.\n"
            "• Executes via LangGraph with artifact passing.\n"
            "• Optional filters: sensor, types; optional overrides: download_dir, api_key_env, db_url \n"
            "• Supports dry-run and plan-only modes."
        )
        print(_render_box(f"Query Assist AI — {args.model_id}", body))

    # Build LangGraph
    builder = StateGraph(WFState)
    builder.add_node("plan", node_plan)
    builder.add_node("execute", node_execute)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", after_plan, {"execute": "execute", END: END})
    builder.add_conditional_edges(
        "execute", check_done, {"execute": "execute", END: END}
    )
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    def run_once(nl: str) -> int:
        st = WFState(
            nl=nl,
            plan=[],
            dry_run=bool(args.dry_run),
            plan_only=bool(args.plan_only),
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
        final_state = graph.invoke(
            st, config={"configurable": {"thread_id": f"tid-{time.time_ns()}"}}
        )
        logs = (
            final_state.get("log")
            if isinstance(final_state, dict)
            else getattr(final_state, "log", [])
        ) or []
        trailing = [
            l
            for l in logs
            if any(
                t in l
                for t in (
                    "[dry-run]",
                    "No CSVs found",
                    "non-zero exit",
                    "Planner",
                    "No actionable plan",
                )
            )
        ]
        if trailing:
            print("\n" + "\n".join(l.strip() for l in trailing))
        actions = (
            final_state.get("actions")
            if isinstance(final_state, dict)
            else getattr(final_state, "actions", [])
        ) or []
        if actions and level <= logging.INFO:
            print("\nACTIONS (LangGraph Execution):")
            for a in actions:
                argv = " ".join(shlex.quote(x) for x in a.get("argv", []))
                rc = a.get("rc")
                em = ", ".join(a.get("emitted_csvs", []) or [])
                print(
                    f"  {a.get('index')}. {a.get('command')}  [rc={rc}]\n      argv: {argv}"
                )
                if em:
                    print(f"     emitted CSVs: {em}")
        return 0 if not any("non-zero exit" in l for l in logs) else 1

    try:
        if args.once:
            rc = run_once(args.once)
            if rc != 0:
                logging.warning("Workflow exited with code %d", rc)
            return
        if level <= logging.INFO:
            print(
                "LangGraph NL workflow for query_assist.py. Type 'exit' or Ctrl-D to quit."
            )
        while True:
            try:
                nl = input("> ")
            except EOFError:
                break
            if not nl.strip():
                continue
            if nl.strip().lower() in {"exit", "quit"}:
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
