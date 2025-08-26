#!/usr/bin/env python3
"""
Natural-language CLI router for query_assist.py using an Ollama-served model.

- Sends a compact routing prompt (system + few shots + user) to Ollama's /api/chat.
- Expects the model to emit ONLY a JSON object describing the desired command.
- Builds an argv list and shells out to query_assist.py accordingly.

Dependencies:
  pip install -U requests

Assumptions:
  - An Ollama server is reachable (e.g., in a container) at:
      base URL from $OLLAMA_HOST or http://localhost:11434
  - The desired model (e.g., 'llama3.1:8b-instruct') is pulled and available in Ollama.

Notes:
  - We set `format: "json"` in the chat request to bias the model toward a clean JSON response.
  - We still robustly slice the first JSON object from the returned text to tolerate minor deviations.

Environment:
  - OLLAMA_HOST (optional): e.g., http://localhost:11434 or http://ollama:11434
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Tuple, Union

import requests

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

FEW_SHOTS: List[Tuple[str, str]] = [
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
]


def extract_assistant_text_from_ollama(resp: Dict[str, Any]) -> str:
    """
    Extract the assistant's text from Ollama /api/chat or /api/generate response.
    """
    # Preferred: /api/chat schema
    msg = resp.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content

    # Fallback: /api/generate schema
    if isinstance(resp.get("response"), str) and resp["response"].strip():
        return resp["response"]

    # Last resort: stringify
    return json.dumps(resp, ensure_ascii=False)


def slice_first_json_object(text: str) -> str:
    """
    Robustly extract the first top-level JSON object {...} from text.
    Raises ValueError if none is found.
    """
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


def ensure_list_or_path(v: Union[None, str, List[str]]) -> List[str]:
    """
    Convert JSON field (None | string | list) into a list of CLI tokens.
    """
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    return [s]


def build_argv(spec: Dict[str, Any], py: str, qa_path: str) -> List[str]:
    """
    Map the JSON spec to query_assist.py argv.
    """
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


def print_router_help():
    """Print brief guidance for the NL router when no command was inferred."""
    print(
        "[router] Expected one of the commands: download_assets | ods_to_uprn | uprns_by_output_area"
    )
    print("[router] Examples:")
    print(
        "  'Download the merged lidar point cloud for UPRN 5045394' → download_assets"
    )
    print("  'Map ODS G85013 to UPRNs' → ods_to_uprn")
    print(
        "  'List all UPRNs in output areas E00004550 and E00032882' → uprns_by_output_area"
    )
    print(
        "[router] You can also be concise, e.g.: '5045394 merged lidar', 'ODS G85013', 'output areas E00004550 E00032882'."
    )


# --- Heuristic parsing ---
def heuristic_parse(nl: str) -> Union[Dict[str, Any], None]:
    """Attempt to derive a spec dict directly from natural language without model.

    Patterns handled:
      - Output areas: presence of 'output area' or codes like E00012345
      - ODS codes: tokens like a letter followed by 5 digits (e.g., G85013) with 'ODS' keyword
      - UPRN asset downloads: numeric tokens length >=6 plus words like 'download', 'get', 'asset', 'lidar', 'image'
    """
    text = nl.strip()
    if not text:
        return None
    lowered = text.lower()

    output_area_codes = re.findall(r"\bE\d{8}\b", text)
    ods_codes = re.findall(r"\b[A-Z]\d{5}\b", text)
    uprn_candidates = [t for t in re.findall(r"\b\d{6,}\b", text)]

    # Output area heuristic
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

    # ODS heuristic
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

    # Asset download heuristic
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
    }
    if uprn_candidates and any(k in lowered for k in asset_keywords):
        return {
            "command": "download_assets",
            "uprn": uprn_candidates,
            "sensor": None,
            "types": None,
            "download_dir": None,
            "api_key_env": None,
            "db_url": None,
            "ods": None,
            "output_area": None,
        }

    return None


def ollama_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    top_p: float = 0.95,
    num_predict: int = 256,
    num_ctx: Union[int, None] = None,
    keep_alive: Union[str, None] = None,
    request_timeout_s: float = 120.0,
    force_json: bool = True,
) -> Dict[str, Any]:
    """
    Call Ollama's /api/chat with given messages and decoding options.
    Returns the parsed JSON response.
    """
    url = base_url.rstrip("/") + "/api/chat"
    payload: Dict[str, Any] = {
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
        # Instruct Ollama to format the assistant message as strict JSON
        payload["format"] = "json"

    resp = requests.post(url, json=payload, timeout=(5.0, request_timeout_s))
    resp.raise_for_status()
    return resp.json()


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
    num_ctx: Union[int, None],
    keep_alive: Union[str, None],
    force_json: bool,
    debug_model: bool,
) -> int:
    """
    One-shot turn: model → JSON → construct argv → (dry) run query_assist.py.
    """
    # Heuristic fast-path: try to parse without model when patterns are obvious.
    heuristic_spec = heuristic_parse(nl)
    if heuristic_spec:
        print("[router] Heuristic matched; bypassing model.")
        if debug_model:
            print(
                f"[debug] Heuristic spec derived from input: {json.dumps(heuristic_spec)}"
            )
        spec = heuristic_spec
        argv = build_argv(spec, py_exe, qa_path)
        print("\n[router] JSON spec (heuristic):")
        print(json.dumps(spec, indent=2))
        print("\n[router] Command:")
        print(" ".join([shlex.quote(x) for x in argv]))
        if dry_run:
            return 0
        proc = subprocess.run(argv)
        return proc.returncode

    messages = [{"role": "system", "content": SYSTEM_ROUTER_PROMPT}]
    for u, a in FEW_SHOTS:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": nl})

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
        print("[debug] Primary model raw JSON response:")
        try:
            print(json.dumps(resp, indent=2, ensure_ascii=False))
        except Exception:
            print(str(resp))

    content = extract_assistant_text_from_ollama(resp)
    if debug_model:
        print("[debug] Extracted content (primary):")
        print(repr(content))

    # Try strict JSON first; if that fails, slice the first JSON object.
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
    if spec is None:
        # Retry once without forcing JSON if we had forced it and got nothing
        if force_json:
            print(
                "[router] Empty/invalid JSON content; retrying without format=json..."
            )
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
            content2 = extract_assistant_text_from_ollama(resp2)
            if debug_model:
                print("[debug] Secondary model raw JSON response (no format=json):")
                try:
                    print(json.dumps(resp2, indent=2, ensure_ascii=False))
                except Exception:
                    print(str(resp2))
                print("[debug] Extracted content (secondary):")
                print(repr(content2))
            try:
                spec = json.loads(content2)
            except Exception:
                try:
                    blob = slice_first_json_object(content2)
                    spec = json.loads(blob)
                except Exception:
                    spec = None
        # Last resort: apply heuristic after model failure
        if spec is None:
            heuristic_spec = heuristic_parse(nl)
            if heuristic_spec:
                print("[router] Falling back to heuristic after model failure.")
                spec = heuristic_spec

    # Fallback: if no command or unsupported command, emit help and return gracefully.
    if not spec or spec.get("command") not in {
        "download_assets",
        "ods_to_uprn",
        "uprns_by_output_area",
    }:
        print("[router] No actionable command inferred from model output.")
        print("[router] Raw model content:")
        print(content.strip())
        print_router_help()
        return 0

    argv = build_argv(spec, py_exe, qa_path)

    print("\n[router] JSON spec:")
    print(json.dumps(spec, indent=2))
    print("\n[router] Command:")
    print(" ".join([shlex.quote(x) for x in argv]))
    if dry_run:
        return 0

    # Inherit env (so API_KEY etc. is available)
    proc = subprocess.run(argv)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(
        description="NL interface for query_assist.py using Ollama"
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
        help="Only print the derived command; do not execute",
    )
    ap.add_argument(
        "--once", "-q", help="Run a single NL query and exit (non-interactive)"
    )

    # Decoding / runtime knobs (kept simple and model-agnostic)
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
        help="Do not set format='json' in the chat call (not recommended)",
    )
    ap.add_argument(
        "--debug-model",
        action="store_true",
        help="Print raw model JSON responses and extracted text",
    )

    args = ap.parse_args()

    print(f"[init] Ollama base URL: {args.base_url}")
    print(
        f"[init] model={args.model_id} temperature={args.temperature} top_p={args.top_p} "
        f"num_predict={args.num_predict} num_ctx={args.num_ctx} keep_alive={args.keep_alive} "
        f"force_json={not args.no_force_json}"
    )

    py_exe = sys.executable
    qa_path = args.query_assist_path

    if args.once:
        try:
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
            )
            sys.exit(rc)
        except Exception as e:
            print(f"[router] Error: {e}", file=sys.stderr)
            sys.exit(1)

    print("NL router for query_assist.py (Ollama). Type 'exit' or Ctrl-D to quit.")
    while True:
        try:
            nl = input("\n> ").strip()
        except EOFError:
            break
        if not nl:
            continue
        if nl.lower() in {"exit", "quit"}:
            break
        try:
            run_once(
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
            )
        except Exception as e:
            print(f"[router] Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
