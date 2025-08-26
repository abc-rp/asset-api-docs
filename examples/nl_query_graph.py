#!/usr/bin/env python3
"""
nl_query_graph.py — LLM-only planner, plan-first, OA/ODS→assets, supports direct UPRN

Fixes:
- Correct LangGraph conditional mapping: return label strings ("execute"/"end")
  and map {"execute": "execute", "end": END}. This eliminates the infinite loop.
- Remove the external invoke loop; let LangGraph run to END in a single invoke.
- Allow single-step plans when UPRNs are provided directly in the NL query.
- Keep CSV auto-handoff and robust Ollama fallback (/api/chat then /api/generate).

Requirements: pip install langgraph requests
"""

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
import time
from typing import Any, Literal, TypedDict

import requests
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# ─────────────────────────────────────────────────────────────────────────────
# Planner prompt
# ─────────────────────────────────────────────────────────────────────────────
PLAN_SYSTEM_PROMPT = r"""
You are a planning assistant that converts a natural language request about
retrieving built environment asset data into an ordered execution plan for
query_assist.py.

Available CLI commands:
  1. uprns_by_output_area  – given output area code(s) yield UPRN CSV(s)
  2. ods_to_uprn           – given ODS clinical practice code(s) yield UPRN CSV
  3. download_assets       – given UPRN(s) (list or CSV path) download assets

Asset type IRIs (use when mentioned or implied; omit "types" to mean ALL types):
  did:rgb-image, did:lidar-pointcloud-merged, did:lidar-pointcloud-frame,
  did:lidar-range-pano, did:lidar-reflectance-pano, did:lidar-signal-pano,
  did:lidar-nearir-pano, did:ir-false-color-image, did:ir-temperature-array,
  did:ir-count-image, did:celsius-temperature, did:relative-humidity

Synonyms / interpretation guidance (LLM, be decisive):
  "building"/"property" -> UPRN(s)
  "practice"/"gp practice" -> ODS code
  "point clouds" (plural) -> include merged and frame unless otherwise specified
  "thermal image" -> did:ir-false-color-image unless arrays are explicitly requested

CRITICAL RULES:
  • If the user asks to get/download assets and provides ODS or Output Area,
    ALWAYS output a TWO-STEP plan: the mapping step first, then a download_assets
    step that consumes the CSVs from the previous step. Use
    {"uprn_from_previous_csvs": true} on that download step.
  • If the user says "all assets" or does not specify types, omit the "types"
    field entirely on download_assets to indicate ALL types.
  • Prefer being decisive and avoid asking questions.

JSON output schema (emit ONLY one JSON object):
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

Do not include any prose. Output the JSON object only.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────
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


class WFState(TypedDict, total=False):
    nl: str
    plan: list[StepSpec]
    current: int
    artifacts: dict[str, Any]
    log: list[str]
    dry_run: bool
    py_exe: str
    qa_path: str
    base_url: str
    model_id: str
    temperature: float
    top_p: float
    num_predict: int
    num_ctx: int | None
    keep_alive: str | None
    force_json: bool
    verbose_level: int
    max_steps: int


# ─────────────────────────────────────────────────────────────────────────────
# Ollama client (robust)
# ─────────────────────────────────────────────────────────────────────────────
def _ollama_chat(
    base_url: str, payload: dict[str, Any], timeout_s: float = 120.0
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/chat"
    r = requests.post(url, json=payload, timeout=(5.0, timeout_s))
    r.raise_for_status()
    return r.json()


def _ollama_generate(
    base_url: str, payload: dict[str, Any], timeout_s: float = 120.0
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/generate"
    r = requests.post(url, json=payload, timeout=(5.0, timeout_s))
    r.raise_for_status()
    return r.json()


def _extract_content_variants(resp: dict[str, Any]) -> str:
    if isinstance(resp.get("message"), dict):
        c = resp["message"].get("content")
        if c:
            return c
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] or {}
        msg = first.get("message") or {}
        c = msg.get("content") or first.get("text")
        if isinstance(c, str) and c:
            return c
    c = resp.get("response")
    if isinstance(c, str) and c:
        return c
    c = resp.get("content")
    if isinstance(c, str) and c:
        return c
    msgs = resp.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict):
        c = msgs[-1].get("content")
        if isinstance(c, str) and c:
            return c
    return ""


def _extract_first_json(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    s = text.find("{")
    if s == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[s:], start=s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[s : i + 1])
                except Exception:
                    return None
    return None


def ollama_plan(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    num_predict: int,
    num_ctx: int | None,
    keep_alive: str | None,
    force_json: bool,
    timeout_s: float = 120.0,
    retries: int = 2,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Try /api/chat with JSON then without
    for attempt in range(retries + 1):
        payload = {
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
        if attempt == 0 and force_json:
            payload["format"] = "json"
        try:
            resp = _ollama_chat(base_url, payload, timeout_s=timeout_s)
            c = _extract_content_variants(resp)
            if c:
                return c
        except Exception:
            pass
        time.sleep(0.15 * (attempt + 1))
    # Fallback /api/generate
    try:
        prompt = (
            f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\nUser:\n{user_prompt}\n\nAssistant:"
        )
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "num_predict": int(num_predict),
            },
        }
        if force_json:
            payload["format"] = "json"
        resp = _ollama_generate(base_url, payload, timeout_s=timeout_s)
        c = _extract_content_variants(resp)
        if c:
            return c
    except Exception:
        pass
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Planning
# ─────────────────────────────────────────────────────────────────────────────
def llm_plan(state: WFState) -> list[StepSpec]:
    content = ollama_plan(
        base_url=state["base_url"],
        model=state["model_id"],
        system_prompt=PLAN_SYSTEM_PROMPT,
        user_prompt=state["nl"],
        temperature=state.get("temperature", 0.0),
        top_p=state.get("top_p", 0.95),
        num_predict=state.get("num_predict", 256),
        num_ctx=state.get("num_ctx"),
        keep_alive=state.get("keep_alive"),
        force_json=state.get("force_json", True),
    )

    obj: dict | None = None
    if content:
        try:
            obj = json.loads(content)
        except Exception:
            obj = _extract_first_json(content)

    # If LLM output is unusable, synthesize from the NL string
    if not obj or not isinstance(obj, dict) or not isinstance(obj.get("steps"), list):
        nl = state["nl"]
        lowered = nl.lower()
        implies_assets = any(
            w in lowered
            for w in (
                "asset",
                "assets",
                "download",
                "point cloud",
                "rgb",
                "image",
                "lidar",
            )
        )
        # Detect UPRNs (10–14 digit sequences are typical); be conservative
        uprns = re.findall(r"\b\d{10,14}\b", nl)
        ods = re.findall(r"\b[A-Z]\d{5}\b", nl)
        oa = re.findall(r"\bE\d{8}\b", nl)
        if implies_assets and uprns:
            return [{"command": "download_assets", "uprn": uprns}]
        if implies_assets and ods:
            return [
                {"command": "ods_to_uprn", "ods": ods},
                {"command": "download_assets", "uprn_from_previous_csvs": True},
            ]
        if implies_assets and oa:
            return [
                {"command": "uprns_by_output_area", "output_area": oa},
                {"command": "download_assets", "uprn_from_previous_csvs": True},
            ]
        raise RuntimeError(
            "Planner produced no actionable steps; provide UPRN(s), ODS or Output Area."
        )

    # Normalize steps from LLM JSON
    steps: list[StepSpec] = []
    for s in obj["steps"]:
        if not isinstance(s, dict):
            continue
        cmd = s.get("command")
        if cmd not in {"download_assets", "ods_to_uprn", "uprns_by_output_area"}:
            continue
        step: StepSpec = {"command": cmd}  # type: ignore[assignment]
        for key in [
            "uprn",
            "ods",
            "output_area",
            "types",
            "sensor",
            "download_dir",
            "api_key_env",
            "db_url",
            "uprn_from_previous_csvs",
        ]:
            if key in s:
                step[key] = s[key]  # type: ignore[index]
        steps.append(step)

    # Enforce download step only when a mapping step exists (don’t force for direct UPRN)
    lowered = state["nl"].lower()
    implies_assets = any(
        w in lowered
        for w in ("asset", "assets", "download", "point cloud", "rgb", "image", "lidar")
    )
    has_mapping = any(
        st.get("command") in {"ods_to_uprn", "uprns_by_output_area"} for st in steps
    )
    has_download = any(st.get("command") == "download_assets" for st in steps)
    if implies_assets and has_mapping and not has_download:
        steps.append({"command": "download_assets", "uprn_from_previous_csvs": True})

    # If a download step exists but lacks UPRNs, consume prior CSVs
    for st in steps:
        if st.get("command") == "download_assets" and not st.get("uprn"):
            st["uprn_from_previous_csvs"] = True

    return steps


# ─────────────────────────────────────────────────────────────────────────────
# Execution helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_argv(spec: StepSpec, py: str, qa_path: str) -> list[str]:
    cmd = [py or sys.executable, qa_path]
    command = spec.get("command")

    if command == "download_assets":
        uprn = spec.get("uprn")
        if isinstance(uprn, list):
            uprn_list = [str(x) for x in uprn]
        elif isinstance(uprn, str):
            uprn_list = [uprn]
        else:
            raise ValueError(
                "download_assets requires 'uprn' unless 'uprn_from_previous_csvs' is set."
            )
        cmd += ["--uprn"] + uprn_list
        if spec.get("sensor"):
            cmd += ["--sensor", str(spec["sensor"])]
        if spec.get("types"):
            cmd += ["--types", ",".join(spec["types"])]
    elif command == "ods_to_uprn":
        ods = spec.get("ods")
        if isinstance(ods, list):
            ods_list = [str(x) for x in ods]
        elif isinstance(ods, str):
            ods_list = [ods]
        else:
            raise ValueError("ods_to_uprn requires 'ods'.")
        cmd += ["--ods"] + ods_list
    elif command == "uprns_by_output_area":
        oa = spec.get("output_area")
        if isinstance(oa, list):
            oa_list = [str(x) for x in oa]
        elif isinstance(oa, str):
            oa_list = [oa]
        else:
            raise ValueError("uprns_by_output_area requires 'output_area'.")
        cmd += ["--output-area"] + oa_list
    else:
        raise ValueError(f"Unsupported command: {command!r}")

    if spec.get("db_url"):
        cmd += ["--db-url", str(spec["db_url"])]
    if spec.get("download_dir"):
        cmd += ["--download-dir", str(spec["download_dir"])]
    if spec.get("api_key_env"):
        cmd += ["--api-key-env", str(spec["api_key_env"])]
    return cmd


def run_query_assist_step(
    step: StepSpec,
    py_exe: str,
    qa_path: str,
    dry_run: bool,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    argv = _build_argv(step, py_exe, qa_path)
    printable = " ".join(shlex.quote(x) for x in argv)
    logging.info("Executing step: %s", json.dumps(step, ensure_ascii=False))
    logging.info("Command: %s", printable)
    if dry_run:
        return 0, f"[dry-run] {printable}\n"

    try:
        p = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
    except FileNotFoundError as e:
        return 127, f"[spawn-failed] {e}\n"
    except Exception as e:
        return 1, f"[spawn-failed] {e}\n"

    captured: list[str] = []
    assert p.stdout is not None
    for line in p.stdout:
        sys.stdout.write(line)  # stream-through to console
        captured.append(line)
    rc = p.wait()
    return rc, "".join(captured)


def _find_csvs_emitted(text: str) -> list[str]:
    pats = [
        r"✔?\s*Saved\s+(?:ODS.?→?UPRN|ODS.?to.?UPRN)\s*CSV\s*[–\-→>]\s*([^\s]+\.csv)",
        r"✔?\s*Saved\s+(?:OA.?→?UPRN|OA.?to.?UPRN|Output\s*Area.?→?UPRN)\s*CSV\s*[–\-→>]\s*([^\s]+\.csv)",
        r"Saved\s*CSV\s*for\s*.*?[–\-→>]\s*([^\s]+\.csv)",
        r"✔\s*Saved\s*ODS.?→?UPRN\s*CSV\s*→\s*([^\s]+\.csv)",
    ]
    out, seen = [], set()
    for pat in pats:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            path = m.group(1)
            if path not in seen:
                out.append(path)
                seen.add(path)
    return out


def materialize_previous_uprn_csvs(state: WFState) -> list[str]:
    from_logs = state.get("artifacts", {}).get("csvs", [])
    if from_logs:
        return list(from_logs)
    candidates = [
        os.path.join(os.getcwd(), "downloads", "ods_to_uprn.csv"),
        os.path.join(os.getcwd(), "downloads", "oa_to_uprn.csv"),
        os.path.join(os.getcwd(), "downloads", "uprns_by_output_area.csv"),
    ]
    return [p for p in candidates if os.path.isfile(p)]


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph nodes
# ─────────────────────────────────────────────────────────────────────────────
def node_execute(state: WFState) -> WFState:
    state.setdefault("artifacts", {})
    state.setdefault("log", [])
    state.setdefault("current", 0)

    if state["current"] >= len(state.get("plan", [])):
        return state

    step: StepSpec = state["plan"][state["current"]]

    # Inject CSVs for download_assets when needed
    if step.get("command") == "download_assets" and not step.get("uprn"):
        csvs = materialize_previous_uprn_csvs(state)
        if not csvs and step.get("uprn_from_previous_csvs"):
            state["log"].append("No CSVs found from previous step(s).")
            state["current"] = len(state["plan"])
            return state
        if csvs:
            step = dict(step)
            step.pop("uprn_from_previous_csvs", None)
            step["uprn"] = csvs
            state["plan"][state["current"]] = step

    rc, captured = run_query_assist_step(
        step,
        state.get("py_exe", sys.executable),
        state["qa_path"],
        state.get("dry_run", False),
    )
    state["log"].append(captured)

    newly = _find_csvs_emitted(captured)
    if newly:
        existing = state["artifacts"].get("csvs", [])
        state["artifacts"]["csvs"] = list(dict.fromkeys(list(existing) + newly))

    if rc != 0:
        state["log"].append(f"Step {state['current']} returned non-zero exit {rc}.")
        state["current"] = len(state["plan"])
    else:
        state["current"] += 1
    return state


def node_check_done(state: WFState) -> str:
    # IMPORTANT: return string labels, not END sentinel
    if state.get("current", 0) >= len(state.get("plan", [])):
        return "end"
    if state.get("current", 0) >= state.get("max_steps", 8):
        state.setdefault("log", []).append(
            f"Aborting: exceeded max_steps={state.get('max_steps', 8)}"
        )
        return "end"
    return "execute"


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def _render_box(title: str, body: str) -> str:
    term_width = shutil.get_terminal_size(fallback=(100, 24)).columns
    max_width = max(60, min(term_width - 2, 100))
    wrap_width = max_width - 4
    body_lines: list[str] = []
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


def _print_plan(plan: list[StepSpec], level: int) -> None:
    if level == logging.INFO:
        print("Plan:")
        for i, step in enumerate(plan, 1):
            display = {k: v for k, v in step.items() if k != "uprn_from_previous_csvs"}
            print(f"  {i}. {json.dumps(display, ensure_ascii=False)}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="LangGraph NL workflow CLI for query_assist.py (LLM-only planner)"
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
    ap.add_argument("--plan-only", action="store_true", dest="plan_only")
    ap.set_defaults(plan_only=False)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Logging
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Banner
    if level <= logging.INFO:
        body = (
            "- LLM-only planner (no heuristics).\n"
            "- Plan is printed FIRST and only at INFO (-v).\n"
            "- Two-stage ODS/OA -> assets enforced when assets are implied.\n"
            "- Dry-run and plan-only for auditability."
        )
        print(_render_box(f"LangGraph NL Workflow — {args.model_id}", body))

    # Build LangGraph (dict state) — IMPORTANT: label mapping uses string keys
    builder = StateGraph(dict)
    builder.add_node("execute", node_execute)
    builder.add_edge(START, "execute")
    builder.add_conditional_edges(
        "execute", node_check_done, {"execute": "execute", "end": END}
    )
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    def run_once(nl: str) -> int:
        st: WFState = {
            "nl": nl,
            "plan": [],
            "current": 0,
            "artifacts": {},
            "log": [],
            "dry_run": bool(args.dry_run),
            "py_exe": sys.executable,
            "qa_path": args.query_assist_path,
            "base_url": args.base_url,
            "model_id": args.model_id,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
            "keep_alive": args.keep_alive,
            "force_json": (not args.no_force_json),
            "verbose_level": level,
            "max_steps": args.max_steps,
        }

        # PLAN
        try:
            st["plan"] = llm_plan(st)
        except Exception as e:
            logging.info("Planning failed: %s", e)
            return 1

        _print_plan(st["plan"], level)
        if args.plan_only:
            return 0

        # EXECUTE: single invoke to END (no external loop)
        final_state = graph.invoke(
            st, config={"configurable": {"thread_id": f"tid-{time.time_ns()}"}}
        )

        trailing = [
            ln
            for ln in final_state.get("log", [])
            if any(
                k in ln
                for k in (
                    "[dry-run]",
                    "No CSVs found",
                    "non-zero exit",
                    "[spawn-failed]",
                )
            )
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
                logging.info("Workflow exited with code %d", rc)
    except KeyboardInterrupt:
        print()
        logging.info("Interrupted.")
        sys.exit(130)
    except Exception as e:
        logging.error("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
