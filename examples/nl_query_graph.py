#!/usr/bin/env python3
"""
nl_query_graph.py — LLM-only planner, plan-first, two-stage ODS

A lean rewrite of the previous LangGraph CLI that:
- Prints the plan FIRST (and ONLY when running at INFO, i.e., with -v)
- Uses an LLM-only planner (no heuristics) with a stronger prompt
- Guarantees a two-stage plan for ODS/Output Area when the request implies assets
- Keeps dry-run and plan-only modes
- Preserves artifact detection (CSV) and feeds them into download steps
- Adds robust retries and graceful fallback when Ollama returns empty/invalid content
- Understands both Ollama-native and OpenAI-style response shapes (choices[0].message.content)

Requirements:
  pip install langgraph requests

Notes:
- We keep LangGraph for structure/checkpointing, but we do not auto-run it before
  printing the plan. We explicitly call plan -> print -> execute.
- The LLM prompt is opinionated to produce two steps when the user asks for
  assets, even if types are not specified (types omitted === all types).
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
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

# Third-party
import requests
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# ============================================================================
# Strong LLM planning prompt (LLM-only)
# ============================================================================
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


# ============================================================================
# Types & State
# ============================================================================
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


@dataclass
class WFState:
    nl: str
    plan: list[StepSpec] = field(default_factory=list)
    current: int = 0
    artifacts: dict[str, Any] = field(default_factory=dict)  # e.g., {"csvs": [...]}
    log: list[str] = field(default_factory=list)
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


# ============================================================================
# LLM plumbing (robust to sparse/odd Ollama responses)
# ============================================================================


def _post_ollama(
    base_url: str, payload: dict[str, Any], timeout_s: float = 120.0
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/chat"
    r = requests.post(url, json=payload, timeout=(5.0, timeout_s))
    r.raise_for_status()
    return r.json()


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


def _extract_content_variants(resp: dict[str, Any]) -> str:
    # Try common shapes returned by Ollama and proxies
    if isinstance(resp.get("message"), dict):
        c = resp["message"].get("content")
        if c:
            return c
    # OpenAI/vLLM/LM Studio style
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] or {}
        msg = first.get("message") or {}
        c = msg.get("content") or first.get("text")
        if isinstance(c, str) and c:
            return c
    # Some servers use top-level 'response'
    c = resp.get("response")
    if isinstance(c, str) and c:
        return c
    # Rare: top-level 'content'
    c = resp.get("content")
    if isinstance(c, str) and c:
        return c
    # Some wrappers return {'messages': [{'content': ...}]}
    msgs = resp.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict):
        c = msgs[-1].get("content")
        if isinstance(c, str) and c:
            return c
    return ""


def ollama_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    top_p: float = 0.95,
    num_predict: int = 256,
    num_ctx: int | None = None,
    keep_alive: str | None = None,
    force_json: bool = True,
    timeout_s: float = 120.0,
    retries: int = 2,
    logger: logging.Logger | None = None,
) -> str:
    """Return assistant content as a string. Retries with/without JSON forcing."""
    last_err: Exception | None = None
    for attempt in range(retries + 1):
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
        use_json = force_json if attempt == 0 else False
        if num_ctx is not None:
            payload["options"]["num_ctx"] = int(num_ctx)
        if keep_alive:
            payload["keep_alive"] = str(keep_alive)
        if use_json:
            payload["format"] = "json"
        try:
            resp = _post_ollama(base_url, payload, timeout_s=timeout_s)
            content = _extract_content_variants(resp)
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Ollama raw keys: %s", list(resp.keys()))
                logger.debug("Ollama content preview: %r", content[:200])
            if content:
                return content
            last_err = RuntimeError("empty content from Ollama")
        except Exception as e:
            last_err = e
        # brief backoff
        time.sleep(0.2 * (attempt + 1))
    if last_err:
        raise last_err
    raise RuntimeError("Ollama returned no content.")


def llm_plan(state: WFState) -> list[StepSpec]:
    messages = [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": state.nl},
    ]
    # Try calling the planner; on failure, DO NOT abort. Fall through to synthesis.
    try:
        content = ollama_chat(
            base_url=state.base_url,
            model=state.model_id,
            messages=messages,
            temperature=state.temperature,
            top_p=state.top_p,
            num_predict=state.num_predict,
            num_ctx=state.num_ctx,
            keep_alive=state.keep_alive,
            force_json=state.force_json,
            retries=2,
            logger=logging.getLogger(__name__),
        )
    except Exception as e:
        logging.getLogger(__name__).warning("Planner LLM call failed: %s", e)
        content = ""

    obj: dict | None
    try:
        obj = json.loads(content)
    except Exception:
        obj = _extract_first_json(content)
    if not obj or not isinstance(obj, dict):
        # LAST-DITCH guardrail: if the LLM truly failed to produce JSON but the NL
        # clearly requests assets with ODS/OA, synthesize a minimal two-step plan
        # rather than crashing. This is NOT a heuristic router; it's a fail-safe
        # to keep execution usable when the model returns empty text.
        lowered = state.nl.lower()
        ods = re.findall(r"\b[A-Z]\d{5}\b", state.nl)
        oa = re.findall(r"\bE\d{8}\b", state.nl)
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
        if implies_assets and (ods or oa):
            if ods:
                return [
                    {"command": "ods_to_uprn", "ods": ods},
                    {"command": "download_assets", "uprn_from_previous_csvs": True},
                ]
            if oa:
                return [
                    {"command": "uprns_by_output_area", "output_area": oa},
                    {"command": "download_assets", "uprn_from_previous_csvs": True},
                ]
        # If we get here, there is nothing actionable — report the original issue.
        raise RuntimeError("LLM did not return a valid JSON object.")

    steps_raw = obj.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise RuntimeError("LLM returned an empty steps list.")

    steps: list[StepSpec] = []
    for s in steps_raw:
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

    # Ensure two-stage plan if assets implied
    lowered = state.nl.lower()
    implies_assets = any(
        w in lowered
        for w in ["asset", "assets", "download", "point cloud", "rgb", "image", "lidar"]
    )
    has_mapping = any(
        st.get("command") in {"ods_to_uprn", "uprns_by_output_area"} for st in steps
    )
    has_download = any(st.get("command") == "download_assets" for st in steps)
    if implies_assets and has_mapping and not has_download:
        steps.append(
            {
                "command": "download_assets",
                "uprn_from_previous_csvs": True,
            }
        )

    return steps


# ============================================================================
# Execution helpers
# ============================================================================


def _build_argv(spec: StepSpec, py: str, qa_path: str) -> list[str]:
    cmd = [sys.executable if not py else py, qa_path]
    command = spec.get("command")
    if command == "download_assets":
        uprn = spec.get("uprn")
        if isinstance(uprn, list):
            uprn_list = [str(x) for x in uprn]
        elif isinstance(uprn, str):
            uprn_list = [uprn]
        else:
            raise ValueError(
                "download_assets requires 'uprn' unless 'uprn_from_previous_csvs' is used earlier."
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
    """Execute a single step via subprocess, stream logs, and return (rc, captured_text)."""
    argv = _build_argv(step, py_exe, qa_path)
    printable = " ".join(shlex.quote(x) for x in argv)
    logging.info("Command: %s", printable)
    if dry_run:
        return 0, f"[dry-run] {printable}\n"

    p = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
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
    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for pth in csvs:
        if pth not in seen:
            out.append(pth)
            seen.add(pth)
    return out


def materialize_previous_uprn_csvs(state: WFState) -> list[str]:
    """Return list of CSV paths produced by earlier steps."""
    from_logs = state.artifacts.get("csvs", [])
    if from_logs:
        return from_logs
    # Fallback: common default path from ODS mapping
    dl_base = os.path.join(os.getcwd(), "downloads")
    candidate = os.path.join(dl_base, "ods_to_uprn.csv")
    if os.path.isfile(candidate):
        return [candidate]
    return []


# ============================================================================
# LangGraph nodes (used only for the execution loop, not for pre-plan)
# ============================================================================


def node_execute(state: WFState) -> WFState:
    if state.current >= len(state.plan):
        return state
    step = state.plan[state.current]

    # Inject UPRNs from previous CSVs if requested
    if step.get("uprn_from_previous_csvs"):
        csvs = materialize_previous_uprn_csvs(state)
        if not csvs:
            state.log.append("No CSVs found from previous step(s).")
            state.current = len(state.plan)
            return state
        step = dict(step)
        step.pop("uprn_from_previous_csvs", None)
        step["uprn"] = csvs
        state.plan[state.current] = step  # persist the resolved step

    rc, captured = run_query_assist_step(
        step, state.py_exe, state.qa_path, state.dry_run
    )
    state.log.append(captured)

    newly_found = _find_csvs_emitted(captured)
    if newly_found:
        existing = state.artifacts.get("csvs", [])
        state.artifacts["csvs"] = list(dict.fromkeys(existing + newly_found))

    if rc != 0:
        state.log.append(f"Step {state.current} returned non-zero exit {rc}.")
        state.current = len(state.plan)
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


# ============================================================================
# UI helpers
# ============================================================================


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
    # Print ONLY when we are at INFO (-v). Not at WARNING/DEBUG.
    if level == logging.INFO:
        print("Plan:")
        for i, step in enumerate(plan, 1):
            step_disp = {
                k: v for k, v in step.items() if k != "uprn_from_previous_csvs"
            }
            print(f"  {i}. {json.dumps(step_disp, ensure_ascii=False)}")
        print()


# ============================================================================
# CLI
# ============================================================================


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

    # Logging level
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Intro banner (print at INFO or DEBUG)
    if level <= logging.INFO:
        body = (
            "- LLM-only planner (no heuristics).\n"
            "- Plan is printed FIRST and only at INFO (-v).\n"
            "- Two-stage ODS/OA -> assets enforced when assets are implied.\n"
            "- Dry-run and plan-only for auditability."
        )
        print(_render_box(f"LangGraph NL Workflow — {args.model_id}", body))

    # Build the LangGraph for execution loop only
    builder = StateGraph(WFState)
    builder.add_node("execute", node_execute)
    builder.add_edge(START, "execute")
    builder.add_conditional_edges(
        "execute", node_check_done, {"execute": "execute", END: END}
    )
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    def run_once(nl: str) -> int:
        st = WFState(
            nl=nl,
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

        # === PLAN (LLM-only) ===
        try:
            st.plan = llm_plan(st)
        except Exception as e:
            logging.warning("Planning failed: %s", e)
            return 1

        # Print plan FIRST, ONLY at INFO
        _print_plan(st.plan, level)
        if args.plan_only:
            return 0

        # === EXECUTE ===
        while st.current < len(st.plan) and st.current < st.max_steps:
            # Run the node explicitly so we fully control when execution starts
            st = graph.invoke(
                st, config={"configurable": {"thread_id": f"tid-{time.time_ns()}"}}
            )
            if node_check_done(st) == END:
                break

        # Emit trailing notes if any
        trailing = [
            ln
            for ln in st.log
            if any(k in ln for k in ("[dry-run]", "No CSVs found", "non-zero exit"))
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
