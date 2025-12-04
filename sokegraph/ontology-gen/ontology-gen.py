#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a domain ontology via a local Ollama model (Mistral by default).

Features:
- Accepts a domain string (e.g., "battery management systems").
- Starts 'ollama serve' automatically if the service is not reachable.
- Pulls the model automatically if it's not present.
- Builds a strong, format-preserving prompt using your example.json (full or excerpt).
- Calls /api/chat, validates JSON, retries once with an automatic "repair" prompt if needed.
- Writes output to a JSON file or stdout.

Usage examples:
  python ontology_gen.py --domain "battery management systems" --out bms_ontology.json
  python ontology_gen.py --domain "computer vision for retail" --model mistral --include-full-example
  python ontology_gen.py --domain "supply chain logistics" --print-prompt

Assumptions:
- Ollama is installed and on PATH.
- The default Ollama host is http://127.0.0.1:11434 (change with --host or OLLAMA_HOST).
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Tuple

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = "mistral"
DEFAULT_EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), "example.json")


def _http_json(method: str, url: str, payload: Dict[str, Any] = None, timeout: int = 20) -> Dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ollama_alive(host: str) -> bool:
    try:
        # Ping tag list as a quick readiness check
        _http_json("GET", f"{host}/api/tags", None, timeout=2)
        return True
    except Exception:
        return False


def _start_ollama_if_needed(host: str, autostart: bool = True, wait_seconds: int = 30) -> None:
    if _ollama_alive(host):
        return
    if not autostart:
        raise RuntimeError(f"Ollama not reachable at {host}. Start 'ollama serve' or pass --autostart.")

    # Try to start 'ollama serve' in the background.
    # Cross-platform best effort: rely on PATH having 'ollama'
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except FileNotFoundError:
        raise RuntimeError("Could not find 'ollama' on PATH. Please install Ollama first.")

    # Wait until the API is ready or timeout
    start = time.time()
    while time.time() - start < wait_seconds:
        if _ollama_alive(host):
            return
        time.sleep(0.5)
    raise RuntimeError("Timed out waiting for Ollama to become ready.")


def _ensure_model(model: str) -> None:
    # Check if model is already present via `ollama list`
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
        if model in out:
            return
    except Exception:
        # We will try to pull anyway
        pass

    # Pull model if missing
    try:
        print(f"Pulling model '{model}' (this can take a few minutes on first run)...", file=sys.stderr)
        subprocess.check_call(["ollama", "pull", model])
    except FileNotFoundError:
        raise RuntimeError("Could not find 'ollama' on PATH. Please install Ollama first.")


def _load_example(example_path: str, include_full: bool, per_category: int = 2) -> Tuple[str, Dict[str, Any]]:
    if not os.path.exists(example_path):
        raise FileNotFoundError(f"example.json not found at: {example_path}")

    with open(example_path, "r", encoding="utf-8") as f:
        example = json.load(f)

    if include_full:
        return json.dumps(example, ensure_ascii=False, indent=2), example

    # Build a small excerpt: up to `per_category` sub-items per top-level category
    excerpt: Dict[str, Any] = {}
    for top_key, sub in example.items():
        if isinstance(sub, dict):
            small = {}
            for i, (subk, subv) in enumerate(sub.items()):
                if i >= per_category:
                    break
                small[subk] = subv
            excerpt[top_key] = small
        else:
            excerpt[top_key] = sub

    return json.dumps(excerpt, ensure_ascii=False, indent=2), example


PROMPT_CORE = """You are an ontology expert in "{domain}". Your job is to design a compact, high-signal ontology for this domain.

Use the JSON *format and style* demonstrated by the reference example (keys → subkeys → list of synonyms/aliases). 
Do NOT copy the content—ADAPT it to the domain. If a section from the example does not make sense for the domain, omit it.
If the domain needs new sections, introduce them following the same nesting pattern.

Constraints:
- Output MUST be STRICTLY VALID JSON (no comments, no trailing commas, double quotes only).
- Structure: A JSON object with 5–10 top-level sections. Each section is a JSON object whose keys are subdomains/entities and values are lists of 2–6 aliases/synonyms/phrases (domain-relevant).
- Be precise and domain-specific; avoid generic filler.
- Prefer concise, industry-accurate aliases.
- If certain sections from the example are applicable (e.g., Environment, Process, Material, Performance & Stability, Application), rename/adapt them to domain-appropriate labels. Otherwise, skip them.
- Keep naming consistent (Title Case for section names; concise subkey names).

Deliver exactly and only the JSON object.
"""

REPAIR_PROMPT = """Your previous output was not valid JSON. 
Return the SAME ontology content, but as strictly valid JSON (no markdown, no prose)."""

def _build_prompt(domain: str, ref_json_str: str) -> str:
    return (
        PROMPT_CORE.format(domain=domain)
        + "\n\nReference JSON example (format guide only — do not copy content):\n"
        + ref_json_str
    )


def _ollama_chat(host: str, model: str, prompt_text: str, temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a careful assistant that only returns valid JSON when asked."},
            {"role": "user", "content": prompt_text},
        ],
        "options": {"temperature": temperature},
    }
    resp = _http_json("POST", f"{host}/api/chat", payload, timeout=90)
    # Newer Ollama returns message content here:
    try:
        return resp["message"]["content"]
    except Exception:
        # Fallback for older shapes if needed
        if "response" in resp:
            return resp["response"]
        raise RuntimeError("Unexpected Ollama response shape.")


def _try_parse_json(s: str) -> Tuple[bool, Any]:
    try:
        return True, json.loads(s)
    except Exception:
        # Sometimes models wrap JSON in code fences or extra text; try to extract the first {...} block
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return True, json.loads(s[start : end + 1])
            except Exception:
                return False, None
        return False, None


def main():
    ap = argparse.ArgumentParser(description="Generate an ontology JSON for a given domain using Ollama.")
    ap.add_argument("--domain", required=True, help='Domain, e.g. "battery management systems"')
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name (default: mistral)")
    ap.add_argument("--example", default=DEFAULT_EXAMPLE_PATH, help="Path to example.json (format guide)")
    ap.add_argument("--include-full-example", action="store_true", help="Include full example.json in the prompt")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Ollama host, e.g. http://127.0.0.1:11434")
    ap.add_argument("--no-autostart", action="store_true", help="Do not auto-start 'ollama serve'")
    ap.add_argument("--no-pull", action="store_true", help="Do not auto-pull model if missing")
    ap.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    ap.add_argument("--out", default="", help="Output JSON path (default: stdout)")
    ap.add_argument("--print-prompt", action="store_true", help="Print the composed prompt and exit")

    args = ap.parse_args()

    # Load example (full or excerpt)
    ref_json_str, _ = _load_example(args.example, include_full=args.include_full_example, per_category=2)

    # Compose prompt
    prompt = _build_prompt(args.domain, ref_json_str)

    if args.print_prompt:
        print(prompt)
        sys.exit(0)

    # Ensure Ollama is up
    _start_ollama_if_needed(args.host, autostart=(not args.no_autostart), wait_seconds=45)

    # Ensure model
    if not args.no_pull:
        _ensure_model(args.model)

    # Ask the model
    raw = _ollama_chat(args.host, args.model, prompt, temperature=args.temperature)

    ok, parsed = _try_parse_json(raw)
    if not ok:
        # Attempt one repair round
        repair_prompt = REPAIR_PROMPT + "\n\nHere is the content to fix:\n" + raw
        raw2 = _ollama_chat(args.host, args.model, repair_prompt, temperature=0.0)
        ok2, parsed2 = _try_parse_json(raw2)
        if not ok2:
            raise RuntimeError("Model returned non-JSON twice. Use --print-prompt to inspect the prompt.")
        parsed = parsed2

    output = json.dumps(parsed, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output + "\n")
        print(f"Wrote ontology to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
