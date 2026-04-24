#!/usr/bin/env python3
"""
Tool-calling eval for local mlx_lm.server.

Sends a suite of tool-call tasks to an OpenAI-compatible endpoint and scores:
  1. tool_calls field present on the response
  2. correct tool name chosen
  3. required arguments present
  4. argument values pass task-specific predicates

Default endpoint is http://127.0.0.1:8080/v1 (the mlx_lm.server port that
opencode also talks to), so pass/fail here reflects what opencode would see.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error


DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1"
# mlx_lm.server advertises models by their absolute resolved path, so the
# client has to reference the exact same string.
DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "Qwen3.6-27B-MLX-mxfp8",
)


# --- Tool specs shared across tasks -----------------------------------------

GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    },
}

LIST_FILES = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files in a directory on the user's machine.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path"},
                "recursive": {"type": "boolean"},
            },
            "required": ["path"],
        },
    },
}

READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
}

RUN_SHELL = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": "Run a shell command and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string"},
            },
            "required": ["command"],
        },
    },
}

CALCULATOR = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a simple arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}

WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for up-to-date information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
}

ALL_TOOLS = [GET_WEATHER, LIST_FILES, READ_FILE, RUN_SHELL, CALCULATOR, WEB_SEARCH]


# --- Tasks ------------------------------------------------------------------

def city_is(expected):
    return lambda args: str(args.get("city", "")).strip().lower() == expected.lower()

def path_eq(expected):
    return lambda args: args.get("path") == expected

def path_contains(sub):
    return lambda args: sub in str(args.get("path", ""))

def expression_contains(*needles):
    def _chk(args):
        e = str(args.get("expression", ""))
        return all(n in e for n in needles)
    return _chk

def command_contains(*needles):
    def _chk(args):
        c = str(args.get("command", ""))
        return all(n in c for n in needles)
    return _chk

def query_contains(*needles):
    def _chk(args):
        q = str(args.get("query", "")).lower()
        return all(n.lower() in q for n in needles)
    return _chk


TASKS = [
    {
        "name": "weather_city",
        "prompt": "What's the weather in Tokyo right now?",
        "tools": [GET_WEATHER],
        "expected_tool": "get_weather",
        "required": ["city"],
        "predicate": city_is("Tokyo"),
    },
    {
        "name": "weather_with_units",
        "prompt": "Give me the Paris temperature in Fahrenheit.",
        "tools": [GET_WEATHER],
        "expected_tool": "get_weather",
        "required": ["city"],
        "predicate": lambda a: city_is("Paris")(a) and a.get("units") in ("f", "F"),
    },
    {
        "name": "list_files_abs_path",
        "prompt": "List the files in /Users/gregmartin/Desktop.",
        "tools": [LIST_FILES],
        "expected_tool": "list_files",
        "required": ["path"],
        "predicate": path_eq("/Users/gregmartin/Desktop"),
    },
    {
        "name": "read_file_with_offset",
        "prompt": "Read lines 100-150 of /etc/hosts.",
        "tools": [READ_FILE],
        "expected_tool": "read_file",
        "required": ["path"],
        "predicate": lambda a: a.get("path") == "/etc/hosts"
            and a.get("offset") in (100, "100")
            and a.get("limit") in (50, 51, "50", "51"),
    },
    {
        "name": "shell_git_status",
        "prompt": "Run 'git status' in the current repo.",
        "tools": [RUN_SHELL],
        "expected_tool": "run_shell",
        "required": ["command"],
        "predicate": command_contains("git", "status"),
    },
    {
        "name": "calculator_basic",
        "prompt": "What is 2345 * 87?",
        "tools": [CALCULATOR],
        "expected_tool": "calculator",
        "required": ["expression"],
        "predicate": expression_contains("2345", "87"),
    },
    {
        "name": "web_search_recent",
        "prompt": "What is the latest Apple MacBook Pro release?",
        "tools": [WEB_SEARCH],
        "expected_tool": "web_search",
        "required": ["query"],
        "predicate": query_contains("macbook"),
    },
    # Selection tasks — model is given ALL tools and must pick the right one
    {
        "name": "select_weather_from_all",
        "prompt": "What's the weather in London?",
        "tools": ALL_TOOLS,
        "expected_tool": "get_weather",
        "required": ["city"],
        "predicate": city_is("London"),
    },
    {
        "name": "select_read_from_all",
        "prompt": "Show me the contents of /tmp/notes.txt.",
        "tools": ALL_TOOLS,
        "expected_tool": "read_file",
        "required": ["path"],
        "predicate": path_contains("/tmp/notes.txt"),
    },
    {
        "name": "select_shell_from_all",
        "prompt": "Show me the running docker containers.",
        "tools": ALL_TOOLS,
        "expected_tool": "run_shell",
        "required": ["command"],
        "predicate": command_contains("docker"),
    },
    # No-tool case: model should NOT call a tool for a simple chat question
    {
        "name": "no_tool_needed",
        "prompt": "Say hi in exactly three words.",
        "tools": [GET_WEATHER, CALCULATOR],
        "expected_tool": None,
        "required": [],
        "predicate": lambda a: True,
    },
]


# --- HTTP -------------------------------------------------------------------

def chat_completion(endpoint, model, messages, tools, temperature=0.2,
                    max_tokens=384, extra_body=None, timeout=180):
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra_body:
        body.update(extra_body)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# --- Scoring ----------------------------------------------------------------

def score_task(task, response):
    """Returns (status, detail) where status in {PASS, FAIL, ERROR}."""
    try:
        choice = response["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError) as e:
        return ("ERROR", f"malformed response: {e}")

    tool_calls = msg.get("tool_calls") or []

    # No-tool expectation
    if task["expected_tool"] is None:
        if tool_calls:
            return ("FAIL", f"model called {tool_calls[0]['function']['name']} "
                            f"when no tool was needed")
        return ("PASS", "no tool call, as expected")

    # Tool expected
    if not tool_calls:
        content = (msg.get("content") or "").strip()
        snippet = content[:120].replace("\n", " ")
        return ("FAIL", f"no tool_calls; content=<{snippet}>")

    tc = tool_calls[0]["function"]
    name = tc.get("name")
    if name != task["expected_tool"]:
        return ("FAIL", f"wrong tool: got {name!r}, expected {task['expected_tool']!r}")

    raw_args = tc.get("arguments") or "{}"
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except json.JSONDecodeError as e:
        return ("FAIL", f"arguments not valid JSON: {e}; raw={raw_args!r}")

    missing = [k for k in task["required"] if k not in args]
    if missing:
        return ("FAIL", f"missing required args: {missing}; got={args}")

    try:
        ok = task["predicate"](args)
    except Exception as e:
        return ("ERROR", f"predicate raised: {e}; args={args}")
    if not ok:
        return ("FAIL", f"predicate rejected args: {args}")
    return ("PASS", f"args={args}")


# --- Runner -----------------------------------------------------------------

def run_suite(endpoint, model, label, extra_body=None, only=None):
    print(f"\n=== {label} ===")
    results = []
    t0 = time.time()
    for task in TASKS:
        if only and task["name"] not in only:
            continue
        messages = [{"role": "user", "content": task["prompt"]}]
        start = time.time()
        try:
            resp = chat_completion(endpoint, model, messages, task["tools"],
                                   extra_body=extra_body)
            status, detail = score_task(task, resp)
        except urllib.error.URLError as e:
            status, detail = "ERROR", f"network: {e}"
        except Exception as e:
            status, detail = "ERROR", f"{type(e).__name__}: {e}"
        dt = time.time() - start
        tag = {"PASS": "✓", "FAIL": "✗", "ERROR": "!"}[status]
        print(f"  {tag} [{dt:5.1f}s] {task['name']:30s} {detail}")
        results.append((task["name"], status, detail))

    passes = sum(1 for _, s, _ in results if s == "PASS")
    fails = sum(1 for _, s, _ in results if s == "FAIL")
    errors = sum(1 for _, s, _ in results if s == "ERROR")
    total = len(results)
    print(f"  --- {passes}/{total} passed  ({fails} fail, {errors} error)  "
          f"in {time.time()-t0:.1f}s ---")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--only", nargs="*", help="Run only these task names")
    ap.add_argument("--kv-bits", type=int, default=0,
                    help="Request kv_bits via extra body (ignored if server doesn't support it)")
    args = ap.parse_args()

    extra = {}
    if args.kv_bits:
        extra["kv_bits"] = args.kv_bits

    results = run_suite(args.endpoint, args.model, args.label,
                        extra_body=extra or None, only=args.only)

    # Nonzero exit if anything failed, useful for CI
    if any(s != "PASS" for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
