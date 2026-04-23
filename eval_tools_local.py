#!/usr/bin/env python3
"""
In-process tool-calling eval — lets us toggle KV cache quantization settings
that mlx_lm.server doesn't expose as per-request parameters.

Loads the model once, then runs the same TASKS from eval_tools.py under
multiple KV-quant configurations and prints a side-by-side pass/fail table.
This is the test that answers: does TurboQuant's KV quant break tool calling?
"""

import argparse
import json
import re
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call

from eval_tools import TASKS, DEFAULT_MODEL, score_task


CONFIGS = [
    {"label": "KV full-precision",      "kv_bits": 0,                      },
    {"label": "KV 8-bit @ start=512",   "kv_bits": 8, "quantized_kv_start": 512},
    {"label": "KV 4-bit @ start=512",   "kv_bits": 4, "quantized_kv_start": 512},
    {"label": "KV 4-bit @ start=0",     "kv_bits": 4, "quantized_kv_start": 0},
]

# Long-context padding — repeated filler to push prompts above 512 tokens so
# KV-quant actually activates with the default start=512.
_PAD_UNIT = (
    "The following is background context that is not relevant to the task. "
    "Ignore it and answer the user's actual question at the end. "
)
LONG_CONTEXT_PAD = (_PAD_UNIT * 120)  # roughly ~2k tokens


def generate_text(model, tokenizer, messages, tools, kv_bits=0,
                  kv_group_size=64, quantized_kv_start=512, max_tokens=384,
                  enable_thinking=None):
    tmpl_kwargs = dict(tools=tools, tokenize=False, add_generation_prompt=True)
    if enable_thinking is not None:
        tmpl_kwargs["enable_thinking"] = enable_thinking
    formatted = tokenizer.apply_chat_template(messages, **tmpl_kwargs)
    sampler = make_sampler(temp=0.2, top_p=0.9, min_p=0.0)
    kwargs = dict(max_tokens=max_tokens, sampler=sampler)
    if kv_bits > 0:
        kwargs.update(kv_bits=kv_bits,
                      kv_group_size=kv_group_size,
                      quantized_kv_start=quantized_kv_start)

    parts = []
    last = None
    for chunk in stream_generate(model, tokenizer, prompt=formatted, **kwargs):
        parts.append(chunk.text)
        last = chunk
    text = "".join(parts)
    return text, last


_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)


def response_from_generation(text, tools):
    """Mimic the shape of an OpenAI /v1/chat/completions response
    so we can reuse eval_tools.score_task."""
    tool_calls = []
    # Tolerate the common failure mode of missing opening <tool_call>:
    if "<function=" in text and "<tool_call>" not in text:
        text = "<tool_call>" + text
    for m in _TOOL_CALL_BLOCK.finditer(text):
        inner = m.group(1).strip()
        try:
            tc = parse_tool_call(inner, tools)
        except Exception:
            continue
        if not tc:
            continue
        tool_calls.append({
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc.get("arguments", {})),
            }
        })

    # Strip tool-call blocks from content
    content = _TOOL_CALL_BLOCK.sub("", text).strip()
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls

    return {"choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls" if tool_calls else "stop"}]}


def run_config(model, tokenizer, cfg, long_context=False, enable_thinking=None,
               max_tokens=384):
    print(f"\n=== {cfg['label']}{'  (long ctx)' if long_context else ''} ===")
    passes = 0
    total = 0
    t0 = time.time()
    for task in TASKS:
        total += 1
        user_prompt = task["prompt"]
        if long_context:
            user_prompt = LONG_CONTEXT_PAD + "\n\n" + user_prompt
        messages = [{"role": "user", "content": user_prompt}]

        start = time.time()
        text, last = generate_text(
            model, tokenizer, messages, task["tools"],
            kv_bits=cfg.get("kv_bits", 0),
            kv_group_size=cfg.get("kv_group_size", 64),
            quantized_kv_start=cfg.get("quantized_kv_start", 512),
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
        )
        dt = time.time() - start
        resp = response_from_generation(text, task["tools"])
        status, detail = score_task(task, resp)
        tag = {"PASS": "✓", "FAIL": "✗", "ERROR": "!"}[status]
        ptoks = getattr(last, "prompt_tokens", "?") if last else "?"
        gtoks = getattr(last, "generation_tokens", "?") if last else "?"
        gtps = f"{last.generation_tps:.0f}" if last and hasattr(last, "generation_tps") else "?"
        print(f"  {tag} [{dt:5.1f}s  p={ptoks} g={gtoks} @{gtps}t/s] "
              f"{task['name']:30s} {detail[:100]}")
        if status == "PASS":
            passes += 1
    print(f"  --- {passes}/{total} passed in {time.time()-t0:.1f}s ---")
    return passes, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--long-context", action="store_true",
                    help="Prepend ~2k tokens of filler to each prompt so "
                         "KV-quant activates at start=512")
    ap.add_argument("--configs", nargs="*",
                    help="Subset of config labels to run")
    ap.add_argument("--no-thinking", action="store_true",
                    help="Pass enable_thinking=False to the chat template "
                         "(for reasoning models like Qwen3.6-27B — skips the "
                         "<think> prefix so the model emits a tool call directly)")
    ap.add_argument("--max-tokens", type=int, default=384,
                    help="Generation budget per task")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"Loaded in {time.time()-t0:.1f}s  (device: {mx.default_device()})")

    summary = []
    for cfg in CONFIGS:
        if args.configs and cfg["label"] not in args.configs:
            continue
        p, t = run_config(
            model, tokenizer, cfg,
            long_context=args.long_context,
            enable_thinking=False if args.no_thinking else None,
            max_tokens=args.max_tokens,
        )
        summary.append((cfg["label"], p, t))

    print("\n=== Summary ===")
    for label, p, t in summary:
        print(f"  {label:30s}  {p}/{t}  ({100*p/t:.0f}%)")


if __name__ == "__main__":
    main()
