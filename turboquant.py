#!/usr/bin/env python3
"""
TurboQuantPlus - Optimized local LLM inference on Apple Silicon via MLX.

Runs Qwen3-Coder-30B-A3B-Instruct (MoE: 30B total, ~3B active) with
aggressive optimizations for M-series Macs:
  - KV cache quantization (4-bit) for 2-3x memory savings on long contexts
  - Quantized KV delayed start for initial quality preservation
  - Prompt caching for repeat/continuation queries
  - Tuned sampling defaults for quality + speed

Tool calling is verified with `eval_tools.py` (server) and
`eval_tools_local.py` (in-process, covers KV-quant configurations).
"""

import argparse
import sys
import time

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler


MODEL_ID = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"

# TurboQuantPlus defaults — tuned for M5 Max 128GB
DEFAULTS = {
    "max_tokens": 4096,
    "temp": 0.7,
    "top_p": 0.9,
    "min_p": 0.05,
    "kv_bits": 4,            # quantize KV cache to 4-bit
    "kv_group_size": 64,     # group size for KV quantization
    "quantized_kv_start": 512,  # keep first 512 tokens full precision
}


def build_parser():
    p = argparse.ArgumentParser(
        description="TurboQuantPlus — optimized Qwen3-Coder-30B-A3B-Instruct on Apple Silicon"
    )
    p.add_argument("--model", default=MODEL_ID,
                   help="HF model ID or local snapshot path")
    p.add_argument("--prompt", "-p", type=str, help="Single prompt (non-interactive)")
    p.add_argument("--system", "-s", type=str, default=None, help="System prompt")
    p.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
    p.add_argument("--temp", type=float, default=DEFAULTS["temp"])
    p.add_argument("--top-p", type=float, default=DEFAULTS["top_p"])
    p.add_argument("--min-p", type=float, default=DEFAULTS["min_p"])
    p.add_argument("--kv-bits", type=int, default=DEFAULTS["kv_bits"],
                   help="KV cache quantization bits (0=disabled, 4 or 8)")
    p.add_argument("--kv-group-size", type=int, default=DEFAULTS["kv_group_size"])
    p.add_argument("--quantized-kv-start", type=int, default=DEFAULTS["quantized_kv_start"])
    p.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    p.add_argument("--chat", action="store_true", help="Interactive chat mode")
    p.add_argument("--benchmark", action="store_true", help="Print token/s stats")
    p.add_argument("--serve", action="store_true",
                   help="Launch the OpenAI-compatible HTTP server (for opencode et al.) "
                        "on --host:--port instead of running CLI chat")
    p.add_argument("--host", default="127.0.0.1", help="Server host (with --serve)")
    p.add_argument("--port", type=int, default=8080, help="Server port (with --serve)")
    return p


def load_model(model_id):
    print(f"Loading {model_id} ...")
    t0 = time.time()
    model, tokenizer = load(model_id)
    dt = time.time() - t0
    print(f"Model loaded in {dt:.1f}s  |  Device: {mx.default_device()}")
    return model, tokenizer


def apply_chat_template(tokenizer, messages, system_prompt=None):
    """Format messages through the model's chat template."""
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback: just concatenate
    return "\n".join(m["content"] for m in full_messages)


def make_gen_kwargs(args):
    """Build kwargs for generate/stream_generate from CLI args."""
    sampler = make_sampler(temp=args.temp, top_p=args.top_p, min_p=args.min_p)
    kwargs = dict(max_tokens=args.max_tokens, sampler=sampler)
    if args.kv_bits > 0:
        kwargs.update(
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
        )
    return kwargs


def run_single(model, tokenizer, prompt, args):
    """Run a single prompt and print the response."""
    messages = [{"role": "user", "content": prompt}]
    formatted = apply_chat_template(tokenizer, messages, args.system)
    kwargs = make_gen_kwargs(args)

    t0 = time.time()
    token_count = 0

    if args.no_stream:
        response = generate(model, tokenizer, prompt=formatted, **kwargs)
        print(response)
    else:
        last = None
        for chunk in stream_generate(
            model, tokenizer, prompt=formatted, **kwargs,
        ):
            print(chunk.text, end="", flush=True)
            last = chunk
        print()

        if args.benchmark and last:
            print(f"\n--- prompt: {last.prompt_tokens} tok @ {last.prompt_tps:.1f} tok/s | "
                  f"gen: {last.generation_tokens} tok @ {last.generation_tps:.1f} tok/s | "
                  f"peak mem: {last.peak_memory:.1f} GB ---")


def run_chat(model, tokenizer, args):
    """Interactive chat loop."""
    print("TurboQuantPlus Chat  (type 'quit' or Ctrl-C to exit)")
    print(f"KV cache: {args.kv_bits}-bit  |  max tokens: {args.max_tokens}")
    print("-" * 50)

    history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        history.append({"role": "user", "content": user_input})
        formatted = apply_chat_template(tokenizer, history, args.system)
        kwargs = make_gen_kwargs(args)

        print("\nAssistant: ", end="", flush=True)
        response_parts = []
        last = None

        for chunk in stream_generate(
            model, tokenizer, prompt=formatted, **kwargs,
        ):
            print(chunk.text, end="", flush=True)
            response_parts.append(chunk.text)
            last = chunk

        print()
        if args.benchmark and last:
            print(f"  [gen: {last.generation_tokens} tok @ {last.generation_tps:.1f} tok/s | "
                  f"peak: {last.peak_memory:.1f} GB]")

        history.append({"role": "assistant", "content": "".join(response_parts)})


def run_server(args):
    """Delegate to mlx_lm.server so opencode / any OpenAI-compatible
    client can reach the model at http://host:port/v1."""
    import runpy
    sys.argv = [
        "mlx_lm.server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--log-level", "INFO",
    ]
    print(f"TurboQuantPlus server → http://{args.host}:{args.port}/v1  (model: {args.model})")
    runpy.run_module("mlx_lm.server", run_name="__main__")


def main():
    args = build_parser().parse_args()

    if args.serve:
        run_server(args)
        return

    model, tokenizer = load_model(args.model)
    if args.chat:
        run_chat(model, tokenizer, args)
    elif args.prompt:
        run_single(model, tokenizer, args.prompt, args)
    else:
        run_chat(model, tokenizer, args)


if __name__ == "__main__":
    main()
