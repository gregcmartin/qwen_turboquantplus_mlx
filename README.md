# TurboQuantPlus

Run Qwen3.6-27B locally on Apple Silicon with MX-FP8 weights and optimized KV cache quantization, plus an OpenAI-compatible server for tools like opencode.

## What It Does

Quantizes [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) (dense 27B parameters) from BF16 to MX-FP8 using `mlx_lm.convert --q-mode mxfp8`, then runs it through Apple's MLX framework with 4-bit KV cache quantization. MX-FP8 is a true floating-point 8-bit format (OCP Microscaling) — distinct from the int8 grouped quant used by most `mlx-community/...-8bit` models. The 4-bit KV cache keeps generation speed high even at long contexts.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 14+
- Python 3.12+
- 64GB+ unified memory recommended
- ~90GB free disk during conversion (~56GB BF16 download + ~26GB MX-FP8 output)

## Setup

```bash
git clone https://github.com/gregcmartin/qwen_turboquantplus_mlx.git
cd qwen_turboquantplus_mlx

python3.12 -m venv .venv
source .venv/bin/activate
pip install mlx-lm huggingface_hub
```

### Quantize the model (one-time)

```bash
mkdir -p models
python -m mlx_lm convert \
  --hf-path Qwen/Qwen3.6-27B \
  --mlx-path ./models/Qwen3.6-27B-MLX-mxfp8 \
  -q --q-mode mxfp8
```

The resulting directory (~26 GB, 8.25 bits/weight effective) is what `turboquant.py` loads by default.

## Usage

### Chat / single prompt

```bash
./run.sh                              # interactive chat
./run.sh -p "Explain MoE routing."    # single prompt
./run.sh --benchmark --chat           # chat with speed stats
./run.sh --temp 0.3 --kv-bits 8       # custom settings
```

### Serve over HTTP (OpenAI-compatible, for opencode et al.)

```bash
./run.sh --serve                      # 127.0.0.1:8080/v1
./run.sh --serve --port 9000          # custom port
```

Then point opencode at it by putting this in `~/.opencode.json`:

```json
{
  "agents": {
    "coder": { "model": "local.Qwen3.6-27B-MLX-mxfp8" },
    "task":  { "model": "local.Qwen3.6-27B-MLX-mxfp8" },
    "title": { "model": "local.Qwen3.6-27B-MLX-mxfp8" }
  }
}
```

and launching opencode with `LOCAL_ENDPOINT=http://127.0.0.1:8080/v1 opencode`.

### Options

| Flag | Default | Description |
|---|---|---|
| `-p, --prompt` | | Single prompt (skips chat) |
| `-s, --system` | | System prompt |
| `--max-tokens` | 4096 | Max output length |
| `--temp` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling |
| `--min-p` | 0.05 | Min probability cutoff |
| `--kv-bits` | 4 | KV cache quantization (0=off, 4, 8) |
| `--kv-group-size` | 64 | Quantization group size |
| `--quantized-kv-start` | 512 | Full-precision tokens before quantization |
| `--benchmark` | off | Show tok/s and memory stats |
| `--no-stream` | off | Wait for full response |
| `--serve` | off | Launch HTTP server instead of CLI |
| `--host` | 127.0.0.1 | Server host |
| `--port` | 8080 | Server port |

## Tool-calling eval

Two evaluators are included, covering the OpenAI `tools` / `tool_calls` path used by opencode:

- **`eval_tools.py`** — hits the running HTTP server (`./run.sh --serve`) with a suite of tool-call tasks and scores tool selection + argument validity.
- **`eval_tools_local.py`** — loads the model in-process and runs the same suite under four KV-cache configurations (full-precision, 8-bit, 4-bit @ start=512, 4-bit @ start=0), with an optional `--long-context` mode that exercises KV quant at ~3.8k-token prompts.

Qwen3.6-27B is a reasoning model — if `enable_thinking` is left on (the default), the model spends its first few hundred tokens in a `<think>…</think>` block, which can blow through a tight `max_tokens` budget before a tool call ever appears. Pass `--no-thinking` to `eval_tools_local.py` to short-circuit reasoning and measure tool-calling behavior directly.

```bash
# Server eval (needs --serve running in another terminal)
python eval_tools.py

# In-process comparison across KV configs (reasoning off — fair tool-call measurement)
python eval_tools_local.py --no-thinking
python eval_tools_local.py --no-thinking --long-context
```

Results on M5 Max 128GB, Qwen3.6-27B MX-FP8, with `--no-thinking`:

| KV config | Short context (~300 tok) | Long context (~3.4k tok) |
|---|---|---|
| full-precision | 11/11 | 10/11 |
| 8-bit @ start=512 | 11/11 | 10/11 |
| 4-bit @ start=512 | 11/11 | 10/11 |
| 4-bit @ start=0 | 11/11 | 10/11 |

The single long-context miss (`web_search_recent`) is consistent across all four KV configs — the model answers from its own knowledge instead of calling the search tool. Not a quantization artifact.

## How the Optimization Works

Two layers of quantization, attacking different bottlenecks:

1. **Weights → MX-FP8.** The OCP Microscaling FP8 format stores each weight as an 8-bit float (E4M3-style) with a shared exponent per microblock. Closer in dynamic range to BF16 than int8 grouped quant, with the same ~8-bit storage footprint.
2. **KV cache → 4-bit.** The attention cache grows with every generated token; at full precision it becomes the bottleneck at long contexts. TurboQuantPlus keeps the first 512 tokens full-precision (system prompt and early context matter most for quality) and quantizes the rest in groups of 64.

Dense 27B means every token activates all 27B parameters (no MoE sparsity), so generation throughput is lower than the previous 30B-A3B MoE config.

Measured on M5 Max 128GB, Qwen3.6-27B MX-FP8, 512-token generation, `--no-thinking`:

| KV config | gen tok/s | prompt tok/s | peak mem |
|---|---|---|---|
| full-precision | **17.8** | 228 | 28.1 GB |
| 8-bit @ start=512 | 16.9 | 201 | 28.1 GB |
| 4-bit @ start=512 | 16.2 | 195 | 28.1 GB |
| 4-bit @ start=0 | 16.6 | 193 | 28.1 GB |

At this generation length the KV cache is small enough that dequant overhead slightly exceeds the bandwidth savings — full-precision KV is actually fastest. The KV-quant win grows with context length; switch to `--kv-bits 4` for prompts pushing past a few thousand tokens or when peak memory becomes the constraint.

## License

MIT
