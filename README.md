# TurboQuantPlus

Run Qwen3-Coder-30B-A3B-Instruct locally on Apple Silicon with optimized KV cache quantization, plus an OpenAI-compatible server for tools like opencode.

## What It Does

Runs [`mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit`](https://huggingface.co/mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit) (30B params, ~3B active per token, 8-bit MLX weights) through Apple's MLX framework with 4-bit KV cache quantization. This keeps generation speed high even at long contexts, and preserves tool-calling accuracy — verified end-to-end with the included eval harness.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 14+
- Python 3.12+
- 64GB+ unified memory recommended
- ~30GB disk for model weights

## Setup

```bash
git clone https://github.com/gregcmartin/qwen_turboquantplus_mlx.git
cd qwen_turboquantplus_mlx

python3.12 -m venv .venv
source .venv/bin/activate
pip install mlx-lm huggingface_hub
```

The model downloads automatically on first run (~30GB).

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
    "coder": { "model": "local.mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit" },
    "task":  { "model": "local.mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit" },
    "title": { "model": "local.mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit" }
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

On M5 Max 128GB, Qwen3-Coder-30B-A3B-Instruct-8bit scores **11/11 at every KV configuration, at both short and long context** — the KV quant does not degrade tool calling.

```bash
# Server eval (needs --serve running in another terminal)
python eval_tools.py

# In-process comparison across KV configs
python eval_tools_local.py
python eval_tools_local.py --long-context
```

## How the Optimization Works

The KV attention cache grows with every generated token. At full precision it becomes a bottleneck and generation slows down fast. TurboQuantPlus quantizes this cache to 4-bit with two refinements:

- The first 512 tokens stay at full precision (system prompt and early context matter most for quality).
- Values are quantized in groups of 64 to reduce error.

In the included eval, 4-bit KV quantization holds 100% tool-call accuracy at ~3.8k-token prompts while full-precision and 4-bit configurations run within a few percent of each other on throughput.

## License

MIT
