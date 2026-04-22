# TurboQuantPlus

Run **Qwen3.6-35B-A3B** (Mixture-of-Experts: 35B total params, ~3B active per token) locally on Apple Silicon via MLX — with a single optimization that delivers a **4.5x generation speedup** over vanilla inference at longer contexts.

## Why This Exists

We set out to run the largest capable open model on a Mac with the best possible user experience. The stock `mlx-lm` pipeline is fast at short generations, but **degrades dramatically as context grows** — dropping from 108 tok/s to 21 tok/s at 2048 tokens. TurboQuantPlus fixes this with quantized KV caching, keeping generation speed nearly flat regardless of context length.

## Benchmark Results (M5 Max, 128GB)

Tested on a 199-token prompt generating 2048 tokens:

| | TurboQuantPlus | Vanilla mlx-lm | Delta |
|---|---|---|---|
| **Prompt processing** | 527 tok/s | 726 tok/s | -27% (one-time cost) |
| **Generation** | **95.5 tok/s** | **21.3 tok/s** | **4.5x faster** |
| **Peak memory** | 28.5 GB | 28.5 GB | Same |

At shorter contexts (512 tokens), both are comparable (~106-108 tok/s). The optimization matters most where it counts — sustained generation in real conversations and long-form output.

### Why Not mlx-swift-lm?

We evaluated the Swift wrapper ([mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm)) and found it offers **no performance advantage**. Both the Swift and Python paths call the same underlying MLX C++/Metal GPU kernels — the language wrapper adds negligible overhead. The Python path (`mlx-lm`) is simpler to set up (no Xcode required), has better tooling, and receives faster upstream updates.

## Model

[mlx-community/Qwen3.6-35B-A3B-6bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-6bit) — 6-bit quantized safetensors (~27GB).

Equivalent to the [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) Q6_K_XL variant, converted to MLX-native safetensors format (GGUF is not supported by the MLX Swift/Python stack).

## The TurboQuantPlus Optimization

The core enhancement is **4-bit KV cache quantization with delayed start**:

1. **KV cache quantization (4-bit)** — As the model generates tokens, the key-value attention cache grows linearly. At full precision this cache becomes the bottleneck, causing generation speed to fall off a cliff. Quantizing it to 4-bit keeps the cache compact and maintains throughput.

2. **Delayed quantization start (512 tokens)** — The first 512 tokens of the KV cache are kept at full precision. Early tokens carry disproportionate weight in attention (system prompt, initial context), so preserving their precision protects output quality where it matters most.

3. **Grouped quantization (group size 64)** — Rather than quantizing the entire KV cache uniformly, values are quantized in groups of 64, preserving local variance and reducing quantization error.

The result: generation speed stays above 95 tok/s even at 2048+ tokens, where vanilla inference has collapsed to ~21 tok/s.

## Quick Start

```bash
# Interactive chat (default)
./run.sh

# Single prompt
./run.sh -p "Explain quantum computing in simple terms"

# Chat with performance stats
./run.sh --benchmark --chat

# Adjust settings
./run.sh --temp 0.3 --max-tokens 8192 --kv-bits 8
```

### All Options

| Flag | Default | Description |
|---|---|---|
| `--model` | Auto-detected local cache | HuggingFace ID or local path |
| `-p, --prompt` | — | Single prompt (non-interactive) |
| `-s, --system` | — | System prompt |
| `--max-tokens` | 4096 | Max generation length |
| `--temp` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--min-p` | 0.05 | Minimum probability cutoff |
| `--kv-bits` | 4 | KV cache quantization (0=off, 4, or 8) |
| `--kv-group-size` | 64 | Quantization group size |
| `--quantized-kv-start` | 512 | Tokens before quantization kicks in |
| `--no-stream` | off | Disable streaming output |
| `--chat` | on (if no prompt) | Interactive chat mode |
| `--benchmark` | off | Print tok/s and memory stats |

## Project Structure

```
turboquant/
  turboquant.py   # Main inference engine with TurboQuantPlus optimizations
  run.sh          # Convenience launcher (activates venv)
  .venv/          # Python 3.12 virtual environment with mlx-lm
  README.md
```

## Requirements

- macOS 14+ (Sonoma) on Apple Silicon
- Python 3.12 (virtual environment included at `.venv/`)
- ~27GB disk for model weights (cached in `~/.cache/huggingface/`)
- 32GB+ unified memory (128GB recommended for best throughput)

## Setup

The environment is pre-built. If starting fresh:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install mlx-lm huggingface_hub
```

The model downloads automatically on first run (~27GB).

## Origin

Built on [mlx-lm](https://github.com/ml-explore/mlx-lm) (v0.31.2) from Apple's MLX team. Inspired by [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm) — evaluated and set aside in favor of the Python path for identical performance with simpler tooling.
