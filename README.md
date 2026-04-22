# TurboQuantPlus

Run Qwen3.6-35B-A3B locally on Apple Silicon with optimized KV cache quantization for fast, sustained inference.

## What It Does

Runs the [Qwen3.6-35B-A3B-6bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-6bit) model (35B params, ~3B active per token) through Apple's MLX framework with 4-bit KV cache quantization. This keeps generation speed at ~95 tok/s even at 2048+ tokens, compared to ~21 tok/s without it.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 14+
- Python 3.12+
- 32GB+ unified memory
- ~27GB disk for model weights

## Setup

```bash
git clone https://github.com/gregcmartin/qwen_turboquantplus_mlx.git
cd qwen_turboquantplus_mlx

python3.12 -m venv .venv
source .venv/bin/activate
pip install mlx-lm huggingface_hub
```

The model downloads automatically on first run (~27GB).

## Usage

```bash
# Interactive chat
./run.sh

# Single prompt
./run.sh -p "Explain quantum computing"

# Chat with speed stats
./run.sh --benchmark --chat

# Custom settings
./run.sh --temp 0.3 --max-tokens 8192 --kv-bits 8
```

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

## Benchmarks (M5 Max, 128GB)

### Short context (512 tokens generated)

| | TurboQuantPlus | Vanilla mlx-lm |
|---|---|---|
| Prompt | 197 tok/s | 514 tok/s |
| Generation | 106 tok/s | 109 tok/s |
| Peak memory | 28.3 GB | 28.3 GB |

At short context lengths, performance is nearly identical.

### Long context (2048 tokens generated)

| | TurboQuantPlus | Vanilla mlx-lm | Delta |
|---|---|---|---|
| Prompt | 527 tok/s | 726 tok/s | -27% |
| **Generation** | **95.5 tok/s** | **21.3 tok/s** | **4.5x faster** |
| Peak memory | 28.5 GB | 28.5 GB | Same |

Vanilla mlx-lm drops from 109 tok/s to 21 tok/s as context grows. TurboQuantPlus holds at 95+ tok/s.

## How the Optimization Works

The KV attention cache grows with every generated token. At full precision it becomes a bottleneck and generation slows down fast. TurboQuantPlus quantizes this cache to 4-bit with two refinements:

- The first 512 tokens stay at full precision (system prompt and early context matter most for quality)
- Values are quantized in groups of 64 to reduce error

## License

MIT
