# ds4 — GLM-5.2 branch (`ds4-glm`)

This branch retargets the ds4 engine (a DeepSeek-V4 Flash specific inference engine)
to run **GLM-5.2** (`zai-org/GLM-5.2`) on Apple Silicon / Metal. It is a *fork*, not a
runtime config switch: every GLM behavior is gated on `DS4_MODEL_VARIANT == DS4_VARIANT_GLM`,
and the DeepSeek-V4 Flash/Pro and CUDA paths are left bit-identical.

The deep design notes, geometry map, risk register, and a dated status log live in
[`glm-port-plan.md`](glm-port-plan.md). This file is just *getting started*.

## Status

| Path | State |
|---|---|
| Q2 GGUF, Metal, single + multi-token generation | **works** (~11.6 tok/s decode on an M3 Ultra) |
| HTTP server (OpenAI / Anthropic / Responses APIs) answering GLM prompts | **works** |
| GLM chat framing, reasoning effort, multi-EOS, `<tool_call>` render/parse | **works** |
| GPU sigmoid router (no per-MoE-layer CPU sync) | **works**, validated vs the CPU forward |
| CPU reference forward (all 78 layers) | **works** (debug/reference only) |
| Q4 GGUF inference | **runs, slowly** — Q4 (≈409 GiB) streams on 256 GB via the CPU routed-MoE path (validation-only, ~0.3 tok/s); a fast GPU expert cache is still WIP. Q4 confirms Q2's degeneration on hard prompts is just the 2-bit ceiling, not a bug |
| DSA indexer (long context), MTP draft tokens | not wired for GLM yet (dense MLA attention is a correct superset) |

GLM-5.2 vs DeepSeek-V4 Flash, the load-bearing deltas (all in the engine, variant-gated):
no Hyper-Connections (single residual stream), absorbed MLA with **key dim 576 / value dim
512** and partial `kv_a_norm`, plain `o_proj` (no output LoRA), **dense** SwiGLU for the
first 3 layers, **sigmoid** noaux_tc routing over **256 experts, 8 used**, interleaved RoPE
(θ=8e6, no YaRN), and the GLM tokenizer + `[gMASK]<sop>` chat template. See `glm-port-plan.md` §1–§3.

## Hardware

* Apple Silicon with Metal. Developed on an **M3 Ultra, 256 GB**.
* **Q2** weights are ≈219 GiB; they fit resident on 256 GB with a usable context. Raise the
  Metal wired limit if you hit memory pressure: `sudo sysctl iogpu.wired_limit_mb=240000`.
* **Q4** weights are ≈409 GiB and do **not** fit resident on 256 GB — Q4 needs SSD streaming
  (WIP, see Status).
* A CPU-only reference build exists (`make cpu`) but large CPU runs can trip macOS VM limits;
  use it only for debugging.

## Build

```sh
make            # builds ./ds4 (CLI), ./ds4-server, ./ds4-bench, ./ds4-eval, ./ds4-agent (Metal)
make cpu        # CPU-only reference build of the same binaries
make test       # unit/regression tests (some need a model + Metal)
```

Metal kernels are compiled at runtime from `metal/*.metal`, read relative to the current
working directory — run the binaries from the repo root.

## Get a GGUF

The converter `gguf-tools/glm-quantize.c` authors a GLM GGUF directly from the Hugging Face
`zai-org/GLM-5.2` release (bf16 safetensors + `config.json` + `tokenizer.json`). It writes the
`deepseek4.*` metadata namespace, the GLM BPE tokenizer, the absorbed-MLA attention fold
(proven exact in `gguf-tools/check_mla_absorption.py`), and the stacked 256-expert layout.

```sh
cd gguf-tools && make glm-quantize

# Q4 (default: Q4_K experts), the full 78-layer model:
./glm-quantize --hf /path/to/GLM-5.2-hf --write-full glm-5.2-q4.gguf

# Q2 (IQ2_XXS gate/up + Q2_K down — the 256 GB target):
./glm-quantize --hf /path/to/GLM-5.2-hf --q2 --write-full glm-5.2-q2.gguf

# inspect / sanity-check the GGUF without running it:
./glm-quantize --verify glm-5.2-q2.gguf
```

Both recipes keep embeddings/output/router at F16, attention/shared/dense at Q8_0, and norms
at F32. `--layers N` limits the write to the first N layers for a fast partial build.

## Run

The engine auto-detects the GLM variant from the GGUF metadata.

```sh
# Validate the load (binds + checks every tensor, prints the geometry):
./ds4 -m glm-5.2-q2.gguf --inspect

# One-shot generation (greedy is the most reliable on Q2):
./ds4 -m glm-5.2-q2.gguf -p "List the first five planets from the Sun" -n 80 --temp 0 -c 2048

# Interactive chat REPL:
./ds4 -m glm-5.2-q2.gguf -c 4096

# Q4, for validation only — runs via the CPU routed-MoE path streaming experts
# from the demand-paged mmap (≈0.3 tok/s; a fast GPU expert cache is WIP):
DS4_GLM_CPU_ROUTED_MOE=1 ./ds4 -m glm-5.2-q4.gguf --ssd-streaming --ssd-streaming-cold \
  --ssd-streaming-cache-experts 1GB -p "..." -n 80 --temp 0 -c 2048
```

**Sampling defaults.** GLM's nominal config is temp 1.0 / top_p 0.95, but on the 2-bit Q2
quantization temp 1.0 degenerates on complex prompts (verified identical on the CLI and server
paths — it is the quant ceiling, not an engine bug; greedy on simple prompts is clean and
deterministic). The GLM defaults are therefore **temp 0.6 / top_p 0.95**; pass `--temp` /
`--top-p` to override.

### Server

```sh
./ds4-server -m glm-5.2-q2.gguf -c 2048 --port 8000
# then, e.g.:
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"Name the planets, briefly."}],
  "reasoning_effort":"none", "temperature":0.6, "top_p":0.95, "max_tokens":256
}'
```

Endpoints: `/v1/chat/completions`, `/v1/responses`, `/v1/completions`, `/v1/messages`. Thinking
is on by default (`reasoning_effort` `none`/`high`/`max`); GLM tool calls are emitted/parsed as
`<tool_call>{name}<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>`.

## Tests

```sh
./ds4 --glm-attn-test            # standalone 576/512 absorbed-attention kernel test (no model)
./ds4_test --server              # GLM chat render + <tool_call> render/parse unit tests (no model)
# Numerical bring-up against the CPU reference (needs the model + Metal, server stopped):
./ds4 -m glm-5.2-q2.gguf -c 2048 --metal-graph-full-test -p "..."
```

`--glm-attn-test` and `./ds4_test --server` run locally in seconds with no model load and are the
fast regressions to keep green. `--metal-graph-full-test` compares the Metal forward to the
validated CPU forward layer by layer.

## Known limitations

* **Q4 streaming is unfinished** — DeepSeek's evicting expert cache is 6-experts-per-token
  hardwired; GLM routes 8, so Q4 (which must stream on 256 GB) needs a GLM-specific streaming
  dispatch. Q2 fits resident and is the working target.
* **No DSA indexer / no MTP** for GLM yet — attention runs as dense MLA (a correct superset for
  sub-1M contexts); MTP draft tokens are DeepSeek-only.
* **Not yet validated against an external reference** (official GLM API logprobs). The Metal
  forward matches the ds4 CPU forward to fp noise and greedy output is coherent and correct, but
  a golden-vector check vs the upstream model is still open.
* GLM-on-CUDA is unsupported (the GLM GPU entry points carry CUDA stubs so the build still links).
