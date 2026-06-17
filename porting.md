# Porting DS4 to GLM-5.2 — Engineering Handoff

**Purpose:** This document hands off a research/analysis phase to a locally running Claude Code instance. The goal is to **fork `antirez/ds4` (the DeepSeek V4 Flash inference engine) into a separate product that runs GLM-5.2**, with no intent to merge back upstream. It captures architecture findings, the target model’s real `config.json`, memory math, a quantization strategy, and an ordered work plan with the hard parts flagged.

**Audience:** A coding agent with the ds4 source checked out locally and the GLM-5.2 weights/config/tokenizer on disk. Where this doc says “verify in source,” it means: read the actual ds4 file, because this analysis was done from the README + config and has not been checked against the code line-by-line.

**Status of claims:** Numbers for parameter counts and memory are computed from the config below and are good to ~±10%. The exact KV storage precision and some kernel internals are engine decisions that must be confirmed against ds4 source and/or measured.

-----

## 1. The two projects

### 1.1 Base engine: `antirez/ds4` (“DwarfStar 4”)

- A deliberately **narrow, single-model** native inference engine for **DeepSeek V4 Flash**. Not a generic GGUF runner, not a wrapper.
- License **MIT**, with retained **GGML/llama.cpp** copyright (quant layouts/tables, some CPU quant logic and kernels adapted from there). **Keep both notices in `LICENSE`.**
- Backends: **Metal (macOS)** and **CUDA (Linux)** graph executors. A CPU path exists for correctness/debug only (and currently crashes some macOS versions — do not rely on it).
- Distinctive design: **the KV cache is a first-class disk citizen.** Weights are mmap’d; an on-disk KV cache (`.kv` / “KVC” files) persists prefixes across sessions and restarts. This is the “stream from SSD” behaviour — it is about **KV persistence**, not weight streaming.
- Server: OpenAI- and Anthropic-compatible (`/v1/chat/completions`, `/v1/completions`, `/v1/messages`), SSE streaming, native thinking-mode streaming, tool calls via **DeepSeek DSML** with an **exact-replay map** (radix-tree backed) so stateless agent clients keep matching the live KV checkpoint.
- Only loads the specific GGUFs antirez published; tensor layout, quant mix, metadata, and MTP state are all assumed.

**Known file layout (from the repo root):**

|File                            |Role                                                            |
|--------------------------------|----------------------------------------------------------------|
|`ds4.c`, `ds4.h`                |Core engine: loading, prompt rendering, KV state, sampling      |
|`ds4_metal.m`                   |Metal graph/kernels                                             |
|`ds4_cuda.cu`, `ds4_gpu.h`      |CUDA graph/kernels                                              |
|`ds4_iq2_tables_cuda.inc`       |IQ2 quant tables (CUDA)                                         |
|`ds4_server.c`                  |HTTP server, API glue, KV-disk cache, tool replay               |
|`ds4_cli.c`                     |Interactive CLI                                                 |
|`ds4_bench.c`                   |Benchmark harness                                               |
|`rax.c`, `rax.h`, `rax_malloc.h`|Radix tree (used for the tool-id → DSML replay map)             |
|`linenoise.c/.h`                |CLI line editing                                                |
|`download_model.sh`             |Pulls GGUFs from HF                                             |
|`Makefile`                      |`make` (Metal), `make cpu`, `CUDA_ARCH=...`                     |
|`tests/test-vectors`            |Logprob vectors captured from the official DeepSeek V4 Flash API|
|`AGENT.md`                      |Existing agent guidance (model for our `CLAUDE.md`)             |

**DS4 on-disk session payload fields** (these are the hardcoded geometry knobs to map, per the README’s KVC format): saved context size, prefill chunk size, raw KV ring capacity, raw sliding-window length, compressed KV capacity, checkpoint token count, **layer count**, **raw/head KV dimension**, **indexer head dimension**, **vocabulary size**, live raw rows. The payload also stores next-token logits, compressed-attention row counts, and **ratio-4 indexer row counts** — note the term “ratio-4 indexer,” which matters in §3.

### 1.2 Target model: `zai-org/GLM-5.2`

- MoE, **MIT license** (no regional limits) — clean for a separate product.
- ~**755B** params in BF16 (1.51 TB repo, 282 safetensors shards). 1M-token context. Ships an MTP layer for speculative decoding. Same **DSA (DeepSeek Sparse Attention) family** as DS4’s target — this is *why* DS4 is a sensible base.

-----

## 2. GLM-5.2 `config.json` — parsed

```
model_type:               glm_moe_dsa
architectures:            [GlmMoeDsaForCausalLM]
dtype:                    bfloat16

# shape
hidden_size:              6144
intermediate_size:        12288        # dense-layer MLP
num_hidden_layers:        78
first_k_dense_replace:    3            # first 3 layers are dense MLP; rest MoE
vocab_size:               154880
tie_word_embeddings:      false

# attention (MLA)
num_attention_heads:      64
num_key_value_heads:      64
head_dim:                 192
q_lora_rank:              2048
kv_lora_rank:             512
qk_nope_head_dim:         192
qk_rope_head_dim:         64
qk_head_dim:              256          # = nope(192) + rope(64)
v_head_dim:               256
attention_bias:           false

# DSA indexer (IndexShare)
index_head_dim:           128
index_n_heads:            32
index_topk:               2048
index_topk_freq:          4            # one FULL indexer every 4 layers
index_skip_topk_offset:   3
index_share_for_mtp_iteration: true
indexer_rope_interleave:  true
indexer_types:            [full,full,full, (shared,shared,shared,full)*…]  # 1 full : 3 shared

# MoE
n_routed_experts:         256
num_experts_per_tok:      8
n_shared_experts:         1
moe_intermediate_size:    2048
moe_layer_freq:           1
scoring_func:             sigmoid
topk_method:              noaux_tc     # auxiliary-loss-free load balancing
routed_scaling_factor:    2.5
norm_topk_prob:           true
n_group / topk_group:     1 / 1

# rope / context
max_position_embeddings:  1048576      # 1M
rope_theta:               8000000
rope_type:                default
rope_interleave:          true

# MTP / tokens
num_nextn_predict_layers: 1
eos_token_id:             [154820, 154827, 154829]
pad_token_id:             154820
```

-----

## 3. Architecture compatibility — what transfers, what doesn’t

**The fork premise holds.** `glm_moe_dsa` is the DSA family DS4 is built around: MLA compressed KV + a lightning indexer for sparse attention. The single most encouraging field is the indexer pattern.

### Transfers with little/no change

- **MLA compressed-KV representation.** DS4 already caches the MLA latent (`kv_lora_rank` + decoupled RoPE key). GLM uses the same scheme.
- **IndexShare ↔ DS4’s “ratio-4 indexer.”** `index_topk_freq: 4` plus `indexer_types` = one `full` then three `shared`, repeating, means a full index is computed once per 4 layers and reused. DS4’s session format already counts “ratio-4 indexer rows.” **This is likely the lowest-friction part of the port** — but confirm DS4’s shared-indexer semantics match GLM’s (whether “shared” reuses the previous full layer’s index verbatim, and how `index_skip_topk_offset: 3` and the leading dense layers are handled).
- **Server / disk-KV / tool-replay plumbing.** HTTP endpoints, SSE streaming, the KVC disk-cache machinery, and the radix-tree exact-replay map are model-agnostic infrastructure. Reuse wholesale; only the *rendered bytes* (template/tool format) change.
- **MTP path.** DS4 has an optional `--mtp` speculative path; GLM has `num_nextn_predict_layers: 1` and `index_share_for_mtp_iteration: true`. Wire GLM’s MTP weights into it.

### Needs real work

- **Kernel geometry deltas vs DeepSeek.** Same MLA+DSA *algorithm*, but GLM’s dims differ from DeepSeek V3/V4 defaults and DS4’s kernels are tuned for the latter. Confirmed deltas:
  
  |Dim                |DeepSeek V3-ish|GLM-5.2                  |
  |-------------------|---------------|-------------------------|
  |num_attention_heads|128            |**64**                   |
  |qk_nope_head_dim   |128            |**192**                  |
  |v_head_dim         |128            |**256**                  |
  |q_lora_rank        |1536           |**2048**                 |
  |kv_lora_rank       |512            |512 (same)               |
  |qk_rope_head_dim   |64             |64 (same)                |
  |layers             |~61            |**78** (3 dense + 75 MoE)|
  |routed experts     |256            |256 (same)               |
  
  The Metal/CUDA attention kernels must handle the larger `qk_nope`/`v_head` and different head count — at minimum re-parameterization, possibly tiling/shared-memory/register-blocking changes. **Treat this as the main engineering risk, alongside the indexer wiring.**
- **Tokenizer + specials.** New 154880-vocab tokenizer; new special/EOS/pad IDs (`154820/154827/154829`, pad `154820`). DS4 hardcodes vocab size and DeepSeek DSML specials.
- **Chat template + thinking modes.** Implement GLM’s `chat_template.jinja` and GLM’s effort-level thinking format (replacing DeepSeek’s renderer and think/think-max/nothink mapping).
- **Tool-call format.** Replace **DSML** rendering/parsing/canonicalization with GLM’s tool-call format (GLM-4-MoE / glm4moe-style parser, *not* DSML). The exact-replay map stays; the bytes it stores change.
- **GGUF authoring + tensor-name map.** No GLM-5.2 GGUF in DS4’s expected layout exists. Write a converter: GLM safetensors tensor names → DS4’s expected names, with correct MoE expert packing.
- **Quant recipe re-derivation.** See §5.
- **Validation vectors.** Capture official GLM-5.2 logprobs (greedy, thinking off, top-logprobs) at several context lengths; replace `tests/test-vectors`.

-----

## 4. Memory & hardware analysis

**Parameter breakdown (computed from §2):**

- Per routed expert (SwiGLU, moe_inter 2048): gate+up+down = 3 × (6144×2048) ≈ **37.7M**.
- Routed experts: 256 × 37.7M × 75 MoE layers ≈ **725B (~97% of the model).**
- Everything else (attention ~13B, dense MLP ~0.7B, shared experts ~2.8B, embeddings ~1.9B untied, indexer, router, norms) ≈ **~19B.**
- **Total ≈ ~744B**, consistent with the ~755B implied by the 1.51 TB BF16 repo.

The ~97% expert fraction is the key enabler: DS4’s “quantize only the routed experts” strategy covers almost the whole model.

**Weight sizes:**

|Quant                  |Approx weights |Notes                                              |
|-----------------------|---------------|---------------------------------------------------|
|Q2 (experts-only mixed)|**~210–230 GB**|experts ~2.3 bpw + non-experts high precision      |
|Q4 (mixed)             |**~400–440 GB**|experts ~4.5 bpw dominate; needs the 512 GB machine|

**KV cache (the number to get right):**
MLA latent per token per layer = `kv_lora_rank + qk_rope_head_dim` = 512 + 64 = **576 values**. All 78 layers cache it.

- Per token: 576 × 78 ≈ **44,928 values** → ~**88 KB/token at FP16**, ~**44 KB/token at 8-bit**.
- **@ 500k tokens:** ~**44 GB** (FP16) / ~**22 GB** (8-bit).
- **@ 1M tokens:** ~**88 GB** (FP16) / ~**44 GB** (8-bit).
- **Indexer adds only a few GB** thanks to IndexShare (only ~1 layer in 4 keeps a full index; `index_head_dim` 128).

> Correction vs earlier napkin math: an initial estimate of ~13 GB at 500k (extrapolated from the smaller DeepSeek V4 Flash) was **too low**. GLM-5.2 has 78 full-MLA layers, so its KV is materially larger. Use the numbers above and pin the storage precision against ds4 source.

**What fits on the M3 Ultra Mac Studio:**

|Machine|Config                       |Verdict                                                                     |
|-------|-----------------------------|----------------------------------------------------------------------------|
|256 GB |Q2 + ~200–300k ctx (8-bit KV)|Fits; keep the machine lean                                                 |
|256 GB |Q2 + 500k ctx                |**Borderline** (~215 + ~22 KV + ~12 overhead ≈ ~250 GB) — needs quantized KV|
|256 GB |Q2 + 1M ctx                  |**Does not fit** (KV alone 44–88 GB)                                        |
|512 GB |Q2 + 1M ctx                  |Fits comfortably                                                            |
|512 GB |Q4 + long ctx                |Fits (~440 + KV + overhead)                                                 |
|512 GB |native-style INT4 / higher   |Tight to over                                                               |

**Recommendation:** target Q2 for the 256 GB class with context capped ~200–300k and an 8-bit (or better) KV cache; reserve 1M-context and Q4 for the 512 GB machine.

-----

## 5. Quantization strategy

DS4’s Q2 is **not** mechanical — replicate the *philosophy*, re-derive the *specifics*:

1. **Quantize only routed experts.** Leave shared experts, all attention projections, the **router/gate**, embeddings, and norms at high precision. Rationale: an error in the router misroutes *every* token; experts are individually low-traffic and tolerate 2-bit. GLM’s 97% expert fraction makes this especially effective.
1. **Role-aware formats.** DS4 uses `IQ2_XXS` for up/gate and `Q2_K` for down (down writes back into the residual stream and is more sensitive). Start here; re-tune for GLM.
1. **imatrix calibration.** Build an importance matrix from representative activations (include coding-agent traffic, since this is a coding-oriented product) so error concentrates where it doesn’t matter.
1. **Logit-validate.** Gate every recipe change on token-level logprob parity vs the official GLM-5.2 API at multiple context lengths. This is the discipline that makes 2-bit usable under agents.

The IQ2/Q2_K bit-packing kernels and tables come from the llama.cpp/GGML lineage already vendored in ds4 — reuse them; the intellectual work is the **bit allocation**, not the packing.

-----

## 6. Ordered work plan

> Dependencies roughly top-to-bottom. Items marked ⚠️ are the high-risk/high-effort ones.

1. **Repo + license hygiene.** Fork, rename the product, strip DeepSeek branding from CLI/server/README, keep MIT + GGML notices. Update model IDs (`deepseek-v4-flash` → `glm-5.2`).
1. **Land the GLM files locally.** `config.json`, `chat_template.jinja`, tokenizer, `generation_config.json`, and the safetensors. (Weights → external 4 TB volume; see the download note in §8.)
1. **Field-by-field geometry map.** ⚠️ Read ds4 source and map every §2 value to the constant it replaces (loader + KVC session payload). Produce a table: GLM field → ds4 symbol → file:line.
1. **Tensor-name + packing map.** GLM safetensors names → ds4’s expected GGUF names; define MoE expert packing. Write the safetensors→GGUF converter.
1. **Tokenizer + specials.** Swap in GLM’s tokenizer; update vocab size, EOS set, pad.
1. **Attention/indexer kernels.** ⚠️ Adapt Metal + CUDA MLA kernels to GLM’s head dims (64 heads, qk_nope 192, v_head 256, q_lora 2048). Wire the IndexShare pattern (`index_topk_freq` 4, `indexer_types`, `index_skip_topk_offset` 3). Validate against a reference forward pass.
1. **MoE routing.** sigmoid scoring, `noaux_tc`, `routed_scaling_factor` 2.5, `norm_topk_prob`, top-8 of 256, 1 shared, `first_k_dense_replace` 3.
1. **RoPE.** `rope_theta` 8e6, `rope_interleave` / `indexer_rope_interleave` true.
1. **Quantize.** Implement the §5 recipe; produce Q2 (and Q4 for big machines); build imatrix; logit-validate.
1. **Prompt rendering + thinking.** GLM chat template + effort-level thinking modes.
1. **Tool format.** ⚠️ Replace DSML with GLM’s tool-call rendering/parsing; keep the exact-replay map, change the stored bytes.
1. **MTP.** Wire GLM’s single nextn layer into `--mtp`.
1. **Validation vectors.** Capture official GLM-5.2 logprobs; port the `ds4_test` runner.
1. **Server rebrand + agent configs.** Update `/v1/models`, model names, and the opencode/Pi/Claude-Code wrapper examples.

-----

## 7. First actions for Claude Code (do these first)

1. **Clone/open ds4** and produce the **§3 geometry map** as a concrete table (GLM field → ds4 symbol → `file:line`). This is the artifact everything else depends on.
1. **Diff head dims** specifically — locate where the Metal/CUDA MLA kernels hardcode head/dim counts and assess whether GLM’s larger `qk_nope`/`v_head` need only constants or also tiling changes. Report findings before writing kernels.
1. **Locate the indexer code path** and determine whether DS4’s “ratio-4 indexer” maps 1:1 onto GLM’s `indexer_types` / `index_topk_freq` semantics, or differs (esp. the leading dense layers and `index_skip_topk_offset`).
1. **Inventory the tensor names** in the GLM safetensors (`safetensors` header read, no full load) and draft the name-mapping table.
1. **Stand up a CPU reference forward pass** (HF `transformers`, `trust_remote_code`) to generate golden logits for a few short prompts — needed to validate kernels and the converter independently of the official API.

-----

## 8. Open questions / unknowns to resolve

- **KV storage precision in ds4** (drives all the §4 KV numbers). Confirm whether ds4 stores the MLA latent at FP16/FP8/lower, and whether the same applies to the indexer keys.
- **Shared-indexer semantics.** Does GLM’s `shared` indexer reuse the most recent `full` layer’s selection exactly? How are the 3 leading dense layers and `index_skip_topk_offset: 3` handled?
- **Non-expert precision in Q2** (moves the ~210–230 GB figure by tens of GB). Decide Q8 vs F16 for attention/router/embeddings.
- **Shared-expert intermediate size** — assumed `moe_intermediate_size` (2048); confirm from weights.
- **Tokenizer specials / chat-template control tokens** for thinking and tool blocks.
- **Whether ds4 has any dense-attention path at all** (it shouldn’t need one — GLM is sparse/DSA like DeepSeek V4 — but confirm the kernels aren’t hardwired to DeepSeek’s exact sparsity constants).

## 9. Hardware notes (download / build target)

- Weights live on an external **4 TB APFS volume** (`/Volumes/4TB-1`, ~3.9 TB free). Download with `HF_HOME` pointed at that volume so the hub **and** xet chunk caches don’t fill the system disk:
  
  ```bash
  export HF_HOME=/Volumes/4TB-1/hf
  hf download zai-org/GLM-5.2 --local-dir /Volumes/4TB-1/GLM-5.2
  ```
  
  Peak transient usage (xet chunks + assembled files) stays under the free space; reclaim afterwards by removing `$HF_HOME/xet` and `<local-dir>/.cache`.
- Build: `make` (Metal) on macOS; CUDA path is for the Linux/NVIDIA variant. CPU path is debug-only.

## 10. References

- Base engine: <https://github.com/antirez/ds4>
- Target model: <https://huggingface.co/zai-org/GLM-5.2>
- Refusal-direction steering (ds4 `dir-steering`): <https://arxiv.org/abs/2406.11717>
- DeepSeek DSML encoding (the format being *replaced*): referenced from the DeepSeek-V4 model card.

-----

*Prepared as a research/analysis handoff. All parameter and memory figures are computed estimates from the config above and should be confirmed against ds4 source and a reference forward pass before relying on them for kernel or allocation decisions.*