# GLM-5.2 Port — Execution Plan & Tracker

Living tracker for forking ds4 (DeepSeek V4 Flash engine) into a GLM-5.2 engine.
Deep research/analysis lives in [`porting.md`](porting.md); this file is the
**checklist + geometry map + risk register + status log**. Update the status
log at the bottom every working session.

> Premise: a *fork*, not a runtime config switch. We retarget the engine to GLM-5.2
> with no intent to merge upstream (aligns with `AGENT.md`: "do not add permanent
> semantic variants behind flags").

---

## 0. Locations & workflow

| Thing | Location |
|---|---|
| Local code (this machine) | `~/Documents/ds4`, branch **`ds4-glm`** (based on `main`, clean) |
| Remote code (notible) | `notible:/Users/notible/Documents/ds4-glm`, on branch **`ds4-glm`** (kept in sync via `git pull`) |
| Production DeepSeek (do not disturb) | `notible:/Users/notible/Documents/ds4` on `main`, server port 8085 (`jumbo_server`, launchd) |
| Model weights | HF shards `notible:/Volumes/4TB-1/glm-5.2/` (complete); built GGUFs `notible:/Volumes/4TB-1/glm-5.2-q2.gguf` (218.9 GiB) + `glm-5.2-q4.gguf` (408.7 GiB) |
| Sync mechanism | commit on this machine → push `origin` → `git pull` on notible |
| Git remotes | `origin` = `unsaltedbutter-ai/ds4` (fork), `upstream` = `antirez/ds4` |

### 🔴 notible memory / port-8085 server-down gate

The prod DeepSeek server (port 8085, kept alive by a **launchd agent** — a plain SIGTERM
won't free its memory; the agent must be unloaded by the user) can stay up for most of this
work. It only has to come **down** when we load GLM weights into notible's memory / run GLM
on Metal.

- ✅ **Server stays UP:** converter/quantizer runs (CPU + disk, a few GB RAM), all engine
  source edits + `make` builds, small CPU-slice sanity checks.
- 🔴 **Server must come DOWN (flagged before each):** imatrix collection (runs the model on
  Metal), first Metal forward-pass bring-up, any full-model load or Q2/Q4 run on notible.

**Commitment:** flag "this step needs port 8085 down" before any 🔴 step; the user stops the
launchd agent. Phases 0–4 are functionally complete; the gate is now crossed routinely for
full-model Q2/Q4 runs. **Operational pattern that works (use it):** drive each full-model run from
a *detached* script on notible with `trap restart_prod EXIT`, so prod always comes back even if the
SSH/agent connection drops; **wait until `pgrep -f ds4-server` is empty after `--stop`** before
launching ds4 (the launchd `--stop` is async; the instance lock refuses otherwise); each 224 GB Q2
load is ~50–90 s. `setup-launchagents-tts.sh --stop|--start jumbo_server` via `bash -lc`.

Small model config files are already downloaded on notible: `config.json`,
`chat_template.jinja`, `generation_config.json`, `README.md`, `LICENSE`.
**Not yet down:** `model.safetensors.index.json`, `tokenizer_config.json`,
`tokenizer.json`, and most shards.

Build: `make` (Metal) on both machines. CPU path is debug/reference only.
Testing order per `AGENT.md`: Metal default path → SSD streaming → (ask before
distributed/CUDA).

---

## 1. Verified architecture: GLM-5.2 vs DeepSeek V4 (ds4)

From GLM `config.json` (fetched twice, byte-identical) and ds4 source
(`ds4.c:177-249`). **Bold = identical to DS4 Flash → ports directly.**

| Param | GLM-5.2 | DS4 Flash | DS4 Pro | Note |
|---|---|---|---|---|
| Layers | 78 | 43 | 61 | new shape |
| Hidden | 6144 | 4096 | 7168 | new shape |
| Vocab | 154880 | 129280 | 129280 | new tokenizer |
| Attn heads | **64** | **64** | 128 | = Flash |
| **MLA `kv_lora_rank`** | **512** | **512** | 512 | **identical** |
| **`qk_rope_head_dim`** | **64** | **64** | 64 | **identical** |
| `q_lora_rank` | 2048 | 1024 | 1536 | new value |
| `qk_nope_head_dim` / `v_head_dim` | 192 / 256 | (absorbed) | absorbed | q_b/kv_b reshape |
| Output projection | plain o_proj | grouped LoRA (`n_out_group`,`n_lora_o`) | grouped LoRA | **GLM has no output LoRA — new path** |
| **Routed experts** | **256** | **256** | 384 | **= Flash** |
| Experts/token | 8 | 6 | 6 | new value |
| **Shared experts** | **1** | **1** | 1 | **identical** |
| **MoE FFN** | **2048** | **2048** | 3072 | **= Flash** |
| First-k layers | **3 dense** (FFN 12288) | 3 hash-routed | 3 hash-routed | **new dense path** |
| Routing | sigmoid / `noaux_tc` / scale 2.5 | sigmoid / 1.5 | sigmoid / 2.5 | constants + verify |
| Indexer heads / dim / topk | 32 / 128 / 2048 | 64 / 128 / 512 | 64 / 128 / 1024 | + per-layer pattern |
| **Hyper-Connections** | **none (1 stream)** | **4** | 4 | **biggest delta — see R1** |
| RoPE θ / scaling / interleave | 8e6 / none / true | 1e4 / YaRN-16 / tail | 1e4 / YaRN / tail | relax+retarget |
| Context | 1,048,576 | 65,536→YaRN | — | larger |
| MTP layers | 1 | yes | yes | wire GLM weights |
| Weights dtype | bf16 | fp8 (E4M3) | fp8 | converter delta |

## 1b. GLM chat format (from notible's `chat_template.jinja`)

- **Frame:** starts `[gMASK]<sop>`. Roles: `<|user|>`, `<|assistant|>`, `<|system|>`, `<|observation|>`.
- **Thinking:** assistant turn begins `<|assistant|>\n` then `<think>…</think>`; empty `<think></think>` when thinking off. Generation prompt appends `<|assistant|>` + (`<think>` open, or `<think></think>` if disabled).
- **Reasoning effort:** when thinking on, a leading `<|system|>Reasoning Effort: High|Max` (default `max`). Maps to ds4 `DS4_THINK_HIGH` / `DS4_THINK_MAX` (replace the DeepSeek prose prefix at `ds4.c:65`).
- **Tools (NOT DSML):** tool list in a `<|system|>` block inside `<tools>…</tools>`; calls as `<tool_call>{name}<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`; results as `<|observation|><tool_response>…</tool_response>`.
- **Special tokens (strings confirmed; IDs inferred, verify when `tokenizer_config.json` downloads):** `<|endoftext|>`=154820 (eos+pad), `<|user|>`=154827, `<|observation|>`=154829 (the three EOS stops), plus `[MASK]` `[gMASK]` `[sMASK]` `<sop>` `<eop>` `<|system|>` `<|assistant|>`. Default sampling temp 1.0 / top_p 0.95.

---

## 2. Geometry map — GLM field → ds4 symbol → file:line

The artifact everything depends on. Verify each `file:line` before editing (line
numbers from a 35-day baseline; may drift).

| GLM field (value) | ds4 symbol | file:line | Action |
|---|---|---|---|
| hidden_size 6144 | `n_embd` / `DS4_N_EMBD` | ds4.c:144 / 293 | new shape value |
| num_hidden_layers 78 | `n_layer` | ds4.c:143 | new shape value |
| vocab_size 154880 | `n_vocab` | ds4.c:145 | new shape value |
| num_attention_heads 64 | `n_head` | ds4.c:146 | = Flash |
| (MLA single latent kv) | `n_head_kv` = 1 | ds4.c:147 | = Flash |
| kv_lora_rank 512 | `n_head_dim`/`n_value_dim` | ds4.c:148-149 | = Flash (512) |
| qk_rope_head_dim 64 | `n_rot` | ds4.c:150 | = Flash |
| q_lora_rank 2048 | `n_lora_q` | ds4.c:152 | new value |
| output o_proj (no LoRA) | `n_lora_o`,`n_out_group` | ds4.c:151,153 | **new plain-o_proj path** |
| qk_nope 192 / v_head 256 | q_b / kv_b shapes | ds4.c:~6742,6781 | reshape projections |
| n_routed_experts 256 | `n_expert` | ds4.c:154 | = Flash |
| num_experts_per_tok 8 | `n_expert_used` | ds4.c:155 | 6 → 8 |
| n_shared_experts 1 | `n_expert_shared` | ds4.c:156 | = Flash |
| moe_intermediate_size 2048 | `n_ff_exp` | ds4.c:157 | = Flash |
| intermediate_size 12288 (dense) | (none — hash layers) | ds4.c:~3043 | **new dense-FFN layer type** |
| first_k_dense_replace 3 | `n_hash_layer` | ds4.c:158 | dense, not hash |
| routed_scaling_factor 2.5 | `expert_weight_scale` | ds4.c:167 | 1.5 → 2.5 |
| scoring sigmoid/noaux_tc | router gate path | ds4.c:~ (MoE) | verify matches |
| index_n_heads 32 | `n_indexer_head` | ds4.c:160 | 64 → 32 |
| index_head_dim 128 | `n_indexer_head_dim` | ds4.c:161 | = Flash |
| index_topk 2048 + pattern | `n_indexer_top_k` | ds4.c:162 | + per-layer types |
| (hyper-connections) | `n_hc`=4, `DS4_MAX_HC` | ds4.c:131,163,200 | **set 1 / strip — R1** |
| rope_theta 8e6 | `rope_freq_base` (validated ==1e4) | ds4.c:206 / ~3978 | relax + set |
| rope scaling none | yarn factor/betas | ds4.c:207-209 | disable YaRN |
| rope_interleave true | tail/NeoX rope | metal/dsv4_rope.metal | interleave support |
| max_position 1M | `rope_orig_ctx` 65536 | ds4.c:211 | adjust |
| num_nextn_predict_layers 1 | MTP path | ds4.h:276-277 | wire GLM MTP |
| eos_token_id [3] | `ds4_token_eos` (single) | ds4.c:~22267 | multi-stop set |
| Shape gate (exit on mismatch) | `ds4_select_shape_from_metadata` | ds4.c:3736-3794 | add `DS4_SHAPE_GLM` |
| compress_ratios validation | `validate_compress_ratio_metadata` | ds4.c:3796-3830 | all-zero (no compressor) |
| routed expert tensor check | `tensor_expect_routed_expert` | ds4.c:~3626 | reuse (256 experts) |
| per-expert byte offset (streaming) | expert loader | ds4.c:~12917-12956 / ds4_ssd.c | reuse layout |
| tokenizer / specials / chat encode | bpe + `encode_chat_prompt` | ds4.c:~22134-22432 | GLM template |
| tool DSML render/parse | `dsml_syntaxes[]` | ds4_server.c:~4214,5245 | GLM `<tool_call>` format |
| HF→GGUF quantizer | `deepseek4-quantize.c` | gguf-tools/ | bf16 read + GLM names |

## 2b. Verified GLM tensor schema (from shard headers, 2026-06-16)

Per-layer names/shapes (all BF16) the converter maps to ds4's GGUF tensors:

| GLM tensor | shape | ds4 target / note |
|---|---|---|
| `embed_tokens.weight` | [154880,6144] | token embeddings |
| `lm_head.weight` | [154880,6144] | output head (untied) |
| `layers.N.input_layernorm` | [6144] | attn pre-norm |
| `layers.N.post_attention_layernorm` | [6144] | ffn pre-norm (**only 2 norms/layer — confirms no HC**) |
| `self_attn.q_a_proj` | [2048,6144] | q down (q_lora 2048) |
| `self_attn.q_a_layernorm` | [2048] | q latent norm |
| `self_attn.q_b_proj` | [16384,2048] | q up (64h × 256) |
| `self_attn.kv_a_proj_with_mqa` | [576,6144] | kv down (512 latent + 64 rope) |
| `self_attn.kv_a_layernorm` | [512] | kv latent norm (latent only, not rope) |
| `self_attn.kv_b_proj` | [28672,512] | kv up (64h × (192 nope + 256 v)) |
| `self_attn.o_proj` | [6144,16384] | **plain o_proj (delta A)** |
| `self_attn.indexer.wq_b` | [4096,2048] | DSA idx q (32h × 128) — v1: skip |
| `self_attn.indexer.wk` | [128,6144] | DSA idx key — v1: skip |
| `self_attn.indexer.k_norm.{weight,bias}` | [128] | DSA idx norm (+bias, unlike ds4) — v1: skip |
| `self_attn.indexer.weights_proj` | [32,6144] | DSA idx head weights — v1: skip |
| `mlp.{gate,up}_proj` (L<3) | [12288,6144] | **dense SwiGLU (delta C)** |
| `mlp.down_proj` (L<3) | [6144,12288] | dense down |
| `mlp.experts.{0..255}.{gate,up}_proj` (L≥3) | [2048,6144] | **256 separate tensors → STACK** into ds4 3-D expert layout |
| `mlp.experts.{0..255}.down_proj` (L≥3) | [6144,2048] | stack |

**Converter's biggest job:** experts ship as 256 separate tensors per projection per MoE
layer; ds4 expects them stacked `[blocks, features, expert_id]`. **Confirmed by
`glm-quantize --dry-run` (2026-06-16):** router `mlp.gate.weight` → `ffn_gate_inp`; noaux_tc
bias `mlp.gate.e_score_correction_bias` → `exp_probs_b.bias`; shared expert
`mlp.shared_experts.{gate,up,down}_proj` → `ffn_*_shexp`. DSA indexer weights live on only
~20/78 layers (the `full` indexer layers; `shared` layers reuse them — matches IndexShare).
Download complete (2026-06-16): `model.norm.weight` present; MTP (`nextn`) is **layer index 78**
with `eh_proj`, `enorm`, `hnorm`, `shared_head.norm` (+ its own attention/experts) — for Phase 5.

---

## 3. Risk register (ranked)

**Risk posture (the project-killer):** #1 is **forward-pass numerical correctness**, concentrated
in the absorbed-MLA attention (no fallback; subtle errors are plausible-but-wrong). The fold
algebra is now **PROVEN** (2026-06-16, `check_mla_absorption.py`, rel err ~1e-15); residual
correctness risk is semantic-convention matching, validated end-to-end vs the official GLM API
once the engine loads. #2 (risk to the *goal*, not the mechanics) is **Q2 quality** — has
fallbacks (Q4/streaming, smaller ctx), and is moot until correctness holds.

- **R1 — Hyper-Connections bypass (MED — RESOLVED by spike, was the gating unknown).**
  ds4's residual stream uses HC (`n_hc=4`, sinkhorn, trained per-layer tensors
  `hc_attn_*`/`hc_ffn_*`/`output_hc_*`). GLM has standard residuals and **no HC tensors**
  (every layer is just `input_layernorm` + `post_attention_layernorm` — verified from shard
  headers), so we do not run "n_hc=1 with weights" — we **bypass HC entirely** and fall back
  to the standard pre-norm residual (`h += sublayer(norm(h))`). Spike verdict: not a rewrite;
  ~6-8 HC-disabled sites (see §3c), low correctness risk; gate on CPU logit parity.
- **R2 — KV "compression" is trained, GLM lacks it (MED, design).** ds4 ratio-2/4
  compression uses trained `attn_compressor_*`/`indexer_compressor_*` tensors absent in
  GLM. Run `compress_ratio=0` everywhere (raw MLA latent KV; loader already supports it).
  Disk-KV persistence still works, just larger/token. Fine for our sub-1M workloads.
- **R3 — RoPE correctness (MED).** θ 1e4→8e6, drop YaRN, **interleaved vs NeoX** layout.
  Silent corruption if wrong; gate on golden-logit parity.
- **R4 — Indexer/DSA semantics (MED).** GLM 32 heads, topk 2048, per-layer `indexer_types`
  (1 full : 3 shared, `index_topk_freq:4`, `index_skip_topk_offset:3`). ds4's "ratio-4
  indexer" is tied to KV-compression-ratio-4, **likely NOT the same as GLM's
  every-4th-layer full index** (porting.md may be optimistic here). **Recommendation:
  disable the indexer for v1, run dense MLA attention** (correct superset; fine for our
  contexts). Re-add DSA later.
- **R5 — Output projection (MED).** GLM uses plain o_proj; ds4 uses grouped output LoRA.
  New attention-output path needed.
- **R6 — Memory fit (MED).** Q2 ≈ 210-225 GB weights. Fully resident on 256 GB is too
  tight with usable context → **SSD streaming is the target config** (resident ~20 GB
  non-routed + large hot-expert cache). Q4 (~400 GB) does not fit single-box.
- **R7 — Converter (MED).** GLM ships bf16 (ds4 quantizer assumes DeepSeek FP8);
  new tensor-name map + template GGUF. Needs `index.json` (pending download).
- **R8 — Tokenizer/template/tools (LOW-MED).** Pre-tokenizer regex, specials, chat
  render, `<tool_call>` parser. Mechanical but broad.

## 3b. M1 — "Smart Q2" memory-fit study (fit 256 GB with a real context window)

Goal: Q2 resident-enough on a 256 GB M3 Ultra to allow a *significant* context, not
just a token. With SSD streaming the model bytes can exceed RAM, so the real budget is:
`resident non-routed (~20 GB) + KV(ctx) + expert cache (tunable) + scratch ≤ ~248 GB`
(raise `iogpu.wired_limit_mb` toward ~248k). Levers to investigate, biggest first:

1. **Quantized KV cache (biggest lever).** GLM runs uncompressed MLA latent
   (`kv_lora_rank+rope` = 576 vals/layer/token, 78 layers ≈ 88 KB/tok @FP16). Store the
   latent at **8-bit** (ds4 already has FP8 KV quantize kernels, `metal/dsv4_kv.metal`)
   → ~44 KB/tok: 200k ctx ≈ 9 GB, 500k ≈ 22 GB. This is what makes "Q2 + large ctx" fit.
2. **Expert-cache vs context tradeoff.** Full Q2 experts ≈ 202 GB. With 8-bit KV at
   ~200-300k ctx there's room to cache nearly all experts (near-resident speed); push ctx
   higher → shrink cache → more cold streaming (slower). Produce a ctx↔cache↔speed table.
3. **Bit allocation tuning.** Default IQ2_XXS gate/up + Q2_K down (~2.23 bpw). Options:
   selectively keep a few sensitive MoE layers (first/last) at Q4 (+RAM, +quality), or
   push down toward IQ2 (−RAM, −quality). Gate every change on logit parity.
4. **Trim non-expert precision.** Keep router + attention at Q8_0; consider Q6/Q5 for
   embeddings + output head (untied, ~1.9 B params) to reclaim a few GB cheaply.

Deliverable: a recommended 256 GB recipe (quant mix + KV precision + cache budget) and the
ctx/speed table, validated against the Q8 reference.

> **Q2 *quality* (distinct from fit) now has its own doc: [`README_GLM_Q2.md`](README_GLM_Q2.md)** —
> how the current Q2 was built (bf16→2-bit direct; IQ2_XXS g/u synthetic-imatrix + Q2_K down unweighted),
> a run battery showing the looping/decoherence failure mode, and ranked better-Q2 options (real imatrix,
> down-weighting, bit-allocation, new low-bit types, 8-bit KV) with a test plan + results tracker.

## 3c. Spike results (2026-06-16) — HC bypass + attention/FFN deltas

**HC bypass (R1).** Hidden state is a flat `[n_hc*n_embd]` buffer (`ds4.c:~8364`),
stream-major. CPU loops mostly use dynamic `n_hc`; Metal has `n_hc==4` fast paths with live
scalar fallbacks, plus two fused decode kernels that hard-reject `n_hc!=4` and auto-downgrade
to unfused (`ds4_metal.m:~25862,~25981`). HC consumes trained tensors GLM lacks. HC-disabled
path = these sites:
1. Per-layer `hc_pre_*` → plain `RMSNorm(h)` for sublayer input (`ds4.c:~6463,~6497`).
2. Per-layer `hc_post_one` → plain residual add (`ds4.c:~6512`).
3. Embedding expand `hc_from_plain_embedding` → identity / single stream (`ds4.c:~6504`).
4. Output head `output_hc_*` collapse → skip (hidden already `[n_embd]`).
5. Tensor validation: skip required HC tensors when disabled (`ds4.c:3588-3622`).
6. Metal: accept the unfused fallback (drop the `n_hc==4` requirement) — no kernel rewrite.
Fixed buffers `post[4]`/`comb[16]` are safe over-allocations. Session payload does NOT store
HC state, so no payload version bump for this.

**Attention/FFN deltas (effort).**
- **A. Output projection — LOW.** Replace grouped 2-stage output LoRA (`layer_grouped_out_one`
  `ds4.c:7094-7142`; Metal `ds4_metal.m:16051-16380`) with a single o_proj matmul
  `[64*256=16384 → 6144]`. Reuses existing matmul; ~5-10 CPU lines, no new kernel.
- **B. Q/KV up-projection — MED, now characterized (correction to earlier "mostly converter").**
  ds4 packs each head as `n_head_dim=512 = 448 nope + 64 rope (tail)`, value = the same latent,
  grouped output LoRA (`ds4.c:6742-6794, 7057-7112`; rope tail `6856-6889`). GLM keeps
  `kv_lora_rank=512` **plus** a separate `qk_rope=64` (`kv_a_proj_with_mqa` is 576-wide) and
  `v_head_dim=256` ≠ key dim. So GLM in absorbed form ⇒ engine **`n_head_dim=576` (512 c_kv +
  64 rope)**, **`n_value_dim=512`**, **partial `kv_a_norm` (512 of 576)**, rope on the tail 64.
  Converter folds `kv_b_k`→`q_b` (giving `attn_q_b [2048, 64*576]`) and `kv_b_v`→output (plain
  absorbed o_proj); `attn_kv` maps directly from GLM `kv_a_proj_with_mqa`. So this is real
  engine work (head_dim/value split + partial norm), not just a converter fold. **VALIDATED
  2026-06-16** (`gguf-tools/check_mla_absorption.py`): native un-absorbed == absorbed to ~1e-15
  on layers 0/3/40, and the weight-level `attn_q_b` fold reproduces the absorbed query exactly.
  Remaining: confirm ds4's value axpy uses `n_value_dim` (512) not `n_head_dim` (576); semantic
  conventions (rope interleave / norm scope) validated end-to-end vs the GLM API later.
- **C. Dense first-3 layers — LOW-MED.** No dense FFN path exists today (only routed-MoE +
  shared-expert SwiGLU `layer_shared_ffn_one` `ds4.c:7184-7216`, intermediate hardcoded to
  `n_ff_exp`). Add `layer_dense_ffn_one()` reusing the SwiGLU kernel at `intermediate=12288`,
  branched on `il < first_k_dense_replace`.

---

## 4. Phased plan

Legend: [x] done · [~] partial · [ ] not started.

### Phase 0 — Groundwork — ✅ DONE
- [x] Architecture investigation + verified config + geometry map (this doc + porting.md)
- [x] GLM chat template + special tokens (§1b)
- [x] Git sync (commit → push `ds4-glm` → `git pull` on notible)
- [x] `DS4_SHAPE_GLM` profile + shape gate (GGUF loads, reports correct dims)
- [x] R1 spike: HC-bypass path (§3c); attention/FFN graph-delta map (§3c); tensor-name schema (§2b)
- [x] Converter skeleton + `--dry-run` coverage
- [ ] `CLAUDE.md` for the fork (low priority)

### Phase 1 — Converter + load — ✅ DONE
- [x] Tensor-name map; bf16 reader; absorbed-MLA fold **proven** (`check_mla_absorption.py`)
- [x] GGUF schema contract (§5c); converter writer (metadata + tokenizer + absorption + expert stacking + quant)
- [x] Model loads; `ds4 --inspect` binds + validates all 1236 tensors on the real 753 B model

### Phase 2 — CPU numerical bring-up — ✅ DONE
- [x] HC bypass, dense first-3, plain o_proj, GLM RoPE (interleaved), indexer off — **full 78-layer CPU forward, finite/sane logits**
- [~] Reference: matched against ds4's own CPU forward (HF bf16 won't fit 256 GB); **exact-vs-official-API logprob vectors still TODO**

### Phase 3 — Metal + multi-token + server — ✅ DONE
- [x] **Metal single-token decode graph (HC=1) — VALIDATED vs CPU** (2026-06-17): `--metal-graph-full-test` gpu_top==cpu_top==154822, logits rms-diff ~0.01. On GPU: HC bypass, 576/512 absorbed attention (no sinks, kq 1/√256, plain o_proj, partial kv_a_norm, no inverse-rope), dense FFN <3, shared expert, output head. Kernel: `kernel_flash_attn_ext_vec_f16_dk576_dv512` (NE=2).
- [x] **8-expert Metal routed MoE — VALIDATED vs CPU** (2026-06-17): routed experts (≈97% FLOPs) now run on the GPU. The baseline id-based MoE path turned out to be already n_expert-general (per-`(expert,token)` dispatch; `ds4_gpu_encode_moe_sum_experts` reduces any count via a binary add chain) — **no `sum8` kernel was needed**; the only 6-hardwiring was the `n_expert > 6` guard, now `> DS4_GPU_MAX_EXPERT_USED`. GLM decode layer selects on the CPU (`metal_graph_glm_cpu_router`, sigmoid noaux_tc top-8 -> `g->router_selected/weights`) and runs experts on Metal via `ds4_gpu_routed_moe_one_tensor`. `--metal-graph-full-test` on Q2: gpu_top==cpu_top==154822, logits rms-diff 0.0136. `DS4_GLM_CPU_ROUTED_MOE=1` forces the v1 all-CPU path (A/B fallback). DeepSeek 6-expert fast paths bit-identical.
- [x] **Multi-token generation WORKS (greedy + sampled, no crash):** full-attention KV (`n_swa=0`) + sequential-decode prefill (both the standalone argmax driver and the **session** path) + multi-EOS; the multi-row attention Q-stride bug is **fixed** (`--glm-attn-test` exact to n_raw=256) and the session/sampled SIGSEGV (GLM-unadapted Metal batch prefill) is **fixed** (GLM session prefill now sequential). `--temp 0` gives coherent correct text (~8.7 tok/s); sampled needs `--top-p 0.95` (GLM's default; the ds4 default top_p=1.0 does no filtering -> noisy on Q2). Speed follow-ups: GLM Metal **batch** prefill (sequential is O(prompt) single-token evals) and a GPU router to drop the per-MoE-layer sync.
- [x] Tokenizer + chat template: `[gMASK]<sop>` framing + reasoning-effort + `<|assistant|><think>` gen prompt + multi-EOS stop set **ID-verified vs the real tokenizer**; **`<tool_call>` render/parse + multi-turn server chat builders (`ds4_chat_*`) + server `render_chat_prompt_text` GLM framing all landed** (gated on `variant==GLM`, DeepSeek bit-identical; 8 no-model unit tests in `./ds4_test --server`)
- [x] 🔴 **Server answers a real prompt end to end — VALIDATED on notible** (2026-06-17): GLM-Q2 server (`ds4-server -m glm-5.2-q2.gguf -c 2048`) renders the correct GLM frame (`--trace`: `[gMASK]<sop><|system|>Reasoning Effort: High<|user|>...<|assistant|><think>`), answers a prompt **coherently** (temp 0.6/top_p 0.95: correct planet list). Default sampling now temp 0.6/top_p 0.95 (temp 1.0 degenerates on Q2 — see §6).
- [x] **GPU sigmoid router — VALIDATED** (2026-06-17): drops the per-MoE-layer CPU↔GPU sync. `ds4_gpu_router_select_glm_tensor` threads `sigmoid_scoring` into `ds4_gpu_encode_router_select` (bias/top-k/normalize/scale are scoring-agnostic). Numerically identical to the CPU router (logits_rms 0.01356, gpu_top==cpu_top); **Q2 generation 8.9 → 11.6 tok/s**. `DS4_GLM_CPU_ROUTER=1` forces the host router (A/B). DeepSeek path untouched (+ CUDA stub).

### Phase 4 — Quant builds, streaming, Q2-vs-Q4 — ✅ Q2 fast; Q4 streamed (disk-bound) ← **WE ARE HERE (speed)**
- [x] **Q2 GGUF** (`/Volumes/4TB-1/glm-5.2-q2.gguf`, 218.9 GiB) + **Q4 GGUF** (408.7 GiB) built, `--inspect`-validated. **Q2 is the fast resident target: ~11.6 tok/s, the recommended serving config.**
- [x] **Q2-vs-Q4 quality settled** (2026-06-18): on the same greedy long prompt Q4 stays coherent (planets + a fact each) where Q2 loops → Q2's degeneration is the **2-bit quantization ceiling, not an engine bug** (with CLI==server + determinism + Metal==CPU, the engine is exonerated; §6).
- [x] **GLM Q4 GPU expert streaming — FUNCTIONAL on CLI + server** (2026-06-18): `slots8` Q4_K kernels + `ds4_gpu_glm_streaming_routed_moe_tensor` stream the 8 routed experts/layer from the mmap into the GPU under `--ssd-streaming` (else CPU routed fallback). Correct, server-validated. **Disk-bound: ~0.42 cold / 0.76 warm tok/s** — 389 GiB of experts can't be resident on 256 GB, so the per-layer expert load dominates.
- [ ] **Q4 streaming SPEED** (the active speed work): a **non-thrashing GLM-native LRU expert cache** (the DeepSeek 6-expert slab reuse thrashed at 0.06 tok/s → reverted; kept in git) exploiting expert-usage skew + **load/compute overlap**. Fundamentally disk-capped (~1–2 tok/s ceiling — a slice of experts is always cold).
- [ ] **8-bit latent KV** (§3b; ds4 has FP8 KV kernels `metal/dsv4_kv.metal`) — shrinks KV (~88→44 KB/tok) for larger Q2/Q4 context **and frees RAM for the Q4 expert cache**. Independent of the disk wall — the higher-leverage next lever.
- [ ] imatrix calibration (currently synthetic per-column-energy importance) — real activation imatrix for better Q2.

### Phase 5 — Later
- [ ] Re-enable DSA indexer (IndexShare) for long context
- [ ] Wire GLM MTP into `--mtp`
- [ ] GLM validation vectors (official API); rebrand CLI/server/README; license hygiene

---

## 5. Open questions
- Special token numeric IDs (pending `tokenizer_config.json` download; inferred in §1b).
- **R1:** does ds4 run correctly at `n_hc=1`, or is HC=4 baked into kernel shapes?
- Indexer "shared" semantics vs ds4 ratio-4; handling of 3 leading dense layers + `index_skip_topk_offset:3`.
- Norm placement (GLM post-/sandwich-norm or QK-norm beyond MLA latent norms?) — confirm vs HF.
- RoPE interleave convention exact match.
- Shared-expert intermediate size (assume `moe_intermediate_size` 2048 — confirm from weights).
- Non-expert precision in Q2 (Q8 vs F16 for attention/router/embeddings) — moves total by tens of GB.
- KV-cache storage precision in ds4 today (FP16 vs FP8 latent) and whether 8-bit KV is wired for the uncompressed (`compress_ratio=0`) path — drives the M1 context math.
- Does building **both Q2 and Q4** need two converter runs from the same imatrix, or can one pass emit both? (affects Phase 4 time.)
- **Golden reference strategy:** full GLM-5.2 bf16 (~1.5 TB) won't fit a 256 GB box, so we can't run a local HF reference of the whole model. Plan: official GLM API for end-to-end logprob vectors + a **CPU slice** (first few layers) for intermediate-tensor parity during Phase 2.

---

## 5c. GGUF schema contract (GLM) — DRAFT

`config_validate_model` (ds4.c:3888-3997) requires these `deepseek4.*` keys and checks each
**== the selected shape's constant**. So the contract = add `DS4_SHAPE_GLM` with these values,
emit matching metadata, and gate behavior on `variant==GLM`.

| metadata key | GLM value | note |
|---|---|---|
| block_count / embedding_length / vocab_size | 78 / 6144 / 154880 | |
| attention.head_count / head_count_kv | 64 / 1 | absorbed MQA-in-latent |
| attention.key_length / value_length | **576 / 512** | key=512 c_kv+64 rope; value=c_kv only |
| rope.dimension_count / q_lora_rank | 64 / 2048 | |
| attention.output_lora_rank / output_group_count | 0 / 1 | sentinels: plain o_proj |
| expert_count / used / ff_length / shared | 256 / 8 / 2048 / 1 | |
| hash_layer_count | 0 | GLM dense, not hash |
| expert_group_count / used | 0 / 0 | n_group=1 |
| attention.sliding_window | TBD | GLM full attn — verify ratio-0 raw-KV sizing |
| attention.indexer.{head_count,key_length,top_k} | 32 / 128 / 2048 | nominal; v1 skips indexer |
| hyper_connection.count / sinkhorn_iterations | 1 / 0 | HC bypassed |
| rope.freq_base / scaling.factor | 8e6 / 1.0 | no YaRN |
| attention.compress_rope_freq_base | nominal | compression off |
| expert_weights_scale | 2.5 | routed_scaling_factor |
| attention.layer_norm_rms_epsilon | 1e-5 | (ds4 was 1e-6) |
| hyper_connection.epsilon | nominal | HC off |
| expert_weights_norm | true | norm_topk_prob |
| compress_ratios[78] | all 0 | need `ds4_expected_layer_compress_ratio`→0 for GLM |
| swiglu_clamp_exp[78] | disable | GLM plain SwiGLU, no clamp |
| + NEW first_k_dense_count | 3 | new key for dense-first-3 |

**Tensors emitted** (`blk.N.`): `attn_norm`, `ffn_norm`, `attn_q_a`, `attn_q_a_norm`, `attn_q_b`
(absorbed `[2048,64*576]`), `attn_kv` (=`kv_a_proj_with_mqa` `[6144,576]`), `attn_kv_a_norm` (`[512]`,
partial), `attn_output` (plain absorbed `[64*512,6144]`); MoE≥3: `ffn_gate_inp`, `exp_probs_b.bias`,
`ffn_*_shexp`, `ffn_{gate,up,down}_exps` (stacked 256); dense<3: `ffn_{gate,up,down}_dense`; top:
`token_embd`, `output_norm`, `output`. **NOT emitted** (engine skips for GLM): `hc_*`/`output_hc_*`,
`attn_compressor_*`, `indexer*`, `attn_sinks`, `ffn_gate_tid2eid`, `attn_output_a/b`.

**Engine deltas gated on `variant==GLM`** (the bulk of the work): HC bypass; plain `attn_output`;
dense FFN <3; attention key=576/value=512 + partial `kv_a_norm`; full-attention KV (compression off);
GLM tokenizer/template/`<tool_call>`; multi-EOS. `general.architecture` reuses `deepseek4` (unvalidated).

## 7. Metal-graph port — remaining work to run GLM at SPEED

The CPU reference forward is GLM-complete and validated (all 78 layers, finite logits). To run
GLM on notible at usable speed, the **Metal graph** (`ds4_metal.m` + `metal/*.metal`) needs the
SAME variant-gated deltas, mirroring the CPU ones. Each below = a known site + the change.
Quant kernels (IQ2_XXS/Q2_K/Q4_K MoE, Q8 matmul) already exist; this is graph wiring + a few
kernel params. Gate everything on `DS4_MODEL_VARIANT == DS4_VARIANT_GLM`.

1. **HC bypass.** Metal HC pipelines: `g_hc_split_sinkhorn_pipeline`, `g_hc_split_weighted_sum[_norm]`,
   `ds4_gpu_hc_*` encoders, `kernel_dsv4_hc_*` (`metal/dsv4_hc.metal`), and the fused decode HC
   (`ds4_metal.m` ~25862, already `if (n_hc!=4) return 0` → unfused fallback). For GLM (`n_hc=1`):
   the per-layer graph must replace `hc_pre`(split+weighted_sum) with **just the RMSNorm** of the
   single `[n_embd]` residual, and `hc_post` with a **plain residual add** (block_out + residual).
   Skip binding/encoding all `hc_*`/`output_hc_*` tensors. Hidden buffer is `[n_embd]` not `[4*n_embd]`.
2. **Attention 576/512.** Flash-attn kernels are templated `dk512_dv512` (`metal/flash_attn.metal`).
   GLM needs **key/score dim 576, value dim 512** — add a `dk576_dv512` template (or a GLM MLA path),
   softmax scale **1/√256** (not 1/√head_dim), and **no attn_sinks** (GLM lacks them). The decoupled
   rope tail (`kernel_dsv4_rope_tail_f32`) applies to dims [512:576]; **skip the inverse-rope on the
   value** for GLM (value = rope-free c_kv). `kv_a_norm` normalizes only the first 512 of the 576 latent.
3. **Plain o_proj.** Grouped output LoRA (`ds4_gpu_attention_output_q8_batch`, `ds4_attention_output_group_ids`)
   → GLM plain matmul `attn_output [64*512=32768 → 6144]` (reuse the Q8 matmul, no grouping).
4. **Dense FFN (layers <3).** `metal_graph_encode_layer_ffn_batch` → GLM dense layers run a SwiGLU on
   `ffn_{gate,up,down}_dense` (intermediate 12288); skip routed MoE + shared + router for those layers.
5. **Router scoring — DONE (2026-06-17, GPU; see §6).** Threaded `sigmoid_scoring` into
   `ds4_gpu_encode_router_select` (`g_unary_sigmoid_pipeline` vs `sqrt(softplus)`); the noaux_tc bias +
   flat top-8 + normalized × 2.5 are scoring-agnostic. New public `ds4_gpu_router_select_glm_tensor`
   (+ CUDA stub) selects on the GPU with no per-MoE-layer CPU sync; validated numerically identical to the
   CPU router (rms 0.01356 both) and lifts generation 8.9 -> 11.6 tok/s.
6. **Output head.** Skip the Metal HC collapse (`output_hc_fn`); GLM does final RMSNorm(`output_norm`) +
   `output` matmul only.
7. **Raw-KV path / no compression.** GLM runs `compress_ratio=0` (raw latent KV). Verify the Metal raw-KV
   store/read (`metal/dsv4_kv.metal`) works at `head_dim=576`/`value=512` (it stores the 576 latent; value
   reads first 512). The FP8/compressor and the DSA indexer paths are skipped for GLM v1.
8. **SSD streaming** (for Q2 + big context on 256 GB): the routed-expert streaming cache is MoE-structural
   and should apply directly once the Metal forward runs (same 256-expert stacked layout); tune per §3b.

Shared fixes already in place (help both backends): `DS4_SHAPE_GLM`, `DS4_MAX_EXPERT_USED=8`, load path.
Validation: once the Metal forward runs, compare its logits to the CPU forward (ds4 has metal-graph-vs-cpu
test harnesses) on a few tokens — they should match within quant/precision noise.

## 7b. Known gaps / cautions for whoever continues

> **UPDATE 2026-06-18 — §7/§7b/§7c are now LARGELY HISTORICAL; read the §6 status-log top first.**
> The Metal bring-up these sections planned is **done and validated on notible**: single-token decode,
> 8-expert MoE, multi-token generation (greedy + sampled), full chat I/O + `<tool_call>` render/parse,
> the **HTTP server** (answers GLM prompts end to end), the **GPU sigmoid router** (~11.6 tok/s on Q2),
> and **Q4 GPU expert streaming** (functional, disk-bound ~0.76 tok/s warm). The remaining work is
> **speed/scale**, not bring-up: a non-thrashing GLM expert cache + load/compute overlap for Q4 (which
> is fundamentally SSD-bound on 256 GB), 8-bit latent KV, and DSA/MTP (Phase 5). The bullets below are
> kept only as historical context; trust §6 + §4 over them. NOTE: the "CPU generation NOT done" bullet
> is **stale** — the Metal generation path (CLI + server) is the working release path; the CPU backend
> remains debug/reference only.

- **RoPE convention: VERIFIED OK for GLM.** ds4's `rope_tail` (ds4.c ~7010, `kernel_dsv4_rope_tail_f32`)
  rotates *adjacent pairs* `tail[i],tail[i+1]` — the **interleaved** convention, which matches GLM's
  `rope_interleave: true` (on the 64 rope dims at the head tail [512:576], θ=8e6 set, YaRN off since
  `rope_scale_factor=1` → `freq_scale=mscale=1`). So rope is correct at all positions, not just pos 0.
  (Still worth a multi-token sanity check once generation runs, but no convention mismatch.)
- **CPU generation (prefill+decode) is NOT done.** `generate_raw_swa_cpu` → `prefill_layer_major_cpu` + a
  decode loop use BATCH attention variants (ds4.c ~9086/9149/9215 `layer_attention_prefix_batch`, ~9549
  `layer_attention_raw_swa_batch`) and decode-scratch fns (`layer_kv_projection_normed_one_decode_scratch`
  ~6971, `layer_grouped_out_one_decode_scratch`) that still need the GLM deltas (no sinks, value=512 / key=576,
  kq_scale 1/√256, skip inverse-rope, plain o_proj, partial kv_a_norm). PLUS the **KV cache**: GLM `n_swa=0`
  means the raw-KV sliding window retains 0 rows → must make GLM retain ALL positions (full attention). Until
  these land, only the single-token forward runs.
- **MoE routing math** (sigmoid + `exp_probs_b` bias + flat top-8 + normalized×2.5) matches GLM's noaux_tc on
  inspection and produces finite logits, but is **not bit-checked** vs a reference. `kq_scale=1/√256` is the
  standard MLA value (high confidence) but unverified-exact.
- The first-token-test top token for a bare OOD single token is `[gMASK]` (GLM's sequence-start token) —
  expected, not a bug; a real continuation needs `[gMASK]<sop>` framing + prefill.

## 7c. Metal decode-path implementation spec (code-grounded, 2026-06-17)

Reverse-engineered against the current source (line numbers verified this session). This turns §7
into a mechanical, site-by-site spec for the **single-token decode** Metal graph (the milestone:
"HC bypass + 576/512 attention, validated vs CPU"). Prefill/batch + multi-token KV retention are a
later milestone.

**Design decision — isolated GLM functions, DeepSeek/CUDA hot paths bit-identical.** The decode-heads
and router-select GPU entry points have CUDA implementations (`ds4_cuda.cu`), so changing their
signatures would break the (here-untestable) CUDA build. So: add *new* GLM Metal functions, leave
existing ones untouched, and give each new public GLM GPU fn a CUDA **stub** (`return 0`) so the CUDA
build keeps linking. CUDA+GLM is unsupported for now — flag to user per AGENT.md (no CUDA box to test).

**Validation harness = `--metal-graph-full-test`, NOT `--metal-graph-test`.** `metal_graph_decode_test`
(ds4.c:16770, the single-layer-0 per-stage diff) is hardwired to MoE expert tensors and an
*unconditional* inverse-rope on heads (16848) → wrong for GLM (layer 0 is dense; GLM skips inverse-rope;
GLM layer-0 has no expert tensors). Use `metal_graph_first_token_full_test` (ds4.c:17013): it compares
each layer's GPU hidden state to `layer_forward_self_one` (the validated GLM CPU forward). Env knobs:
`DS4_METAL_GRAPH_TRACE_LAYERS=1` (per-layer max/rms diff), `DS4_METAL_GRAPH_TEACHER_FORCE=1` (reset GPU
to CPU state each layer → isolates per-layer error, no accumulation), `DS4_METAL_GRAPH_TRACE_STAGE_LAYER=N`
(+`metal_graph_trace_layer_stages`, stage diffs for layer N). NB: single token at pos 0 ⇒ 1 KV row, so
attention softmax is degenerate (weight 1.0) — it still validates dispatch/scale/value-extract/no-sinks,
but multi-row softmax is inherited from the proven 512/512 kernel body.

**Sites:**

1. **Flash-attn kernel — DONE this session.** `kernel_flash_attn_ext_vec_f16_dk576_dv512` added to
   `metal/flash_attn.metal` (vec template `<...,576,512,1>`, signature-only `flash_attn_ext_vec_t`,
   static_asserts pass: 576%32=18, 512%32=16). Prefill non-vec `dk576_dv512` deferred (decode milestone
   uses only the vec kernel). RISK: register/threadgroup budget for DK=576 only surfaces at
   `newComputePipelineState` time (the gated run), not at metal-source compile.

2. **GLM raw-heads encoder** — clone `ds4_gpu_encode_flash_attention_raw_heads` (ds4_metal.m:16793) as
   `_glm`, isolated (drop the `head_dim!=512` reject at 16805). Dim split (the whole point):
   - **KEY dims = 576** (q is 576-wide incl rope tail; K/V share one f16 buffer with 576-wide rows):
     `q_bytes=n_head*576*4`, `raw_bytes=raw_cap*576*4`, ring buffer, f16-kv copy `n_raw*576`,
     `row_bytes_f16=576*2`, `ns10=ns20=576`, `nb11=nb21=576*2` and the `*12/*13` strides, pad sizing,
     `shared_bytes` (align by 576 — safe over-alloc).
   - **VALUE dims = 512** (output heads are the rope-free c_kv): `heads_bytes=n_head*512*4`, output
     strides `nb01=n_head*512*4`/`nb02=512*4`/`nb03=...`, tmp partial width 512,
     `reduce_pipeline = ds4_gpu_get_flash_attn_reduce_pipeline(512, nwg)`.
   - `vec_args.scale = 1/sqrtf(256.0f)`; `has_sinks=false` (vec-pipeline-getter 2nd arg; no sinks buffer
     wrapped); pipeline name `kernel_flash_attn_ext_vec_f16_dk576_dv512`.

3. **Public `ds4_gpu_attention_decode_heads_glm_tensor`** (Metal + `ds4_gpu.h` extern + CUDA stub) — like
   ds4_metal.m:19434 but no `sinks_offset` (GLM `attn_sinks` is NULL), `n_comp==0` path only (no
   compressor/indexer for v1), key_dim=576/value_dim=512.

4. **Sigmoid router** — thread `bool sigmoid_scoring` into the static `ds4_gpu_encode_router_select`
   (ds4_metal.m ~21869; at the softplus_sqrt site ~21996 branch to `g_unary_sigmoid_pipeline`, which
   already exists). Add a public `ds4_gpu_router_select_glm_tensor` wrapper (+ CUDA stub). Keep noaux_tc
   bias + flat top-8 + normalized×2.5 (already general once `n_expert_used=8`).

5. **`metal_graph_encode_decode_layer_glm(...)`** — dispatch at the TOP of
   `metal_graph_encode_decode_layer` (ds4.c:15074): `if (DS4_MODEL_VARIANT==DS4_VARIANT_GLM) return
   metal_graph_encode_decode_layer_glm(...)`. `cur_hc` is the single `[n_embd]` residual (DS4_N_HC=1).
   Sequence (mirrors the proven CPU `layer_forward_self_one`):
   - a. `attn_norm = rms_norm_weight(cur_hc, attn_norm)` — HC-pre bypass; residual stays `cur_hc`.
   - b. Q: `qr=matmul_q8(attn_q_a,attn_norm)`; `qr_norm=rms_norm_weight(qr,attn_q_a_norm,q_rank)`;
        `q=matmul_q8(attn_q_b,qr_norm)`; `head_rms_norm(q,64,576)`; `rope_tail(q,inverse=false)`.
   - c. KV: `kv=matmul_q8(attn_kv,attn_norm)` [576]; `rms_norm_weight(kv,kv,kv_a_norm,512)` IN-PLACE over
        first 512 (tail 512:576 stays raw rope — verify in-place safe, else norm kv_raw→kv[0:512] +
        `ds4_gpu_encode_cpy_f32_f32_1d` for [512:576]); `rope_tail(kv,1 head,576,inverse=false)`;
        `metal_graph_decode_kv_store(kv,raw_cache,raw_cap,raw_row)`.
   - d. `heads = ds4_gpu_attention_decode_heads_glm_tensor(q,raw_cache,n_raw,...)`; **SKIP** the
        inverse-rope-on-heads (ds4.c:15659 in the DS4 path) — GLM value is rope-free.
   - e. `attn_out = matmul_q8(attn_output, heads)` [plain o_proj, 64*512=32768 → 6144].
   - f. `after_attn_hc = add(attn_out, cur_hc)` — HC-post bypass = plain residual add.
   - g. `ffn_norm = rms_norm_weight(after_attn_hc, ffn_norm)` — HC-pre bypass.
   - h. FFN: if `il < DS4_N_FIRST_K_DENSE`: dense `matmul_pair(ffn_gate_dense,ffn_up_dense)`→
        `swiglu`→`matmul(ffn_down_dense)`; `ffn_out=dense`. else: `router_select_glm`(sigmoid) →
        `ds4_gpu_routed_moe_one_tensor`(ffn_*_exps, simple non-overlap path 16180) + shared expert
        (`ffn_*_shexp` gate/up/swiglu/down); `ffn_out = add(shared, routed)`.
   - i. `after_ffn_hc = add(ffn_out, after_attn_hc)`; write to `g->after_ffn_hc` (caller swaps).
   Skip ALL fused-HC and expert-streaming-overlap branches (they need DS4_N_HC==4 and HC tensors).

6. **Output head** — `metal_graph_encode_output_head` (ds4.c:16303) GLM branch: skip the `output_hc`
   collapse (GLM lacks `output_hc_*`); `output_embd = cur_hc` (already `[n_embd]`);
   `output_norm = rms_norm_weight(output_embd, output_norm)`; `logits = matmul_q8(output, output_norm)`.

**Out of decode-milestone scope (later):** prefill/batch Metal (`metal_graph_encode_layer_batch`
ds4.c:19493), multi-token KV full-attention retention (`n_swa=0`), expert streaming/overlap, MTP,
CUDA+GLM.

## 6b. FINAL SUMMARY (end of 2026-06-16/17 ~8h unattended run) — ⚠️ SUPERSEDED

> ⚠️ This summary is a point-in-time snapshot from the FIRST unattended run (CPU-only forward, no Metal
> generation/server). It is **superseded** by the §6 status log above and §4: since then the Metal
> decode graph, multi-token generation, the server, the GPU router, and Q4 GPU streaming all landed and
> were validated on notible. Read §6 (top) + §4 for the current truth; this section is historical.

**Outcome: GLM-5.2 loads and runs (forward pass) on ds4, validated end-to-end on the real 753 B model
on the CPU backend. Two quantized GGUFs were produced as deliverables.** Everything is on branch
`ds4-glm` (local + `notible:~/Documents/ds4-glm`), DeepSeek Flash/Pro path untouched (all GLM behavior
gated on `DS4_MODEL_VARIANT == DS4_VARIANT_GLM`). Prod server port 8085 was **never stopped** — all work
stayed within mmap/CPU-light bounds.

### What works (validated)
- **Converter** `gguf-tools/glm-quantize.c` (threaded ~20×): authors a full GLM GGUF from HF bf16
  safetensors + config + tokenizer — metadata (`deepseek4.*` namespace), GLM BPE tokenizer
  (154880 tokens + 321649 merges), absorbed-MLA attention (folds `kv_b` into `q_b`/`o_proj` —
  proven exact in `gguf-tools/check_mla_absorption.py`), stacked routed experts. Modes: default **Q4**
  (Q4_K experts) and **`--q2`** (IQ2_XXS gate/up + Q2_K down + synthetic importance).
- **Deliverables:** `notible:/Volumes/4TB-1/glm-5.2-q4.gguf` (408.7 GiB) and `glm-5.2-q2.gguf` (218.9 GiB).
  Recipe both: F16 embed/output/router, Q8_0 attn+shared+dense, F32 norms, experts Q4_K **or** IQ2_XXS+Q2_K.
- **Load path:** `ds4 --inspect` binds + validates all 1236 tensors on both real GGUFs (correct shape
  78L/6144/head_dim 576/256 experts, 753 B params, correct types).
- **Full CPU forward:** `ds4 --cpu --first-token-test` runs all 78 layers (dense 0-2 + MoE 3-77 with
  sigmoid noaux_tc routing) on the real model → finite, sane logits. The #1 risk (forward correctness)
  is retired on CPU. RoPE verified interleaved (matches GLM), correct at all positions.
- **Engine deltas in `ds4.c`** (all variant-gated): `DS4_SHAPE_GLM` + shape selection; HC bypass (single
  residual stream via `hc_pre_from_state_one_scratch`); absorbed attention key=576/value=512, no
  attn_sinks, `kq_scale=1/√256`, partial `kv_a_norm`, skip inverse-rope on value; plain `attn_output`;
  dense FFN for `il<3`; sigmoid noaux_tc router; output-head HC-skip; GLM vocab special tokens;
  `config_validate_model` before `vocab_load`; **`DS4_MAX_EXPERT_USED` 6→8** (GLM uses 8/tok — was a
  stack overflow). Plus converter type fixes (F16 embed/router, Q4_K/IQ2_XXS+Q2_K experts).

### What remains (to run GLM *usefully*, i.e. at speed / multi-token)
- **Metal-graph port** — the main remaining work; CPU forward is correct but slow. Every site mapped in
  §7 (HC bypass, 576/512 attention kernel, plain o_proj, dense FFN, sigmoid router, output head, raw-KV).
- **CPU batch/decode + KV-cache** for real multi-token generation (§7b): batch attention variants + the
  decode-scratch fns need the GLM deltas, and the KV cache must do full attention (`n_swa=0`). Not started
  because CPU test cycles on the 430 GB model are 5-15 min — too slow to debug-iterate in-budget.
- **Exactness/quality:** functional but not bit-checked vs a reference; MoE routing matches noaux_tc on
  inspection; `kq_scale=1/√256` is the standard MLA value (high confidence). For Q2 on 256 GB with big
  context: SSD streaming + 8-bit KV (§3b) once the Metal forward runs.

## 6. Status log

- 2026-06-19 (**Q2 campaign finale — all levers tested; two tiers, no usable third; mixed-precision
  streaming fix landed**): Tested the remaining quality levers (`README_GLM_Q2.md` §4.4/§5 for detail).
  **`--q4-layers`** (lift selected layers' gate/up to Q4_K): resident partials (≤~12 layers, e.g.
  last8 at 234 GiB / 11.4 t/s) run but don't clear the hard prompt; bigger configs are too big for
  resident. **Found + fixed a streaming bug:** the slots8 streaming kernels were Q4_K-only, so a
  Q4-gate/up + Q2_K-down model garbled — added `kernel_mul_mv_slots8_q2_K_sum8_f32` (slots6_q2_K_sum6
  widened to 8) + a `down_type` arg to `ds4_gpu_glm_streaming_routed_moe_tensor` (full-Q4 path
  unchanged; CUDA stub/header updated). With the fix, **q4gu-all (Q4 gate/up + Q2_K down, 356 GiB)
  runs and reaches Q4-class quality** (produces the planet facts) → **gate/up is the quality driver**.
  But it does NOT beat Q4: terser facts (Q2_K down) and ~same disk-bound ~1 t/s (only 13% smaller).
  **External HF GGUFs** (unsloth UD-IQ2, REAP50 Q2/Q3) surveyed: glm-dsa / patched-llama.cpp-only
  (ds4 can't load them), REAP50 is expert-pruned + degraded — none higher quality than our Q4.
  **Outcome: `glm-5.2-q2.gguf` (now the imatrix build, 219 GiB resident ~11 t/s, simple prompts) and
  `glm-5.2-q4.gguf` (409 GiB streamed ~1 t/s, complex prompts) are the two tiers; no usable middle.**
  Converter gained `--gu-type/--down-type/--q4-layers`; inferior variants deleted (benchmarks in the
  README). DeepSeek path untouched; DeepSeek prod (8085) cycled via the trap pattern, left UP.
- 2026-06-18 (**Q2 IMPROVEMENT CAMPAIGN — real activation imatrix is a clear resident-Q2 win; full
  detail in `README_GLM_Q2.md`**): Ran an autonomous experiment campaign to improve Q2. **(1) Eval
  metric:** built `--perplexity-file` but found raw-prose ppl is uninformative for GLM (Q4 3079 ≈ Q2
  3184 — the reasoning model is OOD on un-framed prose, masking the quant signal); pivoted to the
  greedy/behavioral battery. **(2) Option A (real imatrix) — the win:** the imatrix collector hooked
  the GLM-unadapted batch prefill (SIGSEGV), so re-pointed it at GLM **sequential decode**; a per-layer
  post-MoE GPU read exhausted the Metal command-buffer pool (~layer 44), so collect **gate/up importance
  inside the CPU router** (one already-present read/layer) and let the converter fall back to synthetic
  for down. New: converter `--imatrix`/`--gu-type`/`--down-type`/`--q4-layers` flags; `ds4_engine_collect_imatrix`
  GLM path; `imatrix_accum_gate_up` + `imatrix_collector_save` omits zero-count entries. Collected a
  20k-token / 12M-observation imatrix and rebuilt: **`glm-5.2-q2-imatrix-dense.gguf` (219 GiB, resident,
  ~11 t/s) is the recommended Q2** — greedy gives a clean correct planet list where baseline loops on
  "Mercury". **(3) Ceiling:** long/complex prompts still degenerate; down→Q4_K (272 GiB, streamed ~0.65
  t/s) only marginally helps → the bottleneck is the **2-bit IQ2_XXS gate/up** (gate/up can only be
  IQ2_XXS or Q4_K — no Q2_K pair_swiglu kernel), fixable only by Q4-level gate/up (full Q4, or `--q4-layers`
  on a few sensitive layers — landed, untested). A0 (synthetic down-weighting) was dropped as subsumed.
  All engine changes variant-gated (DeepSeek path untouched; no new public GPU entry → no CUDA stub).
  Reusable scripts in `scripts/` (build/eval/collect/iterate, all trap-protected). DeepSeek prod (8085)
  stopped/restarted many times via the trap pattern; left UP.
- 2026-06-18 (**Q2 characterized + documented; better-Q2 options + test plan written → `README_GLM_Q2.md`**):
  Per user, ran the current Q2 and documented what works/doesn't, how it was built, and how to make it
  better. **Run** (5-prompt battery on notible via a trap-protected detached script `scripts/glm-q2-characterize.sh`;
  prod stopped, polled `pgrep -f ds4-server` empty, restarted via `trap EXIT` — verified back UP):
  SHORT/LONG × temp {0, 0.6, 1.0} / top_p 0.95, `-c 2048`, ~11.4 t/s greedy / ~9.9 sampled. **Finding:**
  every mode emits the **correct planets in order** then **fails to stop — loops/repeats** (even greedy on
  the SHORT prompt; LONG@0.6 collapsed to word-salad). Confirms the user's recollection (temp + length →
  decoherence) and broadens it (looping is pervasive, not just temp 1.0). Generations are **thinking-on**
  (CLI default) so much of the loop is inside the reasoning trace — flagged a thinking-off retest. **Creation
  method (from source, corrects two priors):** the converter goes **bf16 → f32 → 2-bit directly, NO Q8
  intermediate** (`glm-quantize.c:918`/`:790`; user's "Q8→Q2?" hypothesis is moot — already minimal lossy
  steps); recipe = IQ2_XXS gate/up (2.06 bpw, **synthetic per-column-energy** imatrix `:783`) + Q2_K down
  (2.625 bpw, **imatrix=NULL → unweighted** — a real gap); experts are ~93% of the 219 GiB. (Also noted the
  output head is **Q8_0**, not F16 as README-GLM.md/§6b say.) **Better-Q2 options documented + ranked:**
  A real activation imatrix (largest lever; −1.95% NLL on DeepSeek Q4, more at Q2 — **blocked**:
  `ds4_engine_collect_imatrix` uses the GLM-unadapted batch prefill that SIGSEGVs, so needs the collector
  re-pointed at GLM sequential decode *or* the GLM batch prefill adapted; + port `--imatrix` from
  `deepseek4-quantize.c` into `glm-quantize.c`); A0 weight the down_proj with the existing synthetic imatrix
  (free); B bit-allocation (Q2_K g/u, sensitivity-aware Q4 layers, down→Q4 — mostly off-budget on 256 GB
  without D); C new low-bit types (IQ2_S/Q3_K/IQ3_XXS — need a converter quantizer AND a Metal MoE kernel,
  expensive); D 8-bit latent KV (frees RAM to fund B); E direct-vs-staged = already optimal. **Test plan**:
  eval = load-check + decoherence battery (thinking on/off) + NLL/first-tok/greedy-LCP vs Q4 anchor +
  official GLM API (`gguf-tools/quality-testing/`, DeepSeek-shaped → adapt); iterate with `--layers` slices,
  full builds only for quality; results tracker seeded with the baseline. Order: A0 → B2 → A → D → C. Docs
  only this session (no engine change); DeepSeek path untouched.
- 2026-06-17 (**GLM Q4 GPU expert streaming WORKS on the CLI + server; Q4 is fundamentally disk-bound on
  256 GB**): Built the full GLM Q4 GPU streaming path: `slots8` Q4_K kernels (pair-swiglu + sum8, clones of
  the validated slots6/sum6) + encoders + `ds4_gpu_glm_streaming_routed_moe_tensor` which routes on the CPU
  (ids locate experts in the mmap), stages the 8 selected experts into a resident slab each MoE layer, and
  runs slots8 on the GPU.  Wired into the GLM decode layer under `--ssd-streaming` + Q4_K (else CPU routed
  fallback), + CUDA stubs.  **Validated on notible (CLI + server):** coherent output (Mercury..Jupiter), the
  slots8 kernels compile + run, the **server answers a Q4-streamed prompt end to end** (ready in 6s, decode
  through the streaming path).  **Speed: ~0.42 tok/s cold, ~0.76 warm** (vs ~0.3 CPU).  A warm-page-cache
  A/B (0.37->0.76 gen, 0.13->0.51 prefill) confirms **Q4 is disk-bound**: 389 GiB of experts can't be
  resident on 256 GB, so the per-layer expert load (cold->disk) dominates, not GPU compute, and single-token
  decode has little compute to overlap the load with.  Also fixed `DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER`
  61->78 (GLM has 78 layers; the cache rejected layer 61+).  **Tried** swapping the per-layer memcpy for the
  DeepSeek resident slab cache (peek + 6-wide load in 2 groups) to skip the copy + exploit usage skew, but it
  **thrashed** (0.06 tok/s -- the 6-expert-oriented per-layer cap/eviction doesn't fit GLM's 8-expert
  pattern); **reverted** to the faster memcpy path (kept in git for a proper GLM-specific cache later).
  **Conclusion:** Q4 streaming is functional (server/agent) but disk-bound (~0.76 warm); interactive speed
  needs a non-thrashing GLM expert cache (exploit skew) + load/compute overlap -- substantial.  For FAST GLM
  serving, Q2 (resident, ~11.6 tok/s) is the practical target; Q4 streaming is for quality/validation.
  Remaining: a GLM-native LRU expert cache, prefetch/overlap, and 8-bit latent KV (Phase 4, independent of
  the disk wall -- shrinks KV for larger Q4/Q2 context).
- 2026-06-17 (**Phase 1 of fast Q4 GPU streaming begun: slots8 Q4_K kernels landed (foundation)**): Goal
  (per user): make the ds4 server + agent fast on Q4 streamed from SSD -- build it, cache it, then KV-quant.
  DeepSeek's streaming MoE dispatch is slots6/sum6 (6 experts hardwired: `n_expert==6` gates + `for i<6`
  loops in `ds4_gpu_routed_moe_one_tensor`); GLM routes 8, so GLM needs its own 8-expert dispatch.  **Landed:**
  `kernel_mul_mv_slots8_q4_K_pair_swiglu_f32` + `kernel_mul_mv_slots8_q4_K_sum8_f32` in `metal/moe.metal`
  (exact clones of the validated slots6/sum6, widened to 8 gate/up/down buffers + switch 0..7), their encoders
  `ds4_gpu_encode_mul_mv_slots8_{pair_swiglu,sum8}` in `ds4_metal.m`, and non-fatal pipelines
  `g_moe_mul_mv_slots8_q4_k_{pair_swiglu,sum8}_pipeline` (nil on build failure -> GLM streaming falls back to
  CPU routing; Q2/DeepSeek init unaffected).  Builds clean; kernels compile at runtime so they're
  runtime-validated once wired.  **Remaining Phase 1 (next):** an isolated `ds4_gpu_glm_streaming_routed_moe`
  that, per MoE layer: (1) stages the 8 selected experts' gate/up/down into 8 GPU slot buffers (pread from the
  GGUF), (2) sets up `ds4_gpu_mul_mv_id_args` (ne00=expert_in 6144, ne01=mid 2048, nei0=8, nei1=1, nb01=row
  bytes) + `swiglu_weight_args` (mid_row_stride/weight_stride/clamp) mirroring the slots6 dispatch setup in
  `ds4_gpu_routed_moe_one_tensor` (~ds4_metal.m:24600), (3) calls slots8 pair_swiglu -> mid then slots8 sum8
  -> routed_out; + a public entry (+ CUDA stub) wired into `metal_graph_encode_decode_layer_glm` when
  `g->ssd_streaming` (else the current resident/CPU path).  Validate on notible: `--metal-graph-full-test` on
  Q4 (GPU stream vs CPU) then generation speed.  **Phase 2:** add the LRU expert cache (reuse the slab/peek/
  pread primitives `ds4_gpu_stream_expert_cache_peek` etc.) so hot experts skip disk; then overlap cold loads
  with shared-expert compute.  **Phase 4:** 8-bit latent KV (`metal/dsv4_kv.metal`) for Q4 + big context.
- 2026-06-17 (**Q4 RUNS (minimal CPU-routed path) and SETTLES the Q2-quality question: Q2's degeneration
  is 2-bit quantization, not a bug**): Found a zero-new-code path to run the 409 GiB Q4 on the 256 GB box:
  `--ssd-streaming` (Metal residency restricted to the token embedding + non-routed weights; experts stay
  mmap'd, NOT GPU-wired -> no OOM) + `DS4_GLM_CPU_ROUTED_MOE=1` (the routed MoE runs on the CPU via
  `layer_routed_moe_one`, which reads experts through `tensor_data` over the demand-pageable mmap -- the CPU
  *can* page a 409 GiB map; the GPU can't).  This sidesteps the unimplemented GPU expert cache entirely.
  **Q4 runs** at ~0.14-0.67 tok/s (CPU MoE + cold-disk streaming; page cache warms across runs) -- far too
  slow for production but enough for validation.  **Quality cross-check (the whole point):** on the same
  greedy long prompt where Q2 loops after the planet list, **Q4 stays coherent** -- lists Mercury..Jupiter
  then gives a correct fact per planet, no degeneration.  So Q2's long-prompt degeneration is the **2-bit
  quantization ceiling**, confirmed against Q4 on the identical engine/prompt/settings -- NOT an engine or
  GGUF bug.  Together with CLI==server, run-to-run determinism, and `--metal-graph-full-test` (Metal==CPU),
  the code is fully exonerated and the user's Q4 reference is satisfied.  **Remaining:** a GLM GPU expert
  cache (the 8-expert streaming dispatch DeepSeek hardwires to 6) is now purely a *speed* investment to make
  Q4 practical; the quality question no longer needs it.
- 2026-06-17 (**SPEED: GLM router moved to the GPU — generation 8.9 -> 11.6 tok/s, validated**): The v1 GLM
  decode path selected experts on the host every MoE layer (`metal_graph_glm_cpu_router`: end the command
  buffer, read `ffn_norm` back, top-k on the CPU, upload `selected/weights`) -- a GPU<->CPU sync per layer.
  Threaded a `sigmoid_scoring` flag into the static `ds4_gpu_encode_router_select` (the bias add / flat
  top-k / weight gather / normalize / x scale are scoring-agnostic, so only the unary op differs:
  `g_unary_sigmoid_pipeline` vs `sqrt(softplus)`), added a focused public `ds4_gpu_router_select_glm_tensor`
  (no hash/groups, always noaux_tc bias) encoded into the in-flight command buffer, and wired the GLM decode
  layer to select on the GPU by default (`DS4_GLM_CPU_ROUTER=1` forces the v1 host router for A/B). DeepSeek's
  `ds4_gpu_router_select_tensor` signature is untouched (+ CUDA stub for the new GLM entry). **Validated on
  notible** (`--metal-graph-full-test -p "..."`, server stopped/restarted via the trap script): GPU router vs
  CPU router give **numerically identical logits** -- both `cpu_top==gpu_top==154822`, logits_rms 0.0135575
  (GPU) vs 0.0135582 (CPU), differing by 7e-7, reproducing the §6 0.0136 baseline. **Generation 8.88 -> 11.56
  tok/s, prefill 8.74 -> 11.41** (greedy, glm-5.2-q2.gguf). Greedy *text* diverges from the CPU router (prose
  vs numbered list) -- both coherent and correct: identical logits to 7 sig-figs, but a near-tied formatting
  token flips under greedy on Q2 and cascades. Remaining speed item: the GLM Metal **batch** prefill
  (currently sequential single-token; `metal_graph_encode_layer_batch`).
- 2026-06-17 (**SERVER FINALE: GLM server answers a prompt end to end on notible; code exonerated vs a
  Q2-quality suspicion**): Implemented the full GLM server chat I/O (all gated on `variant==GLM`, DeepSeek
  bit-identical, 8 no-model unit tests via `./ds4_test --server`): (1) `ds4_server.c`
  `render_chat_prompt_text` dispatches to a GLM renderer mirroring `chat_template.jinja` — `[gMASK]<sop>`
  head, reasoning-effort + tool schemas in their own `<|system|>` turns, `<|user|>`/`<|assistant|><think>`
  turns, `<|observation|><tool_response>` results, no per-turn EOS; (2) GLM `<tool_call>` render
  (`append_glm_tool_calls_text`) + parse (`parse_generated_message_glm`, recovers arg types by trying each
  value as JSON) + streaming markers; (3) `ds4.c` token builders `ds4_chat_*` given the same GLM framing
  (CLI/agent path); (4) a small `ds4_is_glm()` / `ds4_glm_reasoning_effort_text()` seam in `ds4.h`. Fetched
  the real `chat_template.jinja` from notible to pin the multi-turn semantics (reasoning kept only for the
  turn after the last user; `<|observation|>` once per consecutive tool run; string args verbatim / others
  as JSON). **E2E on notible** (prod 8085 stopped via the launchd script, restarted after — done via a
  trap-protected detached script so prod always comes back): GLM-Q2 server loads (~80s), `--trace` confirms
  the exact GLM frame is rendered, answers coherently at **~7.8 tok/s** decode / ~8.7 prefill.
  **Sampling investigation (user flagged temp 1.0 garbage as a possible bug):** at first the server's default
  (thinking-HIGH, which forces temp 1.0) produced degenerate `</think>`-spam output. Isolated it with the
  CLI generate path: greedy `--temp 0` is coherent + **byte-identical run-to-run (deterministic)** and
  reproduces the §6 planet answer; the **long** "list AND give a fact about each" prompt degenerates the SAME
  way on BOTH the CLI and the server (lists correctly then loops; CLI temp 1.0 even reproduces the "2. Uran"
  truncation), while the **short** prompt stays coherent on both. So it is **not a code/path bug** (CLI ==
  server, deterministic, greedy-correct, and §6's `--metal-graph-full-test` already shows Metal == CPU) — it
  is the **Q2 2-bit quality ceiling** on complex prompts, worse at temp 1.0. **Default GLM sampling changed
  to temp 0.6 / top_p 0.95** (request default + thinking-mode override + CLI), which keeps Q2 coherent. The
  one thing still unsettled is whether Q2 itself is miscalibrated vs Q4 — that needs the Q4 comparison, still
  blocked on unimplemented GLM expert streaming. **Remaining this session:** GPU sigmoid router (speed).
- 2026-06-17 (**Q4 quality cross-check attempted -> found GLM SSD streaming is unimplemented; output
  glitches are Q2/greedy, not engine bugs**): Tried to run Q4 (408.7 GiB, streaming-only on 256 GB) to
  prove the Q2 generation glitches are quantization, not a code bug. **Q4 streaming is blocked**: the
  streaming decode-span builder (`model_map_span_vec_include_layer_decode`) only listed DeepSeek's
  tensors, so a streamed GLM layer hit `attn_output`/dense FFN ranges "not covered by mapped model
  views" (o_proj failed at layer 0). Added GLM's `attn_output` + `ffn_*_dense` to the span set (correct,
  kept) -> got to layer 64, then **OOM**: GLM has no routed-expert streaming (it reads experts via
  `ds4_gpu_routed_moe_one_tensor` over the model map, not the DeepSeek evicting cache), and mapping all
  408 GiB of experts can't fit. **Conclusion: GLM SSD streaming needs the proper selected-expert cache
  (load the 8 routed experts/layer, evict cold) before Q4 can stream** -- a substantial follow-up;
  reverted the map-all-experts stopgap. **On the glitches** (asked: are they Q2 or a bug?): the
  evidence points to Q2 + greedy behavior, not engine bugs -- (a) at `--temp 0.05` (near-greedy) the
  output is clean and correct, "Uranus" spelled in full; (b) the "Uran" truncation appeared only at
  `--temp 0.7` (sampling the flatter Q2 tail), not under greedy; (c) ">5 planets returned" is the model
  getting the 5 correct then over-generating (greedy doesn't stop) -- a stopping/instruction-following
  weakness Q2 amplifies. A definitive Q4 comparison still needs GLM expert streaming. The kept span fix
  (attn_output/dense) is a prerequisite for that work.
- 2026-06-17 (**session/sampled prefill SIGSEGV fixed -- temp>0 + the server path now run**): The
  default CLI path (`temp>0` -> `run_sampled_generation`) and the server both prefill via
  `ds4_session_sync`, whose cold + resumed prefill called `metal_graph_prefill_layer_major` (the
  GLM-unadapted Metal batch prefill) and **SIGSEGV'd on GLM's absent HC/compressor tensors**. Fix:
  route GLM's session prefill through sequential decode (`metal_graph_eval_token_raw_swa` per token,
  the validated single-token forward) for both the cold path and the checkpoint-extension loop, gated
  on `variant==GLM` (DeepSeek unchanged). **Validated** (`glm-5.2-q2.gguf`, default temp, prompt ->
  sampled generation): **no crash**, sequential prefill runs ("processing 33 input tokens..."), and
  decode produces a correct start ("Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus,"). The
  remaining low quality of *sampled* output (it fragments after the correct start) is **not a bug and
  not a regression -- verified empirically**: (A) the SESSION path at `--temp 0.05` (near-greedy)
  produces coherent, correct output IDENTICAL to the generate-path greedy ("...Mercury, Venus, Earth,
  Mars, Jupiter, Saturn, Uranus..."), proving the session decode is correct; (B) the session path at
  `--temp 0.7 --top-p 0.95` produces a coherent numbered planet list. The earlier fragmentation was
  purely `DS4_DEFAULT_TOP_P=1.0` (no nucleus filtering -> samples the Q2 tail past the natural stop).
  `--top-p 0.95` (GLM's `generation_config` default) fixes it; greedy (`--temp 0`) is already coherent.
  (Reasonable follow-up: default GLM sampling to temp 1.0 / top_p 0.95.)
  The crash fix also unblocks the **server** (it uses sessions); remaining for the finale are the GLM
  server chat builders (`ds4_chat_*`) + `<tool_call>` and an actual end-to-end server run.
- 2026-06-17 (**multi-row attention bug FOUND + FIXED -- multi-token generation now produces coherent,
  correct output**): The garbage was a one-line stride bug in the GLM flash-attn encoder
  (`ds4_gpu_encode_flash_attention_raw_heads_glm`): the **query** strides `nb01/nb02/nb03` (which the
  vec kernel uses as `q += iq2*nb02`) were set from `val_row_bytes` (512*4) instead of `key_row_bytes`
  (576*4). GLM's absorbed MLA has `key_dim=576 != value_dim=512`, so every head past head 0 read its
  query from the wrong offset (drift 64 floats/head) -> corrupt q.k scores. **Invisible at `n_raw==1`**
  (softmax weight is 1.0 regardless of the score), which is why the single-token forward validated but
  any pos>0 decode was garbage. DeepSeek never hit it (key==value==512). **Found by a new standalone
  `--glm-attn-test`** (no model load, runs locally in seconds): random `q[64x576]` + `KV[n_raw x 576]`
  vs a CPU `softmax(q.k/sqrt(256)).v[:512]` reference. The diagnostic showed *whole heads* wrong (not
  dims) -> pointed straight at the per-head query stride. Before: n_raw=1 matched (1e-4), n_raw>=2 max
  ~0.2-0.37. After: **n_raw 1..256 all match within f16 precision (max <2e-4)** -- covers the C=32
  block boundary and the multi-workgroup split + reduce. **End-to-end** (`glm-5.2-q2.gguf -p "List the
  first five planets from the Sun" -n 220 --temp 0 -c 2048`): the model now answers **"The first five
  planets from the Sun are Mercury, Venus, Earth, Mars, Jupiter..."** (coherent + factually correct) at
  ~8.7 tok/s. It then loops/degrades on continued greedy decoding -- expected argmax-repetition + Q2
  quantization, **not** an attention bug (the kernel is exact to n_raw=256). Keep `--glm-attn-test` as a
  fast regression. **Multi-token generation works on Metal.** Remaining for quality/usability: the
  default (temp>0) sampled path still routes through the GLM-unadapted Metal batch prefill
  (`metal_graph_prefill_layer_major`) which SIGSEGVs -- so sampling (which would break the greedy loop)
  needs either a GLM batch prefill or routing the sampled path through the sequential-decode driver;
  plus the server wiring + `<tool_call>` for the finale.
- 2026-06-17 (**multi-token generation + GLM chat I/O implemented; blocked on a multi-row attention
  bug**): Landed three things toward usable generation. (1) **GLM chat I/O** (commit cherry-picked from
  a parallel subagent): `encode_chat_prompt` now emits the real GLM frame `[gMASK]<sop>` + optional
  `<|system|>Reasoning Effort: High|Max` + `<|user|>` + prompt + `<|assistant|><think>` (NO newline
  before `<think>` -- corrects plan 1b); vocab gained `sop_id`/`system_id`/`observation_id`;
  `special_token_at` has a GLM variant; `ds4_token_is_stop` + the generation loop stop on GLM's three
  EOS (`<|endoftext|>`=154820, `<|user|>`=154827, `<|observation|>`=154829). **All special-token IDs +
  the stop set were verified against the real `tokenizer.json`/`generation_config.json` on notible.**
  (2) **Multi-token KV**: GLM is full-attention (`n_swa=0`); `metal_graph_raw_span_for_batch` returned
  `n_raw=1` and `metal_graph_raw_cap_for_context` sized the ring to the (zero) SWA window, so only
  single-token forwards worked. Added GLM branches: full-attention span `n_raw=min(last_pos+1,raw_cap)`
  and `raw_cap=full ctx` (capped 8192 for memory). (3) **Generation driver**: `generate_metal_graph_raw_swa`
  prefills GLM by **sequential decode** (loops the validated single-token forward while the KV
  accumulates) since the Metal batch prefill is not GLM-adapted. Plus `DS4_IGNORE_EOS` debug gate.
  **STATUS: generation runs end to end** (`ds4 -m glm-5.2-q2.gguf -p "..." --temp 0 -c 2048`,
  ~8.8 tok/s, reaches the decode loop) **but the output is GARBAGE** (`\n\n`, repeated `<|endoftext|>`,
  stray CJK). **Root cause: the GLM multi-row attention (`n_raw>1`, i.e. ANY pos>0) is wrong; the
  single-token path (`n_raw=1`, pos 0, softmax weight 1.0) was the only thing ever validated** -- the
  §7c-flagged risk. Step-0 (last-prefill, multi-row) logits already show garbage byte-tokens mixed in;
  decode degrades fully. **Ruled out by code review this session** (all read as correct): `freq_base`
  (8e6, plumbed consistently to CPU+GPU), the interleaved rope kernel math, the `dk576_dv512` vec
  attention kernel (DK/DV/NE=2 score+softmax+value reductions), the pad kernel (n_raw<32 single partial
  block), the KV cache row layout, and `raw_start=0` (no ring for full attn). The bug is subtle and
  needs an **empirical 2-token GPU-vs-CPU per-layer isolation** (extend `metal_graph_first_token_full_test`
  to pos 1 with KV accumulation on both sides) to pinpoint rope(pos>0) vs attention(n_raw>1) vs
  KV-store. **Separately:** the default CLI path (temp>0) routes through the session/batch prefill
  (`metal_graph_prefill_layer_major`) which SIGSEGVs for GLM (not adapted) -- the `--temp 0` argmax
  driver is the working multi-token path; a GLM Metal batch-prefill is a later item (like the GPU
  router). Commits on `ds4-glm`: full-attn KV + sequential prefill; chat framing + multi-EOS; stop-check
  refactor + DS4_IGNORE_EOS. `make ds4` clean throughout; DeepSeek path untouched (all variant-gated).
- 2026-06-17 (**8-expert Metal routed MoE VALIDATED vs CPU — routed experts now on the GPU**): The
  routed MoE (≈97% of FLOPs) was the #1 speed item; it ran on the CPU because
  `ds4_gpu_routed_moe_one_tensor` rejected `n_expert > 6` and GLM routes 8. **The fix was far smaller
  than the handoff feared:** the baseline id-based MoE path is already n_expert-general (the
  gate/up/down `mul_mv_id` kernels dispatch one threadgroup-set per `(expert, token)` via
  `pairs = nei0*nei1`, and `ds4_gpu_encode_moe_sum_experts` already reduces any expert count with a
  binary add chain for `n_expert != 6`). So **no `sum8`/`group8` reduction kernel and no dispatch
  rework were needed** (correcting the §7/handoff framing) — the only 6-hardwiring in GLM's path was
  the entry guard. Changes: (1) ds4_metal.m guard `n_expert > 6` -> `> DS4_GPU_MAX_EXPERT_USED` (8) and
  `selected_ids[6]` -> `[DS4_GPU_MAX_EXPERT_USED]` (defensive; the slot/stream arrays it feeds are all
  `n_expert==6`-gated, so DeepSeek is bit-identical and only GLM's n_expert=8 newly passes to the
  generic baseline). (2) ds4.c `metal_graph_glm_cpu_router` mirrors the DeepSeek
  `metal_graph_decode_cpu_router` but uses GLM routing (`layer_topk_selected_experts`: sigmoid
  noaux_tc + `exp_probs_b` bias + top-8 + normalized x2.5), one sync/MoE-layer to read `ffn_norm`,
  writing `g->router_selected/weights`. (3) the GLM decode layer's MoE branch now calls
  `ds4_gpu_routed_moe_one_tensor` (same wiring as the DeepSeek caller: `g->routed_gate/up/mid/down`
  scratch, `ffn_{gate,up,down}_exps` offsets/types, IQ2_XXS gate/up + Q2_K down, clamp
  `DS4_SWIGLU_CLAMP_EXP`) instead of `metal_graph_glm_cpu_routed_moe`; `DS4_GLM_CPU_ROUTED_MOE=1`
  forces the v1 all-CPU path for A/B. **Validation** (`--metal-graph-full-test` on `glm-5.2-q2.gguf`,
  `-c 2048`, prod server stopped then restarted): **gpu_top==cpu_top==154822** (`[gMASK]`),
  gpu_top_logit 27.1759 vs cpu 27.1943, logits max-diff 0.0648 / rms 0.0136 (vs 0.0110 when the MoE
  was CPU-uploaded -- the small rise is the expected GPU/CPU expert-arithmetic noise, top token
  unchanged with healthy margin). `make ds4` clean on both machines; DeepSeek/CUDA paths untouched
  (no new public GPU entry point, so no new CUDA stub). **Remaining for end-to-end tokens/sec:** the
  router still costs one GPU<->CPU sync per MoE layer (a GPU sigmoid router per §7c.4 would remove it),
  and multi-token generation (CPU batch/decode + KV full-attention + Metal prefill) is still needed to
  actually measure tokens/sec (single-token `--metal-graph-full-test` is load-dominated, not a t/s
  benchmark).
- 2026-06-17 (**Metal decode forward VALIDATED end-to-end vs CPU on the real Q2 model**): The GLM
  single-token Metal graph now runs all 78 layers and matches the validated CPU forward. Natural
  (non-teacher-forced) `--metal-graph-full-test` on `glm-5.2-q2.gguf`: **gpu_top == cpu_top == 154822**
  (`[gMASK]`, the expected sequence-start prediction), gpu_top_logit 27.1895 vs cpu 27.1943, logits
  max-diff 0.053 / rms 0.011. Teacher-forced per-layer diffs are ~1e-4 (early dense layers) growing to
  ~0.4-0.6 max in the last layers (accumulated quant + GPU/CPU reduction-order noise; small vs the
  residual magnitude). **On the GPU:** HC bypass (single residual stream), 576/512 absorbed MLA
  attention (no sinks, kq_scale 1/√256, plain o_proj, partial kv_a_norm, no inverse-rope on the value),
  dense SwiGLU for `il<3`, shared expert, output head (HC-collapse skipped). **On the CPU (v1 stopgap):**
  the routed MoE — the Metal routed-MoE path is hardwired to 6 experts/token (`selected_ids[6]`, `sum6`
  down-reduction kernels) and GLM routes 8, so `metal_graph_glm_cpu_routed_moe` runs the validated CPU
  routed MoE and uploads the result. **An 8-expert Metal routed MoE (new `sum8` kernels + dispatch) is
  now the main remaining speed item** (experts are ~97% of FLOPs). New kernel:
  `kernel_flash_attn_ext_vec_f16_dk576_dv512` (NE=2 → NL=16 divides DK4=144; DK 576 is not a multiple of
  128). **Real bugs found + fixed this session:** (1) `DS4_MAX_LAYER` 61→78 — GLM's 78 layers overflowed
  `ds4_weights.layer[]` into following `ds4_engine` members; the prior "validated" CPU forward had been
  running on UB (benign only because the clobbered MTP/backend fields were unread). (2)
  `ds4_gpu_tensor_alloc(0)` returned NULL → now a minimal placeholder, so GLM's disabled-feature scratch
  (`n_lora_o=0`, `compress_ratio=0` → zero-size attn_low/compressor/indexer buffers) allocates. (3)
  routed-MoE selected-override capped at 6 → 8. (4) `metal_graph_alloc` sized MoE work buffers from the
  dense layer[0] (NULL expert tensors for GLM) → now from the first routed layer. Validation workflow:
  prod server (port 8085) stopped via `setup-launchagents-tts.sh --stop jumbo_server` (needs `bash -lc`
  for brew PATH), test run with `-c 2048` so graph buffers fit under the 240 GB wired limit alongside the
  224 GB model, server restarted (`--start jumbo_server`) after. Diagnostics left in place: a `GLM_STEP`
  label macro in the GLM decode layer and stage-failure prints in the full-test loop (both fire only on
  failure). DeepSeek/CUDA paths untouched; CUDA carries a stub for the new GLM attention entry point.
- 2026-06-17 (Metal port begun): **Reverse-engineered the decode-path Metal graph and wrote the
  code-grounded implementation spec (§7c).** Confirmed the per-layer Metal flow lives in
  `metal_graph_encode_decode_layer` (ds4.c:15074, ~1230 lines of HC-fusion + expert-streaming
  optimizations), called from C; the kernels live in `ds4_metal.m`/`metal/*.metal`. Key findings: (1)
  validation must use `--metal-graph-full-test` (`metal_graph_first_token_full_test`, vs the GLM-aware
  CPU `layer_forward_self_one`), NOT `--metal-graph-test` (hardwired to MoE experts + unconditional
  inverse-rope → not GLM-correct). (2) The flash-attn raw-heads encoder (ds4_metal.m:16793) rejects
  `head_dim!=512` and is parameterized by a single `head_dim` assuming DK==DV; GLM needs DK=576/DV=512
  threaded separately (full dim-split mapped in §7c), scale 1/√256, no sinks. (3) decode-heads +
  router-select have CUDA impls → keep their signatures untouched, add isolated GLM fns + CUDA stubs.
  **Landed:** `kernel_flash_attn_ext_vec_f16_dk576_dv512` in metal/flash_attn.metal (the riskiest novel
  piece; DK=576 register budget verifiable only at the gated pipeline-creation run). Architecture
  decided: a dedicated `metal_graph_encode_decode_layer_glm()` + output-head GLM branch (keeps the
  DeepSeek/CUDA hot path bit-identical per AGENT.md). **Remaining (all compile-checkable with the prod
  server UP; numerical validation needs the prod server DOWN — a gated step requiring the user):** the
  GLM raw-heads encoder, public GLM decode-heads + sigmoid-router wrappers (+ CUDA stubs),
  `metal_graph_encode_decode_layer_glm`, and the output-head branch — each spec'd site-by-site in §7c.
- 2026-06-17 ~01:25: **Q2 validated through the FULL forward + Q2-vs-Q4 comparison.** Q2 first-token-test
  (all 78 layers incl IQ2_XXS gate/up + Q2_K down MoE) **completed cleanly**: finite logits min=-8.98
  max=27.19 rms=2.12, top `[gMASK]` (27.19) — consistent with Q4's full-forward behavior (top `[gMASK]`
  28.7, rms 2.20). Q2 (235 GB) ran clean where the Q4 (430 GB) CPU forward again hit a transient
  memory/VM kill (no output) — confirming Q2 fits RAM and is the better full-run target. **Q2 quality
  preserved at the gross level** (same sequence-start prediction for OOD bare input, similar logit
  magnitudes through the quantized MoE); a rigorous Q2-vs-Q4 quality delta needs proper `[gMASK]<sop>`
  prefill + a reference (awaits the batch path). **RUN COMPLETE** — loop ended. Both GGUFs delivered;
  forward validated end-to-end on CPU; Metal port + multi-token generation documented in §7/§7b/§6b.
- 2026-06-16: Investigation complete. Confirmed GLM-5.2 = `glm_moe_dsa` (MLA+DSA+MoE, DeepSeek-V4 cousin). Verified config, geometry map, GLM chat template. Identified **Hyper-Connections removal (R1)** as the gating risk (missing from porting.md). Branch `ds4-glm` ready locally; notible checkout exists on `main`; model ~18% downloaded.
- 2026-06-16: Per user, added **Q2 and Q4 both as Phase 4 build targets** (Q4 = SSD-streaming-only on 256 GB) for a real quality/speed comparison, and the **§3b M1 memory-fit study** (quantized KV is the key lever for Q2 + large context on 256 GB). Pushed groundwork to `origin`; checked out `ds4-glm` on notible (sync loop proven).
- 2026-06-16: Ran two spikes. **R1 (HC) resolved → bypass path, ~6-8 sites, not a rewrite (§3c).** Attention/FFN deltas mapped: o_proj LOW, dense-first-3 LOW-MED, q/kv up-proj MED (§3c). Captured **verified GLM tensor schema from shard headers (§2b)** — key converter job is stacking 256 per-expert tensors. Download now ~73% (207/282 shards). Next: converter skeleton + `DS4_SHAPE_GLM`/HC-bypass scaffolding once `index.json` lands (router/shared/MTP names).
- 2026-06-16: Recorded the **port-8085 server-down gate** (§0): converter/build/source work keeps the prod server up; only loading/running GLM on notible Metal (imatrix, first forward pass, Q2/Q4 runs) needs it down — will flag before each. Built **`gguf-tools/glm-quantize.c` skeleton**; `--dry-run` over 255 shards validated the GLM→ds4 map and **confirmed router/shared/noaux_tc-bias names** (§2b) + the ~20/78-layer IndexShare indexer. Gaps are just download-incompleteness + final `model.norm` (last shard). Converter is standalone (no GLM template GGUF exists). Next: converter body (dequant + MLA absorption + expert stacking + GGUF authoring, Q8 first).
- 2026-06-16: Built the converter's **safetensors reader** (`stdb`: scans shard headers directly, no `index.json` needed; bf16/f32 → f32) with a `--read` mode. **Validated on real tensors** (norm gains all-positive 0.004-0.22; `q_a_proj`/expert weights symmetric ±0.17, mean≈0 → bf16 decode correct). Next design step: pin the **GGUF schema contract** (metadata + tensor set) jointly with the engine loader, and resolve the absorbed-MLA representation (how GLM's un-absorbed `kv_b` folds into ds4's `attn_q_b`/`attn_kv`/`attn_output`) by reading ds4's attention forward — that gates the converter writer and the engine attention changes.
- 2026-06-16: **Model download complete** (1.4 TB, 282 shards, 59,585 tensors); `tokenizer.json` + `index.json` now local. `glm-quantize --dry-run` now shows the full GLM→ds4 map satisfied (only the by-design indexer SKIP is partial). MTP = layer 78 (`eh_proj`/`enorm`/`hnorm`/`shared_head.norm`). **Absorption characterized (§3c B):** GLM ⇒ engine `n_head_dim=576` (512 c_kv + 64 rope), `n_value_dim=512`, partial `kv_a_norm`; converter folds `kv_b`. This is real engine work, not just a converter fold — **next: resolve the exact fold/dims against a CPU-slice transformers reference, then write the GGUF schema contract** (gates the writer + engine attention). Mechanical converter pieces (config→metadata, tokenizer arrays, GGUF writer, expert stacking, quant) are unblocked and can proceed for non-attention tensors in parallel.
- 2026-06-16: **De-risked the #1 risk.** `check_mla_absorption.py` on real weights (server up) shows the MLA absorption fold is exact (rel ~1e-15, layers 0/3/40) — activation-level (native==absorbed) and the weight-level `attn_q_b` fold. **Engine attention config locked:** `n_head_dim=576`, `n_value_dim=512`, partial `kv_a_norm`; converter folds `kv_b`→`q_b`/`o_proj`, `attn_kv` direct. (numpy in `.venv` via brew python3.12.) Next: GGUF schema contract → converter writer (config→metadata, tokenizer arrays, expert stacking, attention fold, Q8 authoring) → engine load changes. Port 8085 still up (no GPU/RAM step reached yet).
- 2026-06-16 ~21:27: **Entering UNATTENDED mode** (user authorized ~8h; deadline epoch in `/tmp/glm-loop-deadline`, ~05:27 next day). Self-paced via ScheduleWakeup. Drafted the **GGUF schema contract (§5c)** from the exact `config_validate_model` key list. Loop plan each iteration: read this status log → do next chunk (converter writer → engine load deltas → convert+load+validate logits → Q8 ref → imatrix → Q2/Q4 + verify) → commit/push/pull → log → reschedule until deadline. Server stays up except GPU/RAM steps (stop/start `setup-launchagents-tts.sh ... jumbo_server`); **always restart it before stopping the loop.**
- 2026-06-16 ~21:55: Built **GGUF writer core + metadata emission** in `glm-quantize.c` (`--write-gguf`/`--verify`). Emits the 36 `deepseek4.*`/`general.*` KVs per §5c (key_length=576, value_length=512, hash=0, first_k_dense=3, hc.count=1, rope 8e6, expert_scale 2.5, compress_ratios=[0]*78, swiglu_clamp=[30]*78); round-trips correctly. GLM nominal float constants `#define`d to match the forthcoming `DS4_SHAPE_GLM`. `general.architecture` reuses `deepseek4` (unvalidated). Next: `tokenizer.json`→`tokenizer.ggml.{tokens,merges,token_type}` + special ids; then tensor data (norms/embed → attention absorption fold → expert stacking) at Q8.
- 2026-06-16 ~21:45: Added **tokenizer emission** (tokenizer.json → `tokenizer.ggml.{model=gpt2,pre,tokens[154880],token_type,merges,eos/pad}`); JSON unescaping handles `\" \\ \u`. Verified: 154880 tokens (24 reserved placeholders), samples correct (`t[0]='!'`, `t[154820]='<|endoftext|>'`, `t[154822]='[gMASK]'`), 321649 merges; metadata GGUF now 43 KVs / 9.4 MB, round-trips. **Metadata + tokenizer authoring DONE.** Next: tensor DATA — link `quants.c`, write the tensor directory + 32-byte-aligned data; norms F32, embed/output/dense/shared/router Q8_0, routed experts stacked 256→3-D Q8_0, attention absorption fold (`attn_q_b`/`attn_kv`/`attn_output`). Build a Q8-everything reference GGUF (~800 GB; or a layer-sliced subset first for fast engine bring-up).
- 2026-06-16 ~21:55: Built the **tensor-data writer** (`glm-quantize --write-full`/`--layers`, links `quants.c`). Analytic offsets (`ds4q_row_size`) ⇒ streams data, memory-light. Passthrough transforms: token_embd/output/q_a/kv_a/dense-ffn/shared → Q8_0; norms/router/bias → F32. 1-layer test: 12 tensors / 2.29 GB in 3.5 s, offsets 32-aligned, `--verify` reads the directory back (attn_kv ne=[6144,576] ✓). **Remaining for a loadable GGUF:** `attn_q_b` + `attn_output` (absorption fold, math in `check_mla_absorption.py`) and routed experts (stack 256→3-D). TODO during load: confirm ds4's expected types for `token_embd`/`output` and router `ffn_gate_inp` (may be F16 not F32).
- 2026-06-16 ~22:05: **Converter COMPLETE.** Added the absorption fold (`attn_q_b` [2048,36864], `attn_output` [32768,6144]) and expert stacking (`ffn_*_exps`, streamed per-expert since the stacked f32 is 12.9 GB). 4-layer test: 52 tensors / 14.3 GB / 35 s, `--verify` shapes correct, no quant-size errors. Folds = the proven `check_mla_absorption.py` math in C. Full `--layers 78` Q8 ≈ ~800 GB / ~1 hr (background-able; **HOLD until engine type-expectations confirmed**). **Next: ENGINE LOAD DELTAS in `ds4.c`** — start with `DS4_SHAPE_GLM` + `DS4_VARIANT_GLM` + shape selection (values in §5c), then HC bypass, plain `attn_output` bind, dense-first-3 FFN, key=576/value=512 + partial `kv_a_norm`, GLM tokenizer/specials/template/multi-EOS. Bring up on a small `--layers` slice; validate logits vs a numpy reference. This is the largest remaining chunk and the #1 risk's final (semantic) validation.
- 2026-06-16 ~22:25: Engine LOAD PATH committed (88d6eca): GLM weights_bind/validate, variant-gated. Converter types validated (token_embd/ffn_gate_inp→F16, experts→Q4_K — what ds4 requires). **Threaded the converter** (parallel_for over heads/experts): `--layers 4` 7min→27.5s (20× on 28 cores). **Launched full Q4 conversion in background** on notible (PID 29570 → /Volumes/4TB-1/glm-5.2-q4.gguf, ~400 GB, ~35-40 min; poll /tmp/glm-q4-convert.log). Q4 = deliverable + the GGUF to `--inspect` for load-path validation. Next: forward-pass engine deltas (HC bypass, plain attn_output matmul, dense FFN, 576/512 attention) while Q4 converts; then build ds4 + `--inspect` the Q4 GGUF.
- 2026-06-16 ~22:45: **HC bypass DONE (CPU forward).** All ~30 hc_pre/hc_post sites funnel through `hc_pre_from_state_one_scratch`; added a GLM branch there (out=residual, post=1, comb=1 → `hc_post` becomes a plain residual add; attn_norm/ffn_norm still applied by caller). One edit, compiles, DeepSeek path intact. Remaining forward-pass deltas: plain `attn_output` (branch `layer_grouped_out_one` ~7245 → single matmul `attn_output @ heads[64*512]`), dense FFN (branch `layer_ffn_one` ~7954 for `il < DS4_N_FIRST_K_DENSE` → SwiGLU on `ffn_*_dense` at intermediate 12288), attention value=512/key=576 + partial `kv_a_norm` (first 512 of 576), output-head HC-collapse skip (final norm + lm_head only). Then CPU-slice first-token logit validation vs numpy. NOTE: Metal-graph deltas are a separate, large effort after CPU correctness.
- 2026-06-16 ~22:55: GLM **single-token attention deltas** (ds4.c): `kv_a_norm` over first 512 + copy rope tail; `layer_attention_rows_one` — no `attn_sinks` (GLM lacks them; was a NULL-deref), value over `n_value_dim`=512 / key over 576, `kq_scale=1/√256` **[VERIFY vs reference]**; plain `attn_output` matvec (`layer_grouped_out_one` GLM branch). Compiles clean, DeepSeek path intact (`595eb1c`). Remaining CPU forward: dense FFN (`layer_ffn_one` for `il<first_k_dense`), output-head HC-collapse skip, and the batch/decode attention+kv variants (`layer_attention_rows` other sites ~9056/9063/9119, `layer_kv_projection_*_decode_scratch`, `layer_grouped_out_*_batch/decode`). Then first-token logit validation on a 2-layer slice. **Metal graph port = separate large effort (the main thing left to actually RUN on notible at speed).**
- 2026-06-16 ~23:05: GLM **dense FFN** (`layer_dense_ffn_one` + `layer_ffn_one` branch for `il<first_k_dense`) and **output head** (skip HC collapse → final norm + lm_head, both `output_hc_head_one` + decode_scratch variants) done; compile clean (`f59e5bb`). **The single-token CPU forward is now GLM-complete** (HC bypass + attention + dense FFN + output head); MoE layers reuse the existing routed+shared path with GLM tensors (routing math UNVERIFIED vs GLM noaux_tc). The head-test (ds4.c ~25408) uses these single-token functions → a CPU first-token validation on a small slice is now reachable. Remaining: batch/decode variants (for prefill/generation), routing-math check, then the Metal-graph port. Next: --inspect the Q4 GGUF (load-path validation) when it finishes, then attempt CPU first-token logit validation vs numpy on a 2-layer slice.
- 2026-06-16 ~23:05: **🎉 LOAD PATH VALIDATED ON THE REAL 753B MODEL.** Q4 conversion finished (408.7 GiB). Built `ds4` on notible (Metal); `./ds4 -m /Volumes/4TB-1/glm-5.2-q4.gguf --inspect` **succeeds cleanly**: model=GLM-5.2, arch=deepseek4, 78 layers, heads=64/kv=1/head_dim=576, indexer 32/128/2048, experts=256/8, 753.16 B params; types f16 (embed+router, 2 GiB), q8_0 (attn/shared/dense, 27 GiB), q4_k (experts, 380 GiB), f32 (norms). All 1236 tensors **bind + validate** → the converter↔engine contract is PROVEN end-to-end on the full model. Q4 GGUF is a complete deliverable. Next: CPU first-token forward validation (need `--load-layers` for a safe 2-layer slice + numpy reference) to validate forward correctness; then Metal-graph port (to run at speed).
- 2026-06-16 ~23:20: **🎉 GLM CPU FORWARD NUMERICALLY VALIDATED.** `./ds4 --cpu --head-test -p "..."` on the real Q4 model runs layer-0 (dense) + output head and is FINITE/SANE throughout: `q` rms=1.0 (per-head norm ✓), kv/attn/ffn reasonable, **logits min=-9.0 max=9.5 rms=1.84** (no NaN/inf), top tokens are real BPE tokens. The absorbed-MLA attention (576 key/512 value, kq_scale 1/√256, no sinks) + HC bypass + plain o_proj + partial kv_a_norm + dense FFN + output head all work end-to-end on the real model. Session fixes to get here: GLM vocab special tokens (optional lookup), `config_validate_model` BEFORE `vocab_load` (so variant is set), head-test inverse-rope skip. CAVEATS: layer-0-only (no meaningful prediction; MoE path untested); not yet exact-vs-reference. Next: full first-token prediction (server DOWN for the 430 GB model) and/or numpy exactness; MoE; Q2; Metal.
- 2026-06-16 ~23:35: First-token-test (full forward, Q4) **isolated a MoE bug**: cleared dense layers 0-2 + attention, FAILED at first MoE layer (3) "expert id outside expert tensor". Two GLM bugs found+fixed (`5c1e6dc`): (1) `layer_forward_self_one` (the real per-layer forward) applied the **inverse rope** to GLM's `[64×512]` value with 576-stride → corruption (only `head_test` had the skip); now skipped for GLM. (2) router used DeepSeek `sqrt(softplus)`; GLM `scoring_func=sigmoid` (noaux_tc) → now sigmoid for GLM, so bias + flat top-8 + normalized weights (`norm_topk_prob` × `routed_scaling_factor` 2.5) match GLM's noaux_tc. Re-running first-token-test. (Lesson: forward-path deltas must be applied to the real forward fns, not just head_test; batch/decode variants still need the same.)
- 2026-06-16 ~23:40: **ROOT CAUSE of the MoE crash found:** `DS4_MAX_EXPERT_USED` was **6** (DeepSeek's max) but GLM uses **8** experts/token → the `selected[6]` / `gate_base[6]` / `expert_weight[6]` stack arrays overflowed when `topk_desc` wrote 8 entries → garbage expert ids ≥256 → "expert id outside tensor". Fixed to 8 (`370a4f5`); DeepSeek still uses 6 at runtime. (The inverse-rope + sigmoid-router fixes were also real and needed.) Re-running first-token-test → /tmp/glm-ft3.log.
- 2026-06-16 ~23:50: **🎉🎉 FULL GLM FORWARD WORKS END-TO-END ON THE REAL 753B MODEL (CPU).** `--first-token-test` ran ALL 78 layers (dense 0-2 + MoE 3-77 with sigmoid noaux_tc routing) on the Q4 GGUF and produced finite logits (min=-9.3 max=28.7 rms=2.2; final_hc rms=41, normal residual growth). Top token = `[gMASK]` (28.7) then Sign/comments/add — `[gMASK]`-top is EXPECTED for a bare out-of-distribution single token (GLM trains on `[gMASK]<sop>` framing). **#1 forward-correctness risk RETIRED on CPU:** absorbed-MLA attention + HC bypass + dense/MoE FFN + noaux_tc routing + output head all compose correctly over 78 layers. The earlier "silent crash" was a transient memory/VM kill (430 GB on CPU), NOT a code bug. Bugs fixed to get here this session: inverse-rope skip (real fwd), sigmoid router, **DS4_MAX_EXPERT_USED 6→8** (overflow), GLM vocab tokens, config_validate before vocab_load. CAVEAT: numerically-sound + plausible, NOT exact-vs-reference; a meaningful continuation needs `[gMASK]<sop>` framing + full-prompt prefill (batch path, not yet GLM-adapted). NEXT: build Q2 (primary target, fits RAM for a cleaner run), then exactness check / batch path / Metal.
- 2026-06-16 ~23:55: Added converter **--q2 mode** (IQ2_XXS gate/up + Q2_K down + synthetic per-column-energy importance). Validated --layers 4 (type=16/10, no errors). GLM Q2 reuses DeepSeek's exact expert quant types → the proven IQ2/Q2_K CPU+Metal kernels apply directly. **Launched full Q2 conversion** (PID 30279 → /Volumes/4TB-1/glm-5.2-q2.gguf, ~210 GB; poll /tmp/glm-q2-convert.log). Q2 fits 256 GB → cleaner full-model validation run. Next: --inspect + head-test + first-token-test on Q2; compare Q2 vs Q4 top tokens.
- 2026-06-16 ~22:10: **Engine step 1 (shape) DONE.** Added `DS4_VARIANT_GLM` + `DS4_SHAPE_GLM` (§5c values) + `n_first_k_dense` field/macro to ds4.c; extended `ds4_select_shape_from_metadata` to recognize GLM; `ds4_expected_layer_compress_ratio`→0 for GLM. `ds4.o` compiles clean (additive; DeepSeek path untouched). Next behavior deltas (each gated on `variant==GLM`, own commit): HC bypass (§3c site list), plain `attn_output` bind, dense-first-3 FFN, attention key=576/value=512 + partial `kv_a_norm`. Then CPU-slice bring-up + logit validation. (Pre-existing clang-tidy int-division lint at ds4.c:6882 is unrelated.)
