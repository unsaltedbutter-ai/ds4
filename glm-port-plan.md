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
| Remote code (notible) | `notible:/Users/notible/Documents/ds4-glm` (checkout exists, currently on `main`) |
| Production DeepSeek (do not disturb) | `notible:/Users/notible/Documents/ds4` on `main`, server port 8085 |
| Model weights | `notible:/Volumes/4TB-1/glm-5.2/` (downloading — ~18% / 31 of 282 shards / 276 GB; 3.3 TB free) |
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
launchd agent. Phase 0/1 and most of 2 are ✅; the gate is first crossed in Phase 3/4.

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

### Phase 0 — Groundwork (pre-model) ← **WE ARE HERE**
- [x] Architecture investigation + verified config + geometry map (this doc + porting.md)
- [x] Read GLM chat template + special tokens (groundwork for §1b)
- [ ] Establish git sync: commit groundwork, push `ds4-glm`, check out on notible
- [ ] Add `DS4_SHAPE_GLM` profile + relaxed shape gate so a GLM GGUF *loads* and reports correct dims (compiles; untestable until model)
- [x] R1 spike: HC traced; verdict = HC-bypass path, ~6-8 sites, not a rewrite (see §3c)
- [x] Attention/FFN graph-delta map: o_proj (LOW), q/kv up-proj (MED), dense first-3 (LOW-MED) (see §3c)
- [ ] Draft GLM prompt-rendering + `<tool_call>` parser design (have the template)
- [x] Tensor-name schema captured from shard headers (§2b) — router/shared/MTP names + `index.json` still pending
- [x] Converter skeleton `gguf-tools/glm-quantize.c`: safetensors scanner + GLM→ds4 map + `--dry-run` coverage (validated on partial download; router/shared/bias names confirmed)
- [ ] Add `CLAUDE.md` for the fork (GLM context, points here)
- [ ] Capture golden-logit plan (HF `transformers` reference; can't run until weights complete)

### Phase 1 — Converter + load (needs weights)
- [x] Tensor-name map validated via `--dry-run`; router/shared/noaux_tc-bias names confirmed
- [x] Converter reader: bf16/f32 dequant + `--read` (validated on real tensors: norms all-positive, projections ~±0.17 mean≈0)
- [x] Absorbed-MLA representation resolved + fold **proven** (`check_mla_absorption.py`); attention config locked (`n_head_dim=576`/`n_value_dim=512`/partial `kv_a_norm`)
- [ ] **GGUF schema contract** doc: finalize metadata keys + full tensor set (HC-off / plain o_proj / dense-first-3 / no compressor·sinks·hash)
- [ ] Converter writer: config→metadata + GGUF authoring + MLA absorption (fold kv_b) + expert stacking + quantize; Q8-everything first
- [ ] Model loads in ds4, `--inspect` reports GLM dims, tensors bind

### Phase 2 — CPU numerical bring-up ⚠️ (riskiest, cheapest place to prove it)
- [ ] HC=1 path, dense first-3 layers, plain o_proj, GLM RoPE, indexer OFF (dense attn)
- [ ] Single correct forward pass: logits match HF reference on short prompts

### Phase 3 — Metal + server
- [ ] Metal graph at HC=1; tokenizer + chat template + multi-EOS + `<tool_call>`
- [ ] 🔴 Server answers a real prompt end to end (first Metal run on notible — needs port 8085 down)

### Phase 4 — Q2 + Q4 build & streaming on notible
- [ ] 🔴 Build imatrix (runs the model on Metal — needs port 8085 down)
- [ ] Produce **Q2** GGUF: IQ2_XXS gate/up + Q2_K down + Q8_0 rest (~210-225 GB)
- [ ] Produce **Q4** GGUF: Q4_K experts + Q8_0 rest (~400-440 GB; on 256 GB this is **SSD-streaming only**)
- [ ] **Compare Q2 vs Q4**: logit/quality parity vs reference + inference speed (expected: keep Q2 if good enough, but make the comparison real)
- [ ] Apply the §3b (M1) memory-fit levers; tune SSD-streaming expert-cache vs context budget

### Phase 5 — Later
- [ ] Re-enable DSA indexer (IndexShare) for long context
- [ ] Wire GLM MTP into `--mtp`
- [ ] Capture GLM validation vectors; rebrand CLI/server/README; license hygiene

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

## 6. Status log
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
- 2026-06-16 ~22:10: **Engine step 1 (shape) DONE.** Added `DS4_VARIANT_GLM` + `DS4_SHAPE_GLM` (§5c values) + `n_first_k_dense` field/macro to ds4.c; extended `ds4_select_shape_from_metadata` to recognize GLM; `ds4_expected_layer_compress_ratio`→0 for GLM. `ds4.o` compiles clean (additive; DeepSeek path untouched). Next behavior deltas (each gated on `variant==GLM`, own commit): HC bypass (§3c site list), plain `attn_output` bind, dense-first-3 FFN, attention key=576/value=512 + partial `kv_a_norm`. Then CPU-slice bring-up + logit validation. (Pre-existing clang-tidy int-division lint at ds4.c:6882 is unrelated.)
