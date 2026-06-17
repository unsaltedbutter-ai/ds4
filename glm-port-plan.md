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

---

## 3. Risk register (ranked)

- **R1 — Hyper-Connections removal (HIGH, gating).** ds4's residual stream uses HC
  (`n_hc=4`, sinkhorn; ~562 refs in `ds4.c`, plus `metal/dsv4_hc.metal`, KV payload,
  distributed proto). GLM uses standard residuals. Must run `n_hc=1`. Unknown whether
  that is a clean degenerate path or needs graph surgery. **Not mentioned in porting.md —
  prove this first (Phase 2 spike) before committing to a timeline.**
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

---

## 4. Phased plan

### Phase 0 — Groundwork (pre-model) ← **WE ARE HERE**
- [x] Architecture investigation + verified config + geometry map (this doc + porting.md)
- [x] Read GLM chat template + special tokens (groundwork for §1b)
- [ ] Establish git sync: commit groundwork, push `ds4-glm`, check out on notible
- [ ] Add `DS4_SHAPE_GLM` profile + relaxed shape gate so a GLM GGUF *loads* and reports correct dims (compiles; untestable until model)
- [ ] R1 spike: trace HC through the graph; decide `n_hc=1` degeneracy vs strip. Write findings here.
- [ ] Draft GLM prompt-rendering + `<tool_call>` parser design (have the template)
- [ ] Draft converter skeleton: bf16 safetensors reader + tensor-name map (full names pending `index.json`)
- [ ] Add `CLAUDE.md` for the fork (GLM context, points here)
- [ ] Capture golden-logit plan (HF `transformers` reference; can't run until weights complete)

### Phase 1 — Converter + load (needs weights)
- [ ] safetensors(bf16) → GGUF converter, Q8_0-everything first (correctness ref, ignore size)
- [ ] Model loads in ds4, `--inspect` reports GLM dims, tensors bind

### Phase 2 — CPU numerical bring-up ⚠️ (riskiest, cheapest place to prove it)
- [ ] HC=1 path, dense first-3 layers, plain o_proj, GLM RoPE, indexer OFF (dense attn)
- [ ] Single correct forward pass: logits match HF reference on short prompts

### Phase 3 — Metal + server
- [ ] Metal graph at HC=1; tokenizer + chat template + multi-EOS + `<tool_call>`
- [ ] Server answers a real prompt end to end

### Phase 4 — Q2 + Q4 build & streaming on notible
- [ ] Build imatrix (include coding-agent traffic per porting.md §5)
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

---

## 6. Status log
- 2026-06-16: Investigation complete. Confirmed GLM-5.2 = `glm_moe_dsa` (MLA+DSA+MoE, DeepSeek-V4 cousin). Verified config, geometry map, GLM chat template. Identified **Hyper-Connections removal (R1)** as the gating risk (missing from porting.md). Branch `ds4-glm` ready locally; notible checkout exists on `main`; model ~18% downloaded.
- 2026-06-16: Per user, added **Q2 and Q4 both as Phase 4 build targets** (Q4 = SSD-streaming-only on 256 GB) for a real quality/speed comparison, and the **§3b M1 memory-fit study** (quantized KV is the key lever for Q2 + large context on 256 GB). Pushing groundwork to `origin`; checking out `ds4-glm` on notible. Next: R1 HC spike + attention/FFN graph-delta map.
