# GLM-5.2 Q2 — what it is, how it was built, and how to make it better

Scope: the **Q2** GGUF on branch `ds4-glm` (`/Volumes/4TB-1/glm-5.2-q2.gguf`, 218.9 GiB)
— the fast, resident serving target on a 256 GB M3 Ultra (~11.4 tok/s greedy decode). This doc
covers (1) exactly how the current Q2 was produced, (2) what works and what doesn't when
you run it, and (3) a set of candidate recipes for a *better* Q2 plus a plan to test them.

It is the Q2-quality companion to `README-GLM.md` (getting-started) and `glm-port-plan.md`
(the port tracker). For why Q2 (not Q4) is the speed target, see `glm-port-plan.md` §4/§6:
Q4 (≈409 GiB) cannot be resident on 256 GB and is disk-bound (~0.76 tok/s); Q2 fits.

> **RECOMMENDATION (2026-06-18, after the improvement campaign).** Two tiers, pick by priority:
> - **Best quality → `glm-5.2-q4.gguf`** (408.7 GiB, full 256 experts Q4_K, streamed ~1.1 t/s).
>   It is the **only** build that completes a complex instruction: on "list 5 planets *and give a
>   fact about each*" it actually produces the facts (Mercury smallest/no moons, Venus hottest,
>   Jupiter largest). Every Q2 recipe fails this — they loop on the list and emit no fact. If
>   quality is paramount and tok/s is acceptable (you said it is), use Q4.
> - **Best resident/fast → `glm-5.2-q2.gguf`** (219 GiB, ~11 t/s). This canonical name now ships the
>   **imatrix-weighted** Q2 (the original synthetic-importance baseline was promoted out and deleted):
>   IQ2_XXS gate/up weighted by a real 20k-token activation imatrix. Clean and correct on *simple*
>   prompts (greedy gives a correct planet list where the old Q2 looped on "Mercury"), but it cannot
>   do complex prompts — the **2-bit IQ2_XXS gate/up ceiling**, which no Q2 recipe (denser imatrix,
>   down→Q4_K) lifts; only Q4-level gate/up does. Reproduce via `scripts/glm-imatrix-iterate.sh`.
>
> **No usable middle tier exists** between these two (tested exhaustively — §5): lifting *some*
> layers' gate/up to Q4 (`--q4-layers`) either doesn't help (≤~12 layers, the most that fits
> resident) or can't run (more layers → too big for resident, and the Q4_K-only streaming kernels
> garble a model whose down is still Q2_K). The only thing that streams correctly with Q4 gate/up is
> *uniform* Q4_K — i.e. full Q4. External HuggingFace GGUFs (unsloth UD-IQ2, REAP50 Q2/Q3) **don't
> load in ds4** (glm-dsa, patched-llama.cpp-only) and aren't higher quality (REAP50 is pruned +
> degraded; unsloth is 2-bit). A middle tier (Q4 gate/up + Q2_K down, 356 GiB) was built and made
> to stream via a new mixed-precision kernel (§5): it reaches **Q4-class quality** (it produces the
> facts — gate/up is the quality driver) but **does not beat Q4** — terser facts (Q2_K down) and
> ~the same disk-bound ~1 t/s, only 13% smaller. So the two tiers above are the answer; there is no
> usable third tier on this hardware.

---

## 1. How the current Q2 was created

> **Note (post-campaign):** §1 describes the **original** Q2 recipe (synthetic importance), which was
> the shipping `glm-5.2-q2.gguf` through the campaign. As of 2026-06-18 the canonical
> `glm-5.2-q2.gguf` is the **imatrix-weighted** build (§5): identical recipe except IQ2_XXS gate/up
> is weighted by a real 20k-token activation imatrix. Everything in §1 still describes the layout;
> only the gate/up importance source changed. Rebuild either via `scripts/glm-imatrix-iterate.sh`
> (imatrix) or `scripts/glm-q2-build.sh OUT.gguf` (synthetic).

### 1.1 One direct pass: bf16 → f32 → 2-bit. There is **no** Q8 intermediate.

The converter is `gguf-tools/glm-quantize.c`. It reads the Hugging Face `zai-org/GLM-5.2`
release (bf16 safetensors) and authors the GGUF in a single pass. For every tensor it:

1. reads the bf16 source and widens it to f32 (`stdb_read_f32`; bf16→f32 is **exact** —
   f32's mantissa is a superset of bf16's, so this step loses nothing), then
2. quantizes that f32 **directly** to the tensor's target type (`ds4q_quantize_chunk`,
   `glm-quantize.c:918` / the per-expert path at `:790`).

So the answer to "did we go full → Q8 → Q2, with an extra lossy hop?" is **no**. The only
lossy step is the final f32 → 2-bit quantization. We are already at the theoretical minimum
number of lossy steps; there is no intermediate-precision GGUF to eliminate. (A Q8
"reference" GGUF was *contemplated* during bring-up — `glm-port-plan.md` §6, 2026-06-16 —
but it was never built or used as a quantization source; Q2 and Q4 were each quantized
straight from the bf16 weights.)

The command that produced the file:

```sh
cd gguf-tools && make glm-quantize
./glm-quantize --hf /path/to/GLM-5.2-hf --q2 --write-full glm-5.2-q2.gguf
```

`--q2` flips two globals (`glm-quantize.c:1089`): routed-expert **gate/up → IQ2_XXS**,
routed-expert **down → Q2_K**. Without `--q2` the experts are Q4_K (that is the Q4 build).
The converter is threaded (`parallel_for` over heads/experts, ~28 cores on notible); a full
78-layer Q2 write is roughly 40–60 min. `--layers N` writes only the first N layers for a
fast partial build (load/plumbing checks; not end-to-end coherent).

### 1.2 The exact per-tensor recipe (`build_plan`, `glm-quantize.c:801`)

| Tensor group | Type | bits/weight | Notes |
|---|---|---|---|
| `token_embd` (embeddings) | F16 | 16 | |
| `output` (lm_head, untied) | **Q8_0** | 8.5 | README-GLM.md rounds this to "F16"; the converter emits Q8_0 |
| `output_norm`, all `*_norm`, `exp_probs_b.bias` | F32 | 32 | tiny |
| `ffn_gate_inp` (router) | F16 | 16 | per-layer 6144×256 |
| attention `attn_q_a/q_b/kv/output` | Q8_0 | 8.5 | absorbed MLA fold |
| dense FFN `ffn_{gate,up,down}_dense` (layers 0–2) | Q8_0 | 8.5 | the 3 dense layers |
| shared expert `ffn_{gate,up,down}_shexp` | Q8_0 | 8.5 | |
| **routed experts gate/up** `ffn_{gate,up}_exps` (L≥3) | **IQ2_XXS** | **2.0625** | 256 experts × 75 layers |
| **routed experts down** `ffn_down_exps` (L≥3) | **Q2_K** | **2.625** | 256 experts × 75 layers |

**The routed experts are ~93% of the file** (≈204 GB of 219): gate/up at IQ2_XXS ≈ 124 GB,
down at Q2_K ≈ 79 GB; everything non-routed ≈ 15 GB. So Q2 quality is, to first order,
*entirely* about how well those 2-bit expert tensors are quantized. Everything else is
already at 8–16 bits and is not the bottleneck.

### 1.3 The importance weighting — and a gap

Low-bit quantizers choose per-block scales/codes to minimize error, and they do it *better*
when told which input columns matter (an "importance matrix" / imatrix). The current build:

- **IQ2_XXS gate/up:** IQ2_XXS *requires* an imatrix (`quants.c:54`,
  `requires_imatrix=true`). With no real activation imatrix available, the converter
  synthesizes one: **per-column weight energy**, `imat[c] = Σ_rows weight[r,c]²`
  (`glm-quantize.c:783`). This is a *proxy* for importance derived from the weights
  themselves — it has no knowledge of which columns the model actually activates.
- **Q2_K down:** Q2_K does **not** require an imatrix (`requires_imatrix=false`), so the
  converter passes `NULL` and Q2_K falls back to its unweighted reference path
  (`ds4q_write_q2_k_block_ref`, `quants.c:656`). **The down projection is currently
  quantized with no importance information at all** — even the cheap synthetic one.

This is the single clearest weakness in the current recipe and the starting point for §3.

---

## 2. Running the current Q2: what works, what doesn't

### 2.1 What works

- **Loads and runs resident.** `--inspect` binds + validates all 1236 tensors; the model
  fits in 256 GB with a usable context and decodes at **~11.4 tok/s** greedy / ~9.9 sampled
  (GPU sigmoid router, routed MoE on Metal). The server answers GLM prompts end to end.
- **The right facts come out, in order.** Every run in §2.3 produced the correct planets
  (Mercury → Jupiter) at the start — the model's knowledge and the engine are intact. Q2 is
  deterministic under greedy (byte-identical run-to-run).
- **Engine correctness is not the issue.** The Metal forward matches the CPU reference to fp
  noise (`--metal-graph-full-test`), CLI == server, and Q4 (same engine/prompt/settings)
  stays coherent where Q2 degrades — so degradation is the **quantization**, not a bug
  (`glm-port-plan.md` §6, 2026-06-17/18).

### 2.2 What doesn't — the temperature × length decoherence

The known failure mode (and the reason the GLM default was moved to **temp 0.6 / top_p
0.95**, not GLM's nominal temp 1.0):

- **Looping / not-stopping is pervasive; temperature and length set the severity.** Even
  greedy on the *short* prompt loops (§2.3, repeats "...Mercury."). On the long instruction
  ("list the planets *and give a fact about each*") greedy lists them then repeats with
  errors; **temp 0.6 collapsed into word-salad** in our draw; temp 1.0 lists then loops.
  Higher temperature and longer / more-complex prompts make it worse — confirming the
  recollection — and at the extreme it fragments (repeated tokens, stray CJK, truncations
  like "2. Uran").
- **Mechanism:** 2-bit quantization flattens the model's output distribution. Greedy still
  picks the right top token on easy steps but doesn't know when to *stop* (over-generates /
  loops); sampling at temp 1.0 then draws from a noisy, over-flat tail. `top_p 0.95` (nucleus
  filtering) is what keeps sampled Q2 usable — the earlier "garbage at temp>0" was largely
  the old `top_p=1.0` default doing no filtering at all.

### 2.3 Empirical battery (this branch, Q2, `-c 2048`)

Run with `scripts/glm-q2-characterize.sh` on notible. Prompts: **SHORT** = "List the first
five planets from the Sun."; **LONG** = "...and give one interesting fact about each one."

<!-- RESULTS:BEGIN (filled from the 2026-06-18 run) -->
Generations are **thinking-on** (CLI one-shot default): the captured text is the model's
reasoning trace, and Q2 tends to loop *inside* it without emitting `</think>` + a clean final
answer. Speed: prefill ~11.4 t/s; decode ~11.4 (greedy) / ~9.9 (sampled).

| Run | temp | top_p | gen t/s | Behavior |
|---|---|---|---|---|
| short_t0  | 0   | —    | 11.44 | Identifies Mercury, then **loops** "The first planet from the Sun is Mercury." — never completes the list |
| short_t10 | 1.0 | 0.95 | 9.86  | Emits the correct list (1–5, Mercury→Jupiter) once, then repeats and degrades ("1. Jupiter") |
| long_t0   | 0   | —    | 11.36 | Lists 1–5 correctly, then repeats the list with an error ("4. Jupiter"); gives no facts; loops |
| long_t06  | 0.6 | 0.95 | 9.87  | Correct list, then **word-salad**: "...11. Mercury,V Mars,11.1 Mercury / 1 Mercury, Mars, Mercury" — most decoherent of the five |
| long_t10  | 1.0 | 0.95 | 9.87  | Lists 1–5 correctly, repeats the preamble + list, loops |

**What the battery shows:**
- Every mode produces the **correct planets in the correct order at the start** — the model's
  knowledge and the engine are intact. Q2's failure here is **degeneration / looping and not
  stopping**, not factual error.
- **Nothing completes the task cleanly** (5 planets + a fact each, then stop) — not even
  greedy. Greedy is deterministic but loops; sampling can get the list out once but then
  repeats or fragments.
- **Decoherence worsens with length and temperature**, confirming the recollection. The long
  prompt at temp 0.6 collapsed into word-salad in this draw (sampling is stochastic — one
  sample is not definitive, but it shows how fragile Q2's tail is on a complex instruction).
- Caveat / follow-up: because these are thinking-on, part of the looping is the model
  spinning in its reasoning trace. A fair quality read should **also test thinking-off**
  (`reasoning_effort: none` via the server) — added to §4.
<!-- RESULTS:END -->

---

## 3. Options for a better Q2

Goal: improve Q2 quality (less looping / decoherence, lower NLL vs reference) **while
staying resident on 256 GB** (≈248 GB usable budget for weights + KV + scratch). Because the
routed experts are 93% of the bytes and already at ~2 bits, the leverage is overwhelmingly in
*how* those experts are quantized — not in the 8–16 bit remainder. Options, best
quality-per-effort first.

### Option A — Real activation imatrix (highest quality leverage)

Replace the synthetic per-column-energy proxy with a **real** importance matrix collected by
running the model over a calibration set. This is the standard win for 2-bit MoE quants. On
DeepSeek, the same tooling gave **−1.95% NLL** on 100 official continuations at *Q4*; the
effect is expected to be **larger at Q2**, where bit budget is scarce and importance-aware
scale/code selection matters most (`gguf-tools/imatrix/README.md`).

Three pieces are needed; the infrastructure exists for DeepSeek but was **not** ported to GLM:

1. **Converter: add `--imatrix FILE`.** `glm-quantize.c` has no imatrix loader; the full
   loader (llama.cpp legacy `.dat`, per-expert 256-vector slicing) already exists in
   `gguf-tools/deepseek4-quantize.c` (`imatrix_load`, `imatrix_find`). Port it. **LOW–MED**
   (the GLM expert layout is identical: 256 vectors per routed tensor).
2. **Also weight the down_proj.** Pass the imatrix to Q2_K down (today it gets `NULL`). Q2_K
   already supports it (`ds4q_write_q2_k_block_weighted`, `quants.c:654`). **Trivial.**
3. **Collect a GLM imatrix — the blocker.** `ds4_engine_collect_imatrix` (`ds4.c:25844`)
   hooks `metal_graph_prefill_layer_major` / `..._chunked_range`, i.e. the **batch prefill
   that is not GLM-adapted and SIGSEGVs on GLM** (no HC/compressor tensors; commit `7701d0a`
   routed GLM prefill through sequential decode for exactly this reason). So collection does
   not work for GLM today. Two ways forward:
   - **(b) Re-point the collector at GLM sequential decode** — accumulate the same per-expert
     `Σ x[c]²` during `metal_graph_eval_token_raw_swa`. **MED** engine work, no batch prefill
     needed. *Practical cost:* sequential decode is ~11 tok/s, so a large 1.5M-token imatrix
     would take ~38 h — infeasible. A **small** budget (`--imatrix-max-tokens` ≈ 32k–130k,
     ~1–3 h at decode speed) is the realistic target and is the small-budget shape
     DeepSeek-Pro used (`--imatrix-max-tokens 32768`).
   - **(a) Adapt the GLM Metal batch prefill** (a deferred speed item anyway) — then
     collection is fast and a large imatrix becomes feasible. **MED–HIGH**, but it unblocks
     both speed and a high-quality imatrix.
   - Secondary: the tracked calibration dataset is DeepSeek/DSML-rendered; re-render it with
     GLM `[gMASK]<sop>` framing (`build_ds4_imatrix_dataset.py`) for representativeness.

**A0 (free first step):** even without a real imatrix, pass the *existing synthetic*
importance to the Q2_K down_proj (currently `NULL`). One-line change, full rebuild, measure.
It isolates "does importance on `down` help at all" before investing in collection.

### Option B — Bit-allocation recipe changes (uses only already-implemented types)

The converter implements quantizers for Q8_0, Q4_K, **Q2_K, IQ2_XXS**, F16, F32 — and the
engine has Metal MoE kernels for exactly these. So these recipes need **no new kernels**:

- **B1 — gate/up Q2_K instead of IQ2_XXS** (2.0625 → 2.625 bpw). Q2_K is a non-IQ K-quant
  (no codebook grid) and can be more robust; cost ≈ **+33 GB** (→ ~252 GB total). Likely too
  big to stay resident with real context — test on a `--layers` slice / measure fit.
- **B2 — sensitivity-aware Q4 layers.** Keep the first few and last few MoE layers' experts
  at Q4_K (empirically the most error-sensitive), rest at IQ2_XXS/Q2_K. ≈ **+2.7 GB per
  upgraded layer**; ~8 layers ≈ +22 GB (→ ~241 GB) — tight but may fit at small ctx,
  especially if paired with Option D. This is a tunable quality/RAM knob.
- **B3 — down_proj Q4_K** (whole-model). down is often the most sensitive expert projection;
  but 2.625 → 4.5 bpw is **+57 GB** (→ ~276 GB) — resident-infeasible, streaming-only. List
  for completeness; probably out of budget without D.

Each B recipe is a few lines in `build_plan` + a full rebuild. Gate every change on the §4
eval, not vibes.

### Option C — Better low-bit quant types (needs new quantizer **and** Metal kernel)

Types in the enum but `can_quantize=false` today (`quants.c:39`): **IQ2_S** (2.5625 bpw,
needs imatrix), **IQ2_XS** (2.3125), **Q3_K** (3.4375), **IQ3_XXS** (3.0625). IQ2_S/IQ2_XS
are the natural "better than IQ2_XXS at ~the same size" upgrades for gate/up; Q3_K/IQ3_XXS
are mid-bit options for down or sensitive layers. **Caveat — this is the most expensive
option:** each new type needs *both* a converter quantizer in `quants.c` *and* a Metal MoE
dequant/matmul kernel in `metal/moe.metal` (the runtime only has IQ2_XXS/Q2_K/Q4_K expert
kernels).

> **Status (2026-06-19): not pursued — blocked by the same wall as `--q4-layers`.** A 3-bit
> gate/up model (e.g. all Q3_K gate/up, ~280 GiB) is too big to be resident (~240 GiB cap) and
> would need a Q3_K *streaming* slots8 kernel (which doesn't exist) — the same non-uniform-precision
> streaming limitation that kills the `--q4-layers` middle tier (§5). So implementing IQ2_S/Q3_K
> would *also* require the streaming-path fix to be usable. The cheaper prerequisite is therefore
> the streaming fix itself; new quant *types* are only worth it after that, and only if a 3-bit
> point proves better quality-per-GiB than the Q4-gate/up point.

### Option D — 8-bit latent KV (enabler, not a direct expert-quality lever)

Already a Phase 4 item (`glm-port-plan.md` §3b; ds4 has FP8 KV kernels in
`metal/dsv4_kv.metal`). Storing the MLA latent KV at 8-bit halves KV (~88 → ~44 KB/tok). It
doesn't improve expert quantization directly, but it **frees RAM**, which is what makes
Option B2 (a few Q4 expert layers) fit resident, or buys a larger context. Include it as the
budget-maker that unblocks B.

### Option E — Direct-vs-staged (the user's hypothesis): already optimal

Covered in §1.1: the converter already goes bf16 → f32 → 2-bit directly, with the f32 step
exact. There is no Q8 (or other intermediate-precision) hop to remove. **No action** — this
option is closed; the win is in importance (A) and allocation (B), not in restructuring the
pipeline.

### Summary table

| Opt | Change | New kernels? | Rebuild | Fits 256 GB? | Effort | Expected gain |
|---|---|---|---|---|---|---|
| A0 | synthetic imatrix on `down` too | no | full | yes (same size) | trivial | small–med |
| A | real activation imatrix (gate/up + down) | no | full + collect | yes (same size) | med (collector) | **largest** |
| B1 | gate/up IQ2_XXS→Q2_K | no | full | ~no (+33 GB) | low | unknown |
| B2 | first/last MoE layers Q4_K | no | full | tight (+~22 GB) | low | med (targeted) |
| B3 | down IQ2/Q2_K→Q4_K | no | full | no (+57 GB) | low | med, off-budget |
| C | IQ2_S / Q3_K / IQ3_XXS | **yes** | full | depends | high | med–large |
| D | 8-bit latent KV | (KV path) | none | frees RAM | med | enables B/ctx |
| E | skip Q8 stage | — | — | — | none | none (already direct) |

---

## 4. Test plan

We will build candidate GGUFs, run an identical evaluation on each, and record the numbers in
the tracker below. "Better" = lower NLL vs reference **and** fewer decoherence failures, at a
size that stays resident.

### 4.1 Evaluation (run identically on every candidate)

1. **Load check.** `./ds4 -m CAND.gguf --inspect` — binds/validates, prints size + types.
2. **Decoherence battery.** `scripts/glm-q2-characterize.sh` (SHORT/LONG × temp {0, 0.6,
   1.0}, top_p 0.95). Pass/fail = coherent, stops cleanly, no loop/CJK/`</think>`-spam. Run
   it **both thinking-on and thinking-off** (`reasoning_effort: none` via the server): the
   2026-06-18 baseline ran thinking-on and much of the looping was inside the reasoning
   trace, so thinking-off isolates how much of the degeneration is reasoning-loop vs answer.
3. **Greedy fidelity vs Q4** (the quantitative metric — see the metric note below). Q4 is the
   high-quality local anchor (same engine). For each prompt, generate Q4 greedy once (cache
   it), then generate each Q2 candidate greedy and measure **agreement with Q4** (longest
   common token prefix / token-overlap). Higher agreement = less quantization damage. This is
   the DeepSeek imatrix table's "greedy LCP" metric, and it is in-distribution and sensitive.
   For an external gold metric, `gguf-tools/quality-testing/` scores against official-API
   continuations by NLL, but it is DeepSeek-shaped and needs GLM adaptation.
4. **Speed + fit.** Record load time, decode tok/s, file size, and resident-vs-streaming.

> **Metric note (2026-06-18): raw-prose perplexity does NOT discriminate quant quality for
> GLM — do not use it as the primary metric.** I built `--perplexity-file` scoring on held-out
> English (`scripts/glm-ppl-heldout.txt`) and found **Q4 ppl ≈ 3079 and Q2 ppl ≈ 3184** on the
> same text (avg_nll 8.03 vs 8.07) — a near-lossless Q4 scores essentially the same catastrophic
> ppl as Q2. The logprob math is correct (verified) and tokenization is healthy (~5 char/token),
> so this is not a bug: GLM-5.2 is a heavily post-trained *reasoning* model and is deeply out of
> distribution on un-framed prose continuation, so its own uncertainty (high entropy everywhere)
> swamps the quantization signal. Prepending the `[gMASK]<sop>` BOS frame (now in the harness)
> only moved it 6310→5227. **Conclusion:** use greedy-fidelity-vs-Q4 + the behavioral battery,
> which are in the model's competence zone, as the real metrics. Prose-ppl is kept only as a
> weak tertiary (it is deterministic, so a *large* consistent move would still be a signal).

Metrics recorded per candidate: file size (GiB), resident vs streaming, decode tok/s, greedy
agreement vs Q4, the decoherence battery result, and (tertiary) prose avg_nll.

### 4.2 Build/iterate discipline

- **Plumbing first, quality second.** Validate a recipe change with a `--layers 12` partial
  build (load + quant-size correctness, seconds–minutes) before a full ~40–60 min build.
- **Disk budget.** Candidate GGUFs all live on `/Volumes/4TB-1` (≈3.4 TB free after q2+q4),
  built with `scripts/glm-q2-build.sh` (prod-safe). Keep ~2–3 at once; delete losers.
- **Larger-than-RAM is allowed.** Per the user, a Q2 that exceeds 256 GB and streams a bit is
  acceptable for better quality — so higher-bit recipes are in scope (with a streaming speed
  cost), not just resident ones.
- **Gate on the eval, one variable at a time.** Don't stack recipe changes within a single
  build or we can't attribute the delta.

> **Kernel constraint (2026-06-18, from `metal/moe.metal`).** The gate/up fused `pair_swiglu`
> MoE kernel exists only for **IQ2_XXS and Q4_K** — there is **no Q2_K pair_swiglu**. So
> gate/up can be IQ2_XXS (2.06 bpw, resident ~219 GB) or Q4_K (4.5 bpw, ~366 GB ≈ full Q4),
> with nothing in between without a new kernel. The `down` proj is a plain id-matmul and
> supports Q2_K / Q4_K / IQ2_XXS / Q8_0. **Implication:** the only sub-Q4 gate/up option is
> IQ2_XXS, which is exactly the quant that most needs a good imatrix — so **Option A (real
> imatrix) is the main lever for a better *resident* Q2.** A Q2_K-gate/up "B1" would need a new
> `q2_K_pair_swiglu` kernel first.

### 4.3 Order (revised after the metric + kernel findings)

1. **A0** — `down` (Q2_K) gets the synthetic importance it lacked (resident, same ~219 GB).
   Cheap first check; *building now*.
2. **A** — the main event: re-point the imatrix collector at GLM sequential decode, port
   `--imatrix` into `glm-quantize.c`, collect a small (~32k–130k token) GLM imatrix, rebuild
   IQ2_XXS gate/up + Q2_K down weighted by it. Stays resident (~219 GB) and fast — the right
   lever given the kernel constraint.
3. **down → Q4_K** (gate/up still IQ2_XXS, ~275 GB, streams via the CPU-routed mmap path since
   the GPU slots8 stream needs Q4_K gate/up). Tests whether spending bits on the most
   sensitive projection helps; accepts the streaming speed hit.
4. **q2_K_pair_swiglu kernel → B1** (gate/up Q2_K, ~251 GB near-resident) — only if A leaves
   quality short; needs a new Metal kernel (clone the iq2_xxs/q4_K pair_swiglu).
5. **C / D** — new low-bit types (IQ2_S/Q3_K) and 8-bit KV, last resort.

### 4.4 Results tracker (newest first; fill as we execute)

Prose `avg_nll` is the held-out-corpus number (tertiary metric; see the §4.1 note — it barely
moves between Q4 and Q2, so treat only *large* changes as signal). Greedy-vs-Q4 is the primary
quantitative metric (TBD as candidates land).

| Date | Variant | Size | Fit | tok/s | prose avg_nll | greedy-vs-Q4 | Battery | Notes |
|---|---|---|---|---|---|---|---|---|
| 06-18 | **baseline** Q2 (IQ2_XXS g/u synth-imat, Q2_K down unweighted) | 218.9 GiB | resident | ~11.4 | 8.56 (1258 tok) / 8.07 (320 tok) | ref | loops (§2.3) | reference point |
| 06-18 | **Q4 (Q4_K experts, full 256)** | 408.7 GiB | streaming | ~1.08 | 8.03 (noise) | ref | **clears the hard prompt: lists 5 AND gives the facts** (Mercury smallest/no moons, Venus hottest, Jupiter largest) | **best quality** — the only build that does the complex task; no Q2 recipe produces any fact |
| 06-18 | **A-dense (imatrix gate/up, 20k-tok calib)** | 218.9 GiB | resident | ~11 | 8.73 (noise) | — | **short greedy: clean correct numbered list 1–5 (repeated cleanly)** — best Q2 | **recommended Q2**; long/complex still degenerates (gate/up 2-bit ceiling) |
| 06-18 | A (imatrix, gate/up, 1399-tok calib) | 218.9 GiB | resident | ~11 | 8.67 (noise) | — | short greedy lists all 5 planets (baseline looped on "Mercury") | clear win vs baseline; dense is a bit cleaner |
| 06-18 | down4 (imatrix gate/up + **down Q4_K**) | 272 GiB | **streaming ~0.3 t/s** | 0.3 | — | — | short: lists 5; long: enumerates but still repeats/no facts | **not worth it** — more down bits don't lift the gate/up 2-bit ceiling; 35x slower for a marginal long-prompt gain |
| 06-19 | q4gu-last8 (last 8 layers' gate/up → Q4_K + imatrix) | 234 GiB | **resident ~11.4 t/s** | 11.4 | — | — | short: correct list; long: lists, **no facts** (= imatrix-dense) | 8 Q4 layers insufficient; **broken under `--ssd-streaming`** (mixed expert sizes → garbage), so resident-only → capped at ~8–12 layers |
| 06-19 | q4gu-all (all gate/up → Q4_K + Q2_K down) **+ streaming fix** | 356 GiB | streaming | ~0.6–1.4 | — | — | **gives the facts** ("Mercury: closest to the Sun", "Venus: hottest") then over-generates | **Q4-class quality** (confirms gate/up drives quality) but **doesn't beat Q4**: terser facts (Q2_K down), ~same disk-bound speed, only 13% smaller. Needed a new `slots8_q2_K_sum8` kernel to stream at all |
| 06-18 | A0 (down Q2_K weighted) | dropped | — | — | — | — | — | subsumed by A (real imatrix weights down too once collected) |

### 4.5 Running an iteration (notible)

All scripts manage the prod server themselves (stop with a trap that always restarts it) and
write artifacts to `/Volumes/4TB-1`. Launch each detached (`( nohup bash … & )`) and poll its log.

- **One full imatrix iteration** (collect → build → eval), parameterized by a label, token
  budget, and (optional) dataset:
  ```sh
  scripts/glm-imatrix-iterate.sh dense 20000   # 20k-token imatrix from rendered_prompts.txt
  ```
  Produces `/Volumes/4TB-1/glm-5.2.imatrix-<label>.dat`,
  `glm-5.2-q2-imatrix-<label>.gguf`, and an eval under `/tmp/glm-q2-eval/imatrix-<label>/`.
- **Recipe sweep** (no imatrix): `scripts/glm-q2-build.sh OUT.gguf --down-type q4_k` then
  `scripts/glm-q2-eval.sh OUT.gguf LABEL` (or chain with `glm-after-build-eval.sh`).
- **Collection knobs:** `DATASET=…` overrides the calibration corpus; `--imatrix-max-tokens N`
  bounds it. Builds are ~95–110 min (IQ2_XXS codebook search dominates); collection ~1 min per
  ~150–600 tokens (slower as the per-prompt KV grows). Promote a winner by copying it over
  `glm-5.2-q2.gguf`.

---

## 5. Campaign log (newest first)

- **2026-06-19 — streaming fix landed; q4gu-all is Q4-class quality but does NOT beat Q4.**
  Implemented the mixed-precision streaming fix: a new `kernel_mul_mv_slots8_q2_K_sum8_f32`
  (slots6_q2_K_sum6 widened to 8 experts) + threaded the down type through
  `ds4_gpu_glm_streaming_routed_moe_tensor` so a **Q4_K gate/up + Q2_K down** model streams
  correctly (gate/up stay Q4_K; full-Q4 down path unchanged; CUDA stub + header updated). With
  the fix, **q4gu-all (356 GiB) runs coherently** and on the hard prompt **produces the facts**
  ("Mercury: closest to the Sun", "Venus: hottest planet") — confirming **gate/up is the quality
  driver** (down precision barely matters). **But it does not beat Q4:** the facts are terser than
  Q4's (Q2_K down), the streaming speed is ~0.6–1.4 t/s (≈ Q4 — both disk-bound on the Q4_K
  gate/up, so the cheaper down doesn't speed it up), and it's only 13% smaller. So Q4 dominates
  (richer facts, similar speed). **Final outcome of the whole campaign: two tiers — imatrix-Q2
  (219 GiB resident, ~11 t/s, simple prompts) and Q4 (409 GiB streamed, ~1 t/s, complex prompts);
  no usable third tier.** The streaming fix is kept (it's a correct, general capability for
  non-uniform expert precision) but yields no model that beats Q4. Deleted q4gu-all.
- **2026-06-19 — `--q4-layers` sweep + a streaming bug found.** Testing the last untested lever
  (lift selected layers' gate/up to Q4_K on top of the imatrix). **Engine bug found:** a model with
  *mixed* expert sizes across layers (some Q4_K gate/up, some IQ2_XXS) is **broken under
  `--ssd-streaming`** — the streaming expert cache has a single slab size class, so the off-size-class
  layers "bypass the cache and read via mapped model views" and produce garbage (token-0 `!!!!`). Such
  mixed models must run **resident**. **q4gu-last8** (last 8 layers' gate/up → Q4_K, 234 GiB) runs
  fine resident at **11.4 t/s** and is coherent, but on the hard prompt it **still gives no facts**
  (= imatrix-dense) — 8 Q4 layers don't lift the ceiling, and the streaming bug caps resident partials
  at ~8–12 layers, so partial-`--q4-layers` is a dead end for the hard prompt. **q4gu-all** (all 75
  layers' gate/up → Q4_K + Q2_K down, 356 GiB) **also came back garbage** (`!!!!`): too big to be
  resident, and the streaming slots8 kernels are **Q4_K-only** (gate/up *and* down) so they misread
  the Q2_K down. **Conclusion — the `--q4-layers` lever yields no usable middle tier:** resident
  partials (≤~12 layers) don't help the hard prompt, and any config with enough Q4 gate/up is too
  big for resident and only streams correctly if *everything* (incl. down) is Q4_K — i.e. the full
  Q4 model. So the usable options remain **imatrix-Q2 (219 GiB resident, simple prompts) and full Q4
  (409 GiB streamed, complex prompts)**, with nothing usable in between. Unlocking a middle tier
  (e.g. Q4 gate/up + cheap Q2_K down, ~356 GiB, which streamed at ~3.5 t/s before producing garbage —
  3× faster than Q4 *if* it worked) would require a **streaming-path fix**: slots8 kernels / an expert
  cache that handle non-uniform expert precision (Q2_K down with Q4_K gate/up). That is a real engine
  project, flagged as the follow-up. Deleted q4gu-last8 + q4gu-all (benchmarks captured).
- **2026-06-18 — Q4 battery confirms Q4 is the quality answer (only build that does complex tasks).**
  Ran the battery on the full-expert **Q4** (streamed, ~1.08 t/s gen). On the **hard** prompt Q4
  lists 1–5 and then **produces the interesting facts** ("Mercury is the smallest planet and closest
  to the Sun, no moons", "Venus is the hottest planet", "Jupiter is the largest planet") — the actual
  task. **No Q2 variant (baseline, imatrix, dense, down4) ever produced a single fact** — they loop on
  the list. Q4 still over-generates/repeats after the first pass (greedy-not-stopping, fixable with a
  stop + sampling), but the *content* clears the 2-bit ceiling. **Final recommendation:** Q4 for
  quality (speed acceptable per user), imatrix-dense Q2 for resident speed. Deleted the superseded
  inferior variants (1399-tok imatrix, down4) after capturing their benchmarks here. **File cleanup:**
  also deleted the synthetic-importance baseline and **promoted the 20k-token imatrix build to the
  canonical `glm-5.2-q2.gguf`** (default Q2 name = best Q2). Remaining: `glm-5.2-q2.gguf` (best Q2) and
  `glm-5.2-q4.gguf` (best quality); 1.4 TiB free on /Volumes/4TB-1.
- **2026-06-18 — external off-the-shelf GGUFs surveyed; none beats our Q4 for quality, none loads
  in ds4.** Checked three HuggingFace GLM-5.2 GGUFs (user request): `unsloth/GLM-5.2-GGUF`
  (UD-IQ2_XXS 238 GB / UD-IQ2_M 239 GB, full 256 experts), `pipenetwork/GLM-5.2-REAP50-Q2_K`
  (139 GB) and `-Q3_K_M` (182 GB). **All are `glm-dsa` format requiring a *patched* llama.cpp**
  (stock llama.cpp can't load GLM-5.2; ds4 can't load them at all — it needs its own `deepseek4.*`
  metadata + absorbed-MLA + stacked experts). **REAP50 is 50%-expert-*pruned*** (128/256, ~394 B
  params) — architecturally unsupported by ds4, and its cards say "not a quality champion / fragile,
  ~+37.5% perplexity vs full GLM-5.2, collapses to repetition on greedy." unsloth's are full-expert
  but 2-bit (our Q2 class). **Conclusion:** for *best quality* none of these wins — our full-expert
  **Q4** (4-bit, 256 experts) is higher quality than any pruned/2-bit external build, and adopting an
  external one means switching to a separate patched-llama.cpp stack. (notible has no llama.cpp / hf-cli.)
  So the quality answer is our own Q4; battery-testing it to confirm it clears the hard prompt.
- **2026-06-18 — down→Q4_K (streaming) confirms gate/up is the bottleneck; campaign conclusion.**
  Built `down4` = dense-imatrix gate/up (IQ2_XXS) + **down Q4_K** (272 GiB, streams via the
  CPU-routed mmap path since gate/up isn't Q4_K). On the **short** prompt it lists all five
  planets (≈ imatrix-dense). On the **hard** prompt it *enumerates* the planets (better than
  imatrix-dense's "1." collapse) but **still repeats with errors and never gives the facts** —
  a marginal gain for **+53 GiB and ~35× slower** (~0.3 t/s streaming). So spending bits on the
  down_proj does **not** lift the hard-prompt ceiling: the bottleneck is the **2-bit IQ2_XXS
  gate/up**, and the only fix is Q4-level gate/up. Added a `--q4-layers` converter flag (upgrade
  selected layers' gate/up to Q4_K) as the principled next lever for the hard prompt — a few
  sensitive layers at Q4 gate/up, mostly resident — to try if/when desired. **Recommendation:
  `glm-5.2-q2-imatrix-dense.gguf` is the best resident Q2 (219 GiB, ~11 t/s); use Q4 (streamed)
  for hard prompts where quality must hold.**
- **2026-06-18 — denser imatrix (20k tok) is the best resident Q2; the hard-prompt ceiling is
  the 2-bit gate/up.** Collected a 20k-token / 12M-observation imatrix (~625 obs/expert/layer,
  ~14x the first) from the 4700-block corpus and rebuilt. **Short greedy: a clean, correct
  numbered planet list 1–5** (the first imatrix had a repeat-error; the baseline looped on
  "Mercury") — the best Q2 yet, same 219 GiB / resident / ~11 t/s. **But the long/complex prompt
  still degenerates** (collapses to "1." spam), same as baseline/first. **Conclusion:** the
  imatrix maximizes resident-Q2 coherence on simpler prompts, but it cannot lift the hard-prompt
  ceiling, which is the **IQ2_XXS 2-bit gate/up** capacity limit — and gate/up can only be
  IQ2_XXS or Q4_K (no Q2_K kernel), so the only thing that fixes hard prompts is Q4-level gate/up
  (i.e. the streaming Q4 model). `glm-5.2-q2-imatrix-dense.gguf` is the **recommended Q2**.
  Prose ppl drifted 8.56→8.67→8.73 across the imatrix builds (wrong direction) — final proof
  that prose ppl is the wrong metric here; greedy behavior is the signal. Next: test the user's
  "bigger Q2 / streaming OK" lever (down→Q4_K) to confirm down bits don't lift the gate/up ceiling.
- **2026-06-18 — Option A result: the real imatrix improves Q2 coherence (clear win).**
  Built `glm-5.2-q2-imatrix.gguf` (same recipe as baseline Q2 but IQ2_XXS gate/up weighted by
  the real 1399-token activation imatrix; down stays synthetic). Same 219 GiB, resident, ~11 t/s.
  **Greedy short prompt: it now lists all five planets correctly (Mercury→Jupiter); the baseline
  loops on "Mercury" and never completes the list.** The long/complex prompt still degenerates
  (preamble repetition) and sampled@0.6 still decoheres — the 2-bit ceiling on hard instructions
  is not removed, but coherence on simpler prompts is clearly better. Prose ppl moved 8.56→8.67
  (the wrong direction) — further confirmation that prose ppl is uninformative here; the behavior
  is the real signal. **Next: a denser imatrix** (12k tokens from the 4700-block corpus) to push
  the harder prompts; then promote the winner to `glm-5.2-q2.gguf`.
- **2026-06-18 — Option A (real imatrix) built and collecting.** Implemented the full GLM
  imatrix path: converter `--imatrix` consumes a per-expert `.dat`; the engine collects via
  GLM **sequential decode** (the batch-prefill collector SIGSEGVs on GLM). First collector
  read the routed activations with a per-layer GPU read *after* the MoE and crashed at layer
  ~44 — `ds4_gpu_tensor_read` consumes a Metal command buffer, and ~3 reads/layer exhaust the
  pool. Fix: collect **gate/up importance inside the CPU router** (where ffn_norm is already
  read back, one sync/layer — the validated path), and skip the down_proj (its input only
  exists on the GPU post-MoE); the converter falls back to synthetic importance for down.
  Smoke test clean (134 tok, 80400 routed observations, valid 900M `.dat`). A0 (synthetic
  down-weighting) was dropped as **subsumed** by this (the real imatrix weights gate/up far
  better, at the same 219 GB / resident / ~11 t/s). Collecting on the 16-prompt GLM calib set
  (~134 obs/expert/layer), then rebuilding IQ2_XXS gate/up with the real imatrix.
- **2026-06-18 — metric pivot + kernel reality; A0 building.** Built the perplexity harness and
  hit the wall that **raw-prose ppl is uninformative for GLM** (Q4 3079 ≈ Q2 3184 on 320 tok;
  see §4.1 note) — switched the primary metric to greedy-fidelity-vs-Q4 + the behavioral battery.
  Fixed the harness to prepend `[gMASK]<sop>` (full-attention poisons an un-framed position 0).
  Mapped the **kernel constraint**: gate/up `pair_swiglu` exists only for IQ2_XXS/Q4_K (no Q2_K),
  so a better *resident* Q2 hinges on **Option A (real imatrix)** for IQ2_XXS. Added converter
  `--gu-type/--down-type` flags and a prod-safe build runner. Launched **A0** (down Q2_K weighted)
  as the cheap first check; implementing the GLM imatrix collector (Option A) next.

## 6. Benchmark: DeepSeek V4 Flash (prod) vs GLM-5.2 Q2/Q4 (2026-06-19)

All three served via the same OpenAI API (`notible:8085`, `ds4-server`), same 5-prompt general
battery (factual; factual+facts; math; code; transaction-categorization), `max_tokens` capped.
DeepSeek V4 Flash is the resident **prod** model (`deepseek-v4-flash`, Q4); GLM Q2 is resident;
GLM Q4 streams. Two rounds: **R1** = matched greedy (temp 0) + reasoning off; **R2** = temp 0.6 +
reasoning=high (GLM's *recommended* serving mode). Scripts: `scripts/glm-vs-deepseek-bench.sh`,
`glm-bench-round2.sh`, `glm-bench-client.py`.

### Speed (decode tok/s — short prompts are latency-bound; long ones show true decode)
| Model | R1 (greedy / reason off) | R2 (temp 0.6 / reason high) | stops cleanly? |
|---|---|---|---|
| **DeepSeek V4 Flash Q4** (resident, prod) | **~26–31** | **~27–29** | **yes** (finish=stop both rounds) |
| GLM-5.2 Q2 (resident) | ~9–10 | ~9 | **no** — hit the token cap on *every* prompt |
| GLM-5.2 Q4 (streaming) | ~0.4–0.9 | (≈ same) | no |

### Quality (answer content)
- **DeepSeek**: correct, complete, clean, and **terminates**. categorize → "Groceries"; planets with
  rich correct facts ("Mercury – smallest, no atmosphere, 430°C"); math right; code right.
- **GLM-5.2 Q2**: **degenerates in both modes** — R1 greedy gave a *wrong* "OTHER" + control-token
  spam (`<|user|></think>…`); R2 (its recommended mode) thought briefly then spewed `</think>` and
  never produced the answer or stopped. **Not production-viable on this box.**
- **GLM-5.2 Q4**: best GLM quality — gives the facts (Mercury water-ice/no-moons) with minor errors +
  repetition — but **~0.5 t/s streaming is impractical**, and greedy still doesn't stop.

### Conclusion
**DeepSeek V4 Flash Q4 dominates this box: ~3× faster than GLM-5.2 Q2, ~30–60× faster than GLM-5.2
Q4, and the only one that reliably yields a clean, correct, *terminated* answer.** GLM-5.2 Q2 is
both slower and degenerate; GLM-5.2 Q4 is higher quality but impractically slow. **There is no reason
to switch from DeepSeek V4 Flash to GLM-5.2 for general serving here** — GLM-5.2's value would have to
come from specific capabilities/context beyond this battery, not speed or general answer quality.

## 7. Pointers

- Converter: `gguf-tools/glm-quantize.c` (`--q2` at `:1089`, `build_plan` at `:801`,
  per-expert quant at `:770`, synthetic importance at `:783`).
- Quantizers: `gguf-tools/quants.c` (type traits `:39`; Q2_K `:641`, IQ2_XXS `:997`,
  Q4_K `:506`).
- Imatrix (Option A): loader `imatrix_load`/`imatrix_find` in `gguf-tools/deepseek4-quantize.c`
  (to port into `glm-quantize.c`); collector `ds4_engine_collect_imatrix` (`ds4.c:25844`) +
  `imatrix_collect_layer_batch` (`:20788`, reads the *batch* buffers — needs a single-token GLM
  variant reading `g->ffn_norm`/`g->routed_mid`/`g->router_selected`); `.dat` save format
  `imatrix_collector_save` (`:20877`); dataset + docs `gguf-tools/imatrix/`.
- Perplexity / logprobs: `run_perplexity_file` (`ds4_cli.c:790`, `--perplexity-file`, now
  GLM-BOS-framed); `--dump-logprobs`; `ds4_session_token_logprob` (`ds4.c:27908`).
- Eval: `gguf-tools/quality-testing/` (`collect_official.py`/`score_official.c`/
  `compare_scores.py`/`prompts.jsonl`; DeepSeek-shaped — adapt for GLM). `misc/quant_eval.c`
  is referenced by the imatrix README but is not in this tree.
- Scripts (`scripts/`): `glm-q2-build.sh` (prod-safe candidate build → /Volumes/4TB-1),
  `glm-q2-eval.sh MODEL LABEL` (prod-down ppl + battery; `PPLONLY=1` for ppl only),
  `glm-q2-characterize.sh` (battery), `glm-ppl-heldout.txt` (held-out ppl corpus).
