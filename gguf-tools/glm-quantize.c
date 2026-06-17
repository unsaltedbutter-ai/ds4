/*
 * glm-quantize.c -- GLM-5.2 HF-safetensors -> ds4 GGUF converter.
 *
 * Fork of the deepseek4-quantize pipeline for the GLM-5.2 (`glm_moe_dsa`) port.
 * Unlike deepseek4-quantize.c, which copies metadata/tensor-order from an existing
 * template GGUF, this tool authors the GGUF directly from GLM's config + safetensors,
 * because no GLM template GGUF exists yet.
 *
 * STATUS: skeleton. Implements `--dry-run`, which scans the GLM safetensors shard
 * headers and reports whether every source tensor the converter needs is present,
 * classifying each ds4 GGUF target as:
 *   MAPPED  - direct GLM source -> ds4 tensor (dequant bf16 -> quantize)
 *   ABSORB  - needs MLA absorption (fold kv_b into q_b / o_proj) before writing
 *   SKIP    - not needed for v1 (DSA indexer; ds4-only HC/compressor/sink/hash tensors)
 * Dequantization, MLA absorption, expert stacking, and GGUF authoring land next.
 *
 * Build: make -C gguf-tools glm-quantize
 * Run:   gguf-tools/glm-quantize --hf /Volumes/4TB-1/glm-5.2 --dry-run
 */

#include <dirent.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GLM-5.2 shape, from config.json (see glm-port-plan.md sec 1). Hardcoded here
 * for the skeleton; the real converter will read these from config.json. */
#define GLM_N_LAYER       78
#define GLM_N_EXPERT      256
#define GLM_FIRST_K_DENSE 3   /* layers [0,3) are dense MLP; [3,78) are MoE */

static void die(const char *msg) {
    fprintf(stderr, "glm-quantize: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

/* ---- tensor-name set, populated from safetensors shard headers ---- */

typedef struct {
    char **names;
    size_t len, cap;
} name_set;

static void name_set_push(name_set *s, const char *name, size_t n) {
    if (s->len == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 1024;
        s->names = realloc(s->names, s->cap * sizeof(*s->names));
        if (!s->names) die("out of memory");
    }
    char *copy = xmalloc(n + 1);
    memcpy(copy, name, n);
    copy[n] = '\0';
    s->names[s->len++] = copy;
}

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static void name_set_sort(name_set *s) {
    qsort(s->names, s->len, sizeof(*s->names), cmp_str);
}

static bool name_set_has(const name_set *s, const char *key) {
    if (s->len == 0) return false;
    size_t lo = 0, hi = s->len;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int c = strcmp(s->names[mid], key);
        if (c == 0) return true;
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return false;
}

/*
 * Scan one safetensors header. Format: 8-byte LE length, then a JSON object whose
 * top-level keys are tensor names mapping to `{...}` descriptors (plus an optional
 * "__metadata__"). Every top-level entry -- and only those -- is keyed by `"<name>":{`,
 * so we collect each quoted key immediately followed by `:{`. Inner keys ("dtype",
 * "shape", "data_offsets") are followed by `:"` or `:[`, never `:{`.
 */
static void scan_shard(const char *path, name_set *out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "glm-quantize: cannot open %s\n", path); return; }
    uint64_t hlen = 0;
    if (fread(&hlen, 8, 1, fp) != 1 || hlen == 0 || hlen > (uint64_t)512 * 1024 * 1024) {
        fclose(fp);
        die("bad safetensors header length");
    }
    char *hdr = xmalloc((size_t)hlen + 1);
    if (fread(hdr, 1, (size_t)hlen, fp) != (size_t)hlen) die("short safetensors header read");
    hdr[hlen] = '\0';
    fclose(fp);

    for (uint64_t i = 2; i < hlen; i++) {
        if (hdr[i] == '{' && hdr[i - 1] == ':' && hdr[i - 2] == '"') {
            /* key closes at i-2; walk back to its opening quote (no escapes in names) */
            int64_t j = (int64_t)i - 3;
            while (j >= 0 && hdr[j] != '"') j--;
            if (j < 0) continue;
            const char *start = hdr + j + 1;
            size_t klen = (size_t)((int64_t)i - 2 - (j + 1));
            if (klen == 0) continue;
            if (klen == 12 && strncmp(start, "__metadata__", 12) == 0) continue;
            name_set_push(out, start, klen);
        }
    }
    free(hdr);
}

static void scan_hf_dir(const char *dir, name_set *out) {
    DIR *d = opendir(dir);
    if (!d) { fprintf(stderr, "glm-quantize: cannot open dir %s\n", dir); exit(1); }
    struct dirent *e;
    int shards = 0;
    char path[4096];
    while ((e = readdir(d)) != NULL) {
        const char *n = e->d_name;
        size_t ln = strlen(n);
        if (ln < 12 || strncmp(n, "model-", 6) != 0 ||
            strcmp(n + ln - 12, ".safetensors") != 0) continue;
        snprintf(path, sizeof(path), "%s/%s", dir, n);
        scan_shard(path, out);
        shards++;
    }
    closedir(d);
    name_set_sort(out);
    fprintf(stderr, "glm-quantize: scanned %d shard(s), %zu tensors\n", shards, out->len);
}

/* ---- converter coverage map ---- */

typedef enum { CLASS_MAPPED, CLASS_ABSORB, CLASS_SKIP } map_class;

static const char *class_str(map_class c) {
    switch (c) {
        case CLASS_MAPPED: return "MAPPED";
        case CLASS_ABSORB: return "ABSORB";
        case CLASS_SKIP:   return "SKIP  ";
    }
    return "?";
}

/* Check a per-layer source-name template (with one %d for the layer) across
 * [lo, hi), and report how many layers carry it. */
static void report_layer_tensor(const name_set *s, const char *ds4_target,
                                map_class cls, const char *src_fmt,
                                int lo, int hi) {
    int present = 0, first_missing = -1;
    char buf[256];
    for (int l = lo; l < hi; l++) {
        snprintf(buf, sizeof(buf), src_fmt, l);
        if (name_set_has(s, buf)) present++;
        else if (first_missing < 0) first_missing = l;
    }
    int total = hi - lo;
    const char *flag = (present == total) ? "ok" : (present == 0 ? "ABSENT" : "PARTIAL");
    printf("  [%s] %-22s <- %-44s  %3d/%-3d %s",
           class_str(cls), ds4_target, src_fmt, present, total, flag);
    if (first_missing >= 0 && present != 0) printf(" (first miss L%d)", first_missing);
    printf("\n");
}

static void report_top_tensor(const name_set *s, const char *ds4_target,
                              map_class cls, const char *src) {
    printf("  [%s] %-22s <- %-44s  %s\n", class_str(cls), ds4_target, src,
           name_set_has(s, src) ? "present" : "MISSING");
}

static void dry_run_report(const name_set *s) {
    printf("\n== GLM-5.2 -> ds4 converter coverage (dry run) ==\n");

    printf("\nTop-level:\n");
    report_top_tensor(s, "token_embd.weight",  CLASS_MAPPED, "model.embed_tokens.weight");
    report_top_tensor(s, "output_norm.weight",  CLASS_MAPPED, "model.norm.weight");
    report_top_tensor(s, "output.weight",        CLASS_MAPPED, "lm_head.weight");

    printf("\nAttention + norms (all %d layers):\n", GLM_N_LAYER);
    report_layer_tensor(s, "attn_norm",     CLASS_MAPPED, "model.layers.%d.input_layernorm.weight", 0, GLM_N_LAYER);
    report_layer_tensor(s, "ffn_norm",      CLASS_MAPPED, "model.layers.%d.post_attention_layernorm.weight", 0, GLM_N_LAYER);
    report_layer_tensor(s, "attn_q_a",      CLASS_MAPPED, "model.layers.%d.self_attn.q_a_proj.weight", 0, GLM_N_LAYER);
    report_layer_tensor(s, "attn_q_a_norm", CLASS_MAPPED, "model.layers.%d.self_attn.q_a_layernorm.weight", 0, GLM_N_LAYER);
    report_layer_tensor(s, "attn_q_b",      CLASS_ABSORB, "model.layers.%d.self_attn.q_b_proj.weight", 0, GLM_N_LAYER);
    report_layer_tensor(s, "attn_kv",       CLASS_MAPPED, "model.layers.%d.self_attn.kv_a_proj_with_mqa.weight", 0, GLM_N_LAYER);
    report_layer_tensor(s, "attn_kv_a_norm",CLASS_MAPPED, "model.layers.%d.self_attn.kv_a_layernorm.weight", 0, GLM_N_LAYER);
    report_layer_tensor(s, "(fold kv_b)",   CLASS_ABSORB, "model.layers.%d.self_attn.kv_b_proj.weight", 0, GLM_N_LAYER);
    report_layer_tensor(s, "attn_output",   CLASS_ABSORB, "model.layers.%d.self_attn.o_proj.weight", 0, GLM_N_LAYER);

    printf("\nDSA indexer (SKIP for v1 -- run dense MLA):\n");
    report_layer_tensor(s, "indexer.q_b",   CLASS_SKIP, "model.layers.%d.self_attn.indexer.wq_b.weight", 0, GLM_N_LAYER);
    report_layer_tensor(s, "indexer.wk",    CLASS_SKIP, "model.layers.%d.self_attn.indexer.wk.weight", 0, GLM_N_LAYER);
    report_layer_tensor(s, "indexer.knorm", CLASS_SKIP, "model.layers.%d.self_attn.indexer.k_norm.weight", 0, GLM_N_LAYER);
    report_layer_tensor(s, "indexer.proj",  CLASS_SKIP, "model.layers.%d.self_attn.indexer.weights_proj.weight", 0, GLM_N_LAYER);

    printf("\nDense MLP (layers [0,%d)):\n", GLM_FIRST_K_DENSE);
    report_layer_tensor(s, "ffn_gate_dense",CLASS_MAPPED, "model.layers.%d.mlp.gate_proj.weight", 0, GLM_FIRST_K_DENSE);
    report_layer_tensor(s, "ffn_up_dense",  CLASS_MAPPED, "model.layers.%d.mlp.up_proj.weight", 0, GLM_FIRST_K_DENSE);
    report_layer_tensor(s, "ffn_down_dense",CLASS_MAPPED, "model.layers.%d.mlp.down_proj.weight", 0, GLM_FIRST_K_DENSE);

    printf("\nMoE routing + shared expert (layers [%d,%d)) -- names to CONFIRM:\n",
           GLM_FIRST_K_DENSE, GLM_N_LAYER);
    report_layer_tensor(s, "ffn_gate_inp",  CLASS_MAPPED, "model.layers.%d.mlp.gate.weight", GLM_FIRST_K_DENSE, GLM_N_LAYER);
    report_layer_tensor(s, "exp_probs_b",   CLASS_MAPPED, "model.layers.%d.mlp.gate.e_score_correction_bias", GLM_FIRST_K_DENSE, GLM_N_LAYER);
    report_layer_tensor(s, "ffn_gate_shexp",CLASS_MAPPED, "model.layers.%d.mlp.shared_experts.gate_proj.weight", GLM_FIRST_K_DENSE, GLM_N_LAYER);
    report_layer_tensor(s, "ffn_up_shexp",  CLASS_MAPPED, "model.layers.%d.mlp.shared_experts.up_proj.weight", GLM_FIRST_K_DENSE, GLM_N_LAYER);
    report_layer_tensor(s, "ffn_down_shexp",CLASS_MAPPED, "model.layers.%d.mlp.shared_experts.down_proj.weight", GLM_FIRST_K_DENSE, GLM_N_LAYER);

    printf("\nRouted experts (stack %d per MoE layer -> ds4 3-D tensor): sampling experts 0 and %d:\n",
           GLM_N_EXPERT, GLM_N_EXPERT - 1);
    char fmt[256];
    for (int xid = 0; xid <= GLM_N_EXPERT - 1; xid += (GLM_N_EXPERT - 1)) {
        snprintf(fmt, sizeof(fmt), "model.layers.%%d.mlp.experts.%d.gate_proj.weight", xid);
        report_layer_tensor(s, "ffn_gate_exps", CLASS_MAPPED, fmt, GLM_FIRST_K_DENSE, GLM_N_LAYER);
        snprintf(fmt, sizeof(fmt), "model.layers.%%d.mlp.experts.%d.up_proj.weight", xid);
        report_layer_tensor(s, "ffn_up_exps",   CLASS_MAPPED, fmt, GLM_FIRST_K_DENSE, GLM_N_LAYER);
        snprintf(fmt, sizeof(fmt), "model.layers.%%d.mlp.experts.%d.down_proj.weight", xid);
        report_layer_tensor(s, "ffn_down_exps", CLASS_MAPPED, fmt, GLM_FIRST_K_DENSE, GLM_N_LAYER);
    }

    printf("\nLegend: MAPPED=direct dequant+quantize  ABSORB=MLA fold (kv_b into q_b/o_proj)  SKIP=v1 omits\n");
    printf("Note: ds4-only tensors (HC, KV compressor, attn_sinks, hash tid2eid) are intentionally NOT emitted for GLM.\n");
}

static void usage(void) {
    fprintf(stderr,
        "usage: glm-quantize --hf <dir> [--dry-run]\n"
        "  --hf <dir>   directory with GLM-5.2 model-*.safetensors shards\n"
        "  --dry-run    scan shard headers and report converter coverage (default)\n");
    exit(1);
}

int main(int argc, char **argv) {
    const char *hf_dir = NULL;
    bool dry_run = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--hf") == 0 && i + 1 < argc) hf_dir = argv[++i];
        else if (strcmp(argv[i], "--dry-run") == 0) dry_run = true;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) usage();
        else { fprintf(stderr, "glm-quantize: unknown arg: %s\n", argv[i]); usage(); }
    }
    if (!hf_dir) usage();

    name_set tensors = {0};
    scan_hf_dir(hf_dir, &tensors);

    /* Skeleton only knows --dry-run; the quantize/author path is the next step. */
    (void)dry_run;
    dry_run_report(&tensors);
    return 0;
}
