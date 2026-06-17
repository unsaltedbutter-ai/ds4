/*
 * glm-quantize.c -- GLM-5.2 HF-safetensors -> ds4 GGUF converter.
 *
 * Fork of the deepseek4-quantize pipeline for the GLM-5.2 (`glm_moe_dsa`) port.
 * Unlike deepseek4-quantize.c, which copies metadata/tensor-order from an existing
 * template GGUF, this tool authors the GGUF directly from GLM's config + safetensors,
 * because no GLM template GGUF exists yet.
 *
 * Modes:
 *   --dry-run        scan shard headers and report converter coverage (MAPPED/ABSORB/SKIP)
 *   --read <name>    read one HF tensor, dequantize to f32, print shape/dtype + stats
 *
 * Tensors are located by scanning each shard's own header (no model.safetensors.index.json
 * required), so this works against a partially-downloaded model.
 *
 * The JSON tokenizer and safetensors layout follow gguf-tools/deepseek4-quantize.c.
 * Quantization, MLA absorption (fold kv_b into q_b/o_proj), expert stacking, and GGUF
 * authoring land next.
 *
 * Build: make -C gguf-tools glm-quantize
 */

#include <ctype.h>
#include <dirent.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GLM-5.2 shape, from config.json (see glm-port-plan.md sec 1). Hardcoded here
 * for now; the real converter will read these from config.json. */
#define GLM_N_LAYER       78
#define GLM_N_EXPERT      256
#define GLM_FIRST_K_DENSE 3   /* layers [0,3) are dense MLP; [3,78) are MoE */
#define GLM_MAX_DIMS      8

static void die(const char *msg) {
    fprintf(stderr, "glm-quantize: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die("out of memory");
    return q;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static uint16_t load_u16_le(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static float bf16_to_f32(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* =====
 * Minimal JSON tokenizer (adapted from deepseek4-quantize.c). Safetensors headers
 * and config.json are ordinary JSON; tensor names contain no escapes.
 */

typedef enum { JT_OBJECT, JT_ARRAY, JT_STRING, JT_PRIMITIVE } json_type;

typedef struct {
    json_type type;
    int start, end, parent, size;
} json_tok;

typedef struct {
    json_tok *v;
    int len, cap;
    const char *js;
} json_doc;

static int json_add(json_doc *d, json_type type, int start, int end, int parent) {
    if (d->len == d->cap) {
        d->cap = d->cap ? d->cap * 2 : 4096;
        d->v = xrealloc(d->v, (size_t)d->cap * sizeof(d->v[0]));
    }
    int id = d->len++;
    d->v[id] = (json_tok){ .type = type, .start = start, .end = end, .parent = parent, .size = 0 };
    if (parent >= 0) d->v[parent].size++;
    return id;
}

static json_doc json_parse_text(const char *js, size_t len) {
    json_doc d = { .js = js };
    int parent = -1;
    for (int i = 0; i < (int)len; i++) {
        unsigned char c = (unsigned char)js[i];
        if (isspace(c) || c == ':' || c == ',') continue;
        if (c == '{' || c == '[') {
            parent = json_add(&d, c == '{' ? JT_OBJECT : JT_ARRAY, i, -1, parent);
            continue;
        }
        if (c == '}' || c == ']') {
            if (parent < 0) die("bad JSON: unmatched close");
            d.v[parent].end = i + 1;
            parent = d.v[parent].parent;
            continue;
        }
        if (c == '"') {
            int start = i + 1;
            i++;
            bool esc = false;
            for (; i < (int)len; i++) {
                if (esc) esc = false;
                else if (js[i] == '\\') esc = true;
                else if (js[i] == '"') break;
            }
            if (i >= (int)len) die("bad JSON: unterminated string");
            json_add(&d, JT_STRING, start, i, parent);
            continue;
        }
        int start = i;
        while (i < (int)len && !isspace((unsigned char)js[i]) &&
               js[i] != ',' && js[i] != ']' && js[i] != '}') i++;
        json_add(&d, JT_PRIMITIVE, start, i, parent);
        i--;
    }
    if (parent != -1) die("bad JSON: unterminated object/array");
    return d;
}

static void json_free(json_doc *d) { free(d->v); memset(d, 0, sizeof(*d)); }

static bool json_tok_eq(const json_doc *d, int tok, const char *s) {
    const json_tok *t = &d->v[tok];
    int n = t->end - t->start;
    return t->type == JT_STRING && (int)strlen(s) == n &&
           memcmp(d->js + t->start, s, (size_t)n) == 0;
}

static char *json_strdup_tok(const json_doc *d, int tok) {
    const json_tok *t = &d->v[tok];
    return xstrndup(d->js + t->start, (size_t)(t->end - t->start));
}

static bool json_is_descendant(const json_doc *d, int tok, int parent) {
    for (int p = d->v[tok].parent; p >= 0; p = d->v[p].parent)
        if (p == parent) return true;
    return false;
}

static int json_skip(const json_doc *d, int tok) {
    int i = tok + 1;
    while (i < d->len && json_is_descendant(d, i, tok)) i++;
    return i;
}

static int json_obj_get(const json_doc *d, int obj, const char *key) {
    if (obj < 0 || d->v[obj].type != JT_OBJECT) return -1;
    for (int i = obj + 1; i < d->len && d->v[i].parent == obj;) {
        int k = i, v = i + 1;
        if (v >= d->len || d->v[v].parent != obj) return -1;
        if (json_tok_eq(d, k, key)) return v;
        i = json_skip(d, v);
    }
    return -1;
}

static int64_t json_i64(const json_doc *d, int tok) {
    char tmp[64];
    int n = d->v[tok].end - d->v[tok].start;
    if (n <= 0 || n >= (int)sizeof(tmp)) die("bad JSON integer");
    memcpy(tmp, d->js + d->v[tok].start, (size_t)n);
    tmp[n] = '\0';
    return strtoll(tmp, NULL, 10);
}

/* =====
 * Safetensors tensor database, built by scanning every shard's own header.
 */

typedef struct {
    char *name;
    char dtype[16];
    int ndim;
    int64_t shape[GLM_MAX_DIMS];
    uint64_t begin, end;   /* byte range within the shard's data section */
    int shard;
} gtensor;

typedef struct {
    char    *path;
    uint64_t data_base;    /* 8 + header_len */
} gshard;

typedef struct {
    gtensor *t; size_t nt, capt;
    gshard  *s; size_t ns, caps;
} stdb;

static int stdb_add_shard(stdb *db, const char *path, uint64_t data_base) {
    if (db->ns == db->caps) {
        db->caps = db->caps ? db->caps * 2 : 64;
        db->s = xrealloc(db->s, db->caps * sizeof(*db->s));
    }
    db->s[db->ns].path = xstrndup(path, strlen(path));
    db->s[db->ns].data_base = data_base;
    return (int)db->ns++;
}

static void stdb_add_tensor(stdb *db, const gtensor *t) {
    if (db->nt == db->capt) {
        db->capt = db->capt ? db->capt * 2 : 4096;
        db->t = xrealloc(db->t, db->capt * sizeof(*db->t));
    }
    db->t[db->nt++] = *t;
}

static void stdb_scan_shard(stdb *db, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "glm-quantize: cannot open %s\n", path); return; }
    uint64_t hlen = 0;
    if (fread(&hlen, 8, 1, fp) != 1 || hlen == 0 || hlen > (uint64_t)512 * 1024 * 1024) {
        fclose(fp); die("bad safetensors header length");
    }
    char *hdr = xmalloc((size_t)hlen + 1);
    if (fread(hdr, 1, (size_t)hlen, fp) != (size_t)hlen) die("short safetensors header read");
    hdr[hlen] = '\0';
    fclose(fp);

    int shard = stdb_add_shard(db, path, 8 + hlen);
    json_doc d = json_parse_text(hdr, (size_t)hlen);
    if (d.len < 1 || d.v[0].type != JT_OBJECT) die("bad safetensors header");
    for (int i = 1; i < d.len && d.v[i].parent == 0;) {
        int k = i, v = i + 1;
        if (v >= d.len || d.v[v].parent != 0) die("bad safetensors header object");
        if (!json_tok_eq(&d, k, "__metadata__")) {
            int dt = json_obj_get(&d, v, "dtype");
            int sh = json_obj_get(&d, v, "shape");
            int of = json_obj_get(&d, v, "data_offsets");
            if (dt < 0 || sh < 0 || of < 0) die("bad safetensors tensor entry");
            gtensor t = {0};
            t.shard = shard;
            t.name = json_strdup_tok(&d, k);
            char *dtype = json_strdup_tok(&d, dt);
            snprintf(t.dtype, sizeof(t.dtype), "%s", dtype);
            free(dtype);
            int nd = 0;
            for (int j = sh + 1; j < d.len && d.v[j].parent == sh; j = json_skip(&d, j)) {
                if (nd >= GLM_MAX_DIMS) die("too many dims");
                t.shape[nd++] = json_i64(&d, j);
            }
            t.ndim = nd;
            int no = 0;
            for (int j = of + 1; j < d.len && d.v[j].parent == of; j = json_skip(&d, j)) {
                int64_t x = json_i64(&d, j);
                if (no == 0) t.begin = (uint64_t)x; else if (no == 1) t.end = (uint64_t)x;
                no++;
            }
            if (no != 2) die("bad data_offsets");
            stdb_add_tensor(db, &t);
        }
        i = json_skip(&d, v);
    }
    json_free(&d);
    free(hdr);
}

static int cmp_gtensor(const void *a, const void *b) {
    return strcmp(((const gtensor *)a)->name, ((const gtensor *)b)->name);
}

static void stdb_open(stdb *db, const char *dir) {
    memset(db, 0, sizeof(*db));
    DIR *dp = opendir(dir);
    if (!dp) { fprintf(stderr, "glm-quantize: cannot open dir %s\n", dir); exit(1); }
    struct dirent *e;
    int shards = 0;
    char path[4096];
    while ((e = readdir(dp)) != NULL) {
        const char *n = e->d_name;
        size_t ln = strlen(n);
        if (ln < 12 || strncmp(n, "model-", 6) != 0 ||
            strcmp(n + ln - 12, ".safetensors") != 0) continue;
        snprintf(path, sizeof(path), "%s/%s", dir, n);
        stdb_scan_shard(db, path);
        shards++;
    }
    closedir(dp);
    qsort(db->t, db->nt, sizeof(*db->t), cmp_gtensor);
    fprintf(stderr, "glm-quantize: scanned %d shard(s), %zu tensors\n", shards, db->nt);
}

static const gtensor *stdb_find(const stdb *db, const char *name) {
    size_t lo = 0, hi = db->nt;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int c = strcmp(db->t[mid].name, name);
        if (c == 0) return &db->t[mid];
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return NULL;
}

static bool stdb_has(const stdb *db, const char *name) { return stdb_find(db, name) != NULL; }

/* Read a tensor and dequantize to a freshly malloc'd f32 buffer. */
static float *stdb_read_f32(const stdb *db, const gtensor *t, int64_t *n_out) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    size_t nbytes = (size_t)(t->end - t->begin);
    uint8_t *raw = xmalloc(nbytes ? nbytes : 1);
    const gshard *s = &db->s[t->shard];
    FILE *fp = fopen(s->path, "rb");
    if (!fp) die("cannot reopen shard");
    if (fseeko(fp, (off_t)(s->data_base + t->begin), SEEK_SET) != 0) die("seek failed");
    if (nbytes && fread(raw, 1, nbytes, fp) != nbytes) die("short tensor read");
    fclose(fp);

    float *out = xmalloc((size_t)n * sizeof(float));
    if (strcmp(t->dtype, "F32") == 0) {
        if (nbytes != (size_t)n * 4) die("bad F32 byte size");
        memcpy(out, raw, nbytes);
    } else if (strcmp(t->dtype, "BF16") == 0) {
        if (nbytes != (size_t)n * 2) die("bad BF16 byte size");
        for (int64_t i = 0; i < n; i++) out[i] = bf16_to_f32(load_u16_le(raw + (size_t)i * 2));
    } else {
        fprintf(stderr, "glm-quantize: unsupported dtype %s for %s\n", t->dtype, t->name);
        exit(1);
    }
    free(raw);
    if (n_out) *n_out = n;
    return out;
}

/* =====
 * --read: sanity-check the reader against a real tensor.
 */
static void read_tensor_cmd(const stdb *db, const char *name) {
    const gtensor *t = stdb_find(db, name);
    if (!t) { fprintf(stderr, "glm-quantize: tensor not found: %s\n", name); exit(1); }
    int64_t n = 0;
    float *v = stdb_read_f32(db, t, &n);
    float mn = v[0], mx = v[0];
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (v[i] < mn) mn = v[i];
        if (v[i] > mx) mx = v[i];
        sum += v[i];
    }
    printf("%s\n  dtype=%s shape=[", name, t->dtype);
    for (int i = 0; i < t->ndim; i++) printf("%s%lld", i ? "," : "", (long long)t->shape[i]);
    printf("] n=%lld\n", (long long)n);
    printf("  min=%.6g max=%.6g mean=%.6g\n", mn, mx, n ? sum / (double)n : 0.0);
    printf("  first:");
    for (int64_t i = 0; i < n && i < 6; i++) printf(" %.6g", v[i]);
    printf("\n");
    free(v);
}

/* =====
 * --dry-run: converter coverage map.
 */
typedef enum { CLASS_MAPPED, CLASS_ABSORB, CLASS_SKIP } map_class;

static const char *class_str(map_class c) {
    switch (c) {
        case CLASS_MAPPED: return "MAPPED";
        case CLASS_ABSORB: return "ABSORB";
        case CLASS_SKIP:   return "SKIP  ";
    }
    return "?";
}

static void report_layer_tensor(const stdb *db, const char *ds4_target,
                                map_class cls, const char *src_fmt, int lo, int hi) {
    int present = 0, first_missing = -1;
    char buf[256];
    for (int l = lo; l < hi; l++) {
        snprintf(buf, sizeof(buf), src_fmt, l);
        if (stdb_has(db, buf)) present++;
        else if (first_missing < 0) first_missing = l;
    }
    int total = hi - lo;
    const char *flag = (present == total) ? "ok" : (present == 0 ? "ABSENT" : "PARTIAL");
    printf("  [%s] %-22s <- %-44s  %3d/%-3d %s",
           class_str(cls), ds4_target, src_fmt, present, total, flag);
    if (first_missing >= 0 && present != 0) printf(" (first miss L%d)", first_missing);
    printf("\n");
}

static void report_top_tensor(const stdb *db, const char *ds4_target,
                              map_class cls, const char *src) {
    printf("  [%s] %-22s <- %-44s  %s\n", class_str(cls), ds4_target, src,
           stdb_has(db, src) ? "present" : "MISSING");
}

static void dry_run_report(const stdb *db) {
    printf("\n== GLM-5.2 -> ds4 converter coverage (dry run) ==\n");

    printf("\nTop-level:\n");
    report_top_tensor(db, "token_embd.weight",  CLASS_MAPPED, "model.embed_tokens.weight");
    report_top_tensor(db, "output_norm.weight",  CLASS_MAPPED, "model.norm.weight");
    report_top_tensor(db, "output.weight",        CLASS_MAPPED, "lm_head.weight");

    printf("\nAttention + norms (all %d layers):\n", GLM_N_LAYER);
    report_layer_tensor(db, "attn_norm",     CLASS_MAPPED, "model.layers.%d.input_layernorm.weight", 0, GLM_N_LAYER);
    report_layer_tensor(db, "ffn_norm",      CLASS_MAPPED, "model.layers.%d.post_attention_layernorm.weight", 0, GLM_N_LAYER);
    report_layer_tensor(db, "attn_q_a",      CLASS_MAPPED, "model.layers.%d.self_attn.q_a_proj.weight", 0, GLM_N_LAYER);
    report_layer_tensor(db, "attn_q_a_norm", CLASS_MAPPED, "model.layers.%d.self_attn.q_a_layernorm.weight", 0, GLM_N_LAYER);
    report_layer_tensor(db, "attn_q_b",      CLASS_ABSORB, "model.layers.%d.self_attn.q_b_proj.weight", 0, GLM_N_LAYER);
    report_layer_tensor(db, "attn_kv",       CLASS_MAPPED, "model.layers.%d.self_attn.kv_a_proj_with_mqa.weight", 0, GLM_N_LAYER);
    report_layer_tensor(db, "attn_kv_a_norm",CLASS_MAPPED, "model.layers.%d.self_attn.kv_a_layernorm.weight", 0, GLM_N_LAYER);
    report_layer_tensor(db, "(fold kv_b)",   CLASS_ABSORB, "model.layers.%d.self_attn.kv_b_proj.weight", 0, GLM_N_LAYER);
    report_layer_tensor(db, "attn_output",   CLASS_ABSORB, "model.layers.%d.self_attn.o_proj.weight", 0, GLM_N_LAYER);

    printf("\nDSA indexer (SKIP for v1 -- run dense MLA):\n");
    report_layer_tensor(db, "indexer.q_b",   CLASS_SKIP, "model.layers.%d.self_attn.indexer.wq_b.weight", 0, GLM_N_LAYER);
    report_layer_tensor(db, "indexer.wk",    CLASS_SKIP, "model.layers.%d.self_attn.indexer.wk.weight", 0, GLM_N_LAYER);
    report_layer_tensor(db, "indexer.knorm", CLASS_SKIP, "model.layers.%d.self_attn.indexer.k_norm.weight", 0, GLM_N_LAYER);
    report_layer_tensor(db, "indexer.proj",  CLASS_SKIP, "model.layers.%d.self_attn.indexer.weights_proj.weight", 0, GLM_N_LAYER);

    printf("\nDense MLP (layers [0,%d)):\n", GLM_FIRST_K_DENSE);
    report_layer_tensor(db, "ffn_gate_dense",CLASS_MAPPED, "model.layers.%d.mlp.gate_proj.weight", 0, GLM_FIRST_K_DENSE);
    report_layer_tensor(db, "ffn_up_dense",  CLASS_MAPPED, "model.layers.%d.mlp.up_proj.weight", 0, GLM_FIRST_K_DENSE);
    report_layer_tensor(db, "ffn_down_dense",CLASS_MAPPED, "model.layers.%d.mlp.down_proj.weight", 0, GLM_FIRST_K_DENSE);

    printf("\nMoE routing + shared expert (layers [%d,%d)):\n", GLM_FIRST_K_DENSE, GLM_N_LAYER);
    report_layer_tensor(db, "ffn_gate_inp",  CLASS_MAPPED, "model.layers.%d.mlp.gate.weight", GLM_FIRST_K_DENSE, GLM_N_LAYER);
    report_layer_tensor(db, "exp_probs_b",   CLASS_MAPPED, "model.layers.%d.mlp.gate.e_score_correction_bias", GLM_FIRST_K_DENSE, GLM_N_LAYER);
    report_layer_tensor(db, "ffn_gate_shexp",CLASS_MAPPED, "model.layers.%d.mlp.shared_experts.gate_proj.weight", GLM_FIRST_K_DENSE, GLM_N_LAYER);
    report_layer_tensor(db, "ffn_up_shexp",  CLASS_MAPPED, "model.layers.%d.mlp.shared_experts.up_proj.weight", GLM_FIRST_K_DENSE, GLM_N_LAYER);
    report_layer_tensor(db, "ffn_down_shexp",CLASS_MAPPED, "model.layers.%d.mlp.shared_experts.down_proj.weight", GLM_FIRST_K_DENSE, GLM_N_LAYER);

    printf("\nRouted experts (stack %d per MoE layer -> ds4 3-D tensor): sampling experts 0 and %d:\n",
           GLM_N_EXPERT, GLM_N_EXPERT - 1);
    char fmt[256];
    for (int xid = 0; xid <= GLM_N_EXPERT - 1; xid += (GLM_N_EXPERT - 1)) {
        snprintf(fmt, sizeof(fmt), "model.layers.%%d.mlp.experts.%d.gate_proj.weight", xid);
        report_layer_tensor(db, "ffn_gate_exps", CLASS_MAPPED, fmt, GLM_FIRST_K_DENSE, GLM_N_LAYER);
        snprintf(fmt, sizeof(fmt), "model.layers.%%d.mlp.experts.%d.up_proj.weight", xid);
        report_layer_tensor(db, "ffn_up_exps",   CLASS_MAPPED, fmt, GLM_FIRST_K_DENSE, GLM_N_LAYER);
        snprintf(fmt, sizeof(fmt), "model.layers.%%d.mlp.experts.%d.down_proj.weight", xid);
        report_layer_tensor(db, "ffn_down_exps", CLASS_MAPPED, fmt, GLM_FIRST_K_DENSE, GLM_N_LAYER);
    }

    printf("\nLegend: MAPPED=direct dequant+quantize  ABSORB=MLA fold (kv_b into q_b/o_proj)  SKIP=v1 omits\n");
    printf("Note: ds4-only tensors (HC, KV compressor, attn_sinks, hash tid2eid) are intentionally NOT emitted for GLM.\n");
}

static void usage(void) {
    fprintf(stderr,
        "usage: glm-quantize --hf <dir> [--dry-run | --read <tensor>]\n"
        "  --hf <dir>        directory with GLM-5.2 model-*.safetensors shards\n"
        "  --dry-run         scan shard headers and report converter coverage (default)\n"
        "  --read <tensor>   read one HF tensor, dequantize to f32, print shape + stats\n");
    exit(1);
}

int main(int argc, char **argv) {
    const char *hf_dir = NULL, *read_name = NULL;
    bool dry_run = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--hf") == 0 && i + 1 < argc) hf_dir = argv[++i];
        else if (strcmp(argv[i], "--read") == 0 && i + 1 < argc) read_name = argv[++i];
        else if (strcmp(argv[i], "--dry-run") == 0) dry_run = true;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) usage();
        else { fprintf(stderr, "glm-quantize: unknown arg: %s\n", argv[i]); usage(); }
    }
    if (!hf_dir) usage();

    stdb db;
    stdb_open(&db, hf_dir);

    if (read_name) read_tensor_cmd(&db, read_name);
    else { (void)dry_run; dry_run_report(&db); }
    return 0;
}
