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

#include "quants.h"

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

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) die("out of memory");
    return p;
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

/* File + JSON-string helpers for config/tokenizer parsing. */
static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "glm-quantize: cannot open %s\n", path); exit(1); }
    fseeko(f, 0, SEEK_END);
    off_t sz = ftello(f);
    fseeko(f, 0, SEEK_SET);
    char *b = xmalloc((size_t)sz + 1);
    if (sz && fread(b, 1, (size_t)sz, f) != (size_t)sz) die("short file read");
    b[sz] = '\0';
    fclose(f);
    if (len) *len = (size_t)sz;
    return b;
}
static int hex4(const char *p) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        int dgt = (c>='0'&&c<='9')?c-'0':(c>='a'&&c<='f')?c-'a'+10:(c>='A'&&c<='F')?c-'A'+10:0;
        v = v * 16 + dgt;
    }
    return v;
}
static int utf8_enc(unsigned cp, char *o) {
    if (cp < 0x80) { o[0]=(char)cp; return 1; }
    if (cp < 0x800) { o[0]=(char)(0xC0|(cp>>6)); o[1]=(char)(0x80|(cp&0x3F)); return 2; }
    if (cp < 0x10000) { o[0]=(char)(0xE0|(cp>>12)); o[1]=(char)(0x80|((cp>>6)&0x3F)); o[2]=(char)(0x80|(cp&0x3F)); return 3; }
    o[0]=(char)(0xF0|(cp>>18)); o[1]=(char)(0x80|((cp>>12)&0x3F)); o[2]=(char)(0x80|((cp>>6)&0x3F)); o[3]=(char)(0x80|(cp&0x3F)); return 4;
}
/* JSON string token -> unescaped UTF-8 (byte-level BPE tokens have no embedded NUL). */
static char *json_unescape(const json_doc *d, int tok) {
    int s = d->v[tok].start, e = d->v[tok].end;
    const char *p = d->js;
    char *out = xmalloc((size_t)(e - s) * 3 + 1);
    size_t o = 0;
    for (int i = s; i < e; i++) {
        char c = p[i];
        if (c != '\\') { out[o++] = c; continue; }
        if (++i >= e) break;
        char x = p[i];
        switch (x) {
            case 'n': out[o++]='\n'; break;
            case 't': out[o++]='\t'; break;
            case 'r': out[o++]='\r'; break;
            case 'b': out[o++]='\b'; break;
            case 'f': out[o++]='\f'; break;
            case 'u':
                if (i + 4 < e) {
                    int cp = hex4(p + i + 1); i += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 < e && p[i+1]=='\\' && p[i+2]=='u') {
                        int lo = hex4(p + i + 3); i += 6;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    o += (size_t)utf8_enc((unsigned)cp, out + o);
                }
                break;
            default: out[o++] = x; break;  /* \" \\ \/ and any other */
        }
    }
    out[o] = '\0';
    return out;
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
 * GGUF writer. Metadata is authored directly from GLM's config per the schema
 * contract in glm-port-plan.md sec 5c (these `deepseek4.*` values MUST match the
 * DS4_SHAPE_GLM constants the engine will validate against). Tensor data is added
 * incrementally; this milestone emits metadata + 0 tensors.
 */
enum {
    GT_U8 = 0, GT_I8, GT_U16, GT_I16, GT_U32, GT_I32, GT_F32,
    GT_BOOL, GT_STR, GT_ARR, GT_U64, GT_I64, GT_F64
};
#define GGUF_MAGIC   0x46554747u  /* "GGUF" little-endian */
#define GGUF_VERSION 3u

typedef struct { uint8_t *d; size_t n, cap; } buf;
static void bput(buf *b, const void *p, size_t n) {
    if (b->n + n > b->cap) {
        b->cap = b->cap ? b->cap : 4096;
        while (b->cap < b->n + n) b->cap *= 2;
        b->d = xrealloc(b->d, b->cap);
    }
    memcpy(b->d + b->n, p, n);
    b->n += n;
}
static void b_u32(buf *b, uint32_t v) { bput(b, &v, 4); }
static void b_u64(buf *b, uint64_t v) { bput(b, &v, 8); }
static void b_str(buf *b, const char *s) { uint64_t n = strlen(s); b_u64(b, n); if (n) bput(b, s, n); }

static void kv_str (buf *b, int *nk, const char *k, const char *v) { b_str(b,k); b_u32(b,GT_STR);  b_str(b,v); (*nk)++; }
static void kv_u32 (buf *b, int *nk, const char *k, uint32_t v)    { b_str(b,k); b_u32(b,GT_U32);  b_u32(b,v); (*nk)++; }
static void kv_f32 (buf *b, int *nk, const char *k, float v)       { b_str(b,k); b_u32(b,GT_F32);  bput(b,&v,4); (*nk)++; }
static void kv_bool(buf *b, int *nk, const char *k, int v)         { b_str(b,k); b_u32(b,GT_BOOL); uint8_t u=v?1:0; bput(b,&u,1); (*nk)++; }
static void kv_arr_u32(buf *b, int *nk, const char *k, const uint32_t *v, uint64_t n) {
    b_str(b,k); b_u32(b,GT_ARR); b_u32(b,GT_U32); b_u64(b,n);
    for (uint64_t i=0;i<n;i++) b_u32(b,v[i]); (*nk)++;
}
static void kv_arr_f32(buf *b, int *nk, const char *k, const float *v, uint64_t n) {
    b_str(b,k); b_u32(b,GT_ARR); b_u32(b,GT_F32); b_u64(b,n);
    for (uint64_t i=0;i<n;i++) bput(b,&v[i],4); (*nk)++;
}
static void kv_arr_str(buf *b, int *nk, const char *k, char **v, uint64_t n) {
    b_str(b,k); b_u32(b,GT_ARR); b_u32(b,GT_STR); b_u64(b,n);
    for (uint64_t i=0;i<n;i++) b_str(b,v[i]); (*nk)++;
}
static void kv_arr_i32(buf *b, int *nk, const char *k, const int32_t *v, uint64_t n) {
    b_str(b,k); b_u32(b,GT_ARR); b_u32(b,GT_I32); b_u64(b,n);
    for (uint64_t i=0;i<n;i++) bput(b,&v[i],4); (*nk)++;
}

/* GLM nominal float constants (must equal DS4_SHAPE_GLM). YaRN/compression/HC are
 * off for GLM, so their bases/epsilons are nominal placeholders. */
#define GLM_ROPE_FREQ_BASE      8000000.0f
#define GLM_ROPE_SCALE_FACTOR   1.0f
#define GLM_YARN_BETA_FAST      1.0f
#define GLM_YARN_BETA_SLOW      1.0f
#define GLM_COMPRESS_ROPE_BASE  8000000.0f
#define GLM_EXPERT_WEIGHT_SCALE 2.5f
#define GLM_RMS_EPS             1e-5f
#define GLM_HC_EPS              1e-5f
#define GLM_SWIGLU_CLAMP_EXP    30.0f   /* large => effectively no clamp (GLM plain SwiGLU) */

static void emit_metadata(buf *kv, int *nk) {
    kv_str (kv, nk, "general.architecture", "deepseek4");  /* ds4 does not validate this */
    kv_str (kv, nk, "general.name", "GLM-5.2");
    kv_u32 (kv, nk, "general.alignment", 32);
    kv_u32 (kv, nk, "deepseek4.block_count", GLM_N_LAYER);
    kv_u32 (kv, nk, "deepseek4.embedding_length", 6144);
    kv_u32 (kv, nk, "deepseek4.vocab_size", 154880);
    kv_u32 (kv, nk, "deepseek4.attention.head_count", 64);
    kv_u32 (kv, nk, "deepseek4.attention.head_count_kv", 1);
    kv_u32 (kv, nk, "deepseek4.attention.key_length", 576);   /* 512 c_kv + 64 rope (absorbed) */
    kv_u32 (kv, nk, "deepseek4.attention.value_length", 512); /* value = c_kv only */
    kv_u32 (kv, nk, "deepseek4.rope.dimension_count", 64);
    kv_u32 (kv, nk, "deepseek4.attention.q_lora_rank", 2048);
    kv_u32 (kv, nk, "deepseek4.attention.output_lora_rank", 0);    /* sentinel: plain o_proj */
    kv_u32 (kv, nk, "deepseek4.attention.output_group_count", 1);  /* sentinel */
    kv_u32 (kv, nk, "deepseek4.expert_count", GLM_N_EXPERT);
    kv_u32 (kv, nk, "deepseek4.expert_used_count", 8);
    kv_u32 (kv, nk, "deepseek4.expert_feed_forward_length", 2048);
    kv_u32 (kv, nk, "deepseek4.expert_shared_count", 1);
    kv_u32 (kv, nk, "deepseek4.hash_layer_count", 0);
    kv_u32 (kv, nk, "deepseek4.first_k_dense_count", GLM_FIRST_K_DENSE);  /* NEW key for GLM */
    kv_u32 (kv, nk, "deepseek4.attention.sliding_window", 0);   /* TBD: full-attention KV sizing */
    kv_u32 (kv, nk, "deepseek4.attention.indexer.head_count", 32);
    kv_u32 (kv, nk, "deepseek4.attention.indexer.key_length", 128);
    kv_u32 (kv, nk, "deepseek4.attention.indexer.top_k", 2048);
    kv_u32 (kv, nk, "deepseek4.hyper_connection.count", 1);            /* HC bypassed */
    kv_u32 (kv, nk, "deepseek4.hyper_connection.sinkhorn_iterations", 0);
    kv_f32 (kv, nk, "deepseek4.rope.freq_base", GLM_ROPE_FREQ_BASE);
    kv_f32 (kv, nk, "deepseek4.rope.scaling.factor", GLM_ROPE_SCALE_FACTOR);
    kv_f32 (kv, nk, "deepseek4.rope.scaling.yarn_beta_fast", GLM_YARN_BETA_FAST);
    kv_f32 (kv, nk, "deepseek4.rope.scaling.yarn_beta_slow", GLM_YARN_BETA_SLOW);
    kv_f32 (kv, nk, "deepseek4.attention.compress_rope_freq_base", GLM_COMPRESS_ROPE_BASE);
    kv_f32 (kv, nk, "deepseek4.expert_weights_scale", GLM_EXPERT_WEIGHT_SCALE);
    kv_f32 (kv, nk, "deepseek4.attention.layer_norm_rms_epsilon", GLM_RMS_EPS);
    kv_f32 (kv, nk, "deepseek4.hyper_connection.epsilon", GLM_HC_EPS);
    kv_bool(kv, nk, "deepseek4.expert_weights_norm", 1);
    uint32_t cr[GLM_N_LAYER]; for (int i=0;i<GLM_N_LAYER;i++) cr[i]=0;
    kv_arr_u32(kv, nk, "deepseek4.attention.compress_ratios", cr, GLM_N_LAYER);
    float sc[GLM_N_LAYER]; for (int i=0;i<GLM_N_LAYER;i++) sc[i]=GLM_SWIGLU_CLAMP_EXP;
    kv_arr_f32(kv, nk, "deepseek4.swiglu_clamp_exp", sc, GLM_N_LAYER);
}

/* GGUF token types (llama.cpp convention). */
#define GLM_VOCAB 154880
enum { TOK_NORMAL=1, TOK_UNKNOWN=2, TOK_CONTROL=3, TOK_USER=4, TOK_UNUSED=5 };

/* Author tokenizer.ggml.* from tokenizer.json (GPT-2 byte-level BPE). */
static void emit_tokenizer(buf *kv, int *nk, const char *hf_dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/tokenizer.json", hf_dir);
    size_t len = 0;
    char *text = read_file(path, &len);
    json_doc d = json_parse_text(text, len);

    int model  = json_obj_get(&d, 0, "model");
    if (model < 0) die("tokenizer.json: no model");
    int vocab  = json_obj_get(&d, model, "vocab");
    int merges = json_obj_get(&d, model, "merges");
    int added  = json_obj_get(&d, 0, "added_tokens");
    if (vocab < 0 || merges < 0) die("tokenizer.json: missing vocab/merges");

    char **tokens   = xcalloc(GLM_VOCAB, sizeof(char *));
    int32_t *ttype  = xmalloc(GLM_VOCAB * sizeof(int32_t));
    for (int i = 0; i < GLM_VOCAB; i++) ttype[i] = TOK_UNUSED;

    /* vocab: { "<token>": id } */
    for (int i = vocab + 1; i < d.len && d.v[i].parent == vocab;) {
        int k = i, v = i + 1;
        if (v >= d.len || d.v[v].parent != vocab) break;
        int64_t id = json_i64(&d, v);
        if (id >= 0 && id < GLM_VOCAB) { tokens[id] = json_unescape(&d, k); ttype[id] = TOK_NORMAL; }
        i = json_skip(&d, v);
    }
    /* added_tokens: [ { "id":N, "content":"..", "special":true } ] — all special => CONTROL */
    if (added >= 0) {
        for (int i = added + 1; i < d.len && d.v[i].parent == added; i = json_skip(&d, i)) {
            int idt = json_obj_get(&d, i, "id"), ct = json_obj_get(&d, i, "content");
            if (idt < 0 || ct < 0) continue;
            int64_t id = json_i64(&d, idt);
            if (id < 0 || id >= GLM_VOCAB) continue;
            free(tokens[id]); tokens[id] = json_unescape(&d, ct); ttype[id] = TOK_CONTROL;
        }
    }
    /* reserved ids with no token -> placeholders */
    int placeholders = 0;
    for (int i = 0; i < GLM_VOCAB; i++) {
        if (!tokens[i]) { char b[32]; snprintf(b, sizeof(b), "<|unused_%d|>", i); tokens[i] = xstrndup(b, strlen(b)); placeholders++; }
    }

    /* merges: [ ["a","b"] ] -> "a b"  (also accept plain "a b" strings) */
    uint64_t n_merges = 0;
    for (int i = merges + 1; i < d.len && d.v[i].parent == merges; i = json_skip(&d, i)) n_merges++;
    char **mstr = xmalloc((size_t)(n_merges ? n_merges : 1) * sizeof(char *));
    uint64_t mi = 0;
    for (int i = merges + 1; i < d.len && d.v[i].parent == merges; i = json_skip(&d, i)) {
        if (d.v[i].type == JT_STRING) { mstr[mi++] = json_unescape(&d, i); continue; }
        if (d.v[i].type == JT_ARRAY && i + 1 < d.len && d.v[i+1].parent == i) {
            int a = i + 1, b = json_skip(&d, a);
            char *sa = json_unescape(&d, a);
            char *sb = (b < d.len && d.v[b].parent == i) ? json_unescape(&d, b) : xstrndup("", 0);
            size_t la = strlen(sa), lb = strlen(sb);
            char *j = xmalloc(la + 1 + lb + 1);
            memcpy(j, sa, la); j[la] = ' '; memcpy(j + la + 1, sb, lb); j[la + 1 + lb] = '\0';
            free(sa); free(sb); mstr[mi++] = j;
        } else mstr[mi++] = xstrndup("", 0);
    }

    kv_str    (kv, nk, "tokenizer.ggml.model", "gpt2");
    kv_str    (kv, nk, "tokenizer.ggml.pre", "gpt2");
    kv_arr_str(kv, nk, "tokenizer.ggml.tokens", tokens, GLM_VOCAB);
    kv_arr_i32(kv, nk, "tokenizer.ggml.token_type", ttype, GLM_VOCAB);
    kv_arr_str(kv, nk, "tokenizer.ggml.merges", mstr, mi);
    kv_u32    (kv, nk, "tokenizer.ggml.eos_token_id", 154820);      /* <|endoftext|> */
    kv_u32    (kv, nk, "tokenizer.ggml.padding_token_id", 154820);

    fprintf(stderr, "glm-quantize: tokenizer: %d tokens (%d placeholders), %llu merges | "
            "t[0]='%s' t[154820]='%s' t[154822]='%s'\n",
            GLM_VOCAB, placeholders, (unsigned long long)mi, tokens[0], tokens[154820], tokens[154822]);

    for (int i = 0; i < GLM_VOCAB; i++) free(tokens[i]);
    free(tokens); free(ttype);
    for (uint64_t i = 0; i < mi; i++) free(mstr[i]);
    free(mstr);
    json_free(&d); free(text);
}

static void write_gguf_meta(const char *out, const char *hf_dir) {
    buf kv = {0};
    int nk = 0;
    emit_metadata(&kv, &nk);
    if (hf_dir) emit_tokenizer(&kv, &nk, hf_dir);
    FILE *f = fopen(out, "wb");
    if (!f) die("cannot open output gguf");
    uint32_t magic = GGUF_MAGIC, ver = GGUF_VERSION;
    uint64_t n_tensors = 0, n_kv = (uint64_t)nk;
    fwrite(&magic, 4, 1, f); fwrite(&ver, 4, 1, f);
    fwrite(&n_tensors, 8, 1, f); fwrite(&n_kv, 8, 1, f);
    fwrite(kv.d, 1, kv.n, f);
    fclose(f);
    free(kv.d);
    printf("wrote %s: %d KV records, 0 tensors, %zu metadata bytes\n", out, nk, kv.n);
}

/* =====
 * Tensor-data writer: GGUF tensor directory + 32-byte-aligned data section.
 * This iteration writes passthrough tensors (dequant bf16 -> quantize). Absorbed
 * attention (attn_q_b/attn_output) and stacked routed experts are TODO.
 */
typedef enum { SRC_PASS, SRC_QB_ABSORB, SRC_O_ABSORB, SRC_EXPERT } src_kind;
typedef struct {
    char name[192];
    ds4q_type type;
    int ndim;
    int64_t ne[4];
    src_kind kind;
    char hf[256];          /* PASS: source name; EXPERT: name template with one %d */
    int layer;             /* for ABSORB / EXPERT */
    uint64_t offset, nbytes;
} tplan;

static int64_t plan_nrows(const tplan *t) {
    int64_t r = 1;
    for (int i = 1; i < t->ndim; i++) r *= t->ne[i];
    return r;
}

static tplan *plan_add(tplan **arr, int *n, int *cap, const char *name, ds4q_type type,
                       int ndim, int64_t e0, int64_t e1, int64_t e2, const char *hf) {
    if (*n == *cap) { *cap = *cap ? *cap * 2 : 256; *arr = xrealloc(*arr, (size_t)*cap * sizeof(tplan)); }
    tplan *t = &(*arr)[(*n)++];
    memset(t, 0, sizeof(*t));
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->type = type; t->ndim = ndim; t->ne[0]=e0; t->ne[1]=e1; t->ne[2]=e2;
    t->kind = SRC_PASS; t->layer = -1;
    snprintf(t->hf, sizeof(t->hf), "%s", hf);
    return t;
}

/* attn_q_b absorbed [out=64*576, in=2048]: per head, top 512 rows = W_UK_h^T @ q_b_nope_h,
 * next 64 rows = q_b_rope_h. (See gguf-tools/check_mla_absorption.py — proven.) */
static float *produce_qb_absorb(stdb *db, int L) {
    char nm[256];
    snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_b_proj.weight", L);
    const gtensor *gq = stdb_find(db, nm); if (!gq) die("no q_b_proj");
    snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.kv_b_proj.weight", L);
    const gtensor *gk = stdb_find(db, nm); if (!gk) die("no kv_b_proj");
    int64_t a=0,b=0;
    float *qb = stdb_read_f32(db, gq, &a);    /* [16384,2048] */
    float *kvb = stdb_read_f32(db, gk, &b);   /* [28672,512]  */
    float *out = xcalloc((size_t)36864 * 2048, sizeof(float));
    for (int h = 0; h < 64; h++) {
        for (int m = 0; m < 192; m++) {
            const float *uk = kvb + ((size_t)h*448 + m) * 512;   /* W_UK_h row m: [512] */
            const float *qn = qb  + ((size_t)h*256 + m) * 2048;  /* q_b_nope_h row m: [2048] */
            for (int l = 0; l < 512; l++) {
                float av = uk[l];
                float *orow = out + ((size_t)h*576 + l) * 2048;
                for (int j = 0; j < 2048; j++) orow[j] += av * qn[j];
            }
        }
        for (int r = 0; r < 64; r++)
            memcpy(out + ((size_t)h*576 + 512 + r) * 2048,
                   qb + ((size_t)h*256 + 192 + r) * 2048, 2048 * sizeof(float));
    }
    free(qb); free(kvb);
    return out;
}

/* attn_output absorbed [out=6144, in=64*512]: per head, cols [h*512..] = o_proj_h @ W_UV_h. */
static float *produce_o_absorb(stdb *db, int L) {
    char nm[256];
    snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.o_proj.weight", L);
    const gtensor *go = stdb_find(db, nm); if (!go) die("no o_proj");
    snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.kv_b_proj.weight", L);
    const gtensor *gk = stdb_find(db, nm); if (!gk) die("no kv_b_proj");
    int64_t a=0,b=0;
    float *op = stdb_read_f32(db, go, &a);    /* [6144,16384] */
    float *kvb = stdb_read_f32(db, gk, &b);   /* [28672,512]  */
    float *out = xcalloc((size_t)6144 * 32768, sizeof(float));
    for (int h = 0; h < 64; h++) {
        for (int i = 0; i < 6144; i++) {
            const float *opi = op + (size_t)i*16384 + (size_t)h*256;  /* [256] */
            float *orow = out + (size_t)i*32768 + (size_t)h*512;       /* [512] */
            for (int c = 0; c < 256; c++) {
                float av = opi[c];
                const float *uv = kvb + ((size_t)h*448 + 192 + c) * 512;  /* W_UV_h row c: [512] */
                for (int l = 0; l < 512; l++) orow[l] += av * uv[l];
            }
        }
    }
    free(op); free(kvb);
    return out;
}

static tplan *build_plan(int n_layers, int *count) {
    tplan *p = NULL; int n = 0, cap = 0;
    char nm[192], hf[192];
    plan_add(&p,&n,&cap,"token_embd.weight", DS4Q_TYPE_F16,  2, 6144, 154880, 0, "model.embed_tokens.weight");
    plan_add(&p,&n,&cap,"output.weight",     DS4Q_TYPE_Q8_0, 2, 6144, 154880, 0, "lm_head.weight");
    plan_add(&p,&n,&cap,"output_norm.weight",DS4Q_TYPE_F32,  1, 6144, 0, 0, "model.norm.weight");
    for (int L = 0; L < n_layers; L++) {
        #define NM(s) (snprintf(nm,sizeof(nm),"blk.%d.%s",L,s), nm)
        #define HF(s) (snprintf(hf,sizeof(hf),"model.layers.%d.%s",L,s), hf)
        plan_add(&p,&n,&cap,NM("attn_norm.weight"),     DS4Q_TYPE_F32, 1,6144,0,0,    HF("input_layernorm.weight"));
        plan_add(&p,&n,&cap,NM("ffn_norm.weight"),      DS4Q_TYPE_F32, 1,6144,0,0,    HF("post_attention_layernorm.weight"));
        plan_add(&p,&n,&cap,NM("attn_q_a.weight"),      DS4Q_TYPE_Q8_0,2,6144,2048,0, HF("self_attn.q_a_proj.weight"));
        plan_add(&p,&n,&cap,NM("attn_q_a_norm.weight"), DS4Q_TYPE_F32, 1,2048,0,0,    HF("self_attn.q_a_layernorm.weight"));
        plan_add(&p,&n,&cap,NM("attn_kv.weight"),       DS4Q_TYPE_Q8_0,2,6144,576,0,  HF("self_attn.kv_a_proj_with_mqa.weight"));
        plan_add(&p,&n,&cap,NM("attn_kv_a_norm.weight"),DS4Q_TYPE_F32, 1,512,0,0,     HF("self_attn.kv_a_layernorm.weight"));
        { tplan *t = plan_add(&p,&n,&cap,NM("attn_q_b.weight"),    DS4Q_TYPE_Q8_0,2,2048,36864,0,""); t->kind=SRC_QB_ABSORB; t->layer=L; }
        { tplan *t = plan_add(&p,&n,&cap,NM("attn_output.weight"), DS4Q_TYPE_Q8_0,2,32768,6144,0,"");  t->kind=SRC_O_ABSORB;  t->layer=L; }
        if (L < GLM_FIRST_K_DENSE) {
            plan_add(&p,&n,&cap,NM("ffn_gate_dense.weight"),DS4Q_TYPE_Q8_0,2,6144,12288,0, HF("mlp.gate_proj.weight"));
            plan_add(&p,&n,&cap,NM("ffn_up_dense.weight"),  DS4Q_TYPE_Q8_0,2,6144,12288,0, HF("mlp.up_proj.weight"));
            plan_add(&p,&n,&cap,NM("ffn_down_dense.weight"),DS4Q_TYPE_Q8_0,2,12288,6144,0, HF("mlp.down_proj.weight"));
        } else {
            plan_add(&p,&n,&cap,NM("ffn_gate_inp.weight"),  DS4Q_TYPE_F16, 2,6144,256,0, HF("mlp.gate.weight"));
            plan_add(&p,&n,&cap,NM("exp_probs_b.bias"),     DS4Q_TYPE_F32, 1,256,0,0,    HF("mlp.gate.e_score_correction_bias"));
            plan_add(&p,&n,&cap,NM("ffn_gate_shexp.weight"),DS4Q_TYPE_Q8_0,2,6144,2048,0, HF("mlp.shared_experts.gate_proj.weight"));
            plan_add(&p,&n,&cap,NM("ffn_up_shexp.weight"),  DS4Q_TYPE_Q8_0,2,6144,2048,0, HF("mlp.shared_experts.up_proj.weight"));
            plan_add(&p,&n,&cap,NM("ffn_down_shexp.weight"),DS4Q_TYPE_Q8_0,2,2048,6144,0, HF("mlp.shared_experts.down_proj.weight"));
            char tmpl[256];
            snprintf(tmpl,sizeof(tmpl),"model.layers.%d.mlp.experts.%%d.gate_proj.weight",L);
            { tplan *t = plan_add(&p,&n,&cap,NM("ffn_gate_exps.weight"),DS4Q_TYPE_Q4_K,3,6144,2048,256,tmpl); t->kind=SRC_EXPERT; t->layer=L; }
            snprintf(tmpl,sizeof(tmpl),"model.layers.%d.mlp.experts.%%d.up_proj.weight",L);
            { tplan *t = plan_add(&p,&n,&cap,NM("ffn_up_exps.weight"),  DS4Q_TYPE_Q4_K,3,6144,2048,256,tmpl); t->kind=SRC_EXPERT; t->layer=L; }
            snprintf(tmpl,sizeof(tmpl),"model.layers.%d.mlp.experts.%%d.down_proj.weight",L);
            { tplan *t = plan_add(&p,&n,&cap,NM("ffn_down_exps.weight"),DS4Q_TYPE_Q4_K,3,2048,6144,256,tmpl); t->kind=SRC_EXPERT; t->layer=L; }
        }
        #undef NM
        #undef HF
    }
    *count = n;
    return p;
}

static void write_gguf_full(const char *out, const char *hf_dir, int n_layers) {
    stdb db; stdb_open(&db, hf_dir);
    int n = 0;
    tplan *plan = build_plan(n_layers, &n);

    /* offsets are analytic: sizes depend only on type+shape (ds4q_row_size). */
    uint64_t off = 0;
    for (int i = 0; i < n; i++) {
        int64_t nrows = plan_nrows(&plan[i]);
        size_t rs = ds4q_row_size(plan[i].type, plan[i].ne[0]);
        plan[i].nbytes = (uint64_t)rs * (uint64_t)nrows;
        plan[i].offset = off;
        off = ds4q_pad(off + plan[i].nbytes, 32);
    }

    buf kv = {0}; int nk = 0;
    emit_metadata(&kv, &nk);
    emit_tokenizer(&kv, &nk, hf_dir);

    FILE *f = fopen(out, "wb");
    if (!f) die("cannot open output gguf");
    uint32_t magic = GGUF_MAGIC, ver = GGUF_VERSION;
    uint64_t nt = (uint64_t)n, nkv = (uint64_t)nk, pos = 0;
    fwrite(&magic,4,1,f); fwrite(&ver,4,1,f); fwrite(&nt,8,1,f); fwrite(&nkv,8,1,f); pos += 24;
    fwrite(kv.d,1,kv.n,f); pos += kv.n; free(kv.d);
    for (int i = 0; i < n; i++) {
        tplan *t = &plan[i];
        uint64_t nl = strlen(t->name);
        fwrite(&nl,8,1,f); fwrite(t->name,1,nl,f); pos += 8 + nl;
        uint32_t nd = (uint32_t)t->ndim; fwrite(&nd,4,1,f); pos += 4;
        for (int dd = 0; dd < t->ndim; dd++) { uint64_t e=(uint64_t)t->ne[dd]; fwrite(&e,8,1,f); pos += 8; }
        uint32_t ty = (uint32_t)t->type; fwrite(&ty,4,1,f); pos += 4;
        fwrite(&t->offset,8,1,f); pos += 8;
    }
    while (pos < ds4q_pad(pos, 32)) { fputc(0, f); pos++; }   /* align data section */

    uint64_t dpos = 0;
    for (int i = 0; i < n; i++) {
        tplan *t = &plan[i];
        while (dpos < t->offset) { fputc(0, f); dpos++; }
        int64_t ncols = t->ne[0];
        ds4q_quantize_init(t->type);

        if (t->kind == SRC_EXPERT) {
            /* stream 256 experts: the full stacked tensor is too big to hold in RAM */
            int64_t per_rows = t->ne[1];
            size_t rs = ds4q_row_size(t->type, ncols);
            for (int E = 0; E < (int)t->ne[2]; E++) {
                char nm[256]; snprintf(nm, sizeof(nm), t->hf, E);
                const gtensor *g = stdb_find(&db, nm);
                if (!g) { fprintf(stderr, "glm-quantize: missing %s\n", nm); exit(1); }
                int64_t ne = 0; float *src = stdb_read_f32(&db, g, &ne);
                if (ne != per_rows * ncols) { fprintf(stderr, "glm-quantize: %s size mismatch\n", nm); exit(1); }
                void *dst = xmalloc(rs * (size_t)per_rows);
                size_t got = ds4q_quantize_chunk(t->type, src, dst, 0, per_rows, ncols, NULL);
                fwrite(dst, 1, got, f); dpos += got;
                free(dst); free(src);
            }
        } else {
            float *src;
            int64_t nrows = plan_nrows(t);
            if (t->kind == SRC_QB_ABSORB)     src = produce_qb_absorb(&db, t->layer);
            else if (t->kind == SRC_O_ABSORB) src = produce_o_absorb(&db, t->layer);
            else {
                const gtensor *g = stdb_find(&db, t->hf);
                if (!g) { fprintf(stderr, "glm-quantize: missing source %s\n", t->hf); exit(1); }
                int64_t ne = 0; src = stdb_read_f32(&db, g, &ne);
                if (ne != nrows * ncols) {
                    fprintf(stderr, "glm-quantize: %s size %lld != %lld*%lld\n",
                            t->name, (long long)ne, (long long)nrows, (long long)ncols);
                    exit(1);
                }
            }
            if (t->type == DS4Q_TYPE_F32) {
                fwrite(src, 4, (size_t)(nrows*ncols), f); dpos += (uint64_t)nrows*ncols*4;
            } else if (t->type == DS4Q_TYPE_F16) {
                uint16_t *d = xmalloc((size_t)(nrows*ncols) * 2);
                ds4q_f32_to_f16_row(src, d, nrows*ncols);
                fwrite(d, 2, (size_t)(nrows*ncols), f); dpos += (uint64_t)nrows*ncols*2; free(d);
            } else {
                void *dst = xmalloc(t->nbytes);
                size_t got = ds4q_quantize_chunk(t->type, src, dst, 0, nrows, ncols, NULL);
                if (got != t->nbytes) {
                    fprintf(stderr, "glm-quantize: %s quant %zu != %llu\n",
                            t->name, got, (unsigned long long)t->nbytes);
                    exit(1);
                }
                fwrite(dst, 1, got, f); dpos += got; free(dst);
            }
            free(src);
        }
        if ((i % 100) == 0 || i == n - 1)
            fprintf(stderr, "  [%d/%d] %s\n", i + 1, n, t->name);
    }
    fclose(f);
    printf("wrote %s: %d tensors, %d KV, %d layers\n", out, n, nk, n_layers);
    free(plan);
}

/* Minimal GGUF reader for --verify: lists KV keys, confirms structural validity. */
static uint32_t r_u32(FILE *f){ uint32_t v; if(fread(&v,4,1,f)!=1) die("read u32"); return v; }
static uint64_t r_u64(FILE *f){ uint64_t v; if(fread(&v,8,1,f)!=1) die("read u64"); return v; }
static char *r_str(FILE *f){ uint64_t n=r_u64(f); char *s=xmalloc(n+1); if(n&&fread(s,1,n,f)!=n) die("read str"); s[n]='\0'; return s; }
static size_t gguf_scalar_size(uint32_t t){
    switch(t){ case GT_U8: case GT_I8: case GT_BOOL: return 1;
               case GT_U16: case GT_I16: return 2;
               case GT_U32: case GT_I32: case GT_F32: return 4;
               case GT_U64: case GT_I64: case GT_F64: return 8; default: return 0; }
}
static void skip_value(FILE *f, uint32_t t){
    if (t==GT_STR){ uint64_t n=r_u64(f); if(fseeko(f,(off_t)n,SEEK_CUR)) die("seek"); return; }
    if (t==GT_ARR){ uint32_t et=r_u32(f); uint64_t n=r_u64(f);
        if (et==GT_STR){ for(uint64_t i=0;i<n;i++){ uint64_t l=r_u64(f); if(fseeko(f,(off_t)l,SEEK_CUR)) die("seek"); } return; }
        size_t s=gguf_scalar_size(et); if(!s) die("bad array elem type");
        if(fseeko(f,(off_t)(s*n),SEEK_CUR)) die("seek"); return; }
    size_t s=gguf_scalar_size(t); if(!s) die("bad value type");
    if(fseeko(f,(off_t)s,SEEK_CUR)) die("seek");
}
static void verify_gguf(const char *path){
    FILE *f=fopen(path,"rb"); if(!f) die("open gguf");
    uint32_t magic=r_u32(f), ver=r_u32(f);
    uint64_t nt=r_u64(f), nk=r_u64(f);
    printf("magic=%.4s version=%u n_tensors=%llu n_kv=%llu\n",
           (char*)&magic, ver, (unsigned long long)nt, (unsigned long long)nk);
    if (magic != GGUF_MAGIC) die("not a GGUF file");
    for (uint64_t i=0;i<nk;i++){ char *k=r_str(f); uint32_t t=r_u32(f); printf("  KV %s\n", k); free(k); skip_value(f,t); }
    for (uint64_t i=0;i<nt;i++){
        char *nme=r_str(f); uint32_t nd=r_u32(f);
        int64_t ne[4]={1,1,1,1};
        for (uint32_t j=0;j<nd;j++){ uint64_t e=r_u64(f); if(j<4) ne[j]=(int64_t)e; }
        uint32_t ty=r_u32(f); uint64_t of=r_u64(f);
        if (i < 24 || i+4 >= nt)
            printf("  T %-28s type=%u ne=[%lld,%lld,%lld] off=%llu\n", nme, ty,
                   (long long)ne[0],(long long)ne[1],(long long)ne[2],(unsigned long long)of);
        free(nme);
    }
    fclose(f);
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
        "usage: glm-quantize [--hf <dir>] [--dry-run | --read <tensor> | --write-gguf <out> | --verify <gguf>]\n"
        "  --hf <dir>        directory with GLM-5.2 model-*.safetensors shards\n"
        "  --dry-run         scan shard headers and report converter coverage (default)\n"
        "  --read <tensor>   read one HF tensor, dequantize to f32, print shape + stats\n"
        "  --write-gguf <o>  author a GLM GGUF (metadata + tokenizer, 0 tensors) to <o>\n"
        "  --write-full <o>  author a GLM GGUF with tensor data (passthrough; --layers N)\n"
        "  --layers <N>      limit --write-full to the first N layers (default 78)\n"
        "  --verify <gguf>   read back a GGUF header; list metadata keys + tensor infos\n");
    exit(1);
}

int main(int argc, char **argv) {
    const char *hf_dir = NULL, *read_name = NULL, *write_out = NULL, *write_full = NULL, *verify_in = NULL;
    int n_layers = 78;
    bool dry_run = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--hf") == 0 && i + 1 < argc) hf_dir = argv[++i];
        else if (strcmp(argv[i], "--read") == 0 && i + 1 < argc) read_name = argv[++i];
        else if (strcmp(argv[i], "--write-gguf") == 0 && i + 1 < argc) write_out = argv[++i];
        else if (strcmp(argv[i], "--write-full") == 0 && i + 1 < argc) write_full = argv[++i];
        else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) n_layers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--verify") == 0 && i + 1 < argc) verify_in = argv[++i];
        else if (strcmp(argv[i], "--dry-run") == 0) dry_run = true;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) usage();
        else { fprintf(stderr, "glm-quantize: unknown arg: %s\n", argv[i]); usage(); }
    }

    /* --verify needs only the gguf; the writers need --hf. */
    if (verify_in) { verify_gguf(verify_in); return 0; }
    if (write_out) { if (!hf_dir) usage(); write_gguf_meta(write_out, hf_dir); return 0; }
    if (write_full) { if (!hf_dir) usage(); if (n_layers < 0) n_layers = 0; write_gguf_full(write_full, hf_dir, n_layers); return 0; }

    if (!hf_dir) usage();
    stdb db;
    stdb_open(&db, hf_dir);
    if (read_name) read_tensor_cmd(&db, read_name);
    else { (void)dry_run; dry_run_report(&db); }
    return 0;
}
