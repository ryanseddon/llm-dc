//
// llm.h - tinyStories Llama-2 260K transformer, ported from Karpathy's llama2.c
// (via the esp32-llm fork). FreeRTOS dual-core, the Xtensa S3 dotprod, and the
// mmap shims are stripped for Windows CE on Dreamcast. Phase 1 = scalar matmul;
// Phase 2 will swap the matmul inner loop for a hand-tuned SH-4 FTRV kernel.
//
// One pragmatic change vs. upstream: llama2.c names every fp32 weight/activation
// `v4sf` (a GCC `aligned(16)` float attribute). MSVC's 1999 cl.exe doesn't parse
// `__attribute__`, so `v4sf` becomes a plain `float` typedef. ABI for the model
// blob is unchanged (fp32 weights, sequential hear layout).
//
#ifndef LLM_H
#define LLM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <windows.h>
#include "floatmathlib.h"   // SH-4 soft-float: sins/coss/exps/sqrts/pows

// llama2.c writes the math with the C99 *f suffix; CE coredll has no expf/sinf/
// cosf/sqrtf/powf. floatmathlib.h only aliases the un-suffixed names, so map the
// f-suffixed ones here. Macros guarded so a future build overlay can override.
#ifndef expf
#define expf(v)     exps(v)
#endif
#ifndef sinf
#define sinf(v)     sins(v)
#endif
#ifndef cosf
#define cosf(v)     coss(v)
#endif
#ifndef sqrtf
#define sqrtf(v)    sqrts(v)
#endif
#ifndef powf
#define powf(a,b)   pows((a),(b))
#endif

// Kept as a plain float alias so the vendored body compiles with minimal churn.
// If/when Phase 2 needs aligned weight rows, malloc+round-up will be applied at
// buffer build time, not via a typed-pointer qualifier.
typedef float v4sf;

typedef struct
{
	float prob;
	int index;
} ProbIndex;

typedef struct
{
	char *str;
	int id;
} TokenIndex;

typedef struct
{
	char **vocab;
	float *vocab_scores;
	TokenIndex *sorted_vocab;
	int vocab_size;
	unsigned int max_token_length;
	unsigned char byte_pieces[512];
} Tokenizer;

typedef struct
{
	int dim;          // transformer dimension
	int hidden_dim;  // ffn inner
	int n_layers;
	int n_heads;
	int n_kv_heads;  // GQA: <= n_heads
	int vocab_size;
	int seq_len;
} Config;

typedef struct
{
	float *token_embedding_table; // (vocab_size, dim)
	float *rms_att_weight;       // (layer, dim)
	float *rms_ffn_weight;       // (layer, dim)
	float *wq, *wk, *wv, *wo;    // (layer, ...)
	float *w1, *w2, *w3;          // (layer, ...)
	float *rms_final_weight;     // (dim,)
	float *wcls;                 // shared with token_embedding_table if shared_weights
} TransformerWeights;

typedef struct
{
	float *x, *xb, *xb2, *hb, *hb2, *q;
	float *k, *v, *att, *logits;
	float *key_cache, *value_cache;
} RunState;

typedef struct
{
	Config config;
	TransformerWeights weights;
	RunState state;
	float *data;        // the malloc'd weight blob (heap, not mmap'd)
	size_t file_size;
} Transformer;

typedef struct
{
	int vocab_size;
	ProbIndex *probindex;
	float temperature;
	float topp;
	unsigned long long rng_state;
} Sampler;

// File paths are wide on CE (\CD-ROM\<file>.bin). coredll fopen wants ANSI and CE
// doesn't auto-convert the \CD-ROM prefix reliably, so we use CreateFileW+ReadFile
// (the dcwplay.c/dcgfx.c idiom) instead of fopen/fread.
void build_transformer(Transformer *t, const WCHAR *checkpoint_path);
void build_tokenizer(Tokenizer *t, const WCHAR *tokenizer_path, int vocab_size);
void build_sampler(Sampler *s, int vocab_size, float temperature, float topp,
                    unsigned long long rng_seed);

// Single-token step: the app drives the decode loop (Phase 1 mark) instead of the
// upstream blocking generate(). This lets the DCWin render loop pump between
// forwards so the window doesn't freeze for the ~multi-second burst of 64 steps.
float *forward(Transformer *t, int token, int pos);
int sample(Sampler *s, float *logits);
char *decode(Tokenizer *t, int prev_token, int token);
void encode(Tokenizer *t, char *text, int bos, int eos, int *tokens, int *n_tokens);

void free_transformer(Transformer *t);
void free_tokenizer(Tokenizer *t);
void free_sampler(Sampler *s);

#endif