//
// llm.c - llama2.c (TinyStories 260K) inference, ported to Windows CE on
// Dreamcast. Stripped from esp32-llm/main/llm.c: FreeRTOS dual-core, the Xtensa
// S3 dotprod, the mmap shim, and the blocking generate()/chat scaffolding are
// gone; the matmul inner loop is scalar C (Phase 1). The app drives forward()
// one token at a time from its window message loop so DCWin repaints between
// tokens. File I/O for the weight blob + tokenizer is CreateFileW+ReadFile
// (CE coredll's fopen is ANSI-only and unreliable for \CD-ROM prefixed paths).
//
// Math: coredll exposes no expf/sinf/cosf/sqrtf/powf - floatmathlib.h's sins/
// coss/exps/sqrts are wired in via macros in llm.h. The Transformer body
// otherwise reads ~verbatim from Karpathy's llama2.c.
//
#include "llm.h"
#include "syslog.h"

// CE 2.12 coredll ships malloc/free/realloc/qsort/memset/memcpy but no calloc
// and no bsearch; llm.c was written with both. Local shims replace them.
static void *xcalloc(size_t n, size_t sz)
{
	size_t tot = n * sz;
	void *p = malloc(tot);
	if (p)
		memset(p, 0, tot);
	return p;
}

static int xstr_lookup(const char *key, const TokenIndex *a, int n)
{
	int lo = 0, hi = n - 1;
	while (lo <= hi)
	{
		int mid = (lo + hi) >> 1;
		int cmp = strcmp(a[mid].str, key);
		if (cmp == 0)
			return a[mid].id;
		if (cmp < 0)
			lo = mid + 1;
		else
			hi = mid - 1;
	}
	return -1;
}

// ---------------------------------------------------------------------------
// weights + runstate
//
static void malloc_run_state(RunState *s, Config *p)
{
	int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
s->x            = (float *)xcalloc(p->dim, sizeof(float));
	s->xb           = (float *)xcalloc(p->dim, sizeof(float));
	s->xb2          = (float *)xcalloc(p->dim, sizeof(float));
	s->hb           = (float *)xcalloc(p->hidden_dim, sizeof(float));
	s->hb2          = (float *)xcalloc(p->hidden_dim, sizeof(float));
	s->q            = (float *)xcalloc(p->dim, sizeof(float));
	s->key_cache    = (float *)xcalloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
	s->value_cache  = (float *)xcalloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
	s->att          = (float *)xcalloc(p->n_heads * p->seq_len, sizeof(float));
	s->logits       = (float *)xcalloc(p->vocab_size, sizeof(float));
	if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q ||
	    !s->key_cache || !s->value_cache || !s->att || !s->logits)
	{
		SysLog(L"llm: malloc_run_state OOM");
	}
}

static void free_run_state(RunState *s)
{
	free(s->x);
	free(s->xb);
	free(s->xb2);
	free(s->hb);
	free(s->hb2);
	free(s->q);
	free(s->att);
	free(s->logits);
	free(s->key_cache);
	free(s->value_cache);
}

static void memory_map_weights(TransformerWeights *w, Config *p, float *ptr,
                               int shared_weights)
{
	int head_size = p->dim / p->n_heads;
	unsigned long long n_layers = (unsigned long long)p->n_layers;
	w->token_embedding_table = ptr;
	ptr += p->vocab_size * p->dim;
	w->rms_att_weight = ptr;
	ptr += n_layers * p->dim;
	w->wq = ptr;
	ptr += n_layers * p->dim * (p->n_heads * head_size);
	w->wk = ptr;
	ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
	w->wv = ptr;
	ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
	w->wo = ptr;
	ptr += n_layers * (p->n_heads * head_size) * p->dim;
	w->rms_ffn_weight = ptr;
	ptr += n_layers * p->dim;
	w->w1 = ptr;
	ptr += n_layers * p->dim * p->hidden_dim;
	w->w2 = ptr;
	ptr += n_layers * p->hidden_dim * p->dim;
	w->w3 = ptr;
	ptr += n_layers * p->dim * p->hidden_dim;
	w->rms_final_weight = ptr;
	ptr += p->dim;
	ptr += p->seq_len * head_size / 2; // skip freq_cis_real (legacy RoPE blob)
	ptr += p->seq_len * head_size / 2; // skip freq_cis_imag
	w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

static void read_checkpoint(const WCHAR *checkpoint, Config *config,
                            TransformerWeights *weights, float **data,
                            size_t *file_size)
{
	HANDLE h;
	DWORD got;
	int shared_weights;

	h = CreateFileW(checkpoint, GENERIC_READ, FILE_SHARE_READ, NULL,
	                OPEN_EXISTING, 0, NULL);
	if (h == INVALID_HANDLE_VALUE)
	{
		SysLog(L"llm: checkpoint open failed (err=%lu)", GetLastError());
		*data = NULL;
		*file_size = 0;
		return;
	}
	*file_size = GetFileSize(h, NULL);
	*data = (float *)malloc(*file_size);
	if (!*data)
	{
		SysLog(L"llm: cp malloc %u B failed", (DWORD)*file_size);
		CloseHandle(h);
		return;
	}
	if (!ReadFile(h, *data, (DWORD)*file_size, &got, NULL) || got != *file_size)
	{
		SysLog(L"llm: cp read short (got=%lu want=%lu)", got, (DWORD)*file_size);
		free(*data);
		*data = NULL;
		CloseHandle(h);
		return;
	}
	CloseHandle(h);

	memcpy(config, *data, sizeof(Config));
	shared_weights = config->vocab_size > 0 ? 1 : 0;
	config->vocab_size = abs(config->vocab_size);
	memory_map_weights(weights, config, *data + sizeof(Config) / sizeof(float),
	                    shared_weights);
	SysLog(L"llm: checkpoint OK (%u B, vocab=%d dim=%d L=%d)",
	       (DWORD)*file_size, config->vocab_size, config->dim, config->n_layers);
}

void build_transformer(Transformer *t, const WCHAR *checkpoint_path)
{
	memset(t, 0, sizeof(*t));
	read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size);
	if (!t->data)
		return;
	malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t)
{
	free_run_state(&t->state);
	free(t->data);
	t->data = NULL;
}

// ---------------------------------------------------------------------------
// neural net blocks
//
void rmsnorm(float *o, float *x, float *weight, int size)
{
	float ss = 0.0f;
	int j;
	for (j = 0; j < size; j++)
		ss += x[j] * x[j];
	ss /= (float)size;
	ss += 1e-5f;
	ss = 1.0f / sqrts(ss);
	for (j = 0; j < size; j++)
		o[j] = weight[j] * (ss * x[j]);
}

void softmax(float *x, int size)
{
	float max_val = x[0];
	float sum = 0.0f;
	int i;
	for (i = 1; i < size; i++)
		if (x[i] > max_val)
			max_val = x[i];
	for (i = 0; i < size; i++)
	{
		x[i] = exps(x[i] - max_val);
		sum += x[i];
	}
	for (i = 0; i < size; i++)
		x[i] /= sum;
}

// W is [d rows x n cols] row-major; out += W * x. Scalar C, 4-way unrolled.
// This is the Phase-1 baseline; Phase 2 replaces the inner loop with an
// asm FTRV-kernel symbol `ftrv_matmul` (same signature) without touching callers.
void matmul(float *xout, float *x, float *w, int n, int d)
{
	int i, j;
	for (i = 0; i < d; i++)
	{
		float val = 0.0f;
		float *row = &w[(size_t)i * n];
		for (j = 0; j <= n - 4; j += 4)
			val += row[j] * x[j] + row[j+1] * x[j+1]
			     + row[j+2] * x[j+2] + row[j+3] * x[j+3];
		for (; j < n; j++)
			val += row[j] * x[j];
		xout[i] = val;
	}
}

float *forward(Transformer *t, int token, int pos)
{
	Config *p = &t->config;
	TransformerWeights *w = &t->weights;
	RunState *s = &t->state;
	float *x = s->x;
	int dim = p->dim;
	int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
	int kv_mul = p->n_heads / p->n_kv_heads;
	int hidden_dim = p->hidden_dim;
	int head_size = dim / p->n_heads;
	int l, h, i, t2;
	float *content_row;

	content_row = w->token_embedding_table + (size_t)token * dim;
	memcpy(x, content_row, dim * sizeof(*x));

	for (l = 0; l < p->n_layers; l++)
	{
		int loff = l * p->seq_len * kv_dim;
		s->k = s->key_cache   + loff + pos * kv_dim;
		s->v = s->value_cache + loff + pos * kv_dim;

		// attention rmsnorm + qkv matmuls
		rmsnorm(s->xb, x, w->rms_att_weight + (size_t)l * dim, dim);
		matmul(s->q, s->xb, w->wq + (size_t)l * dim * dim,           dim, dim);
		matmul(s->k, s->xb, w->wk + (size_t)l * dim * kv_dim,        dim, kv_dim);
		matmul(s->v, s->xb, w->wv + (size_t)l * dim * kv_dim,        dim, kv_dim);

		// RoPE: rotate q (and k) per head pair
		for (i = 0; i < dim; i += 2)
		{
			int head_dim = i % head_size;
			float freq = 1.0f / pows(10000.0f, (float)head_dim / (float)head_size);
			float val = (float)pos * freq;
			float fcr = coss(val);
			float fci = sins(val);
			int rotn = i < kv_dim ? 2 : 1;
			int vi;
			for (vi = 0; vi < rotn; vi++)
			{
				float *vec = vi == 0 ? s->q : s->k;
				float v0 = vec[i], v1 = vec[i+1];
				vec[i]   = v0 * fcr - v1 * fci;
				vec[i+1] = v0 * fci + v1 * fcr;
			}
		}

		// multihead attention (serial over heads; Phase 2 FTRV slate will accelerate
		// the q*k dot and the a*v reduction separately if needed)
		for (h = 0; h < p->n_heads; h++)
		{
			float *q = s->q + h * head_size;
			float *att = s->att + h * p->seq_len;
			float *xb;
			for (t2 = 0; t2 <= pos; t2++)
			{
				float *k = s->key_cache + loff + (size_t)t2 * kv_dim
				            + (h / kv_mul) * head_size;
				float score = 0.0f;
				int ii;
				for (ii = 0; ii < head_size; ii++)
					score += q[ii] * k[ii];
				score /= sqrts((float)head_size);
				att[t2] = score;
			}
			softmax(att, pos + 1);
			xb = s->xb + h * head_size;
			memset(xb, 0, head_size * sizeof(float));
			for (t2 = 0; t2 <= pos; t2++)
			{
				float *v = s->value_cache + loff + (size_t)t2 * kv_dim
				           + (h / kv_mul) * head_size;
				float a = att[t2];
				int ii;
				for (ii = 0; ii < head_size; ii++)
					xb[ii] += a * v[ii];
			}
		}

		// output projection + residual
		matmul(s->xb2, s->xb, w->wo + (size_t)l * dim * dim, dim, dim);
		for (i = 0; i < dim; i++)
			x[i] += s->xb2[i];

		// ffn rmsnorm + SwiGLU
		rmsnorm(s->xb, x, w->rms_ffn_weight + (size_t)l * dim, dim);
		matmul(s->hb,  s->xb, w->w1 + (size_t)l * dim * hidden_dim, dim, hidden_dim);
		matmul(s->hb2, s->xb, w->w3 + (size_t)l * dim * hidden_dim, dim, hidden_dim);
		for (i = 0; i < hidden_dim; i++)
		{
			float val = s->hb[i];
			val *= (1.0f / (1.0f + exps(-val)));
			val *= s->hb2[i];
			s->hb[i] = val;
		}
		matmul(s->xb, s->hb, w->w2 + (size_t)l * hidden_dim * dim, hidden_dim, dim);
		for (i = 0; i < dim; i++)
			x[i] += s->xb[i];
	}

	// final rmsnorm + classifier
	rmsnorm(x, x, w->rms_final_weight, dim);
	matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
	return s->logits;
}

// ---------------------------------------------------------------------------
// BPE tokenizer
//
static int compare_tokens(const void *a, const void *b)
{
	return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, const WCHAR *tokenizer_path, int vocab_size)
{
	HANDLE h;
	DWORD got;
	int i, len;

	t->vocab_size = vocab_size;
	t->vocab = (char **)malloc(vocab_size * sizeof(char *));
	t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
	t->sorted_vocab = NULL;
	for (i = 0; i < 256; i++)
	{
		t->byte_pieces[i*2]     = (unsigned char)i;
		t->byte_pieces[i*2 + 1] = '\0';
	}

	h = CreateFileW(tokenizer_path, GENERIC_READ, FILE_SHARE_READ, NULL,
	                OPEN_EXISTING, 0, NULL);
	if (h == INVALID_HANDLE_VALUE)
	{
		SysLog(L"llm: tokenizer open failed (err=%lu)", GetLastError());
		return;
	}
	ReadFile(h, &t->max_token_length, sizeof(int), &got, NULL);
	for (i = 0; i < vocab_size; i++)
	{
		ReadFile(h, &t->vocab_scores[i], sizeof(float), &got, NULL);
		ReadFile(h, &len, sizeof(int), &got, NULL);
		t->vocab[i] = (char *)malloc(len + 1);
		ReadFile(h, t->vocab[i], len, &got, NULL);
		t->vocab[i][len] = '\0';
	}
	CloseHandle(h);
	SysLog(L"llm: tokenizer OK (vocab=%d maxtok=%u)", vocab_size,
	       t->max_token_length);
}

void free_tokenizer(Tokenizer *t)
{
	int i;
	for (i = 0; i < t->vocab_size; i++)
		free(t->vocab[i]);
	free(t->vocab);
	free(t->vocab_scores);
	free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token)
{
	char *piece = t->vocab[token];
	unsigned char byte_val;
	if (prev_token == 1 && piece[0] == ' ')
		piece++; // sentencepiece strips leading space after BOS
	if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
		piece = (char *)t->byte_pieces + byte_val * 2;
	return piece;
}

static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size)
{
	return xstr_lookup(str, sorted_vocab, vocab_size);
}

void encode(Tokenizer *t, char *text, int bos, int eos, int *tokens, int *n_tokens)
{
	char *str_buffer;
	size_t str_len = 0;
	char *c;
	int dummy_prefix;

	if (text == NULL)
	{
		*n_tokens = 0;
		return;
	}

	if (t->sorted_vocab == NULL)
	{
		int i;
		t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
		for (i = 0; i < t->vocab_size; i++)
		{
			t->sorted_vocab[i].str = t->vocab[i];
			t->sorted_vocab[i].id = i;
		}
		qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
	}

	str_buffer = (char *)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
	*n_tokens = 0;
	if (bos)
		tokens[(*n_tokens)++] = 1;
	if (text[0] != '\0')
	{
		dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
		tokens[(*n_tokens)++] = dummy_prefix;
	}

	for (c = text; *c != '\0'; c++)
	{
		if ((*c & 0xC0) != 0x80)
			str_len = 0;
		str_buffer[str_len++] = *c;
		str_buffer[str_len] = '\0';
		if ((*(c+1) & 0xC0) == 0x80 && str_len < 4)
			continue;
		{
			int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
			int i;
			if (id != -1)
			{
				tokens[(*n_tokens)++] = id;
			}
			else
			{
				for (i = 0; i < (int)str_len; i++)
					tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
			}
		}
		str_len = 0;
	}

	// BPE merge loop: greedily merge best adjacent pair per pass
	for (;;)
	{
		float best_score = -1e10f;
		int best_id = -1, best_idx = -1, i;
		for (i = 0; i < (*n_tokens - 1); i++)
		{
			int id;
			sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
			id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
			if (id != -1 && t->vocab_scores[id] > best_score)
			{
				best_score = t->vocab_scores[id];
				best_id = id;
				best_idx = i;
			}
		}
		if (best_idx == -1)
			break;
		tokens[best_idx] = best_id;
		for (i = best_idx + 1; i < (*n_tokens - 1); i++)
			tokens[i] = tokens[i+1];
		(*n_tokens)--;
	}

	if (eos)
		tokens[(*n_tokens)++] = 2;
	free(str_buffer);
}

// ---------------------------------------------------------------------------
// sampler
//
static int sample_argmax(float *probabilities, int n)
{
	int max_i = 0, i;
	float max_p = probabilities[0];
	for (i = 1; i < n; i++)
	{
		if (probabilities[i] > max_p)
		{
			max_i = i;
			max_p = probabilities[i];
		}
	}
	return max_i;
}

static int sample_mult(float *probabilities, int n, float coin)
{
	float cdf = 0.0f;
	int i;
	for (i = 0; i < n; i++)
	{
		cdf += probabilities[i];
		if (coin < cdf)
			return i;
	}
	return n - 1;
}

static int sample_cmp(const void *a, const void *b)
{
	ProbIndex *pa = (ProbIndex *)a;
	ProbIndex *pb = (ProbIndex *)b;
	if (pa->prob > pb->prob) return -1;
	if (pa->prob < pb->prob) return 1;
	return 0;
}

static int sample_topp(float *probabilities, int n, float topp,
                       ProbIndex *probindex, float coin)
{
	int n0 = 0;
	int last_idx, i;
	float cutoff = (1.0f - topp) / (float)(n - 1);
	float cumulative_prob, r, cdf;
	for (i = 0; i < n; i++)
	{
		if (probabilities[i] >= cutoff)
		{
			probindex[n0].index = i;
			probindex[n0].prob = probabilities[i];
			n0++;
		}
	}
	qsort(probindex, n0, sizeof(ProbIndex), sample_cmp);
	cumulative_prob = 0.0f;
	last_idx = n0 - 1;
	for (i = 0; i < n0; i++)
	{
		cumulative_prob += probindex[i].prob;
		if (cumulative_prob > topp)
		{
			last_idx = i;
			break;
		}
	}
	r = coin * cumulative_prob;
	cdf = 0.0f;
	for (i = 0; i <= last_idx; i++)
	{
		cdf += probindex[i].prob;
		if (r < cdf)
			return probindex[i].index;
	}
	return probindex[last_idx].index;
}

void build_sampler(Sampler *s, int vocab_size, float temperature, float topp,
                   unsigned long long rng_seed)
{
	s->vocab_size = vocab_size;
	s->temperature = temperature;
	s->topp = topp;
	s->rng_state = rng_seed;
	s->probindex = (ProbIndex *)malloc(vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *s)
{
	free(s->probindex);
}

static unsigned int random_u32(unsigned long long *state)
{
	*state ^= *state >> 12;
	*state ^= *state << 25;
	*state ^= *state >> 27;
	return (unsigned int)((*state * 0x2545F4914F6CDD1Dull) >> 32);
}

static float random_f32(unsigned long long *state)
{
	return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *s, float *logits)
{
	int next, q;
	if (s->temperature == 0.0f)
		return sample_argmax(logits, s->vocab_size);

	for (q = 0; q < s->vocab_size; q++)
		logits[q] /= s->temperature;
	softmax(logits, s->vocab_size);
	{
		float coin = random_f32(&s->rng_state);
		if (s->topp <= 0 || s->topp >= 1)
			next = sample_mult(logits, s->vocab_size, coin);
		else
			next = sample_topp(logits, s->vocab_size, s->topp,
			                   s->probindex, coin);
	}
	return next;
}