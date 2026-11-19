//
// dcwllm.c - TinyStories 260K Llama-2 inference, running in a DCWin window.
// Phase 1 of the llm-port plan: scalar matmul, ASCII OSK prompt entry, one
// forward() per render frame so the DCWin render loop keeps pumping between
// tokens (a Tree-style blocking generate() would freeze the compositor for
// the whole ~64-token burst).
//
// Layout: prompt field at top, a 4x10 OSK + space/bksp/go/clear buttons, the
// execution status line, and a tail-following green-on-black output view.
//
#include "dcwlib.h"
#include "syslog.h"
#include "../llm/llm.h"

#define LW          600
#define LH          460
#define ROW_OUT     12

#define ST_IDLE     0
#define ST_GEN      1
#define ST_DONE     2

#define STEPS_DEF   64

#define KROWS       4
#define KCOLS       10
#define KW          34
#define KH          24
#define KGAP        2
#define KOX         4
#define KOY         28

#define BTN_Y       132
#define BTN_H       26
#define BTN_W       144
#define BTN_GAP     4

#define PROMPT_Y    2
#define PROMPT_H    22
#define PROMPT_X    4
#define PROMPT_W    (LW - 8)

#define STATUS_Y    (BTN_Y + BTN_H + 4)
#define OUT_Y       (STATUS_Y + 14)

static const WCHAR *kbRows[KROWS] = {
	L"1234567890", L"qwertyuiop", L"asdfghjkl-", L"zxcvbnm.:/"
};

static const struct {
	const WCHAR *label;
	int id;          // 0=space 1=bksp 2=go 3=clear
} s_btn[4] = {
	{L"Space", 0},
	{L"Bksp",  1},
	{L"Go",    2},
	{L"Clear", 3},
};

static int s_hotR = -1, s_hotC = -1;     // pointer-hovered key (mouse)
static int s_hotBtn = -1;                // pointer-hovered button
// D-pad selection: rows 0..3 are the keyboard rows (KCOLS cols); row 4 is the
// button row (4 cols). ENTER (real keyboard or the shell-synthesized VK_RETURN
// the controller's A button produces) activates the focused cell.
// Defaults to the "Go" button so an immediate ENTER re-runs generation.
static int s_selRow = 4, s_selCol = 2;
// Last seen pointer client-relative position. We sync s_selRow/s_selCol to the
// hovered cell ONLY when (px,py) has actually changed since last frame - so a
// D-pad-only user's selection isn't dragged back to the analog-stick's resting
// position every render. Initialised to out-of-window sentinel values.
static int s_lastPx = -1000, s_lastPy = -1000;

static Transformer g_tx;
static Tokenizer   g_tk;
static Sampler     g_sm;
static int g_haveModel = 0;
static int g_state = ST_IDLE;

static char  g_prompt[80];
static int   g_nPrompt = 0;
static char  g_out[8192];
static int   g_outLen = 0;
static int   g_ptoks[256];
static int   g_nPtok = 0;
static int   g_token = 0, g_pos = 0, g_steps = STEPS_DEF;
static DWORD g_startTick = 0;
static float g_lastTps = 0.0f;
static int   g_fwdCount = 0;     // forwards completed in the current run

// ---------------------------------------------------------------------------
// helpers
//
static void OutAppend(int ch)
{
	if (g_outLen < (int)sizeof(g_out) - 2)
	{
		g_out[g_outLen++] = (char)ch;
		g_out[g_outLen]   = '\0';
	}
}

static void PromptClear(void)
{
	g_nPrompt = 0;
	g_prompt[0] = '\0';
}

static void PromptAppend(char ch)
{
	if (g_nPrompt < (int)sizeof(g_prompt) - 2)
	{
		g_prompt[g_nPrompt++] = ch;
		g_prompt[g_nPrompt]   = '\0';
	}
}

static void PromptBksp(void)
{
	if (g_nPrompt > 0)
	{
		g_nPrompt--;
		g_prompt[g_nPrompt] = '\0';
	}
}

static void OutClear(void)
{
	g_outLen = 0;
	g_out[0] = '\0';
}

static void StartGenerate(void)
{
	if (g_state == ST_GEN)
		return;                       // already running - ignore extra Go
	if (!g_haveModel)
	{
		SysLog(L"llm: no model loaded");
		return;
	}
	OutClear();
	g_pos = 0;
	g_token = 1;                       // BOS - encode will overwrite below
	g_fwdCount = 0;
	g_lastTps = 0.0f;
	g_startTick = 0;
	encode(&g_tk, g_prompt[0] ? g_prompt : (char *)"", 1, 0, g_ptoks, &g_nPtok);
	if (g_nPtok < 1)
	{
		SysLog(L"llm: encode returned 0 tokens");
		return;
	}
	g_token = g_ptoks[0];
	g_state = ST_GEN;
	SysLog(L"llm: start (%d prompt tokens, %d steps)", g_nPtok, g_steps);
}

static void FinishGenerate(void)
{
	if (g_state == ST_GEN)
	{
		DWORD elapsed = GetTickCount() - g_startTick;
		if (g_pos > 1 && elapsed > 0)
			g_lastTps = (float)(g_pos - 1) * 1000.0f / (float)elapsed;
		else
			g_lastTps = 0.0f;
		SysLog(L"llm: done %d tok, %.2f tok/s", g_pos, g_lastTps);
	}
	// Reset to IDLE so a subsequent Go / ENTER re-runs without a restart.
	// The render path distinguishes "just-finished" from "fresh" via g_lastTps.
	g_state = ST_IDLE;
}

static void StepForward(void)
{
	float *logits;
	int next;
	char *piece, *p;

	if (g_state != ST_GEN)
		return;
	if (g_pos >= g_steps)
	{
		FinishGenerate();
		return;
	}
	logits = forward(&g_tx, g_token, g_pos);
	if (g_pos < g_nPtok - 1)
		next = g_ptoks[g_pos + 1];
	else
		next = sample(&g_sm, logits);
	g_pos++;
	g_fwdCount++;
	if (next == 1)
	{
		FinishGenerate();
		return;
	}
	piece = decode(&g_tk, g_token, next);
	for (p = piece; *p; p++)
		OutAppend(*p);
	g_token = next;
	if (g_startTick == 0)
		g_startTick = GetTickCount();
}

// ---------------------------------------------------------------------------
// hit-testing
//
static int HitKey(int x, int y)
{
	if (y < KOY || y >= KOY + KROWS * (KH + KGAP))
		return -1;
	if (x < KOX || x >= KOX + KCOLS * (KW + KGAP))
		return -1;
	return ((y - KOY) / (KH + KGAP)) * KCOLS + (x - KOX) / (KW + KGAP);
}

static int HitBtn(int x, int y)
{
	int i, bx;
	if (y < BTN_Y || y >= BTN_Y + BTN_H)
		return -1;
	bx = KOX;
	for (i = 0; i < 4; i++)
	{
		if (x >= bx && x < bx + BTN_W)
			return i;
		bx += BTN_W + BTN_GAP;
	}
	return -1;
}

// D-pad: move focus. Row 0..3 = keyboard (KCOLS cols); row 4 = buttons (4 cols).
// Clamp at edges; lateral movement on a key row stops at the row's ends.
static void SelMove(int dRow, int dCol)
{
	int nr = s_selRow + dRow;
	int nc = s_selCol + dCol;
	if (nr < 0) nr = 0;
	if (nr > 4) nr = 4;
	if (nc < 0) nc = 0;
	if (nr == 4)
	{
		if (nc > 3) nc = 3;
	}
	else
	{
		if (nc > KCOLS - 1) nc = KCOLS - 1;
	}
	s_selRow = nr;
	s_selCol = nc;
}

// ENTER / A-button on the focused cell: type the key or fire the button.
// This is the single activation path for both real keyboard Enter and the
// shell-synthesized VK_RETURN the analog-stick A-button produces (see the
// pointer block in DcwMain). Pointer-click hits the same cell on Enter because
// the pointer-hover syncs the same s_selRow/s_selCol whenever the stick moved.
static void ActivateSel(void)
{
	if (s_selRow == 4)
	{
		switch (s_btn[s_selCol].id)
		{
			case 0: PromptAppend(' ');               break;
			case 1: PromptBksp();                   break;
			case 2: StartGenerate();                break;
			case 3: PromptClear(); OutClear();      break;
			default:                                break;
		}
	}
	else
	{
		WCHAR ch = kbRows[s_selRow][s_selCol];
		PromptAppend((char)(ch & 0x7F));
	}
}

// ---------------------------------------------------------------------------
// output line wrap + render
//
static int NextLineBreak(int from, int maxw)
{
	int i = from, lastSpace = -1;
	while (i < g_outLen && (i - from) < maxw)
	{
		if (g_out[i] == '\n')
			return i + 1;
		if (g_out[i] == ' ')
			lastSpace = i;
		i++;
	}
	if (i >= g_outLen)
		return g_outLen;
	if (lastSpace > from)
		return lastSpace + 1;
	return i;
}

static void RenderOutput(DCWin *w, int x, int y, int rowsAvail)
{
	int starts[256];
	int lineCount = 0, from = 0;
	int firstLine, i, k;
	WCHAR wbuf[40];

	while (from < g_outLen && lineCount < 256)
	{
		starts[lineCount++] = from;
		from = NextLineBreak(from, 38);
	}
	firstLine = lineCount > rowsAvail ? lineCount - rowsAvail : 0;
	for (i = firstLine; i < lineCount; i++)
	{
		int s = starts[i];
		int e = (i + 1 < lineCount) ? starts[i+1] : g_outLen;
		int len = e - s;
		if (len > 0 && g_out[s + len - 1] == '\n')
			len--;
		if (len > 38)
			len = 38;
		for (k = 0; k < len; k++)
			wbuf[k] = (WCHAR)(unsigned char)g_out[s + k];
		wbuf[len] = 0;
		DCWinText(w, x, y + (i - firstLine) * ROW_OUT,
		          RGB(0, 255, 80), RGB(0, 0, 0), wbuf);
	}
}

static void RenderPromptField(DCWin *w)
{
	WCHAR wbuf[80];
	int i, n = g_nPrompt;
	DCWinFill(w, PROMPT_X, PROMPT_Y, PROMPT_W, PROMPT_H, RGB(0, 0, 64));
	if (n > 76)
		n = 76;
	for (i = 0; i < n; i++)
		wbuf[i] = (WCHAR)(unsigned char)g_prompt[i];
	wbuf[n]   = L'|';
	wbuf[n+1] = 0;
	DCWinText(w, PROMPT_X + 4, PROMPT_Y + 5,
	          RGB(80, 255, 255), RGB(0, 0, 64), wbuf);
}

static void RenderKeyboard(DCWin *w)
{
	int r, c, kx, ky;
	WCHAR buf[2];
	int hot, sel;
	COLORREF bg, fg;

	for (r = 0; r < KROWS; r++)
	{
		for (c = 0; c < KCOLS; c++)
		{
			kx = KOX + c * (KW + KGAP);
			ky = KOY + r * (KH + KGAP);
			hot = (s_hotR == r && s_hotC == c);
			sel = (s_selRow == r && s_selCol == c);
			if (sel)      bg = RGB(0, 160, 0);   // D-pad focus (green)
			else if (hot) bg = RGB(0, 80, 160);  // pointer hover (blue)
			else          bg = RGB(0, 40, 80);
			fg = (r == 3) ? RGB(255, 200, 0) : RGB(255, 255, 255);
			DCWinFill(w, kx, ky, KW, KH, bg);
			buf[0] = kbRows[r][c];
			buf[1] = 0;
			DCWinText(w, kx + 10, ky + 6, fg, bg, buf);
		}
	}
}

static void RenderButtons(DCWin *w)
{
	int i, kx, hot, sel;
	COLORREF bg, fg;
	WCHAR buf[12];
	int n;
	DWORD cur;
	DWORD elapsed;

	for (i = 0; i < 4; i++)
	{
		kx = KOX + i * (BTN_W + BTN_GAP);
		hot = (s_hotBtn == i);
		sel = (s_selRow == 4 && s_selCol == i);
		if (sel)      bg = RGB(0, 160, 0);   // D-pad focus (green)
		else if (hot) bg = RGB(80, 0, 120);  // pointer hover (purple)
		else          bg = RGB(40, 0, 80);
		if (s_btn[i].id == 2 && g_state == ST_GEN)
			bg = RGB(120, 0, 0);            // override: Go shows busy-red
		fg = RGB(255, 255, 255);
		DCWinFill(w, kx, BTN_Y, BTN_W, BTN_H, bg);
		for (n = 0; n < 12 && s_btn[i].label[n]; n++)
			buf[n] = s_btn[i].label[n];
		buf[n] = 0;
		DCWinText(w, kx + 24, BTN_Y + 7, fg, bg, buf);
	}

	// status line: state, pos/steps, last tps, current iteration tok/s
	cur = GetTickCount();
	if (g_state == ST_GEN && g_startTick && g_pos > 1)
	{
		elapsed = cur - g_startTick;
		if (elapsed > 0)
			g_lastTps = (float)(g_pos - 1) * 1000.0f / (float)elapsed;
	}
	{
		const WCHAR *st = (g_state == ST_GEN) ? L"GEN" :
		                  (g_state == ST_DONE) ? L"done" : L"idle";
		WCHAR s[40];
		_snwprintf(s, 40, L"%s  %d/%d tok  %.2f t/s  %s",
		           st, g_pos, g_steps, g_lastTps,
		           g_haveModel ? L"model ready" : L"NO MODEL");
		DCWinText(w, 4, STATUS_Y, RGB(160, 160, 160), RGB(0, 0, 0), s);
	}
}

static void RenderModelHint(DCWin *w)
{
	WCHAR s[60];
	if (g_haveModel)
		return;
	_snwprintf(s, 60, L"no \\CD-ROM\\stories260K.bin / tok512.bin");
	DCWinText(w, 4, OUT_Y, RGB(255, 80, 80), RGB(0, 0, 0), s);
}

// ---------------------------------------------------------------------------
// entry
//
int DcwMain(HINSTANCE hInst, LPWSTR lpCmd)
{
	DCWin *w;
	DWORD dwKey;
	int nCw = LW, nCh = LH;
	int prevBtn = 0, px, py, pb;

	(void)hInst; (void)lpCmd;

	w = DCWinOpen(40, 40, LW, LH, L"LLM (stories260K)", ICON_APP);
	if (!w)
	{
		OutputDebugStringW(L"DCWLLM: DCWinOpen failed\r\n");
		return 1;
	}

	SysLog(L"dcwllm: launching, loading model...");
	// Publish an opaque background frame BEFORE the (multi-second on real
	// GD-ROM) model load: the shell starts compositing our window the
	// instant inUse=1, and until our first DCWinEndFrame the window has
	// cmdCount=0 (a transparent client area). On hardware that lets the
	// desktop icons show through during the model load (the depth-sort fix
	// in dcgfx keeps them under us after the first publish, but only AFTER
	// we publish). This frame closes that first-seconds gap on real iron.
	{
		WCHAR szBoot[40];
		DCWinBeginFrame(w);
		DCWinFillBg(w, RGB(0, 0, 0));
		_snwprintf(szBoot, 40, L"loading model...");
		DCWinText(w, 4, 4, RGB(0, 255, 80), RGB(0, 0, 0), szBoot);
		DCWinEndFrame(w);
	}
	build_transformer(&g_tx, L"\\CD-ROM\\stories260K.bin");
	build_tokenizer(&g_tk, L"\\CD-ROM\\tok512.bin", g_tx.config.vocab_size);
	build_sampler(&g_sm, g_tx.config.vocab_size, 0.0f, 0.9f, 0xDEADBEEFCAFEBABEull);
	if (g_tx.data && g_tk.vocab)
	{
		g_haveModel = 1;
		SysLog(L"dcwllm: model ready (dim=%d L=%d heads=%d/%d seq=%d)",
		       g_tx.config.dim, g_tx.config.n_layers,
		       g_tx.config.n_heads, g_tx.config.n_kv_heads, g_tx.config.seq_len);
	}
	else
	{
		SysLog(L"dcwllm: model load failed - app will idle");
	}

	for (;;)
	{
		int bDirty = 1;             // we always re-render for status line tick

		DCWinClientSize(w, &nCw, &nCh);

		while (DCWinPollKey(w, &dwKey))
		{
			if (dwKey == VK_BACK)
			{
				PromptBksp();
			}
			else if (dwKey == VK_RETURN)
			{
				// ENTER activates the D-pad-focused cell. This fires for both
				// real keyboard Enter AND the shell-synthesized VK_RETURN the
				// shell injects on every A-button/body pointer-click (the shell
				// forwards Enter to the focused window). That makes the DC A
				// button and the keyboard Enter key do the same thing here:
				// "activate whatever the D-pad focus is on", NOT "type the cell
				// under the analog cursor". The D-pad focus tracks pointer-hover
				// only when the pointer is actively moving (see DCWinGetPointer
				// below) so a D-pad-only user's selection stays put.
				ActivateSel();
			}
			else if (dwKey == VK_UP)         SelMove(-1, 0);
			else if (dwKey == VK_DOWN)       SelMove(+1, 0);
			else if (dwKey == VK_LEFT)       SelMove(0, -1);
			else if (dwKey == VK_RIGHT)      SelMove(0, +1);
			else if (dwKey >= 0x20 && dwKey < 0x80)
			{
				// ASCII chars from a real keyboard go straight into the prompt;
				// VK_SPACE (=0x20) lands here too (typed space).
				PromptAppend((char)dwKey);
			}
		}

		if (DCWinGetPointer(w, &px, &py, &pb))
		{
			int kid = HitKey(px, py);
			int bid = HitBtn(px, py);
			s_hotR = (kid >= 0) ? kid / KCOLS : -1;
			s_hotC = (kid >= 0) ? kid % KCOLS : -1;
			s_hotBtn = bid;
			// Sync D-pad focus to the hovered cell ONLY when the pointer
			// actually moved this frame (vs. every frame). Without this gate
			// the next D-pad move gets overwritten the very next frame by the
			// stale pointer-hover position, so D-pad-only users can't navigate.
			// When the user IS actively moving the analog stick, focus tracks
			// the cursor (mouse-pointer parity); when they're not, D-pad stays
			// sticky on its own choice.
			if (px != s_lastPx || py != s_lastPy)
			{
				s_lastPx = px;
				s_lastPy = py;
				if (kid >= 0)      { s_selRow = kid / KCOLS; s_selCol = kid % KCOLS; }
				else if (bid >= 0) { s_selRow = 4;           s_selCol = bid; }
			}
			// Don't call a local HandleClick on the pointer click edge. The
			// shell synthesizes VK_RETURN on every body pointer-click anyway,
			// which is consumed by ActivateSel above. Routing both A-button and
			// Enter through the same ActivateSel path is what makes A "select
			// the D-pad focus" instead of "type what's under the analog stick".
			prevBtn = pb;
		}
		else
		{
			s_hotR = s_hotC = -1;
			s_hotBtn = -1;
			prevBtn = 0;
		}

		// drive one forward per frame while generating
		if (g_state == ST_GEN)
			StepForward();

		if (bDirty)
		{
			int rowsAvail = (nCh - OUT_Y - 2) / ROW_OUT;
			if (rowsAvail < 1)
				rowsAvail = 1;
			if (rowsAvail > 240)
				rowsAvail = 240;
			DCWinBeginFrame(w);
			DCWinFillBg(w, RGB(0, 0, 0));
			RenderPromptField(w);
			RenderKeyboard(w);
			RenderButtons(w);
			if (g_haveModel)
				RenderOutput(w, 4, OUT_Y, rowsAvail);
			else
				RenderModelHint(w);
			DCWinEndFrame(w);
		}

		if (DCWinShouldClose(w))
			break;
		Sleep(20);
	}

	if (g_haveModel)
	{
		free_sampler(&g_sm);
		free_tokenizer(&g_tk);
		free_transformer(&g_tx);
	}
	DCWinClose(w);
	return 0;
}