//
// ieproto.cpp - a winsock-backed Asynchronous Pluggable Protocol handler for "http:".
//
// CE Trident's WinInet doesn't fetch on the Dreamcast, and every way of injecting HTML into an
// already-loaded document fails (IPersistStreamInit::Load -> E_UNEXPECTED, IHTMLDocument2::write
// -> fault) and the filesystem is read-only (no file: trick). What DOES work is mshtml's normal
// bind/load pipeline (it renders about:blank and any navigation). So we register an
// IInternetProtocol that OVERRIDES the "http" namespace for this process: the control navigates
// http:// as usual, URLMON routes the bind to us, and we serve bytes fetched over winsock (the
// stack dcwnet proved works). Page, images, CSS and links all flow through here.
//
#include <windows.h>
#include <winsock.h>
#include <urlmon.h> // IInternetProtocol / IInternetSession (C++ interfaces - no CINTERFACE)
#include <string.h>
#include <stdarg.h>
#include "syslog.h" // anchor-hang diagnostics: every protocol callback lands in the System Log

// PLog: one line to BOTH diagnostics channels - the SysLog ring (dcwlog app) and
// OutputDebugString (SCIF console on the debug kernel; visible in Flycast's serial dump).
static void PLog(const WCHAR *pszFmt, ...)
{
	WCHAR awch[240];
	va_list ap;
	va_start(ap, pszFmt);
	wvsprintfW(awch, pszFmt, ap);
	va_end(ap);
	SysLogW(awch);
	OutputDebugStringW(awch);
	OutputDebugStringW(L"\r\n");
}

// {DC5A0001-7700-4d00-9000-000000000001} - our private handler CLSID
static const CLSID CLSID_DcwHttp = {
    0xdc5a0001, 0x7700, 0x4d00, {0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01}};

// ---- winsock HTTP (self-contained; mirrors dcwnet) -------------------------------
#define HTTP_MAXRESP (512 * 1024)

static int AppendA(char *pszDst, const char *pszSrc)
{
	int i = 0;
	while (pszSrc[i])
	{
		pszDst[i] = pszSrc[i];
		i++;
	}
	return i;
}
static const char *StrStrA2(const char *pszHay, const char *pszNdl)
{
	int i, j;
	for (i = 0; pszHay[i]; i++)
	{
		for (j = 0; pszNdl[j] && pszHay[i + j] == pszNdl[j]; j++)
			;
		if (!pszNdl[j])
			return pszHay + i;
	}
	return 0;
}

static char *HttpFetch(const char *pszHost, int nPort, const char *pszPath, int *pcbOut)
{
	SOCKET s = INVALID_SOCKET;
	struct hostent *phe;
	SOCKADDR_IN sa;
	char *pBuf = NULL;
	int cb = 0, n, cbReq;
	char szReq[700];
	__try
	{
		phe = gethostbyname(pszHost);
		if (!phe || !phe->h_addr_list[0])
			return NULL;
		s = socket(AF_INET, SOCK_STREAM, 0);
		if (s == INVALID_SOCKET)
			return NULL;
		memset(&sa, 0, sizeof(sa));
		sa.sin_family = AF_INET;
		sa.sin_port = htons((u_short)nPort);
		memcpy(&sa.sin_addr, phe->h_addr_list[0], 4);
		if (connect(s, (SOCKADDR *)&sa, sizeof(sa)) != 0)
		{
			closesocket(s);
			return NULL;
		}
		cbReq = AppendA(szReq, "GET ");
		cbReq += AppendA(szReq + cbReq, pszPath);
		cbReq += AppendA(szReq + cbReq, " HTTP/1.0\r\nHost: ");
		cbReq += AppendA(szReq + cbReq, pszHost);
		cbReq +=
		    AppendA(szReq + cbReq, "\r\nUser-Agent: Mozilla/4.0 (compatible; DCWin)\r\nAccept: "
		                           "*/*\r\nConnection: close\r\n\r\n");
		send(s, szReq, cbReq, 0);
		pBuf = (char *)LocalAlloc(LMEM_FIXED, HTTP_MAXRESP);
		if (!pBuf)
		{
			closesocket(s);
			return NULL;
		}
		for (;;)
		{
			if (cb >= HTTP_MAXRESP - 4096)
				break;
			n = recv(s, pBuf + cb, 4096, 0);
			if (n <= 0)
				break;
			cb += n;
		}
		closesocket(s);
		pBuf[cb] = 0;
		*pcbOut = cb;
		return pBuf;
	}
	__except (EXCEPTION_EXECUTE_HANDLER)
	{
		if (s != INVALID_SOCKET)
			closesocket(s);
		if (pBuf)
			LocalFree(pBuf);
		return NULL;
	}
}

// Parse an http URL, GET it, follow one-scheme (http) redirects. Returns the LocalAlloc'd
// response (caller LocalFree) with the body offset/length filled in, or NULL.
static char *FetchUrlA(const char *pszUrlIn, int *pBodyOff, int *pBodyLen)
{
	char szUrl[600];
	int iHop, i = 0;
	const char *pszLocalPath = NULL;
	int bHostPath = 0;
	if (!strncmp(pszUrlIn, "dcw://cdrom/", 12) || !strncmp(pszUrlIn, "dcw://local/", 12))
	{
		pszLocalPath = pszUrlIn + 12;
		bHostPath = 1;
	}
	else if (!strncmp(pszUrlIn, "dcw://\\", 7) || !strncmp(pszUrlIn, "dcw:///", 7))
		pszLocalPath = pszUrlIn + 6;
	if (pszLocalPath)
	{
		char szPathA[600];
		WCHAR szPathW[600];
		HANDLE hFile;
		DWORD dwSize, dwRead = 0;
		char *pFile;
		int nPath = 0;
		const char *pszReadPath = pszLocalPath;

		if (bHostPath)
		{
			szPathA[nPath++] = '\\';
			szPathA[nPath++] = 'C';
			szPathA[nPath++] = 'D';
			szPathA[nPath++] = '-';
			szPathA[nPath++] = 'R';
			szPathA[nPath++] = 'O';
			szPathA[nPath++] = 'M';
			szPathA[nPath++] = '\\';
		}
		while (*pszReadPath && nPath < (int)sizeof(szPathA) - 1)
		{
			char ch = *pszReadPath++;
			if (ch == '?' || ch == '#')
				break;
			if (ch == '/')
				ch = '\\';
			szPathA[nPath++] = ch;
		}
		szPathA[nPath] = 0;
		if (!MultiByteToWideChar(CP_ACP, 0, szPathA, -1, szPathW,
		                         sizeof(szPathW) / sizeof(szPathW[0])) ||
			szPathW[0] != L'\\')
			return NULL;
		hFile = CreateFileW(szPathW, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
		                     FILE_ATTRIBUTE_NORMAL, NULL);
		if (hFile == INVALID_HANDLE_VALUE)
			return NULL;
		dwSize = GetFileSize(hFile, NULL);
		if (dwSize == INVALID_FILE_SIZE || dwSize >= HTTP_MAXRESP)
		{
			CloseHandle(hFile);
			return NULL;
		}
		pFile = (char *)LocalAlloc(LMEM_FIXED, dwSize + 1);
		if (!pFile || (dwSize && (!ReadFile(hFile, pFile, dwSize, &dwRead, NULL) || dwRead != dwSize)))
		{
			if (pFile)
				LocalFree(pFile);
			CloseHandle(hFile);
			return NULL;
		}
		CloseHandle(hFile);
		pFile[dwSize] = 0;
		*pBodyOff = 0;
		*pBodyLen = (int)dwSize;
		return pFile;
	}
	while (pszUrlIn[i] && i < 599)
	{
		szUrl[i] = pszUrlIn[i];
		i++;
	}
	szUrl[i] = 0;

	for (iHop = 0; iHop < 4; iHop++)
	{
		const char *pszU = szUrl, *pszH;
		char szHost[256], szPath[512];
		int nPort = 80, k;
		if (pszU[0] == 'h' && pszU[1] == 't' && pszU[2] == 't' && pszU[3] == 'p' && pszU[4] == 's')
			return NULL; // no TLS
		if (pszU[0] == 'd' && pszU[1] == 'c' && pszU[2] == 'w' && pszU[3] == ':' &&
		    pszU[4] == '/' && pszU[5] == '/')
			pszU += 6; // our scheme
		else if (pszU[0] == 'h' && pszU[1] == 't' && pszU[2] == 't' && pszU[3] == 'p' &&
		         pszU[4] == ':' && pszU[5] == '/' && pszU[6] == '/')
			pszU += 7;
		for (k = 0; pszU[k] && pszU[k] != '/' && pszU[k] != ':' && k < 255; k++)
			szHost[k] = pszU[k];
		szHost[k] = 0;
		pszH = pszU + k;
		if (*pszH == ':')
		{
			nPort = 0;
			pszH++;
			while (*pszH >= '0' && *pszH <= '9')
				nPort = nPort * 10 + (*pszH++ - '0');
		}
		if (*pszH != '/')
		{
			szPath[0] = '/';
			szPath[1] = 0;
		}
		else
		{
			k = 0;
			while (pszH[k] && k < 510)
			{
				szPath[k] = pszH[k];
				k++;
			}
			szPath[k] = 0;
		}

		{
			int cb;
			char *pResp = HttpFetch(szHost, nPort, szPath, &cb);
			if (!pResp)
				return NULL;
			if (pResp[9] == '3') // redirect
			{
				const char *pszLoc = StrStrA2(pResp, "\nLocation:");
				if (!pszLoc)
					pszLoc = StrStrA2(pResp, "\nlocation:");
				if (pszLoc)
				{
					int j = 0;
					pszLoc += 10;
					while (*pszLoc == ' ')
						pszLoc++;
					while (*pszLoc && *pszLoc != '\r' && *pszLoc != '\n' && j < 599)
						szUrl[j++] = *pszLoc++;
					szUrl[j] = 0;
					LocalFree(pResp);
					continue;
				}
			}
			{
				const char *pszBody = StrStrA2(pResp, "\r\n\r\n");
				*pBodyOff = pszBody ? (int)(pszBody + 4 - pResp) : 0;
				*pBodyLen = cb - *pBodyOff;
			}
			return pResp;
		}
	}
	return NULL;
}

static int StartsWithW(LPCWSTR pszText, LPCWSTR pszPrefix)
{
	while (*pszPrefix)
	{
		if (*pszText++ != *pszPrefix++)
			return 0;
	}
	return 1;
}

static HRESULT CopyUrlResult(LPCWSTR pszText, LPWSTR pszResult, DWORD cchResult, DWORD *pcchResult)
{
	DWORD cch = (DWORD)lstrlenW(pszText) + 1;
	if (pcchResult)
		*pcchResult = cch;
	if (!pszResult || cchResult < cch)
		return S_FALSE;
	lstrcpyW(pszResult, pszText);
	return S_OK;
}

static void CopyWideN(LPWSTR pszDst, LPCWSTR pszSrc, int cchDst)
{
	int i;
	if (cchDst <= 0)
		return;
	for (i = 0; i < cchDst - 1 && pszSrc[i]; i++)
		pszDst[i] = pszSrc[i];
	pszDst[i] = 0;
}

static int UrlPathEnd(LPCWSTR pszUrl)
{
	int i;
	for (i = 0; pszUrl[i]; i++)
	{
		if (pszUrl[i] == L'?' || pszUrl[i] == L'#')
			return i;
	}
	return i;
}

static int IsLocalDcwUrl(LPCWSTR pszUrl)
{
	return StartsWithW(pszUrl, L"dcw://cdrom/") || StartsWithW(pszUrl, L"dcw://local/") ||
	       StartsWithW(pszUrl, L"dcw://\\") || StartsWithW(pszUrl, L"dcw:///");
}

// bounded append (lstrcatW has no cap; CombineUrl's stack buffer must not overflow)
static void CatWideN(LPWSTR pszDst, LPCWSTR pszSrc, int cchDstCap)
{
	int n = lstrlenW(pszDst), i;
	for (i = 0; pszSrc[i] && n + i < cchDstCap - 1; i++)
		pszDst[n + i] = pszSrc[i];
	pszDst[n + i] = 0;
}

// ---- the per-request protocol object ---------------------------------------------
class CDcwHttp : public IInternetProtocol, public IInternetProtocolInfo
{
	ULONG m_cRefs;
	char *m_pResp;
	int m_nBodyOff, m_nBodyLen, m_nPos;
	IInternetProtocolSink *m_pSink;
	int m_bResultDone;

  public:
	CDcwHttp()
	{
		m_cRefs = 1;
		m_pResp = NULL;
		m_nBodyOff = m_nBodyLen = m_nPos = 0;
		m_pSink = NULL;
		m_bResultDone = 0;
	}
	~CDcwHttp()
	{
		if (m_pResp)
			LocalFree(m_pResp);
		if (m_pSink)
			m_pSink->Release();
	}

	STDMETHOD(QueryInterface)(REFIID riid, void **ppv)
	{
		if (riid == IID_IUnknown || riid == IID_IInternetProtocolRoot ||
		    riid == IID_IInternetProtocol)
		{
			*ppv = (IInternetProtocol *)this;
			AddRef();
			return S_OK;
		}
		if (riid == IID_IInternetProtocolInfo)
		{
			PLog(L"proto: QI Info");
			*ppv = (IInternetProtocolInfo *)this;
			AddRef();
			return S_OK;
		}
		*ppv = NULL;
		return E_NOINTERFACE;
	}
	STDMETHOD_(ULONG, AddRef)(void)
	{
		return ++m_cRefs;
	}
	STDMETHOD_(ULONG, Release)(void)
	{
		if (--m_cRefs == 0)
		{
			delete this;
			return 0;
		}
		return m_cRefs;
	}

	// IInternetProtocolRoot
	STDMETHOD(Start)(LPCWSTR szUrl, IInternetProtocolSink *pSink, IInternetBindInfo *pBI,
	                 DWORD grfPI, DWORD dwReserved)
	{
		char szUrlA[600];
		(void)pBI;
		(void)grfPI;
		(void)dwReserved;
		WideCharToMultiByte(CP_ACP, 0, szUrl, -1, szUrlA, sizeof(szUrlA), NULL, NULL);
		OutputDebugStringW(L"proto: Start ");
		OutputDebugStringW(szUrl);
		OutputDebugStringW(L"\r\n");
		PLog(L"P Start %s", szUrl ? szUrl : L"");

		m_pResp = FetchUrlA(szUrlA, &m_nBodyOff, &m_nBodyLen);
		m_nPos = 0;
		if (!m_pResp)
		{
			PLog(L"P fetch FAIL");
			OutputDebugStringW(L"proto: fetch FAILED\r\n");
			pSink->ReportResult(INET_E_DOWNLOAD_FAILURE, 0, NULL);
			return INET_E_DOWNLOAD_FAILURE;
		}
		{
			WCHAR b[64];
			wsprintfW(b, L"proto: %d bytes\r\n", m_nBodyLen);
			OutputDebugStringW(b);
		}
		PLog(L"P bytes=%d", m_nBodyLen);
		m_pSink = pSink;
		m_pSink->AddRef(); // ReportResult is deferred until Read drains
		{
			int nUrl = lstrlenW(szUrl);
			LPCWSTR pszMime = L"text/html";
			if (nUrl >= 4 && szUrl[nUrl - 4] == L'.' && szUrl[nUrl - 3] == L'g' &&
				szUrl[nUrl - 2] == L'i' && szUrl[nUrl - 1] == L'f')
				pszMime = L"image/gif";
			pSink->ReportProgress(BINDSTATUS_MIMETYPEAVAILABLE, pszMime);
		}
		// FULLYAVAILABLE: all data is ready; mshtml will Read until S_FALSE. We must NOT call
		// ReportResult yet - doing it before Read finishes makes mshtml re-bind + hang.
		pSink->ReportData(BSCF_FIRSTDATANOTIFICATION | BSCF_LASTDATANOTIFICATION |
		                      BSCF_DATAFULLYAVAILABLE,
		                  m_nBodyLen, m_nBodyLen);
		PLog(L"P reported");
		return S_OK;
	}
	STDMETHOD(ParseUrl)(LPCWSTR pwzUrl, PARSEACTION ParseAction, DWORD dwParseFlags,
	                    LPWSTR pwzResult, DWORD cchResult, DWORD *pcchResult, DWORD dwReserved)
	{
		WCHAR szPart[1024];
		int n, i;
		(void)dwParseFlags;
		(void)dwReserved;
		PLog(L"P PU a=%d c=%d %s", (int)ParseAction, (int)cchResult, pwzUrl ? pwzUrl : L"");
		if (ParseAction == PARSE_SCHEMA)
			return CopyUrlResult(L"dcw", pwzResult, cchResult, pcchResult);
		if (ParseAction == PARSE_SERVER)
			return CopyUrlResult(L"cdrom", pwzResult, cchResult, pcchResult);
		if (ParseAction == PARSE_ANCHOR)
		{
			for (i = 0; pwzUrl[i] && pwzUrl[i] != L'#'; i++)
				;
			return CopyUrlResult(pwzUrl[i] ? pwzUrl + i : L"", pwzResult, cchResult, pcchResult);
		}
		if (ParseAction == PARSE_DOCUMENT)
		{
			n = UrlPathEnd(pwzUrl);
			if (n >= (int)(sizeof(szPart) / sizeof(szPart[0])))
				return E_FAIL;
			CopyWideN(szPart, pwzUrl, n + 1);
			return CopyUrlResult(szPart, pwzResult, cchResult, pcchResult);
		}
		if (ParseAction == PARSE_CANONICALIZE || ParseAction == PARSE_FRIENDLY ||
			ParseAction == PARSE_ENCODE || ParseAction == PARSE_DECODE)
			return CopyUrlResult(pwzUrl, pwzResult, cchResult, pcchResult);
		return INET_E_DEFAULT_ACTION;
	}
	STDMETHOD(CombineUrl)(LPCWSTR pwzBaseUrl, LPCWSTR pwzRelativeUrl, DWORD dwCombineFlags,
	                      LPWSTR pwzResult, DWORD cchResult, DWORD *pcchResult, DWORD dwReserved)
	{
		WCHAR szCombined[1200];
		int nEnd, nDir;
		(void)dwCombineFlags;
		(void)dwReserved;
		PLog(L"P CU b=%s", pwzBaseUrl ? pwzBaseUrl : L"");
		PLog(L"P CU r=%s", pwzRelativeUrl ? pwzRelativeUrl : L"");
		if (StartsWithW(pwzRelativeUrl, L"dcw://"))
			return CopyUrlResult(pwzRelativeUrl, pwzResult, cchResult, pcchResult);
		if (!StartsWithW(pwzBaseUrl, L"dcw://"))
			return INET_E_DEFAULT_ACTION;
		nEnd = UrlPathEnd(pwzBaseUrl);
		if (pwzRelativeUrl[0] == L'#' || pwzRelativeUrl[0] == L'?')
		{
			CopyWideN(szCombined, pwzBaseUrl, nEnd + 1);
			CatWideN(szCombined, pwzRelativeUrl, (int)(sizeof(szCombined) / sizeof(szCombined[0])));
		}
		else if (pwzRelativeUrl[0] == L'/')
		{
			int nHostEnd = 6;
			while (pwzBaseUrl[nHostEnd] && pwzBaseUrl[nHostEnd] != L'/')
				nHostEnd++;
			CopyWideN(szCombined, pwzBaseUrl, nHostEnd + 1);
			CatWideN(szCombined, pwzRelativeUrl, (int)(sizeof(szCombined) / sizeof(szCombined[0])));
		}
		else
		{
			nDir = nEnd;
			while (nDir > 0 && pwzBaseUrl[nDir - 1] != L'/')
				nDir--;
			CopyWideN(szCombined, pwzBaseUrl, nDir + 1);
			CatWideN(szCombined, pwzRelativeUrl, (int)(sizeof(szCombined) / sizeof(szCombined[0])));
		}
		return CopyUrlResult(szCombined, pwzResult, cchResult, pcchResult);
	}
	STDMETHOD(CompareUrl)(LPCWSTR pwzUrl1, LPCWSTR pwzUrl2, DWORD dwCompareFlags)
	{
		(void)dwCompareFlags;
		PLog(L"P Cmp");
		return lstrcmpiW(pwzUrl1, pwzUrl2) == 0 ? S_OK : S_FALSE;
	}
	STDMETHOD(QueryInfo)(LPCWSTR pwzUrl, QUERYOPTION QueryOption, DWORD dwQueryFlags,
	                    LPVOID pBuffer, DWORD cbBuffer, DWORD *pcbBuf, DWORD dwReserved)
	{
		LPCWSTR pszMime = L"text/html";
		DWORD dwValue;
		int nUrl;
		(void)dwQueryFlags;
		(void)dwReserved;
		PLog(L"P QI o=%d cb=%d", (int)QueryOption, (int)cbBuffer);
		nUrl = lstrlenW(pwzUrl);
		if (nUrl >= 4 && pwzUrl[nUrl - 4] == L'.' && pwzUrl[nUrl - 3] == L'g' &&
			pwzUrl[nUrl - 2] == L'i' && pwzUrl[nUrl - 1] == L'f')
			pszMime = L"image/gif";
		if (QueryOption == QUERY_CONTENT_TYPE)
		{
			DWORD cbNeed = ((DWORD)lstrlenW(pszMime) + 1) * sizeof(WCHAR);
			if (pcbBuf)
				*pcbBuf = cbNeed;
			if (!pBuffer || cbBuffer < cbNeed)
				return S_FALSE;
			lstrcpyW((LPWSTR)pBuffer, pszMime);
			return S_OK;
		}
		if (QueryOption == QUERY_CAN_NAVIGATE || QueryOption == QUERY_USES_NETWORK ||
			QueryOption == QUERY_USES_CACHE)
		{
			if (!pBuffer || cbBuffer < sizeof(DWORD))
				return S_FALSE;
			dwValue = (QueryOption == QUERY_CAN_NAVIGATE) ? TRUE
			          : (QueryOption == QUERY_USES_NETWORK) ? !IsLocalDcwUrl(pwzUrl) : FALSE;
			*((DWORD *)pBuffer) = dwValue;
			if (pcbBuf)
				*pcbBuf = sizeof(DWORD);
			return S_OK;
		}
		return INET_E_QUERYOPTION_UNKNOWN;
	}
	STDMETHOD(Continue)(PROTOCOLDATA *)
	{
		PLog(L"P Cont");
		return S_OK;
	}
	STDMETHOD(Abort)(HRESULT hrReason, DWORD dwOptions)
	{
		(void)dwOptions;
		PLog(L"P Abort %x", (unsigned)hrReason);
		if (m_pSink && !m_bResultDone)
		{
			m_bResultDone = 1;
			m_pSink->ReportResult(hrReason ? hrReason : E_ABORT, 0, NULL);
		}
		return S_OK;
	}
	STDMETHOD(Terminate)(DWORD)
	{
		PLog(L"P Term");
		if (m_pSink)
		{
			m_pSink->Release();
			m_pSink = NULL;
		}
		return S_OK;
	}
	STDMETHOD(Suspend)(void)
	{
		return E_NOTIMPL;
	}
	STDMETHOD(Resume)(void)
	{
		return E_NOTIMPL;
	}

	// IInternetProtocol
	STDMETHOD(Read)(void *pv, ULONG cb, ULONG *pcbRead)
	{
		ULONG cbAvail, cbRead;
		PLog(L"P R %d/%d", m_nPos, m_nBodyLen);
		if (!m_pResp)
		{
			if (pcbRead)
				*pcbRead = 0;
			return S_FALSE;
		}
		cbAvail = (ULONG)(m_nBodyLen - m_nPos);
		cbRead = (cb < cbAvail) ? cb : cbAvail;
		if (cbRead)
			memcpy(pv, m_pResp + m_nBodyOff + m_nPos, cbRead);
		m_nPos += (int)cbRead;
		if (pcbRead)
			*pcbRead = cbRead;
		if (m_nPos >= m_nBodyLen) // drained -> NOW signal the bind completed
		{
			PLog(L"P R drained");
			if (m_pSink && !m_bResultDone)
			{
				m_bResultDone = 1;
				m_pSink->ReportResult(S_OK, 0, NULL);
			}
			return S_FALSE;
		}
		return S_OK;
	}
	STDMETHOD(Seek)(LARGE_INTEGER, DWORD, ULARGE_INTEGER *)
	{
		return E_NOTIMPL;
	}
	STDMETHOD(LockRequest)(DWORD)
	{
		PLog(L"P Lock");
		return S_OK;
	}
	STDMETHOD(UnlockRequest)(void)
	{
		PLog(L"P Unlock");
		return S_OK;
	}
};

// ---- the class factory (a process-lifetime singleton) ----------------------------
class CDcwFactory : public IClassFactory
{
  public:
	STDMETHOD(QueryInterface)(REFIID riid, void **ppv)
	{
		if (riid == IID_IUnknown || riid == IID_IClassFactory)
		{
			*ppv = (IClassFactory *)this;
			return S_OK;
		}
		*ppv = NULL;
		return E_NOINTERFACE;
	}
	STDMETHOD_(ULONG, AddRef)(void)
	{
		return 2;
	} // never freed
	STDMETHOD_(ULONG, Release)(void)
	{
		return 1;
	}
	STDMETHOD(CreateInstance)(IUnknown *pUnkOuter, REFIID riid, void **ppv)
	{
		CDcwHttp *pHttp;
		if (pUnkOuter)
			return CLASS_E_NOAGGREGATION;
		SysLog(L"proto: CI %08x", (unsigned)riid.Data1);
		pHttp = new CDcwHttp();
		if (!pHttp)
			return E_OUTOFMEMORY;
		{
			HRESULT hr = pHttp->QueryInterface(riid, ppv);
			pHttp->Release();
			return hr;
		}
	}
	STDMETHOD(LockServer)(BOOL)
	{
		return S_OK;
	}
};

static CDcwFactory g_dcwFactory;

// Register our PRIVATE "dcw" scheme. We fully own it (no built-in fallback), so every bind
// phase mshtml does for a dcw:// URL is routed to us - unlike overriding "http", where the
// document-load phase fell back to (dead) WinInet and hung. iehost rewrites http://->dcw://.
extern "C" void IeRegisterWinsockProtocol(void)
{
	IInternetSession *pSess = NULL;
	if (CoInternetGetSession(0, &pSess, 0) == S_OK && pSess)
	{
		// dcw:// = the main document scheme (fully ours, no WinInet fallback). ALSO claim http:// and
		// https:// so the page's SUB-RESOURCE binds - images/CSS and especially the favicon Trident
		// auto-fetches after every page - reach us too, instead of falling into the image's dead
		// WinInet and hanging forever (the document then never reaches DocumentComplete, so the chrome
		// stays stuck on "opening page..."). http:// we fetch over winsock; https:// FetchUrlA refuses
		// (no TLS) -> a fast INET_E failure, so the resource is skipped and the page still completes.
		// (Unlike the earlier failed experiment, the MAIN document is still navigated as dcw://, so
		// this only intercepts resource binds - it doesn't route the document load through http.)
		HRESULT hrD = pSess->RegisterNameSpace(&g_dcwFactory, CLSID_DcwHttp, L"dcw", 0, NULL, 0);
		HRESULT hrH = pSess->RegisterNameSpace(&g_dcwFactory, CLSID_DcwHttp, L"http", 0, NULL, 0);
		HRESULT hrS = pSess->RegisterNameSpace(&g_dcwFactory, CLSID_DcwHttp, L"https", 0, NULL, 0);
		pSess->Release();
		{
			WCHAR b[96];
			wsprintfW(b, L"IE: register handlers dcw=%08x http=%08x https=%08x\r\n", (unsigned)hrD,
			          (unsigned)hrH, (unsigned)hrS);
			OutputDebugStringW(b);
		}
		PLog(L"IE: reg dcw=%x http=%x https=%x", (unsigned)hrD, (unsigned)hrH, (unsigned)hrS);
	}
	else
		OutputDebugStringW(L"IE: CoInternetGetSession FAILED\r\n");
}
