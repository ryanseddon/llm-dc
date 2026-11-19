#!/usr/bin/env python3
"""Neutralize the wininet URL cache: every cache-family export is patched to
`mov #0,r0; rts; nop` (bytes 00 e0 0b 00 09 00) so lookups are instant misses.
Fixes the anchor (<A HREF>) hang, where mshtml re-binds through a cache lookup
("Client UrlCache MMF") that never returns on this image.
Usage: patch-wininet.py <in.dll> <out.dll>   (table is offset/byte-verified)
"""
import struct
import sys

IN, OUT = sys.argv[1], sys.argv[2]

TABLE = [
    ("CommitUrlCacheEntryA", 10364, "e62f224f751f641f"),
    ("CommitUrlCacheEntryW", 13616, "e62f862f962fa62f"),
    ("CreateUrlCacheEntryA", 10196, "862f962fa62fb62f"),
    ("CreateUrlCacheEntryW", 13348, "e62f862f962fa62f"),
    ("CreateUrlCacheGroup", 16644, "862f224f4368d87f"),
    ("DeleteUrlCacheEntryA", 12456, "862f224f4368f07f"),
    ("DeleteUrlCacheEntryW", 15336, "e62f862f962fa62f"),
    ("DeleteUrlCacheGroup", 16700, "511f422f0b0001e0"),
    ("FindCloseUrlCache", 16308, "224f4365f07f09d0"),
    ("FindFirstUrlCacheEntryA", 11828, "862f962fa62f224f"),
    ("FindFirstUrlCacheEntryExA", 12148, "862f962fa62fb62f"),
    ("FindFirstUrlCacheEntryExW", 12948, "e62f862f962fa62f"),
    ("FindFirstUrlCacheEntryW", 14836, "e62f862f962fa62f"),
    ("FindNextUrlCacheEntryA", 11996, "862f962f224f651f"),
    ("FindNextUrlCacheEntryExA", 12312, "862f962f224f761f"),
    ("FindNextUrlCacheEntryExW", 13208, "e62f862f962fa62f"),
    ("FindNextUrlCacheEntryW", 15068, "e62f862f962fa62f"),
    ("FreeUrlCacheSpaceA", 16888, "862f962fa62f224f"),
    ("FreeUrlCacheSpaceW", 15640, "e62f862f962fa62f"),
    ("GetUrlCacheConfigInfoA", 27836, "862f962fa62f224f"),
    ("GetUrlCacheConfigInfoW", 15792, "862f962fa62fb62f"),
    ("GetUrlCacheEntryInfoA", 11328, "862f962fa62f224f"),
    ("GetUrlCacheEntryInfoExA", 11504, "862f962fa62fb62f"),
    ("GetUrlCacheEntryInfoExW", 14244, "e62f862f962fa62f"),
    ("GetUrlCacheEntryInfoW", 14472, "e62f862f962fa62f"),
    ("RetrieveUrlCacheEntryFileA", 10768, "862f962fa62f224f"),
    ("RetrieveUrlCacheEntryFileW", 14016, "e62f862f962fa62f"),
    ("SetUrlCacheConfigInfoA", 27760, "862f962f224f"),
    ("SetUrlCacheEntryGroupA", 16708, "862f962fa62fb62f"),
    ("SetUrlCacheEntryGroupW", 15472, "e62f862f962fa62f"),
    ("SetUrlCacheEntryInfoA", 11680, "862f962fa62f224f"),
    ("SetUrlCacheEntryInfoW", 14676, "e62f862f962fa62f"),
    ("UnlockUrlCacheEntryFileA", 10648, "862f224f4368f07f"),
    ("UnlockUrlCacheEntryFileW", 15196, "e62f862f962fa62f"),
]

PATCH = bytes([0x00, 0xE0, 0x0B, 0x00, 0x09, 0x00])  # mov #0,r0; rts; nop

with open(IN, "rb") as f:
    data = bytearray(f.read())

if len(data) != 333918:
    sys.exit("patch-wininet: unexpected wininet.dll size %d (expected 333918)" % len(data))

for name, off, orig in TABLE:
    got = bytes(data[off:off + 6]).hex()
    if got != orig[:12]:
        sys.exit("patch-wininet: %s @%d: bytes %s != expected %s" % (name, off, got, orig))
    data[off:off + len(PATCH)] = PATCH

with open(OUT, "wb") as f:
    f.write(data)
print("patch-wininet: %d cache exports neutralized -> %s" % (len(TABLE), OUT))
