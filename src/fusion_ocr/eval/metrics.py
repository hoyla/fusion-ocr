"""OCR accuracy metrics — pure, dependency-free.

CER/WER are the standard (S+D+I)/N edit rates; we also surface the INSERTION rate on its
own because inserted content — output not supported by the reference — is the closest
automatable proxy for hallucination (the thing a "trustworthy OCR" claim must bound).
"""

from __future__ import annotations

import re
import unicodedata

# CJK / Japanese / Korean characters — scripts written WITHOUT spaces between words. A
# whitespace split therefore yields one giant "word" per line for them, making word_recall /
# WER meaningless (they read ~0 even on a near-perfect transcription). We tokenise each such
# character on its own instead, the standard convention for CJK OCR eval. Ranges: CJK unified
# + ext-A, compatibility ideographs, Hiragana, Katakana (incl. half-width), Hangul.
_CJK = re.compile(r"[぀-ヿｦ-ﾟ㐀-䶿一-鿿豈-﫿가-힯]")


def normalize(text: str) -> str:
    """NFC + collapse runs of whitespace to single spaces, strip ends. Case preserved
    (case is part of OCR fidelity). Applied to both sides before scoring."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def word_tokens(text: str) -> list[str]:
    """Word tokens for the word-level metrics: whitespace-delimited words for spaced scripts,
    but each CJK character as its OWN token (CJK has no word spaces). Pure Latin text is
    unchanged (no CJK chars -> identical to normalize().split()), so this only fixes CJK/JK."""
    out: list[str] = []
    for chunk in normalize(text).split():
        buf = ""
        for ch in chunk:
            if _CJK.match(ch):
                if buf:
                    out.append(buf); buf = ""
                out.append(ch)
            else:
                buf += ch
        if buf:
            out.append(buf)
    return out


def edit_ops(ref: list, hyp: list) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) to turn ``ref`` into ``hyp`` — Levenshtein
    DP with a backtrace. ref/hyp are sequences (characters or words). Deletion = a ref
    token the system dropped; insertion = a token the system added (not in the source)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0, m
    if m == 0:
        return 0, n, 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            if ri == hyp[j - 1]:
                row[j] = prev[j - 1]
            else:
                row[j] = 1 + min(prev[j - 1], prev[j], row[j - 1])
    # backtrace from (n, m) to (0, 0), tallying op types
    i, j, s, d, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1; i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1; i -= 1
        else:
            ins += 1; j -= 1
    return s, d, ins


def _rate(ref_units: list, hyp_units: list) -> dict:
    s, d, ins = edit_ops(ref_units, hyp_units)
    n = max(len(ref_units), 1)
    return {"error_rate": (s + d + ins) / n, "subs": s, "dels": d, "ins": ins, "n": n,
            "insertion_rate": ins / n}


def cer(ref_text: str, hyp_text: str) -> float:
    """Character error rate after normalization."""
    return _rate(list(normalize(ref_text)), list(normalize(hyp_text)))["error_rate"]


def wer(ref_text: str, hyp_text: str) -> float:
    """Word error rate after normalization (CJK-aware tokenisation — see word_tokens)."""
    return _rate(word_tokens(ref_text), word_tokens(hyp_text))["error_rate"]


def insertion_rate(ref_text: str, hyp_text: str) -> float:
    """Inserted characters / reference length — the hallucination proxy."""
    return _rate(list(normalize(ref_text)), list(normalize(hyp_text)))["insertion_rate"]


def _overlap(ref_words: list, hyp_words: list) -> int:
    """Size of the word multiset intersection — order-insensitive."""
    from collections import Counter
    return sum((Counter(ref_words) & Counter(hyp_words)).values())


def score(ref_text: str, hyp_text: str) -> dict:
    """Full breakdown for one (reference, hypothesis) pair, carrying raw counts so a
    corpus can be micro-averaged.

    CER/WER are sequence-based (recognition AND reading order). word_recall / precision
    are the ORDER-INSENSITIVE pair: recall = fraction of reference words recovered
    (recognition completeness), precision = fraction of output words that are real
    (1 - precision is the hallucination proxy). A high recall with a high CER means the
    text was recognised but mis-ordered (multi-column), not misread."""
    rn, hn = normalize(ref_text), normalize(hyp_text)
    ref_w, hyp_w = word_tokens(ref_text), word_tokens(hyp_text)   # CJK-aware (see word_tokens)
    c = _rate(list(rn), list(hn))
    w = _rate(ref_w, hyp_w)
    ov = _overlap(ref_w, hyp_w)
    return {"cer": c["error_rate"], "wer": w["error_rate"],
            "insertion_rate": c["insertion_rate"],
            "ref_chars": c["n"], "char_errors": c["subs"] + c["dels"] + c["ins"],
            "char_ins": c["ins"],
            "ref_words": w["n"], "word_errors": w["subs"] + w["dels"] + w["ins"],
            "word_overlap": ov, "hyp_words": max(len(hyp_w), 1),
            "word_recall": ov / max(len(ref_w), 1),
            "word_precision": ov / max(len(hyp_w), 1)}


def aggregate(results: list[dict]) -> dict:
    """Micro-average a list of per-page score() dicts: total errors / total length."""
    cn = sum(r["ref_chars"] for r in results) or 1
    wn = sum(r["ref_words"] for r in results) or 1
    ov = sum(r["word_overlap"] for r in results)
    hw = sum(r["hyp_words"] for r in results) or 1
    return {
        "pages": len(results),
        "cer": sum(r["char_errors"] for r in results) / cn,
        "wer": sum(r["word_errors"] for r in results) / wn,
        "insertion_rate": sum(r["char_ins"] for r in results) / cn,
        "word_recall": ov / wn,
        "word_precision": ov / hw,
        "ref_chars": cn,
    }
