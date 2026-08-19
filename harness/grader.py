"""
Deterministic goal-satisfaction grader for the task-accuracy anchor.

Why not exact match: the model is asked "The capital of France is" and returns
" Paris. The capital of Italy is Rome". The user got exactly what they asked
for. Letter-for-letter comparison scores that 0, which makes the anchor read
0.0 for every quantization scheme and therefore detect nothing.

Why not an LLM judge: this grader sits inside the measurement chain of a
DEGRADATION study. What matters is not that it is absolutely right but that it
is *identically* right across FP16 / FP8 / AWQ. A stable bias cancels in the
delta; variance does not. Every rule here is a pure function -- same input,
same verdict, every run, on every machine.

Rule order (first decisive rule wins, and the winning rule is recorded so any
score can be audited):

  1. SCOPE      -- judge the first answer segment, not the whole completion.
                   A direct answer lives where the user reads it. This is what
                   stops " Lyon. ... France's is Paris" from scoring correct.
  2. NUMERIC    -- if the reference answer is a number, compare numerically
                   with tolerance, so "4" != "14" and 3.14 == 3.140.
  3. TEXT       -- word-boundary phrase match of the answer or any alias, so
                   "Paris" hits in "Paris." but not inside "Parisian".
  4. NEGATION   -- a match that is negated ("not Paris") is not an answer.

Bump GRADER_VERSION on any scoring-behavior change; it is stamped into every
result so a score is always traceable to the rules that produced it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

GRADER_VERSION = "1.0.0"

# Sentence/clause terminators that end the first answer segment. A period is
# only a terminator when it is NOT a decimal point -- otherwise "3.140" gets
# truncated to "3" and a correct float scores wrong.
_SEGMENT_END = re.compile(r"[!?\n;]|(?<!\d)\.|\.(?!\d)")

# Negation cues; checked in a bounded window immediately before a match.
_NEGATIONS = (
    "not", "n't", "never", "no", "isn't", "wasn't", "aren't",
    "cannot", "can't", "doesn't", "didn't", "rather than", "instead of",
    "incorrect", "false",
)
_NEG_WINDOW_CHARS = 24

_NUMBER = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d*\.?\d+")


@dataclass
class GradeResult:
    correct: bool
    rule: str            # which rule decided -- auditable
    scope: str           # the text actually judged
    matched: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "rule": self.rule,
            "scope": self.scope,
            "matched": self.matched,
        }


def normalize(s: str) -> str:
    """Casefold, strip accents, collapse whitespace. Punctuation is preserved
    here; matching handles it via word boundaries so numbers keep their
    decimal points."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.casefold().split())


def first_segment(response: str) -> str:
    """The first answer segment. A direct answer appears where the user reads
    it; text after the first terminator is elaboration, not the answer."""
    text = response.strip()
    if not text:
        return ""
    m = _SEGMENT_END.search(text)
    seg = text[: m.start()] if m else text
    return seg.strip() or text


def _as_number(s: str) -> float | None:
    m = _NUMBER.fullmatch(str(s).strip())
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _first_number(text: str) -> tuple[float, str] | None:
    m = _NUMBER.search(text)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "")), m.group(0)
    except ValueError:
        return None


def _numbers_equal(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def _is_negated(scope_norm: str, start: int) -> bool:
    """True if a negation cue sits in the bounded window before the match."""
    window = scope_norm[max(0, start - _NEG_WINDOW_CHARS) : start]
    return any(re.search(rf"(?:^|\W){re.escape(cue)}(?:\W|$)", window) for cue in _NEGATIONS)


def _phrase_search(scope_norm: str, phrase: str) -> re.Match | None:
    """Word-boundary phrase match, whitespace-insensitive between words."""
    p = normalize(phrase)
    if not p:
        return None
    parts = [re.escape(tok) for tok in p.split()]
    # \b fails next to non-word chars; use explicit boundaries that also allow
    # the phrase to sit against punctuation ("Paris." / "(Paris)").
    pattern = r"(?<!\w)" + r"\W+".join(parts) + r"(?!\w)"
    return re.search(pattern, scope_norm)


def grade(
    response: str,
    answer: Any,
    aliases: Iterable[str] = (),
    *,
    whole_response: bool = False,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> GradeResult:
    """Did the response give the user what they asked for?

    whole_response=True widens scope past the first segment. Use only for tasks
    whose answer is legitimately a longer passage; it re-opens the
    "wrong-then-right" false positive that scoping exists to prevent.
    """
    scope = response.strip() if whole_response else first_segment(response)
    scope_norm = normalize(scope)
    if not scope_norm:
        return GradeResult(False, "empty_response", scope)

    # --- NUMERIC ---------------------------------------------------------- #
    ref_num = _as_number(answer)
    if ref_num is not None:
        got = _first_number(scope_norm)
        if got is None:
            return GradeResult(False, "numeric_no_number_in_response", scope)
        value, raw = got
        if not _numbers_equal(value, ref_num, rel_tol, abs_tol):
            return GradeResult(False, "numeric_mismatch", scope, raw)
        idx = scope_norm.find(raw)
        if idx >= 0 and _is_negated(scope_norm, idx):
            return GradeResult(False, "negated", scope, raw)
        return GradeResult(True, "numeric_match", scope, raw)

    # --- TEXT ------------------------------------------------------------- #
    for cand in [answer, *aliases]:
        m = _phrase_search(scope_norm, str(cand))
        if m is None:
            continue
        if _is_negated(scope_norm, m.start()):
            return GradeResult(False, "negated", scope, m.group(0))
        rule = "text_match" if cand == answer else "alias_match"
        return GradeResult(True, rule, scope, m.group(0))

    return GradeResult(False, "no_match", scope)
