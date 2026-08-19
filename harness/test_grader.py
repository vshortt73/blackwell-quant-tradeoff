"""Grader test suite. Run: python harness/test_grader.py

Every case is a scoring decision this study depends on. If a rule changes,
these are the assertions that should force the change to be deliberate.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grader import grade, GRADER_VERSION  # noqa: E402

# (response, answer, aliases, expected_correct, note)
CASES = [
    # --- the real failures that motivated this grader --------------------- #
    (" Paris. The capital of Italy is Rome", "Paris", (), True,
     "correct answer followed by unsolicited elaboration"),
    (" 4, 4 + 2", "4", (), True,
     "correct number followed by continuation"),

    # --- exact / trivial -------------------------------------------------- #
    ("Paris", "Paris", (), True, "bare exact"),
    ("  paris  ", "Paris", (), True, "case + whitespace"),
    ("Paris.", "Paris", (), True, "trailing punctuation"),
    ("(Paris)", "Paris", (), True, "wrapped in parens"),

    # --- false positives a naive `in` check would WRONGLY accept ---------- #
    (" Lyon. The capital of Italy is Rome and France's is Paris", "Paris", (), False,
     "wrong first, right later -- scoping must reject"),
    ("not Paris, it is Lyon", "Paris", (), False, "direct negation"),
    ("The answer is definitely not Paris", "Paris", (), False, "negation, distance"),
    ("Parisian cuisine is famous", "Paris", (), False,
     "substring inside a longer word must not match"),
    ("14", "4", (), False, "4 must not match inside 14"),
    ("The answer isn't 4", "4", (), False, "negated number"),

    # --- numeric semantics ------------------------------------------------ #
    ("3.140", "3.14", (), True, "trailing-zero float equality"),
    ("1,000", "1000", (), True, "thousands separator"),
    ("-5", "-5", (), True, "negative"),
    ("The result is 42 units", "42", (), True, "number embedded in prose"),
    ("41", "42", (), False, "off-by-one"),
    ("no number here", "42", (), False, "numeric answer, no number returned"),

    # --- spelled-out numbers (models answer in words as often as digits) --- #
    ("four", "4", (), True, "bare word number"),
    ("four, but introduces quantization error", "4", (), True,
     "word number followed by elaboration"),
    ("The factor is sixteen", "16", (), True, "word number in prose"),
    ("three", "4", (), False, "wrong word number"),
    ("4", "4", (), True, "digits still work"),
    ("forty", "40", (), True, "tens word"),

    # --- aliases ---------------------------------------------------------- #
    ("NYC is the largest", "New York City", ("NYC",), True, "alias hit"),
    ("New  York   City", "New York City", (), True, "multi-word, odd spacing"),
    ("Rome", "New York City", ("NYC",), False, "neither answer nor alias"),

    # --- degenerate ------------------------------------------------------- #
    ("", "Paris", (), False, "empty response"),
    ("   ", "Paris", (), False, "whitespace-only response"),
]


def main() -> int:
    print(f"grader v{GRADER_VERSION}\n")
    failed = []
    for resp, ans, aliases, want, note in CASES:
        got = grade(resp, ans, aliases)
        ok = got.correct == want
        if not ok:
            failed.append((resp, ans, want, got, note))
        mark = "ok  " if ok else "FAIL"
        print(f"  [{mark}] {note}")
        print(f"         resp={resp!r} ans={ans!r} -> {got.correct} via {got.rule}")

    # Determinism: identical inputs must yield identical verdicts.
    for resp, ans, aliases, _, _ in CASES:
        a, b = grade(resp, ans, aliases), grade(resp, ans, aliases)
        assert a.as_dict() == b.as_dict(), f"non-deterministic on {resp!r}"

    print(f"\n{len(CASES)-len(failed)}/{len(CASES)} passed; determinism check passed")
    for resp, ans, want, got, note in failed:
        print(f"  FAILED: {note}: {resp!r} vs {ans!r} wanted {want}, got {got}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
