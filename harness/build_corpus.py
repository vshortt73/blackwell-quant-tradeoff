"""
Build data/heldout_corpus.txt from private, unpublished prose.

PRIVACY CONTRACT: this script never prints file contents. It emits counts,
token statistics, and file names only. The corpus it writes is gitignored --
`perplexity()` stores only aggregates (mean NLL, perplexity, token count), so
no source text reaches results/raw/ either.

Why a builder rather than a hand-assembled file:
  - "held-out" only holds if the text was never published. Public text is in
    the model's training data; memorised text has artificially low perplexity,
    and -- worse for a DEGRADATION study -- may respond to quantization
    differently than novel text, contaminating the delta rather than just the
    absolute value.
  - Markdown structure is not language. Perplexity over `###`, `|---|` and
    bullet markers measures format predictability, not modelling quality. We
    strip markup and keep flowing prose.
  - Model-written docs (handoffs, transcripts) are in-distribution in a way
    human prose is not, so they are excluded by default.
  - Passage length must be roughly uniform, or length becomes a hidden
    variable in the mean.

Usage:
    python harness/build_corpus.py --root /iris-v3/docs --out data/heldout_corpus.txt
    python harness/build_corpus.py --files a.md b.md --out data/heldout_corpus.txt
"""

from __future__ import annotations

import argparse
import hashlib
import re
import statistics
import sys
from pathlib import Path

# Filenames matching these are model-written or third-party; excluded by default.
DEFAULT_EXCLUDE = [
    r"CC_HANDOFF", r"CLAUDE", r"HANDOFF", r"transcript", r"Meeting",
    r"^\d{4}-\d{2}-\d{2}-claude", r"agenda", r"Agenda", r"_ocr",
    r"README", r"CHANGELOG", r"LICENSE",
]

# Passages containing these are dropped wholesale: private prose can carry
# credentials, hosts and personal identifiers, and none of it belongs in a
# corpus that gets POSTed to a server.
SENSITIVE = [
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",       # email
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                           # IPv4
    r"\b(?:sk|hf|ghp|gho|xox[baprs])[-_][A-Za-z0-9]{16,}",    # token shapes
    r"(?i)\b(api[_-]?key|secret|password|passwd|bearer|token)\b\s*[:=]",
    r"\b[0-9a-fA-F]{32,}\b",                                  # long hex
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"\b\d{3}-\d{2}-\d{4}\b",                                 # SSN-shaped
    r"(?i)\b[A-HJ-NPR-Z0-9]{17}\b",                           # VIN-shaped
]
_SENSITIVE_RE = [re.compile(p) for p in SENSITIVE]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def strip_markdown(text: str) -> list[str]:
    """Markup -> flowing prose paragraphs. Structure is dropped, not scored."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)      # fenced code
    text = re.sub(r"~~~.*?~~~", " ", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)      # html comments
    text = re.sub(r"<[^>]+>", " ", text)                     # html tags
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            out.append("")
            continue
        if line.startswith("#"):                 # headers: fragments, not prose
            continue
        if line.startswith("|") or re.fullmatch(r"[|\-: ]+", line):
            continue                             # tables / rules
        if re.fullmatch(r"[-*_=]{3,}", line):
            continue
        line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line)   # list markers
        line = re.sub(r"^\s*>\s?", "", line)                   # blockquote
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", line)      # images
        line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)   # links -> text
        line = re.sub(r"`([^`]*)`", r"\1", line)               # inline code
        line = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", line)  # emphasis
        line = re.sub(r"\s+", " ", line).strip()
        out.append(line)
    paras, buf = [], []
    for line in out:
        if line:
            buf.append(line)
        elif buf:
            paras.append(" ".join(buf)); buf = []
    if buf:
        paras.append(" ".join(buf))
    return paras


def is_sensitive(s: str) -> bool:
    return any(r.search(s) for r in _SENSITIVE_RE)


def prose_ratio(s: str) -> float:
    """Fraction of characters that are letters/spaces. Filters residual
    config-ish or path-heavy lines that survived markup stripping."""
    if not s:
        return 0.0
    good = sum(1 for c in s if c.isalpha() or c.isspace())
    return good / len(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="*", default=[], help="director(ies) to scan (non-recursive by default)")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--files", nargs="*", default=[], help="explicit file list")
    ap.add_argument("--out", default="data/heldout_corpus.txt")
    ap.add_argument("--manifest", default=None, help="default: <out>.manifest")
    ap.add_argument("--tokenizer", default="/mnt/nvme4tb/llm_models/qwen/Qwen3-8B")
    ap.add_argument("--target", type=int, default=384, help="tokens per passage")
    ap.add_argument("--min-tokens", type=int, default=192)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--min-prose-ratio", type=float, default=0.80)
    ap.add_argument("--no-default-excludes", action="store_true")
    args = ap.parse_args()

    paths: list[Path] = [Path(f) for f in args.files]
    for r in args.root:
        root = Path(r)
        it = root.rglob("*") if args.recursive else root.glob("*")
        paths += [p for p in it if p.suffix.lower() in (".md", ".txt") and p.is_file()]

    excludes = [] if args.no_default_excludes else [re.compile(p) for p in DEFAULT_EXCLUDE]
    kept, skipped = [], []
    for p in sorted(set(paths)):
        if any(r.search(p.name) for r in excludes):
            skipped.append((p.name, "excluded (model-written or third-party)")); continue
        kept.append(p)
    if not kept:
        raise SystemExit("no input files after exclusions")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def ntok(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    passages: list[str] = []
    seen: set[str] = set()
    n_sensitive = n_lowprose = n_dup = n_toolong = 0
    used_files: list[tuple[str, int]] = []

    for p in kept:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            skipped.append((p.name, f"unreadable: {type(e).__name__}")); continue
        before = len(passages)
        # Accumulate across paragraphs WITHIN a file, then chunk. Chunking each
        # paragraph independently discards every paragraph shorter than
        # --min-tokens, which in structured technical markdown is nearly all of
        # them. Passages stay within one document, so context remains coherent.
        stream = " ".join(strip_markdown(text))
        for sent_group in _chunk(stream, tok, args.target, args.max_tokens):
                s = sent_group.strip()
                n = ntok(s)
                if n < args.min_tokens:
                    continue
                if n > args.max_tokens:
                    n_toolong += 1; continue
                if is_sensitive(s):
                    n_sensitive += 1; continue
                if prose_ratio(s) < args.min_prose_ratio:
                    n_lowprose += 1; continue
                h = hashlib.sha256(" ".join(s.lower().split()).encode()).hexdigest()
                if h in seen:
                    n_dup += 1; continue
                seen.add(h)
                passages.append(s)
        used_files.append((p.name, len(passages) - before))

    if not passages:
        raise SystemExit(
            "no passages survived filtering -- try --min-tokens lower, or check "
            "that the inputs contain flowing prose rather than tables/config."
        )

    assert not any("\n" in s for s in passages), "passage contains newline"
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(passages) + "\n")

    counts = [ntok(s) for s in passages]
    scored = sum(counts) - len(counts)   # first token of each passage is unscored
    man = Path(args.manifest) if args.manifest else out.with_suffix(out.suffix + ".manifest")
    man.write_text(
        "# Files contributing to this corpus (passage counts).\n"
        + "\n".join(f"{n}\t{name}" for name, n in used_files) + "\n"
    )

    print(f"wrote {len(passages)} passages -> {out}")
    print(f"  manifest -> {man}")
    print(f"\n  files used    : {len(used_files)}   skipped: {len(skipped)}")
    for name, n in sorted(used_files, key=lambda x: -x[1])[:12]:
        print(f"      {n:>4} passages  {name}")
    if skipped:
        print("  skipped:")
        for name, why in skipped[:10]:
            print(f"      {name}  -- {why}")
    print(f"\n  tokens/passage: min={min(counts)} max={max(counts)} "
          f"mean={statistics.mean(counts):.0f} median={statistics.median(counts):.0f}")
    print(f"  total tokens  : {sum(counts):,}   scored: {scored:,}")
    print(f"  dropped       : {n_sensitive} sensitive, {n_lowprose} low-prose, "
          f"{n_dup} duplicate, {n_toolong} over --max-tokens")
    over = sum(1 for c in counts if c > 8192)
    print(f"  over 8192 ctx : {over}  ({'OK' if over == 0 else 'MUST FIX -- these requests will fail'})")
    print(f"  est. runtime  : ~{len(passages) * 0.15:.0f}s per arm (1 request per passage)")
    if scored < 50_000:
        print(f"\n  [WARN] only {scored:,} scored tokens. 50k-200k is the useful range; "
              "below that the perplexity estimate is noisy relative to quantization deltas.")


def _chunk(stream: str, tok, target: int, hard_max: int):
    """Split a prose stream into ~target-token passages on sentence boundaries.

    Sentence boundaries only: cutting mid-sentence would put a token with no
    real left-context at the start of a passage, which inflates perplexity for
    reasons that have nothing to do with the model."""
    sents = [x for x in _SENTENCE_END.split(stream) if x.strip()]
    buf: list[str] = []
    for s in sents:
        buf.append(s)
        n = len(tok(" ".join(buf), add_special_tokens=False)["input_ids"])
        if n >= target:
            joined = " ".join(buf)
            # A single sentence longer than hard_max would exceed the passage
            # budget; emit it alone and let the length filters judge it.
            yield joined
            buf = []
    if buf:
        yield " ".join(buf)


if __name__ == "__main__":
    main()
