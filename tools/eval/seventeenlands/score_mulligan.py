"""Score coach responses against the 17lands mulligan ground truth.

Reads the prompts.jsonl produced by ``build_mulligan_prompts`` (which has
the bucket's correct option in ``meta.bucket_stats.correct``) plus a
responses.jsonl from ``tools.eval.run`` and prints per-backend metrics:

  - **Higher-WR pick rate** — primary metric. % of decisions where the coach
    picked the option with the higher empirical win rate in its bucket.
    51% = essentially coin-flip. 70%+ = real signal.
  - **Agreement with played decision** — secondary. % of decisions where the
    coach matched what the diamond+ player actually did. Slightly noisier
    (humans aren't always right).
  - **Parse rate** — % of responses where we could classify keep/mull. If
    this is below ~95%, fix the prompt or the parser.

Usage:
    python -m tools.eval.seventeenlands.score_mulligan \\
        --prompts tools/eval/data/mulligan_prompts.jsonl \\
        --responses tools/eval/data/mulligan_responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


_KEEP_RE = re.compile(r"\b(keep|kept|keeping)\b", re.IGNORECASE)
_MULL_RE = re.compile(r"\b(mull(igan)?|mulligan(?:ing)?|throw\s+back)\b", re.IGNORECASE)


def parse_decision(response: str) -> str | None:
    """Classify a free-text coach response as 'keep', 'mull', or None.

    The system prompt asks for KEEP/MULLIGAN on the first line; we use that
    when present. Fall back to keyword search across the whole response.
    """
    if not response:
        return None
    first_line = response.strip().splitlines()[0] if response.strip() else ""
    fl_upper = first_line.strip().upper().rstrip(".:")
    if fl_upper in ("KEEP", "KEEP IT", "KEEP THE HAND", "KEEP HAND"):
        return "keep"
    if fl_upper in ("MULLIGAN", "MULL", "MULLIGAN TO 6"):
        return "mull"
    has_keep = bool(_KEEP_RE.search(response))
    has_mull = bool(_MULL_RE.search(response))
    if has_keep and not has_mull:
        return "keep"
    if has_mull and not has_keep:
        return "mull"
    # Both or neither — try the first line again with looser heuristics.
    if "keep" in first_line.lower() and "mull" not in first_line.lower():
        return "keep"
    if ("mull" in first_line.lower()) and "keep" not in first_line.lower():
        return "mull"
    return None


def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def score(prompts_path: Path, responses_path: Path, json_out: Path | None = None) -> None:
    prompts = {p["id"]: p for p in _read_jsonl(prompts_path)}
    responses = list(_read_jsonl(responses_path))

    if not prompts:
        print(f"no prompts in {prompts_path}", file=sys.stderr)
        sys.exit(1)
    if not responses:
        print(f"no responses in {responses_path}", file=sys.stderr)
        sys.exit(1)

    # backend -> stats
    stats: dict[str, dict] = defaultdict(lambda: {
        "n": 0,
        "parsed": 0,
        "unparsed_examples": [],
        "errors": 0,
        "matches_higher_wr": 0,
        "scorable": 0,
        "matches_played": 0,
        "by_decision": {"keep": 0, "mull": 0},
    })

    # Also break down by whether the bucket was a "keep" or "mull" call —
    # useful to spot lopsided coaches (e.g. always-keep biases).
    by_correct: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n": 0, "match": 0}
    )

    for r in responses:
        be = r.get("backend") or "?"
        pid = r.get("prompt_id")
        prompt = prompts.get(pid)
        if not prompt:
            continue
        meta = prompt.get("meta") or {}
        bucket = (meta.get("bucket_stats") or {}).get("correct")
        played = meta.get("actually_played")

        s = stats[be]
        s["n"] += 1
        if r.get("error"):
            s["errors"] += 1
            continue

        decision = parse_decision(r.get("response") or "")
        if decision is None:
            if len(s["unparsed_examples"]) < 3:
                s["unparsed_examples"].append((pid, (r.get("response") or "")[:120]))
            continue
        s["parsed"] += 1
        s["by_decision"][decision] = s["by_decision"].get(decision, 0) + 1

        if bucket in ("keep", "mull"):
            s["scorable"] += 1
            if decision == bucket:
                s["matches_higher_wr"] += 1
            by_correct[(be, bucket)]["n"] += 1
            if decision == bucket:
                by_correct[(be, bucket)]["match"] += 1

        if played in ("keep", "mull") and decision == played:
            s["matches_played"] += 1

    # Print main table
    print()
    cols = [
        ("backend",             30),
        ("n",                    4),
        ("parse%",               7),
        ("higher_wr%",          11),
        ("played%",              9),
        ("kept%",                7),
        ("mulled%",              8),
    ]
    header = " ".join(f"{name:<{w}}" for name, w in cols)
    print(header)
    print("-" * len(header))
    for be in sorted(stats):
        s = stats[be]
        n = s["n"] or 1
        parse_pct = s["parsed"] / n * 100
        higher_wr_pct = (
            s["matches_higher_wr"] / s["scorable"] * 100 if s["scorable"] else 0.0
        )
        played_pct = s["matches_played"] / s["parsed"] * 100 if s["parsed"] else 0.0
        kept = s["by_decision"]["keep"]
        mulled = s["by_decision"]["mull"]
        total_decisions = kept + mulled or 1
        kept_pct = kept / total_decisions * 100
        mulled_pct = mulled / total_decisions * 100
        row = [
            be[:30], s["n"],
            f"{parse_pct:.0f}%",
            f"{higher_wr_pct:.1f}%",
            f"{played_pct:.0f}%",
            f"{kept_pct:.0f}%",
            f"{mulled_pct:.0f}%",
        ]
        print(" ".join(f"{str(v):<{w}}" for v, (_, w) in zip(row, cols)))

    print()
    print("Per-decision breakdown (how each backend handles 'keep' buckets vs 'mull' buckets):")
    for (be, side) in sorted(by_correct):
        d = by_correct[(be, side)]
        if not d["n"]:
            continue
        pct = d["match"] / d["n"] * 100
        print(f"  {be:30} on '{side}' buckets: {d['match']}/{d['n']} = {pct:.1f}%")

    # Surface a couple of unparsed examples per backend so the user can see
    # whether parser tweaks would lift parse rate.
    print()
    print("Unparsed examples (first 3 per backend):")
    for be in sorted(stats):
        for pid, snippet in stats[be]["unparsed_examples"]:
            print(f"  {be:30} {pid}: {snippet!r}")

    if json_out:
        import time as _time
        backends_payload = []
        for be in sorted(stats):
            s = stats[be]
            n = s["n"] or 1
            higher_wr_pct = (
                s["matches_higher_wr"] / s["scorable"] if s["scorable"] else None
            )
            played_pct = s["matches_played"] / s["parsed"] if s["parsed"] else None
            kept = s["by_decision"]["keep"]
            mulled = s["by_decision"]["mull"]
            total_decisions = (kept + mulled) or 1
            backends_payload.append({
                "backend": be,
                "n": s["n"],
                "errors": s["errors"],
                "parsed": s["parsed"],
                "scorable": s["scorable"],
                "matches_higher_wr": s["matches_higher_wr"],
                "matches_played": s["matches_played"],
                "higher_wr_rate": round(higher_wr_pct, 4) if higher_wr_pct is not None else None,
                "played_agreement_rate": round(played_pct, 4) if played_pct is not None else None,
                "decisions": {"keep": kept, "mull": mulled},
                "decision_share": {
                    "keep": round(kept / total_decisions, 4),
                    "mull": round(mulled / total_decisions, 4),
                },
                "by_correct_bucket": {
                    "keep": (
                        {"n": by_correct[(be, "keep")]["n"],
                         "match": by_correct[(be, "keep")]["match"]}
                        if by_correct.get((be, "keep")) else {"n": 0, "match": 0}
                    ),
                    "mull": (
                        {"n": by_correct[(be, "mull")]["n"],
                         "match": by_correct[(be, "mull")]["match"]}
                        if by_correct.get((be, "mull")) else {"n": 0, "match": 0}
                    ),
                },
            })
        payload = {
            "target": "17lands_mulligan",
            "ts": _time.time(),
            "n_prompts": len(prompts),
            "n_responses": len(responses),
            "backends": backends_payload,
        }
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {json_out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--json", type=Path,
                        help="Optional structured JSON summary (for admin dashboard)")
    args = parser.parse_args()
    score(args.prompts, args.responses, args.json)


if __name__ == "__main__":
    main()
