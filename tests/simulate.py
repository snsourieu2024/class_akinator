"""Deterministic simulation over every classmate in the knowledge matrix.

For each person i we play a game answering honestly (the row of M[i]) and
verify that:

    * the game terminates (either by true certainty or by exhausting Q),
    * the final guess equals person i,
    * termination happens exactly when the consistent set reaches size 1.

Also reports min / avg / max number of questions to identification, plus any
pair of people whose rows in M are identical (inherently indistinguishable).

Run from the project root:

    python tests/simulate.py
"""

from __future__ import annotations

import os
import sys
from statistics import mean
from typing import Dict, List, Tuple

# Make ``data`` and ``engine`` importable when invoked from the project root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from data import PEOPLE  # noqa: E402
from engine import (  # noqa: E402
    K,
    M_int,
    certainty,
    consistent_count,
    cumulative_k_pct,
    guess_index,
    n_people,
    n_questions,
    pick_question,
    should_guess,
)


def simulate_for(target_idx: int) -> Tuple[bool, int, int, int]:
    """Returns (success, n_questions_asked, consistent_at_end, guessed_idx)."""
    truth = M_int[target_idx]
    answers: Dict[int, bool] = {}

    # Pick the opening question once, deterministically.
    j, _ = pick_question(answers)
    while not should_guess(answers) and j >= 0:
        answers[j] = bool(truth[j])
        if should_guess(answers):
            break
        j, _ = pick_question(answers)

    final = consistent_count(answers)
    g = guess_index(answers)
    return (g == target_idx, len(answers), final, g)


def find_duplicate_rows() -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for a in range(n_people):
        for b in range(a + 1, n_people):
            if np.array_equal(M_int[a], M_int[b]):
                pairs.append((a, b))
    return pairs


def demo_certainty_trace(target_name: str = "Salma Alnsour") -> None:
    if target_name not in PEOPLE:
        print(f"[demo] {target_name!r} not in roster; skipping trace")
        return
    target = PEOPLE.index(target_name)
    truth = M_int[target]
    answers: Dict[int, bool] = {}

    print(f"\n=== Certainty trace for target: {target_name} ===")
    j, h = pick_question(answers)
    step = 0
    while not should_guess(answers) and j >= 0:
        step += 1
        answers[j] = bool(truth[j])
        n_cons = consistent_count(answers)
        cert = certainty(answers)
        print(
            f"  step {step:>2} | q#{j:<2} ans={'YES' if truth[j] else 'NO '} "
            f"| consistent={n_cons:<2} | certainty={cert:.4f} "
            f"| should_guess={should_guess(answers)}"
        )
        if should_guess(answers):
            break
        j, h = pick_question(answers)
    g = guess_index(answers)
    print(
        f"  -> terminated after {len(answers)} questions; "
        f"guess = {PEOPLE[g]} "
        f"(correct: {g == target}); "
        f"final certainty = {certainty(answers):.4f}"
    )


def main() -> int:
    print(f"Matrix: {n_people} people x {n_questions} questions")
    print(f"Adaptive k = {K}  (>=90% variance; cumulative = {cumulative_k_pct:.2f}%)")

    dups = find_duplicate_rows()
    if dups:
        print("\n[warn] duplicate rows — inherently indistinguishable:")
        for a, b in dups:
            print(f"  {PEOPLE[a]} <=> {PEOPLE[b]}")
    else:
        print("\nAll 46 rows of M are pairwise distinct.")

    results: List[Tuple[str, bool, int, int, int]] = []
    for i, name in enumerate(PEOPLE):
        ok, nq, final_cons, g = simulate_for(i)
        results.append((name, ok, nq, final_cons, g))

    correct = [r for r in results if r[1]]
    wrong = [r for r in results if not r[1]]
    q_counts = [r[2] for r in results]

    print(f"\nSimulated {len(results)} targets:")
    print(f"  correctly identified  : {len(correct)} / {len(results)}")
    print(f"  misidentified         : {len(wrong)}")
    print(f"  questions asked (min) : {min(q_counts)}")
    print(f"  questions asked (avg) : {mean(q_counts):.2f}")
    print(f"  questions asked (max) : {max(q_counts)}")

    # Distribution histogram
    print("\n  distribution of #questions to identification:")
    hist: Dict[int, int] = {}
    for nq in q_counts:
        hist[nq] = hist.get(nq, 0) + 1
    for nq in sorted(hist):
        bar = "#" * hist[nq]
        print(f"    {nq:>2} q : {hist[nq]:>2}  {bar}")

    # Spot-check: every correctly-identified game ended with exactly one
    # consistent candidate, which is the probability-1.0 stop condition.
    bad_term = [r for r in correct if r[3] != 1]
    if bad_term:
        print("\n[FAIL] some games terminated without reaching certainty:")
        for r in bad_term:
            print(f"  {r[0]}: final consistent set size = {r[3]}")
    else:
        print("\n[PASS] every correct identification ended with |consistent| == 1.")

    if wrong:
        print("\n[FAIL] misidentifications:")
        for name, _, nq, fc, g in wrong:
            print(f"  target={name}  guessed={PEOPLE[g]}  nq={nq}  final_consistent={fc}")

    demo_certainty_trace("Salma Alnsour")
    demo_certainty_trace("Elshan Ragimov")

    return 0 if not wrong and not bad_term else 1


if __name__ == "__main__":
    sys.exit(main())
