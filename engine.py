"""SVD-based scoring + hard consistency filtering for the Class Akinator.

Core ideas
----------
* Hard consistency mask: a candidate that contradicts *any* already-answered
  question has probability exactly 0. With binary M and binary answers, this
  is an exact integer comparison — no epsilon is needed for correctness.
* Ranking among consistent candidates uses cosine similarity in the truncated
  SVD latent space, blended with raw agreement once >= 4 answers are given
  (matches the blend described in section 3.4 of the paper).
* Belief is a proper probability distribution over the 46 candidates with
  inconsistent candidates zeroed out. When exactly one candidate remains
  consistent, the belief assigns it probability 1.0 exactly, so ``should_guess``
  terminates the game immediately.
* Question selection maximises binary entropy H(p_j) with p_j = b^T M[:, j]
  under the *filtered* belief, so ruled-out candidates do not contribute.
* k is chosen adaptively as the smallest dimension capturing >= 90% of the
  spectral energy; this keeps the code honest if data.py changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from data import KNOWLEDGE_MATRIX, PEOPLE

# --- Matrix precomputation ---------------------------------------------------

M = np.asarray(KNOWLEDGE_MATRIX, dtype=np.float64)
n_people, n_questions = M.shape

# Integer view for exact equality checks inside the hard consistency mask.
M_int = M.astype(np.int8)

# Full economy SVD; Vt has shape (n_questions, n_questions) for M 46x36.
_U, _s, _Vt = np.linalg.svd(M, full_matrices=False)


def _pick_k(singular: np.ndarray, threshold: float = 0.90) -> int:
    """Smallest k with cumulative singular-value energy >= ``threshold``."""
    sq = singular ** 2
    total = float(np.sum(sq))
    if total <= 0.0:
        return int(len(singular))
    cum = np.cumsum(sq) / total
    for i, c in enumerate(cum):
        if c >= threshold:
            return int(i + 1)
    return int(len(singular))


K = _pick_k(_s, 0.90)
Vk = _Vt[:K, :]            # (K, n_questions)
Z = M @ Vk.T               # (n_people, K)  -- latent profile per person

# Display-only spectral statistics (used by the "How it works" modal).
_sum_sq = float(np.sum(_s ** 2))
sigma_display = _s[:6].tolist()
variance_sigma1_pct = float(_s[0] ** 2 / _sum_sq * 100.0)
cumulative_k_pct = float(np.sum(_s[:K] ** 2) / _sum_sq * 100.0)


# --- Utilities ---------------------------------------------------------------

def _answers_arrays(answers: Dict[int, bool]) -> Tuple[np.ndarray, np.ndarray]:
    if not answers:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)
    idx = np.fromiter(sorted(answers.keys()), dtype=np.int64)
    vals = np.fromiter(
        (1 if answers[int(j)] else 0 for j in idx),
        dtype=np.int8,
        count=idx.size,
    )
    return idx, vals


def h_binary(p: float) -> float:
    """Binary entropy H(p) in bits. Returns 0 at p in {0, 1}."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


# --- The hard consistency mask (the fix) ------------------------------------

def consistent_mask(answers: Dict[int, bool]) -> np.ndarray:
    """Boolean mask of people consistent with every answered question.

    A person is inconsistent iff ``M_int[i, j] != user_answer`` for at least
    one answered ``j``. Exact integer equality — no floating-point wiggle.
    """
    idx, vals = _answers_arrays(answers)
    if idx.size == 0:
        return np.ones(n_people, dtype=bool)
    sub = M_int[:, idx]                            # (n_people, n_answered)
    return np.all(sub == vals[None, :], axis=1)     # (n_people,)


def consistent_count(answers: Dict[int, bool]) -> int:
    return int(consistent_mask(answers).sum())


# --- Scoring among the consistent set ---------------------------------------

def _raw_agreement(answers: Dict[int, bool]) -> np.ndarray:
    idx, vals = _answers_arrays(answers)
    if idx.size == 0:
        return np.zeros(n_people, dtype=np.float64)
    sub = M[:, idx]
    return np.sum(1.0 - np.abs(sub - vals.astype(np.float64)[None, :]), axis=1)


def _svd_cosine(answers: Dict[int, bool]) -> np.ndarray:
    idx, vals = _answers_arrays(answers)
    if idx.size == 0:
        return np.zeros(n_people, dtype=np.float64)
    z_user = Vk[:, idx] @ vals.astype(np.float64)      # (K,)
    nu = float(np.linalg.norm(z_user))
    if nu < 1e-12:
        return np.zeros(n_people, dtype=np.float64)
    z_norms = np.linalg.norm(Z, axis=1)                 # (n_people,)
    out = np.zeros(n_people, dtype=np.float64)
    good = z_norms >= 1e-12
    out[good] = (Z[good] @ z_user) / (z_norms[good] * nu)
    return out


def _norm_within(vec: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Min-max normalise ``vec`` over ``active`` entries, zero elsewhere."""
    out = np.zeros_like(vec)
    if not active.any():
        return out
    sub = vec[active]
    lo = float(sub.min())
    hi = float(sub.max())
    if hi > lo:
        out[active] = (sub - lo) / (hi - lo)
    else:
        out[active] = 1.0  # all equal: give uniform weight within the set
    return out


# --- Belief: the only "probability" the UI is allowed to trust --------------

def belief(answers: Dict[int, bool]) -> np.ndarray:
    """Posterior probability over the 46 candidates.

    * Inconsistent candidates get exactly 0.
    * One consistent candidate  -> probability 1.0 (true certainty).
    * Many consistent candidates -> SVD cosine ranking (blended with raw
      agreement after >= 4 answers), normalised within the consistent set.
    * No consistent candidates   -> graceful fallback: rank over the full
      population so the UI still has something to show. The app treats this
      as "guess anyway" because no answered question can now be satisfied.
    """
    mask = consistent_mask(answers)
    n_cons = int(mask.sum())

    # Certain: exactly one consistent candidate.
    if n_cons == 1:
        out = np.zeros(n_people, dtype=np.float64)
        out[mask] = 1.0
        return out

    n_ans = len(answers)
    active = mask if n_cons > 0 else np.ones(n_people, dtype=bool)

    # Before any answers: uniform over the whole class.
    if n_ans == 0:
        out = np.zeros(n_people, dtype=np.float64)
        out[active] = 1.0 / float(active.sum())
        return out

    # Rank the active set.
    if n_ans < 4:
        # Paper's rule: raw agreement until the 4th answer.
        scores = _raw_agreement(answers)
        scores = _norm_within(scores, active)
    else:
        svd_s = _svd_cosine(answers)
        raw_s = _raw_agreement(answers)
        scores = 0.6 * _norm_within(svd_s, active) + 0.4 * _norm_within(raw_s, active)

    # Zero outside the active set and renormalise to a probability distribution.
    scores = np.where(active, scores, 0.0)
    scores = np.maximum(scores, 0.0)
    total = float(scores.sum())
    if total <= 0.0:
        out = np.zeros(n_people, dtype=np.float64)
        out[active] = 1.0 / float(active.sum())
        return out
    return scores / total


# --- Question selection -----------------------------------------------------

def pick_question(answers: Dict[int, bool]) -> Tuple[int, float]:
    """Return (question_index, entropy) maximising H(p_j) under the belief.

    Returns ``(-1, 0.0)`` if every question has been answered.
    """
    answered = set(int(k) for k in answers.keys())
    b = belief(answers)
    best_j, best_h = -1, -1.0
    for j in range(n_questions):
        if j in answered:
            continue
        p_yes = float(np.dot(b, M[:, j]))
        h = h_binary(p_yes)
        if h > best_h:
            best_j, best_h = j, h
    if best_j < 0:
        return -1, 0.0
    return best_j, best_h


def question_entropy(answers: Dict[int, bool], q_index: int) -> float:
    b = belief(answers)
    p = float(np.dot(b, M[:, q_index]))
    return h_binary(p)


# --- Stopping & final guess --------------------------------------------------

def should_guess(answers: Dict[int, bool]) -> bool:
    """Stop iff exactly one consistent candidate remains, or nothing to ask."""
    if len(answers) >= n_questions:
        return True
    return consistent_count(answers) <= 1


def guess_index(answers: Dict[int, bool]) -> int:
    """Index of the final guess: unique consistent if any, else argmax belief."""
    return int(np.argmax(belief(answers)))


def certainty(answers: Dict[int, bool]) -> float:
    """Top posterior probability; 1.0 iff a unique consistent candidate."""
    return float(belief(answers).max())


# --- UI helpers --------------------------------------------------------------

@dataclass
class TopCandidate:
    index: int
    name: str
    probability: float    # real posterior in [0, 1]
    bar_pct: float        # same number, 0..100 for CSS width
    leader: bool          # True iff probability == 1.0 (certain)


def top_candidates_for_ui(
    answers: Dict[int, bool], top_n: int = 5
) -> Tuple[List[TopCandidate], float]:
    """Top-N candidates by belief, plus the entropy of the current question."""
    b = belief(answers)
    order = np.argsort(-b)[:top_n]
    rows: List[TopCandidate] = []
    for rank, i in enumerate(order):
        p = float(b[int(i)])
        if p <= 0.0 and rank > 0:
            break  # don't pad with zeros once we've dropped off the consistent set
        rows.append(
            TopCandidate(
                index=int(i),
                name=PEOPLE[int(i)],
                probability=p,
                bar_pct=min(100.0, 100.0 * p),
                leader=(rank == 0 and p >= 0.999999),
            )
        )
    return rows, 0.0  # entropy filled in by the caller that knows q_index


def sorted_by_belief(answers: Dict[int, bool]) -> List[Tuple[int, float]]:
    b = belief(answers)
    order = np.argsort(-b)
    return [(int(i), float(b[int(i)])) for i in order]
