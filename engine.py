"""SVD-based scoring and question selection (numpy.linalg.svd)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from data import KNOWLEDGE_MATRIX, PEOPLE
from rules import IMPLICATIONS, MUTEX_GROUPS, MUTEX_PAIRS

K = 14
M = np.array(KNOWLEDGE_MATRIX, dtype=np.float64)
_n_people, n_questions = M.shape

_U, s, Vt = np.linalg.svd(M, full_matrices=False)
Vk = Vt[:K, :]  # (K, n_questions)
Z = M @ Vk.T  # (n_people, K)

_sum_sq = float(np.sum(s**2))
sigma_display = s[:6].tolist()
variance_sigma1_pct = float(s[0] ** 2 / _sum_sq * 100)
cumulative_k_pct = float(np.sum(s[:K] ** 2) / _sum_sq * 100)


def h_binary(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1 - p) * math.log2(1 - p))


def _norm01(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx == mn:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)


def _answers_dict_to_arrays(answers: Dict[int, bool]) -> Tuple[np.ndarray, np.ndarray]:
    """Indices (sorted) and 0/1 values for answered questions."""
    if not answers:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    idx = np.array(sorted(answers.keys()), dtype=np.int64)
    vals = np.array([1.0 if answers[int(j)] else 0.0 for j in idx], dtype=np.float64)
    return idx, vals


def compute_raw_scores(answers: Dict[int, bool]) -> np.ndarray:
    idx, vals = _answers_dict_to_arrays(answers)
    if idx.size == 0:
        return np.ones(_n_people, dtype=np.float64)
    sub = M[:, idx]
    agree = 1.0 - np.abs(sub - vals.reshape(1, -1))
    return np.sum(agree, axis=1)


def _project_user(answers: Dict[int, bool]) -> np.ndarray:
    idx, vals = _answers_dict_to_arrays(answers)
    if idx.size == 0:
        return np.zeros(K, dtype=np.float64)
    return Vk[:, idx] @ vals


def compute_svd_cosine_scores(answers: Dict[int, bool]) -> np.ndarray:
    zu = _project_user(answers)
    nu = float(np.linalg.norm(zu))
    if nu < 1e-15:
        return np.zeros(_n_people, dtype=np.float64)
    zi_norms = np.linalg.norm(Z, axis=1)
    dots = Z @ zu
    out = np.zeros(_n_people, dtype=np.float64)
    mask = zi_norms >= 1e-15
    out[mask] = dots[mask] / (zi_norms[mask] * nu)
    return out


def final_scores(answers: Dict[int, bool]) -> np.ndarray:
    n_ans = len(answers)
    raw = compute_raw_scores(answers)
    raw_n = _norm01(raw)
    if n_ans < 4:
        return raw_n
    svd_s = compute_svd_cosine_scores(answers)
    svd_n = _norm01(svd_s)
    return 0.6 * svd_n + 0.4 * raw_n


def belief_from_scores(scores: np.ndarray) -> np.ndarray:
    shifted = np.maximum(scores, 1e-12)
    return shifted / np.sum(shifted)


def excluded_questions(answers: Dict[int, bool]) -> set:
    """Return question indices that are redundant given the current answers."""
    excluded: set = set()
    for a, b in MUTEX_PAIRS:
        if answers.get(a) is True:
            excluded.add(b)
        if answers.get(b) is True:
            excluded.add(a)
    for group in MUTEX_GROUPS:
        for member in group:
            if answers.get(member) is True:
                excluded |= (group - {member})
                break
    for q, expected, skip in IMPLICATIONS:
        if q in answers and answers[q] == expected:
            excluded |= skip
    excluded -= set(answers.keys())
    return excluded


def pick_question(answers: Dict[int, bool]) -> Tuple[int, float]:
    answered = set(answers.keys())
    excluded = excluded_questions(answers)
    candidates = [j for j in range(n_questions) if j not in answered and j not in excluded]
    if not candidates:
        return -1, 0.0
    scores = final_scores(answers)
    b = belief_from_scores(scores)
    best_j, best_h = -1, -1.0
    for j in candidates:
        p_yes = float(np.dot(b, M[:, j]))
        h = h_binary(p_yes)
        if h > best_h:
            best_h = h
            best_j = j
    p_cur = float(np.dot(b, M[:, best_j]))
    return best_j, h_binary(p_cur)


def should_guess(scores: np.ndarray, n_answered: int) -> bool:
    if n_answered < 3:
        return False
    order = np.argsort(-scores)
    s0, s1 = float(scores[order[0]]), float(scores[order[1]])
    if s0 <= 0:
        return False
    return s0 - s1 > 0.25 * s0


def scores_excluding(answers: Dict[int, bool], rejected) -> np.ndarray:
    """Return final_scores with rejected person indices zeroed out."""
    scores = final_scores(answers).copy()
    for r in rejected:
        scores[int(r)] = 0.0
    return scores


def best_guess_index(answers: Dict[int, bool], rejected) -> int:
    """Top non-rejected person index based on current answers."""
    scores = scores_excluding(answers, rejected)
    return int(np.argsort(-scores)[0])


def question_entropy_for_display(answers: Dict[int, bool], q_index: int) -> float:
    scores = final_scores(answers)
    b = belief_from_scores(scores)
    p = float(np.dot(b, M[:, q_index]))
    return h_binary(p)


@dataclass
class TopCandidate:
    index: int
    name: str
    score: float
    bar_pct: float
    leader: bool


def top_candidates_for_ui(answers: Dict[int, bool], q_index: int) -> Tuple[List[TopCandidate], float]:
    scores = final_scores(answers)
    order = np.argsort(-scores)[:5]
    max_s = float(scores[order[0]]) if order.size else 1.0
    if max_s <= 0:
        max_s = 1.0
    lead_gap = float(scores[order[0]] - scores[order[1]]) if order.size > 1 else 0.0
    leader_clear = order.size > 0 and float(scores[order[0]]) > 0 and lead_gap > 0.12 * float(scores[order[0]])
    rows: List[TopCandidate] = []
    for i in order:
        si = float(scores[i])
        rows.append(
            TopCandidate(
                index=int(i),
                name=PEOPLE[i],
                score=si,
                bar_pct=min(100.0, 100.0 * si / max_s),
                leader=bool(i == order[0] and leader_clear),
            )
        )
    ent = question_entropy_for_display(answers, q_index)
    return rows, ent


def sorted_scores_indices(scores: np.ndarray) -> List[Tuple[int, float]]:
    order = np.argsort(-scores)
    return [(int(i), float(scores[i])) for i in order]
