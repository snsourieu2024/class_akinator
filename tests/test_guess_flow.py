"""Tests for the guess-and-retry flow: forced first guess, rejection tracking."""

import numpy as np

from data import KNOWLEDGE_MATRIX, PEOPLE
from engine import (
    best_guess_index,
    final_scores,
    pick_question,
    scores_excluding,
    should_guess,
)


def test_scores_excluding_zeroes_rejected_indices():
    answers = {0: True, 9: True}
    rejected = [3, 7]
    scores = scores_excluding(answers, rejected)
    assert scores[3] == 0.0
    assert scores[7] == 0.0


def test_scores_excluding_does_not_mutate_underlying_scores():
    answers = {0: True, 9: True}
    base = final_scores(answers).copy()
    _ = scores_excluding(answers, [3, 7])
    np.testing.assert_array_equal(final_scores(answers), base)


def test_best_guess_excludes_rejected():
    # Use a real person's answers; their own index should be the top guess.
    target = 2  # Sanad Albilleh
    answers = {i: bool(KNOWLEDGE_MATRIX[target][i]) for i in range(36)}
    top = best_guess_index(answers, [])
    assert top == target
    # Reject the top → next pick must be different
    second = best_guess_index(answers, [target])
    assert second != target


def test_simulated_game_forces_guess_at_q14():
    """Play a full game on a target; verify first guess happens at Q14 if reached."""
    target = 2
    true_answers = {i: bool(KNOWLEDGE_MATRIX[target][i]) for i in range(36)}
    answers: dict[int, bool] = {}
    for step in range(14):
        j, _ = pick_question(answers)
        if j < 0:
            break
        answers[j] = true_answers[j]
    assert len(answers) == 14, "expected to reach 14 questions before any guess"
    guess = best_guess_index(answers, [])
    assert 0 <= guess < len(PEOPLE)


def test_simulated_wrong_guess_then_continue():
    """If first guess at Q14 is wrong, simulate continuing until correct or exhausted."""
    target = 2
    true_answers = {i: bool(KNOWLEDGE_MATRIX[target][i]) for i in range(36)}
    answers: dict[int, bool] = {}
    rejected: list[int] = []

    # First 14 questions
    for _ in range(14):
        j, _ = pick_question(answers)
        if j < 0:
            break
        answers[j] = true_answers[j]
    first_guess = best_guess_index(answers, rejected)

    if first_guess == target:
        # Got it on first try — that's fine, but force the wrong-path for the test
        # by rejecting the correct guess and continuing.
        rejected.append(first_guess)
    else:
        rejected.append(first_guess)

    # Continue questioning until should_guess fires or we run out
    while True:
        j, _ = pick_question(answers)
        if j < 0:
            break
        answers[j] = true_answers[j]
        if should_guess(scores_excluding(answers, rejected), len(answers)):
            break

    second = best_guess_index(answers, rejected)
    assert second not in rejected
    # Even if not target, must be a valid index
    assert 0 <= second < len(PEOPLE)
