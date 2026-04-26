"""Tests for engine.excluded_questions and pick_question filtering."""

from engine import excluded_questions, pick_question


def test_no_answers_excludes_nothing():
    assert excluded_questions({}) == set()


def test_yes_on_outgoing_excludes_quiet():
    assert 25 in excluded_questions({24: True})


def test_yes_on_quiet_excludes_outgoing():
    assert 24 in excluded_questions({25: True})


def test_no_on_outgoing_does_not_exclude_quiet():
    assert 25 not in excluded_questions({24: False})


def test_yes_on_long_hair_excludes_short_hair():
    assert 17 in excluded_questions({15: True})


def test_yes_on_dark_eyes_excludes_light_eyes():
    assert 19 in excluded_questions({18: True})


def test_yes_on_front_row_excludes_back_row():
    assert 32 in excluded_questions({31: True})


def test_yes_on_blonde_excludes_other_hair_colors():
    excl = excluded_questions({12: True})
    assert {11, 13, 14}.issubset(excl)
    assert 12 not in excl  # already-answered isn't reported as "excluded"


def test_no_on_blonde_does_not_exclude_other_hair_colors():
    excl = excluded_questions({12: False})
    assert excl.isdisjoint({11, 13, 14})


def test_yes_on_flatmates_excludes_family_and_alone():
    excl = excluded_questions({33: True})
    assert {34, 35}.issubset(excl)


def test_male_no_excludes_facial_hair():
    assert 20 in excluded_questions({9: False})


def test_male_yes_does_not_exclude_facial_hair():
    assert 20 not in excluded_questions({9: True})


def test_ethnicity_yes_excludes_other_ethnicities():
    # Q5 Arab/Middle Eastern YES → all other Q0..Q8 must be excluded
    excl = excluded_questions({5: True})
    for q in range(9):
        if q == 5:
            continue
        assert q in excl, f"Q{q} should be excluded by Q5=YES"


def test_ethnicity_no_does_not_exclude_other_ethnicities():
    # Q5 Arab/ME NO → other ethnicities still in play (a NO doesn't pin anything)
    excl = excluded_questions({5: False})
    for q in range(9):
        assert q not in excl, f"Q{q} should not be excluded by Q5=NO"


def test_no_on_long_hair_excludes_short_hair():
    assert 17 in excluded_questions({15: False})


def test_yes_on_long_hair_still_excludes_short_hair():
    # The mutex pair rule still fires for the YES direction
    assert 17 in excluded_questions({15: True})


def test_pick_question_skips_excluded():
    # After answering Q24 outgoing YES, pick_question must never return 25.
    j, _ = pick_question({24: True})
    assert j != 25
    assert j != 24


def test_pick_question_returns_minus_one_when_all_answered():
    answers = {i: True for i in range(36)}
    j, h = pick_question(answers)
    assert j == -1
    assert h == 0.0


def test_pick_question_with_empty_answers_returns_valid_index():
    j, _ = pick_question({})
    assert 0 <= j < 36
