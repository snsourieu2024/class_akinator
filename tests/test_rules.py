"""Sanity checks on the shape of the rule declarations in rules.py."""

from rules import IMPLICATIONS, MUTEX_GROUPS, MUTEX_PAIRS


def test_mutex_pairs_are_disjoint_index_tuples():
    seen = set()
    for a, b in MUTEX_PAIRS:
        assert isinstance(a, int) and isinstance(b, int)
        assert a != b
        assert (a, b) not in seen and (b, a) not in seen
        seen.add((a, b))


def test_mutex_groups_have_at_least_two_members_and_no_overlap():
    flat: set[int] = set()
    for group in MUTEX_GROUPS:
        assert isinstance(group, frozenset)
        assert len(group) >= 2
        assert flat.isdisjoint(group), f"group {group} overlaps another group"
        flat |= group


def test_implications_have_valid_shape():
    for q_idx, expected_answer, skip_set in IMPLICATIONS:
        assert isinstance(q_idx, int)
        assert isinstance(expected_answer, bool)
        assert isinstance(skip_set, frozenset)
        assert q_idx not in skip_set
        assert all(isinstance(s, int) for s in skip_set)


def test_expected_specific_rules_present():
    pairs = {tuple(sorted(p)) for p in MUTEX_PAIRS}
    assert (15, 17) in pairs
    assert (18, 19) in pairs
    assert (24, 25) in pairs
    assert (31, 32) in pairs

    assert frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8}) in MUTEX_GROUPS
    assert frozenset({11, 12, 13, 14}) in MUTEX_GROUPS
    assert frozenset({33, 34, 35}) in MUTEX_GROUPS

    assert (9, False, frozenset({20})) in IMPLICATIONS
    assert (15, False, frozenset({17})) in IMPLICATIONS


def test_all_referenced_indices_are_in_range():
    from data import QUESTIONS
    n = len(QUESTIONS)  # 36
    for a, b in MUTEX_PAIRS:
        assert 0 <= a < n and 0 <= b < n
    for group in MUTEX_GROUPS:
        for q in group:
            assert 0 <= q < n
    for q_idx, _, skip in IMPLICATIONS:
        assert 0 <= q_idx < n
        for s in skip:
            assert 0 <= s < n
