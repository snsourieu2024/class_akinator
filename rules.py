"""Skip rules for the Akinator question selector.

Three structures encode when a question becomes redundant given prior answers.
Question indices reference data.QUESTIONS.

- MUTEX_PAIRS: (a, b) where YES on either implies NO on the other. So if either
  is answered YES, the other is skipped. (NO on either still leaves the other
  open — the answer could still legitimately be YES.)
- MUTEX_GROUPS: a set of question indices where exactly one can be YES. As soon
  as any member is answered YES, every other member of the group is skipped.
- IMPLICATIONS: (q, expected, skip) — if question q has answer `expected`, skip
  every index in `skip`.
"""

from __future__ import annotations

from typing import FrozenSet, List, Tuple

MUTEX_PAIRS: List[Tuple[int, int]] = [
    (15, 17),  # long hair  ↔ short hair
    (18, 19),  # dark eyes  ↔ light eyes
    (24, 25),  # outgoing   ↔ quiet
    (31, 32),  # front row  ↔ back row
]

MUTEX_GROUPS: List[FrozenSet[int]] = [
    frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8}),  # ethnicity (European, Spanish, Caribbean, E.European, Latin Am., Arab/ME, Asian, N.American, African)
    frozenset({11, 12, 13, 14}),              # hair color: dark, blonde, red, blue
    frozenset({33, 34, 35}),                  # living: flatmates, family, alone
]

IMPLICATIONS: List[Tuple[int, bool, FrozenSet[int]]] = [
    (9, False, frozenset({20})),    # not male → skip facial hair
    (15, False, frozenset({17})),   # not long hair → skip short hair
]
