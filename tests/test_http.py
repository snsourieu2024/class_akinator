"""End-to-end test: drives the real FastAPI app via an ASGI client.

Confirms that:
1. /start transitions to the play screen and renders a question.
2. POSTing truthful answers for a target eventually flips the phase to
   "guess" on the very request where the hard consistency mask collapses
   to size 1 (probability 1.0).
3. The final rendered page shows the correct name *and* the "Certainty: 100%"
   badge.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from data import PEOPLE  # noqa: E402
from engine import M_int, consistent_count, pick_question, should_guess  # noqa: E402
from main import app  # noqa: E402


QUESTION_RE = re.compile(r'<div class="q-text">([^<]+)</div>', re.S)
GUESS_NAME_RE = re.compile(r'data-full="([^"]+)"')
CERTAINTY_RE = re.compile(r"Certainty:\s*100%")
REMAINING_RE = re.compile(r"(\d+)\s+of\s+\d+\s+still possible")


def play_one(target_name: str) -> None:
    assert target_name in PEOPLE, target_name
    target_idx = PEOPLE.index(target_name)
    truth = M_int[target_idx]

    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200 and "Let&rsquo;s play" in r.text, "welcome missing"

        r = client.post("/start", follow_redirects=True)
        assert r.status_code == 200, r.status_code
        m = QUESTION_RE.search(r.text)
        assert m, "first question not rendered"

        # Mirror the server's question selection so we answer the right one.
        answers = {}
        step = 0
        while True:
            step += 1
            assert step <= 40, "simulation did not terminate"
            j, _ = pick_question(answers)
            assert j >= 0, "no next question to ask"
            ans = "yes" if truth[j] else "no"
            answers[j] = bool(truth[j])

            r = client.post(
                "/answer",
                data={"value": ans},
                follow_redirects=True,
            )
            assert r.status_code == 200, r.status_code

            if should_guess(answers):
                # Very step where the mask collapsed to <= 1 or questions ran out.
                assert 'class="guess-card"' in r.text, (
                    "expected guess screen after certainty reached"
                )
                name_match = GUESS_NAME_RE.search(r.text)
                assert name_match, "guess name not rendered"
                rendered = name_match.group(1)
                assert rendered == target_name, (
                    f"wrong name rendered: got {rendered!r}, expected {target_name!r}"
                )
                cons = consistent_count(answers)
                if cons == 1:
                    assert CERTAINTY_RE.search(r.text), (
                        "expected 'Certainty: 100%' badge when |consistent| == 1"
                    )
                print(
                    f"  [{target_name}] terminated at step {step}, "
                    f"consistent={cons}, name rendered OK, "
                    f"certainty-badge={'yes' if CERTAINTY_RE.search(r.text) else 'no'}"
                )
                return

            # Still playing: must be back on the play screen.
            assert 'class="q-text"' in r.text, "expected play screen mid-game"
            m = REMAINING_RE.search(r.text)
            if m:
                rendered_cons = int(m.group(1))
                actual_cons = consistent_count(answers)
                assert rendered_cons == actual_cons, (
                    f"UI shows {rendered_cons} consistent, engine says {actual_cons}"
                )


def main() -> int:
    targets = [
        "Salma Alnsour",
        "Elshan Ragimov",
        "Ignacio Zubizarreta",
        "Tatiana Quinn",
        "Guo Yuting",
        "Ali Samara",
    ]
    print("Driving the real FastAPI app end-to-end...\n")
    for name in targets:
        play_one(name)

    print("\n[PASS] every HTTP playthrough terminated on the step the mask hit 1,")
    print("       the correct name was rendered, and the 100% badge was shown.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
