# Legacy — do not use

These files are **not** part of the active runtime. They're kept here only
for history and to satisfy anyone who goes looking for them.

The canonical app is the FastAPI project at the repo root:

```
main.py
engine.py
data.py
templates/index.html
run.py
requirements.txt
tests/simulate.py
tests/test_http.py
```

Start it with `python run.py` (defaults to `http://127.0.0.1:8080`).

## What's in here and why it's legacy

| File | Why it's legacy |
|------|-----------------|
| `index.html` | A parallel JS implementation of the whole game (with its own embedded matrix and its own `ml-matrix` SVD). Duplicates the Python app, will drift, and uses the same buggy score-gap stopping rule the Python app used to have. Replaced by `templates/index.html` which is server-rendered from the FastAPI engine. |
| `math-logic.py` | Old 38-question design (still references the dropped `"American"` and `"mixed background"` columns). Does not match the current 46×36 matrix. |
| `simulate.mjs` | Node simulation that ships its own stale 46×38 matrix. Superseded by `tests/simulate.py`, which imports the real `data.py` and `engine.py`. |
| `verify-elshan.mjs` | Hardcodes `QUESTIONS_LEN = 38` and reads the matrix by regex-eval'ing the old `index.html`. Stale on every axis. |
| `test-svd.mjs` | Six-line toy (`svd([[1,0],[0,1]])`). Harmless but dead. |
| `package.json`, `package-lock.json` | Exist only to pull `ml-matrix` for the JS files above. Unused by the FastAPI app. |

## What changed in the canonical app

1. **True certainty termination.** The engine now builds an exact hard
   consistency mask over the binary matrix: any candidate contradicting an
   answered question has probability 0. When exactly one candidate is left,
   the belief is 1.0 and the game stops on that turn.
2. **Honest probability.** The UI shows a real posterior (`belief`), not a
   min–max-normalised cosine score dressed up as probability.
3. **Adaptive k.** `K` is now picked as the smallest k with cumulative
   variance ≥ 90% of the current matrix, rather than a hardcoded 14 that
   could drift from the data.
4. **Polished frontend.** Progress chip + "N of 46 still possible" chip
   replace the "Question X of ~8" fiction; yes/no buttons are disabled on
   submit to prevent double-posting; the dead "wrong answer" input is
   removed; the guess screen shows a certainty badge that says *100%* only
   when the mask really is unique.

## Verification

- `tests/simulate.py`: 46/46 classmates correctly identified, min 5 / avg
  5.61 / max 6 questions, every termination at `|consistent| == 1`.
- `tests/test_http.py`: drives the real FastAPI app over HTTP; every
  playthrough terminates on the exact request the mask collapses, renders
  the correct name, and shows the `Certainty: 100%` badge.
