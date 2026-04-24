"""FastAPI app — all game logic runs in Python (engine + NumPy SVD)."""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from data import PEOPLE, QUESTIONS
from engine import (
    cumulative_k_pct,
    final_scores,
    pick_question,
    should_guess,
    sigma_display,
    sorted_scores_indices,
    top_candidates_for_ui,
    variance_sigma1_pct,
)

app = FastAPI(title="Class Akinator")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET", "bcsai-akinator-dev-key-change-me"),
    max_age=86400,
)

templates = Jinja2Templates(directory="templates")


def _get_answers(session: Dict[str, Any]) -> Dict[int, bool]:
    raw = session.get("answers") or {}
    return {int(k): bool(v) for k, v in raw.items()}


def _set_answers(session: Dict[str, Any], answers: Dict[int, bool]) -> None:
    session["answers"] = {str(k): v for k, v in answers.items()}


@app.get("/")
def home(request: Request):
    session = request.session
    phase = session.get("phase", "welcome")
    ctx: Dict[str, Any] = {
        "phase": phase,
        "people": PEOPLE,
        "questions": QUESTIONS,
        "sigma_display": sigma_display,
        "variance_sigma1_pct": round(variance_sigma1_pct, 2),
        "cumulative_k_pct": round(cumulative_k_pct, 2),
    }

    if phase == "play":
        q = int(session["current_q"])
        answers = _get_answers(session)
        n = len(answers) + 1
        top, entropy = top_candidates_for_ui(answers, q)
        ctx.update(
            {
                "question_text": QUESTIONS[q],
                "question_num": n,
                "current_q": q,
                "top_candidates": top,
                "entropy": round(entropy, 3),
            }
        )
    elif phase == "guess":
        answers = _get_answers(session)
        scores = final_scores(answers)
        ranking = sorted_scores_indices(scores)[:3]
        ctx.update(
            {
                "guess_name": PEOPLE[ranking[0][0]],
                "guess_ranking": ranking,
                "celebrated": bool(session.pop("celebrated", False)),
                "show_wrong_form": bool(session.pop("show_wrong_form", False)),
            }
        )

    return templates.TemplateResponse(request, "index.html", ctx)


@app.post("/start")
def start(request: Request):
    session = request.session
    session["phase"] = "play"
    session["answers"] = {}
    j, _ = pick_question({})
    session["current_q"] = j
    session.pop("celebrated", None)
    session.pop("show_wrong_form", None)
    return RedirectResponse(url="/", status_code=303)


@app.post("/answer")
def answer(request: Request, value: str = Form(...)):
    session = request.session
    if session.get("phase") != "play":
        return RedirectResponse(url="/", status_code=303)
    q = int(session["current_q"])
    answers = _get_answers(session)
    answers[q] = value.lower() in ("yes", "y", "1", "true")
    _set_answers(session, answers)

    scores = final_scores(answers)
    n_ans = len(answers)

    if should_guess(scores, n_ans) or n_ans >= len(QUESTIONS):
        session["phase"] = "guess"
        session.pop("current_q", None)
        return RedirectResponse(url="/", status_code=303)

    j, _ = pick_question(answers)
    session["current_q"] = j
    return RedirectResponse(url="/", status_code=303)


@app.post("/guess/correct")
def guess_correct(request: Request):
    request.session["celebrated"] = True
    return RedirectResponse(url="/", status_code=303)


@app.post("/guess/wrong")
def guess_wrong(request: Request):
    request.session["show_wrong_form"] = True
    return RedirectResponse(url="/", status_code=303)


@app.post("/play-again")
def play_again(request: Request):
    request.session.clear()
    request.session["phase"] = "welcome"
    return RedirectResponse(url="/", status_code=303)


# Optional: mount empty static if folder exists (for future assets)
_static = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")
