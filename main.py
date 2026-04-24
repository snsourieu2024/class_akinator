"""FastAPI app for the Class Akinator.

Terminates *immediately* when true certainty (probability = 1.0) is reached,
i.e. when exactly one candidate is still consistent with every answer given.
"""

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
    K,
    certainty,
    consistent_count,
    cumulative_k_pct,
    guess_index,
    n_people,
    pick_question,
    question_entropy,
    should_guess,
    sigma_display,
    sorted_by_belief,
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


def _answers(session) -> Dict[int, bool]:
    return {int(k): bool(v) for k, v in (session.get("answers") or {}).items()}


def _set_answers(session, answers: Dict[int, bool]) -> None:
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
        "k_latent": K,
        "total_people": n_people,
        "total_questions": len(QUESTIONS),
    }

    if phase == "play":
        answers = _answers(session)
        q = int(session["current_q"])
        top, _ = top_candidates_for_ui(answers, top_n=5)
        ent = question_entropy(answers, q)
        ctx.update(
            {
                "question_text": QUESTIONS[q],
                "question_num": len(answers) + 1,
                "current_q": q,
                "top_candidates": top,
                "entropy": round(ent, 3),
                "consistent_n": consistent_count(answers),
            }
        )
    elif phase == "guess":
        answers = _answers(session)
        ranking = sorted_by_belief(answers)
        gi = guess_index(answers)
        top_prob = ranking[0][1] if ranking else 0.0
        ctx.update(
            {
                "guess_name": PEOPLE[gi],
                "guess_ranking": ranking[:3],
                "guess_probability": top_prob,
                "is_certain": top_prob >= 0.999999,
                "consistent_n": consistent_count(answers),
                "questions_asked": len(answers),
                "celebrated": bool(session.pop("celebrated", False)),
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
    return RedirectResponse(url="/", status_code=303)


@app.post("/answer")
def answer(request: Request, value: str = Form(...)):
    session = request.session
    if session.get("phase") != "play":
        return RedirectResponse(url="/", status_code=303)

    q = int(session["current_q"])
    answers = _answers(session)
    answers[q] = value.lower() in ("yes", "y", "1", "true")
    _set_answers(session, answers)

    # Terminate immediately on true certainty or if nothing remains to ask.
    if should_guess(answers) or certainty(answers) >= 0.999999:
        session["phase"] = "guess"
        session.pop("current_q", None)
        return RedirectResponse(url="/", status_code=303)

    j, _ = pick_question(answers)
    if j < 0:
        session["phase"] = "guess"
        session.pop("current_q", None)
    else:
        session["current_q"] = j
    return RedirectResponse(url="/", status_code=303)


@app.post("/guess/correct")
def guess_correct(request: Request):
    request.session["celebrated"] = True
    return RedirectResponse(url="/", status_code=303)


@app.post("/play-again")
def play_again(request: Request):
    request.session.clear()
    request.session["phase"] = "welcome"
    return RedirectResponse(url="/", status_code=303)


_static = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")
