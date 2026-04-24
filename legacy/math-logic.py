"""
Class Akinator - Powered by Eigenvalues and SVD
================================================
BCSAI 2026 | IE University | Matrices and Linear Transformations

HOW TO PLUG IN YOUR DATA
-------------------------
1. Keep PEOPLE list in the same row order as your Excel sheet (46 people).
2. Replace KNOWLEDGE_MATRIX with your 46 x 38 array of 0s and 1s.
   Easiest way: copy all 46 rows from Excel, paste below as a Python list.
3. Run:   python akinator_svd.py
   Or:    python akinator_svd.py --analysis   (shows SVD breakdown first)

HOW IT WORKS
------------
- M is a 46x38 binary matrix. M[i][j] = 1 if person i answers yes to question j.
- SVD decomposes M = U * Sigma * V^T, finding latent "trait dimensions".
- User answers are projected into this latent space: z = V_k^T * a
- Each person's profile is also projected: Z = M * V_k
- Nearest neighbour in latent space = the guess.
- Questions are selected by binary entropy to maximise information per question.
"""

import numpy as np
import sys

# ============================================================
# QUESTIONS  (38 total - matches Excel column order exactly)
# ============================================================
QUESTIONS = [
    # REGION (0-9)
    "Is your person European?",
    "Is your person Spanish?",
    "Is your person American?",
    "Is your person Caribbean?",
    "Is your person Eastern European?",
    "Is your person Latin American?",
    "Is your person Arab or Middle Eastern?",
    "Is your person Asian?",
    "Is your person North American?",
    "Is your person African?",
    # PHYSICAL (10-24)
    "Is your person male?",
    "Is your person tall?",
    "Does your person have dark hair?",
    "Does your person have blonde hair?",
    "Does your person have red or auburn hair?",
    "Does your person have blue or dyed hair?",
    "Does your person have long hair (past shoulders)?",
    "Does your person have curly or wavy hair?",
    "Does your person have short hair?",
    "Does your person have dark eyes?",
    "Does your person have light eyes?",
    "Does your person have facial hair?",
    "Does your person wear glasses?",
    "Does your person usually wear a hoodie?",
    "Does your person usually wear a cap?",
    # PERSONALITY (25-30)
    "Is your person outgoing or loud?",
    "Is your person quiet or reserved?",
    "Is your person funny or jokey?",
    "Is your person very competitive?",
    "Is your person chill and laid back?",
    "Does your person always ask questions in class?",
    # BACKGROUND (31)
    "Does your person have a mixed background?",
    # FUN (32-34)
    "Is your person always late to class?",
    "Does your person sit in the front row?",
    "Does your person sit in the back row?",
    # LIVING (35-37)
    "Does your person live with flatmates?",
    "Does your person live with family?",
    "Does your person live alone?",
]

N_QUESTIONS = len(QUESTIONS)  # 38

# ============================================================
# PEOPLE  (46 total - same order as Excel rows)
# ============================================================
PEOPLE = [
    "Ignacio Zubizarreta",        #  0  Instructor
    "Iciar Adelino",              #  1
    "Sanad Albilleh",             #  2
    "Salma Alnsour",              #  3
    "Issam Arida",                #  4
    "Andres Befeler",             #  5
    "Rodrigo Blanco",             #  6
    "Nikoloz Chachia",            #  7
    "Hernan Chacon",              #  8
    "Xander Chen",                #  9
    "Javier Cruz",                # 10
    "Alesia Dako",                # 11
    "Ralph De Catelan",           # 12
    "Sergio Diez",                # 13
    "Jad El Aawar",               # 14
    "Olivia Espineira",           # 15
    "Adria Fijo",                 # 16
    "Alex Garcia",                # 17
    "Matteo Giorgetti",           # 18
    "Jaime Gomez",                # 19
    "Nicolas Gonzalez Erales",    # 20
    "Laura Gonzalez",             # 21
    "Nicolas Grass",              # 22
    "Lara Iglesias",              # 23
    "Zaid Jumean",                # 24
    "Adam Khoury",                # 25
    "William Lavold",             # 26
    "Georgy Lifshitz",            # 27
    "Ryann Mack",                 # 28
    "Hari Mallela",               # 29
    "Lama Moucattash",            # 30
    "Kiril Petrovski",            # 31
    "Gabriel Queiroz",            # 32
    "Tatiana Quinn",              # 33
    "Elshan Ragimov",             # 34
    "Habib Rahal",                # 35
    "Kohana Rakipi",              # 36
    "Jose Reyes",                 # 37
    "Christoph Rintz",            # 38
    "Ali Samara",                 # 39
    "Laura Somogyi",              # 40
    "Viktoriia Stepanenko",       # 41
    "Declan Van Dam",             # 42
    "Ronan Vaz",                  # 43
    "Guo Yuting",                 # 44
    "Alejandro Zapata",           # 45
]

N_PEOPLE = len(PEOPLE)  # 46

# ============================================================
# KNOWLEDGE MATRIX  M  (46 rows x 38 columns)
# <<PLACEHOLDER>>
# Replace the block below with your actual data.
# Each row = one person, in the same order as PEOPLE above.
# Each column = one question, in the same order as QUESTIONS above.
# Values: 1 = yes, 0 = no.
#
# Easiest way to fill this in:
#   1. Open the filled Excel sheet
#   2. Select all 46 data rows (columns C onwards, 38 columns)
#   3. Copy and paste as a Python list of lists here
#
# Example format:
# KNOWLEDGE_MATRIX = np.array([
#   # Eur  Spa  Ame  Car  EEu  Lat  Arab Asi  NAm  Afr  Mal  Tal  DkH  BlH  RdH  BlDy Lon  Cur  Sho  DkE  LtE  Fac  Gls  Hod  Cap  Out  Qui  Fun  Com  Chl  Aqn  Mix  Lat  Fro  Bck  Flt  Fam  Aln
#   [1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   1,   0,   1,   0,   0,   0,   0,   0,   0,   0,   1,   0],  # Ignacio
#   ...
# ], dtype=float)
# ============================================================

# Temporary random placeholder - REPLACE THIS with your real data
np.random.seed(99)
KNOWLEDGE_MATRIX = np.random.randint(0, 2, (N_PEOPLE, N_QUESTIONS)).astype(float)


# ============================================================
# SVD ENGINE
# ============================================================

def compute_svd(M, k=None):
    """
    Full SVD: M = U * Sigma * V^T
    Automatically chooses k to capture >= 90% of variance.
    Returns U_k, sigma_k, Vt_k, k
    """
    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
    if k is None:
        cumvar = np.cumsum(sigma**2) / np.sum(sigma**2)
        k = int(np.searchsorted(cumvar, 0.90)) + 1
        k = min(k, len(sigma))
    return U[:, :k], sigma[:k], Vt[:k, :], k


def raw_score(M, answers):
    """
    Count agreements: s[i] = number of answered questions
    where person i's attribute matches the user's answer.
    Core operation: 1 - |M[:,j] - a[j]| per answered question.
    """
    s = np.zeros(M.shape[0])
    for j, a in answers.items():
        s += 1 - np.abs(M[:, j] - a)
    return s


def svd_score(M, Vt_k, answer_vec):
    """
    Project user answers and all people into SVD latent space,
    then compute cosine similarity.

    z_user    = V_k^T * a          (user in latent space)
    Z_people  = M * V_k            (all people in latent space)
    score[i]  = cosine(Z[i], z_user)
    """
    Z_people = M @ Vt_k.T                          # (46 x k)
    z_user   = Vt_k @ answer_vec                   # (k,)
    dots     = Z_people @ z_user                   # (46,)
    norms    = (np.linalg.norm(Z_people, axis=1)
                * (np.linalg.norm(z_user) + 1e-9)) # (46,)
    return dots / (norms + 1e-9)


def binary_entropy(p):
    """H(p) = -p*log2(p) - (1-p)*log2(1-p). Peaks at p=0.5."""
    if p < 1e-9 or p > 1 - 1e-9:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def pick_best_question(M, answers, scores):
    """
    Select the unanswered question with highest binary entropy.
    p_all = b @ M  (one matrix-vector product covers all 38 questions)
    """
    asked   = set(answers.keys())
    b       = scores / (scores.sum() + 1e-9)   # belief vector
    p_all   = b @ M                             # prob of yes per question

    best_q, best_h = -1, -1.0
    for j in range(M.shape[1]):
        if j in asked:
            continue
        h = binary_entropy(p_all[j])
        if h > best_h:
            best_h = h
            best_q = j
    return best_q


# ============================================================
# ANALYSIS (run with --analysis flag)
# ============================================================

def print_analysis(M, people, questions, sigma, Vt_k, k):
    print("\n" + "=" * 60)
    print("  SVD ANALYSIS OF THE CLASS KNOWLEDGE MATRIX")
    print("=" * 60)
    print(f"  Matrix      : {M.shape[0]} people x {M.shape[1]} questions")
    print(f"  Rank used k : {k}")
    print()

    total = np.sum(sigma ** 2)
    cum   = 0.0
    print(f"  {'Dim':<5}  {'σᵢ':>8}  {'Var %':>7}  {'Cumul %':>9}")
    print("  " + "-" * 35)
    for i, s in enumerate(sigma):
        v    = s ** 2 / total * 100
        cum += v
        print(f"  {i+1:<5}  {s:>8.3f}  {v:>7.2f}  {cum:>9.2f}")
        if cum >= 95:
            print(f"  ... (95% variance captured at k={i+1})")
            break

    print()
    print("  TOP QUESTION LOADINGS PER LATENT DIMENSION")
    print("  (what each SVD dimension represents)")
    print()
    for dim in range(min(5, k)):
        loadings = Vt_k[dim]
        top_idx  = np.argsort(np.abs(loadings))[::-1][:4]
        top_qs   = [(questions[j][:40], f"{loadings[j]:+.3f}") for j in top_idx]
        print(f"  Dimension {dim+1}  (σ={sigma[dim]:.2f}):")
        for q, load in top_qs:
            print(f"    {load}  {q}")
        print()

    print("  2D PROJECTION (first two SVD dimensions)")
    Z = M @ Vt_k[:2].T
    print(f"  {'Person':<28}  {'Dim1':>7}  {'Dim2':>7}")
    print("  " + "-" * 46)
    for i, name in enumerate(people):
        print(f"  {name:<28}  {Z[i,0]:>7.3f}  {Z[i,1]:>7.3f}")
    print("=" * 60 + "\n")


# ============================================================
# MAIN GAME LOOP
# ============================================================

def play(show_analysis=False):
    M = KNOWLEDGE_MATRIX.copy()

    # Compute SVD once at startup
    U_k, sigma_k, Vt_k, k = compute_svd(M)

    if show_analysis:
        print_analysis(M, PEOPLE, QUESTIONS, sigma_k, Vt_k, k)

    answers    = {}                          # {question_index: 0 or 1}
    answer_vec = np.zeros(N_QUESTIONS)       # full 38-dim answer vector
    MAX_Q      = 10

    print("\n" + "=" * 55)
    print("  CLASS AKINATOR")
    print("  Powered by Eigenvalues and SVD")
    print("=" * 55)
    print("  Think of someone in this class (or the professor).")
    print("  Answer y (yes) or n (no) to each question.\n")

    for turn in range(MAX_Q):
        # Raw agreement score (stable with few answers)
        scores = raw_score(M, answers)

        # Blend with SVD cosine score once we have enough answers
        if len(answers) >= 4:
            sv      = svd_score(M, Vt_k, answer_vec)
            raw_n   = scores / (scores.max() + 1e-9)
            svd_n   = ((sv - sv.min())
                       / (sv.max() - sv.min() + 1e-9))
            scores  = 0.6 * svd_n + 0.4 * raw_n

        # Early guess if one person is clearly ahead
        if len(answers) >= 3:
            top2 = np.sort(scores)[::-1][:2]
            if top2[0] - top2[1] > 0.25 * top2[0]:
                break

        # Pick best question by entropy
        q = pick_best_question(M, answers, scores)
        if q == -1:
            break

        print(f"  Q{turn + 1}: {QUESTIONS[q]}")
        while True:
            ans = input("    Your answer (y/n): ").strip().lower()
            if ans in ("y", "yes"):
                answers[q]     = 1
                answer_vec[q]  = 1.0
                break
            elif ans in ("n", "no"):
                answers[q]     = 0
                answer_vec[q]  = 0.0
                break
            else:
                print("    Please type y or n.")
        print()

    # Final scoring
    scores = raw_score(M, answers)
    if len(answers) >= 4:
        sv     = svd_score(M, Vt_k, answer_vec)
        raw_n  = scores / (scores.max() + 1e-9)
        svd_n  = (sv - sv.min()) / (sv.max() - sv.min() + 1e-9)
        scores = 0.6 * svd_n + 0.4 * raw_n

    ranking = np.argsort(scores)[::-1]

    print("=" * 55)
    print(f"  I think you are thinking of:")
    print(f"  >>> {PEOPLE[ranking[0]].upper()} <<<")
    print()
    print("  Top 3 candidates:")
    for i in range(min(3, N_PEOPLE)):
        idx = ranking[i]
        bar = "#" * int(scores[idx] * 20)
        print(f"    {i+1}. {PEOPLE[idx]:<28}  {scores[idx]:.3f}  {bar}")
    print()

    correct = input("  Was I right? (y/n): ").strip().lower()
    if correct in ("y", "yes"):
        print("\n  The eigenvalues never lie.\n")
    else:
        actual = input("  Who were you thinking of? ").strip()
        print(f"\n  Good to know! I will remember {actual} next time.\n")
    print("=" * 55 + "\n")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    play(show_analysis="--analysis" in sys.argv)