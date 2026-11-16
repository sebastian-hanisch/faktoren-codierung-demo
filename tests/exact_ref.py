"""Zweiter, unabhängiger Rechenweg in Brüchen (`fractions`): Normalgleichung, F-Test der Streichung von Spalten und Wald-Test in Zellmittel-Parametern.

Kein Import aus den Modulen der App. Die Codierungen sind hier noch einmal von Hand aufgeschrieben (Referenz, Summe, Helmert, gewichtet zentriert), damit ein gemeinsamer Fehler in `fcd_design.py`
nicht beide Seiten trifft. Alles Exakte wird als Bruch verglichen."""

from fractions import Fraction as F


def coding(kind, k, ref=0, shares=None):
    """k × (k − 1)-Matrix (Zeilen = Stufen) als Brüche."""
    if kind == "Referenz":
        return [[F(1 if i == j else 0) for j in range(k) if j != ref] for i in range(k)]
    if kind == "Summe":
        return [[F(1 if i == j else (-1 if i == k - 1 else 0)) for j in range(k - 1)] for i in range(k)]
    if kind == "Helmert":
        cols = []
        for j in range(1, k):
            cols.append([F(-1) if i < j else (F(j) if i == j else F(0)) for i in range(k)])
        return [[cols[c][i] for c in range(k - 1)] for i in range(k)]
    if kind == "Gewichtet zentriert":
        cs = [j for j in range(k) if j != ref]
        return [[F(1 if i == j else 0) - F(shares[j]) for j in cs] for i in range(k)]
    raise ValueError(kind)


def design(a, b, ka, kb, kind, ref_a=0, ref_b=0, interaction=True):
    """Entwurfsmatrix in Brüchen; Spalten: Achsenabschnitt, A, B, A mal B; Gewichte der Codierung „gewichtet zentriert“ aus den Daten."""
    n = len(a)
    sa = [F(sum(1 for x in a if x == i), n) for i in range(ka)]
    sb = [F(sum(1 for x in b if x == j), n) for j in range(kb)]
    CA, CB = coding(kind, ka, ref_a, sa), coding(kind, kb, ref_b, sb)
    rows = []
    for ai, bi in zip(a, b):
        r = [F(1)] + CA[ai] + CB[bi]
        if interaction:
            r += [x * z for x in CA[ai] for z in CB[bi]]
        rows.append(r)
    pa, pb = ka - 1, kb - 1
    return rows, list(range(1, 1 + pa)), list(range(1 + pa, 1 + pa + pb)), list(range(1 + pa + pb, len(rows[0])))


def solve(A, bvec):
    n = len(A)
    M = [list(r) + [bb] for r, bb in zip(A, bvec)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * z for x, z in zip(M[r], M[c])]
    return [M[r][n] for r in range(n)]


def ls(X, y, cols=None):
    """Kleinste Quadrate über die Normalgleichung in Brüchen: Koeffizienten und Fehlerquadratsumme."""
    cols = list(range(len(X[0]))) if cols is None else cols
    Xs = [[row[c] for c in cols] for row in X]
    p = len(cols)
    yy = [F(v) for v in y]
    A = [[sum(Xs[i][r] * Xs[i][s] for i in range(len(yy))) for s in range(p)] for r in range(p)]
    bb = [sum(Xs[i][r] * yy[i] for i in range(len(yy))) for r in range(p)]
    beta = solve(A, bb)
    sse = sum((yy[i] - sum(Xs[i][r] * beta[r] for r in range(p))) ** 2 for i in range(len(yy)))
    return beta, sse


def f_drop(X, y, drop):
    """Exakter F-Wert (Bruch), Zähler- und Nenner-Freiheitsgrade der Streichung der Spalten `drop` aus dem vollen Modell."""
    keep = [c for c in range(len(X[0])) if c not in drop]
    _, s1 = ls(X, y)
    _, s0 = ls(X, y, keep)
    dfres = len(y) - len(X[0])
    return ((s0 - s1) / len(drop)) / (s1 / dfres), len(drop), dfres


def cells(a, b, y):
    cn, cs = {}, {}
    for ai, bi, yi in zip(a, b, y):
        cn[(ai, bi)] = cn.get((ai, bi), 0) + 1
        cs[(ai, bi)] = cs.get((ai, bi), 0) + F(yi)
    return cn, {k: cs[k] / cn[k] for k in cn}


def wald_f(cn, mu, sse, dfres, L, ka, kb):
    """F = (Lθ)' (L V L')⁻¹ (Lθ) / q / s² mit V = diag(1 / n_ij), θ = Zellmittel (Zeile i, Spalte j)."""
    cl = [(i, j) for i in range(ka) for j in range(kb)]
    th = [mu[c] for c in cl]
    q = len(L)
    Lt = [sum(L[r][k] * th[k] for k in range(len(cl))) for r in range(q)]
    M = [[sum(L[r][k] * L[s][k] / cn[cl[k]] for k in range(len(cl))) for s in range(q)] for r in range(q)]
    x = solve(M, Lt)
    return sum(Lt[r] * x[r] for r in range(q)) / q / (sse / dfres)


def rows_hypothesis(ka, kb, omega_rows):
    """Zeilen der Hypothese „gewichtete Zeilenmittel gleich“: omega_rows[i] = Gewichte der Stufen von B in Zeile i; verglichen wird jede Zeile i ≥ 1 mit Zeile 0."""
    L = []
    for i in range(1, ka):
        r = [F(0)] * (ka * kb)
        for j in range(kb):
            r[i * kb + j] = F(omega_rows[i][j])
            r[j] = -F(omega_rows[0][j])
        L.append(r)
    return L


def weekend_hypothesis(ka, kb, omega_a):
    """Eine Zeile: Σ_i ω_i (θ_i1 − θ_i0) (kb = 2)."""
    r = [F(0)] * (ka * kb)
    for i in range(ka):
        r[i * kb], r[i * kb + 1] = -F(omega_a[i]), F(omega_a[i])
    return [r]
