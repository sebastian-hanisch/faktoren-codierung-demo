"""Codierung von Faktoren, Entwurfsmatrix, Kleinste-Quadrate-Anpassung und F-Test der Streichung von Spalten (der Rechenweg der App; Wald-Tests in Zellmittel-Parametern stehen in `fcd_tests.py`).

Zwei Faktoren: A (Art, k_A Stufen) und B (Wochenende, k_B Stufen) mit Wechselwirkung. Spalten: Achsenabschnitt, (k_A − 1) Spalten für A, (k_B − 1) für B, (k_A − 1)(k_B − 1) Produktspalten.
„Haupteffekt-Test“ heißt hier immer: F-Test der Streichung der Haupteffektspalten bei erhaltenen Wechselwirkungsspalten.

Codierungen (Spalten eines Faktors mit k Stufen, `coding_matrix` liefert die k × (k − 1)-Matrix, Zeile = Stufe):
  Referenz   Punktvariablen ohne die Referenzstufe r
  Summe      Stufe j (j < k − 1) in Spalte j mit +1, die LETZTE Stufe in jeder Spalte mit −1 (Zeilen summieren sich zu null)
  Helmert    Spalte j − 1: Stufe j mit +j, alle früheren Stufen mit −1 (Stufe j gegen das Mittel der vorigen, nicht skaliert)
  Gewichtet  Referenz-Punktvariablen minus ihr Stichprobenanteil (am Anteil zentriert; eigene Bezeichnung, nicht gleich einem Namen aus einem Programm behauptet)"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

import fcd_constants as C


class RankError(ValueError):
    """Die Entwurfsmatrix hat nicht den vollen Spaltenrang (zum Beispiel eine leere Zelle im Modell mit Wechselwirkung)."""


def coding_matrix(kind: str, k: int, ref: int = 0, shares=None) -> np.ndarray:
    """k × (k − 1)-Matrix der Codierung; `ref` ist die ausgelassene Stufe (Referenz, Gewichtet), `shares` die Stichprobenanteile (Gewichtet)."""
    if kind == C.COD_REFERENZ:
        return np.eye(k)[:, [j for j in range(k) if j != ref]]
    if kind == C.COD_SUMME:
        m = np.zeros((k, k - 1))
        for j in range(k - 1):
            m[j, j] = 1.0
            m[k - 1, j] = -1.0
        return m
    if kind == C.COD_HELMERT:
        m = np.zeros((k, k - 1))
        for j in range(1, k):
            m[:j, j - 1] = -1.0
            m[j, j - 1] = float(j)
        return m
    if kind == C.COD_GEWICHTET:
        cols = [j for j in range(k) if j != ref]
        return np.eye(k)[:, cols] - np.asarray(shares, dtype=float)[cols][None, :]
    raise ValueError(kind)


def hypothesis_weights(kind: str, k: int, ref: int = 0, shares=None) -> np.ndarray:
    """Gewichte ω, mit denen der Haupteffekt-Test des ANDEREN Faktors dessen Stufen mittelt: Referenz = die Referenzstufe allein, Summe und Helmert = ungewichtetes Mittel,
    Gewichtet = Stichprobenanteile. (Der Koeffizient eines Haupteffekts ist sein Effekt an der Stelle, an der alle Spalten des anderen Faktors null sind.)"""
    if kind == C.COD_REFERENZ:
        w = np.zeros(k)
        w[ref] = 1.0
        return w
    if kind in (C.COD_SUMME, C.COD_HELMERT):
        return np.full(k, 1.0 / k)
    if kind == C.COD_GEWICHTET:
        return np.asarray(shares, dtype=float).copy()
    raise ValueError(kind)


def column_labels(kind: str, levels, ref: int = 0, factor: str = "Art") -> list[str]:
    """Namen der k − 1 Spalten eines Faktors."""
    k = len(levels)
    if kind == C.COD_REFERENZ:
        return [f"{factor} {levels[j]}" for j in range(k) if j != ref]
    if kind == C.COD_GEWICHTET:
        return [f"{factor} {levels[j]} (zentriert)" for j in range(k) if j != ref]
    if kind == C.COD_SUMME:
        return [f"{factor} {levels[j]} (Summe)" for j in range(k - 1)]
    if kind == C.COD_HELMERT:
        return [f"{factor} {levels[j]} gegen " + (levels[0] if j == 1 else "Mittel von " + ", ".join(levels[:j])) for j in range(1, k)]
    raise ValueError(kind)


@dataclass(frozen=True)
class Design:
    X: np.ndarray
    names: tuple
    i_int: tuple
    i_a: tuple
    i_b: tuple
    i_ab: tuple
    kind: str
    ref_a: int
    ref_b: int


def build_design(a, b, ka: int, kb: int, kind: str, ref_a: int = 0, ref_b: int = 0, a_levels=None, b_levels=C.WE_LABELS, interaction: bool = True) -> Design:
    """Entwurfsmatrix zu den Stufen-Indizes a und b. Gewichtet: die Anteile kommen aus den Daten (a, b)."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    a_levels = a_levels or tuple(str(i) for i in range(ka))
    n = a.size
    sh_a = np.bincount(a, minlength=ka) / n if n else np.zeros(ka)
    sh_b = np.bincount(b, minlength=kb) / n if n else np.zeros(kb)
    CA = coding_matrix(kind, ka, ref_a, sh_a)[a]
    CB = coding_matrix(kind, kb, ref_b, sh_b)[b]
    cols = [np.ones((n, 1)), CA, CB]
    la, lb = column_labels(kind, a_levels, ref_a, "Art"), column_labels(kind, b_levels, ref_b, "")
    lb = [s.strip() for s in lb]
    names = ["Achsenabschnitt"] + la + lb
    if interaction:
        for i in range(CA.shape[1]):
            for j in range(CB.shape[1]):
                cols.append((CA[:, i] * CB[:, j])[:, None])
                names.append(f"{la[i]} × {lb[j]}")
    X = np.hstack(cols)
    p_a, p_b = CA.shape[1], CB.shape[1]
    return Design(X, tuple(names), (0,), tuple(range(1, 1 + p_a)), tuple(range(1 + p_a, 1 + p_a + p_b)), tuple(range(1 + p_a + p_b, X.shape[1])), kind, ref_a, ref_b)


def cell_design(ka: int, kb: int, kind: str, ref_a: int = 0, ref_b: int = 0, shares_a=None, shares_b=None, a_levels=None) -> Design:
    """Entwurfsmatrix der Zellen (eine Zeile je Zelle, Reihenfolge Stufe von A mal Stufe von B; k_A·k_B Zeilen und Spalten im Modell mit Wechselwirkung)."""
    a = np.repeat(np.arange(ka), kb)
    b = np.tile(np.arange(kb), ka)
    CA = coding_matrix(kind, ka, ref_a, shares_a)[a]
    CB = coding_matrix(kind, kb, ref_b, shares_b)[b]
    la, lb = column_labels(kind, a_levels or tuple(str(i) for i in range(ka)), ref_a, "Art"), [s.strip() for s in column_labels(kind, C.WE_LABELS[:kb], ref_b, "")]
    cols = [np.ones((a.size, 1)), CA, CB] + [(CA[:, i] * CB[:, j])[:, None] for i in range(CA.shape[1]) for j in range(CB.shape[1])]
    X = np.hstack(cols)
    names = ["Achsenabschnitt"] + la + lb + [f"{la[i]} × {lb[j]}" for i in range(len(la)) for j in range(len(lb))]
    p_a, p_b = len(la), len(lb)
    return Design(X, tuple(names), (0,), tuple(range(1, 1 + p_a)), tuple(range(1 + p_a, 1 + p_a + p_b)), tuple(range(1 + p_a + p_b, X.shape[1])), kind, ref_a, ref_b)


def smallest_relative_singular_value(X: np.ndarray) -> float:
    s = np.linalg.svd(X, compute_uv=False)
    return float(s[-1] / s[0]) if s[0] > 0 else 0.0


def full_rank(X: np.ndarray) -> bool:
    return X.shape[0] >= X.shape[1] and smallest_relative_singular_value(X) > C.SING_TOL


def fit(X: np.ndarray, y: np.ndarray):
    """Kleinste Quadrate per `numpy.linalg.lstsq`: Koeffizienten, angepasste Werte, Fehlerquadratsumme. Verlangt vollen Spaltenrang (sonst RankError, nie eine beliebige Lösung)."""
    if not full_rank(X):
        raise RankError("Entwurfsmatrix ohne vollen Spaltenrang (zum Beispiel eine leere Zelle)")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    r = y - fitted
    return beta, fitted, float(r @ r)


def sse(X: np.ndarray, y: np.ndarray) -> float:
    """Fehlerquadratsumme der Anpassung; bei Rangmangel eines (kleineren) Modells wählt lstsq die Lösung kleinster Norm, die Anpassung ist trotzdem eindeutig."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    return float(r @ r)


def drop_test(X: np.ndarray, y: np.ndarray, drop, sse_full: float | None = None):
    """F-Test der Streichung der Spalten `drop` aus dem vollen Modell: (F, p, q, Freiheitsgrade des Rests). F und p sind nan, wenn der Rest keine Freiheitsgrade hat."""
    n, p = X.shape
    dfres = n - p
    q = len(drop)
    if sse_full is None:
        sse_full = fit(X, y)[2]
    if dfres <= 0 or q == 0:
        return float("nan"), float("nan"), q, dfres
    keep = [i for i in range(p) if i not in set(drop)]
    s0 = sse(X[:, keep], y)
    if sse_full <= 0.0:
        return (float("inf") if s0 > 0 else float("nan")), (0.0 if s0 > 0 else float("nan")), q, dfres
    F = max(((s0 - sse_full) / q) / (sse_full / dfres), 0.0)
    return F, float(stats.f.sf(F, q, dfres)), q, dfres


def coefficient_table(d: Design, y: np.ndarray):
    """Koeffizienten, Standardfehler, t-Werte und p-Werte des vollen Modells (nur bei vollem Rang, Rest mit Freiheitsgraden)."""
    X = d.X
    beta, fitted, sse_full = fit(X, y)
    n, p = X.shape
    dfres = n - p
    if dfres <= 0:
        nan = np.full(p, np.nan)
        return beta, nan, nan, nan, dfres
    cov = sse_full / dfres * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    pv = 2.0 * stats.t.sf(np.abs(t), dfres)
    return beta, se, t, pv, dfres


def cell_weights(d_cells: Design) -> np.ndarray:
    """Gewichte der Zellmittel für jeden Koeffizienten im Modell mit Wechselwirkung (volles Modell mit k_A·k_B Spalten): Koeffizient = Σ Gewicht × Zellmittel.
    Zeile c ist die c-te Zeile der Inversen der Zell-Entwurfsmatrix; die Anpassung des Modells sind die Zellmittel, der Koeffizient folgt aus ihnen allein."""
    return np.linalg.inv(d_cells.X)


def typ1_art_F(a, b, y, ka: int, kb: int):
    """Typ I (A zuerst) des Art-Haupteffekts: Modell nur mit Achsenabschnitt gegen Modell mit Art; Nenner aus dem vollen Modell mit Wechselwirkung (Summencodierung)."""
    d = build_design(a, b, ka, kb, C.COD_SUMME)
    _, _, s_full = fit(d.X, y)
    n = len(y)
    dfres = n - d.X.shape[1]
    s1 = sse(np.ones((n, 1)), y)
    sa = sse(d.X[:, [0, *d.i_a]], y)
    if dfres <= 0 or s_full <= 0:
        return float("nan"), float("nan")
    F = ((s1 - sa) / (ka - 1)) / (s_full / dfres)
    return F, float(stats.f.sf(F, ka - 1, dfres))
