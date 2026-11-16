"""Handbeispiele von Hand gerechnet (Bruch gleich Bruch) und gegen die App-Routine (Entwurfsmatrix und `lstsq`) gehalten.

Erfundene ganze Zahlen im Gewand von Art (Import, Export, Umschlag) und Wochenende, nicht aus dem Container-Generator, gewählt, damit man von Hand rechnen kann:
  H1: je Zelle zwei Beobachtungen m − 1 und m + 1; Zellmittel Wochentag 4 / 4 / 4, Wochenende 4 / 8 / 12; SSE 12, Rest 6 Freiheitsgrade, s² = 2.
  H2: dieselben Zellmittel, Zellgrößen Import 2 / 4, Export 3 / 1, Umschlag 2 / 2 (n = 14); SSE 12, Rest 8 Freiheitsgrade, s² = 3/2.
Die Erwartungswerte stehen als Rechnung im Test (zum Beispiel `48 / 2`), nie nur als Ergebnis des Codes."""

from fractions import Fraction as F

import numpy as np
import pytest
from scipy import stats

import exact_ref as X
import fcd_design as D
import fcd_evaluation as ev
import fcd_tests as T
from conftest import H1_VALUES, H2_VALUES, flatten

ART = ("Import", "Export", "Umschlag")


def _fit_float(a, b, y, kind, ref_a=0, ref_b=0, interaction=True):
    d = D.build_design(a, b, 3, 2, kind, ref_a, ref_b, ART, ("Wochentag", "Wochenende"), interaction)
    return d, D.fit(d.X, np.array(y, float))[0]


# --- H1 --------------------------------------------------------------------------------------------------------------------------------------

def test_h1_zellmittel_sse_und_wochenendeffekte(h1):
    a, b, y = h1
    cn, mu = X.cells(a, b, y)
    assert mu == {(0, 0): 4, (1, 0): 4, (2, 0): 4, (0, 1): 4, (1, 1): 8, (2, 1): 12} and len(y) == 12
    Xr, iA, iB, iI = X.design(a, b, 3, 2, "Referenz")
    beta, sse = X.ls(Xr, y)
    assert sse == 6 * 2 and len(y) - 6 == 6                                   # sechs Zellen mit je (m−1−m)² + (m+1−m)² = 2; Rest 12 − 6 Freiheitsgrade
    assert [mu[(i, 1)] - mu[(i, 0)] for i in range(3)] == [0, 4, 8]


def test_h1_referenz_koeffizient_ist_der_effekt_bei_der_referenzart_und_beide_haupteffekte_verschwinden(h1):
    a, b, y = h1
    Xr, iA, iB, iI = X.design(a, b, 3, 2, "Referenz")
    beta, _ = X.ls(Xr, y)
    assert beta[iB[0]] == 0 and beta[iA[0]] == 0 and beta[iA[1]] == 0          # Effekt bei Import = 0; Artunterschiede am Wochentag 4 − 4 = 0
    FB, qb, dfr = X.f_drop(Xr, y, iB)
    FA, qa, _ = X.f_drop(Xr, y, iA)
    assert FB == 0 and FA == 0 and (qb, qa, dfr) == (1, 2, 6)
    assert stats.f.sf(0, 1, 6) == 1.0


def test_h1_summe_f_wochenende_ist_24_und_f_art_ist_8(h1):
    a, b, y = h1
    Xs, iA, iB, iI = X.design(a, b, 3, 2, "Summe")
    beta, sse = X.ls(Xs, y)
    FB, _, _ = X.f_drop(Xs, y, iB)
    FA, _, _ = X.f_drop(Xs, y, iA)
    ss_b = 6 * (2 ** 2 + 2 ** 2)          # Wochenendeffekte 0, 4, 8: Mittel 4; je Zeile 4 Beobachtungen, Zeilenmittel am Wochentag 4, am Wochenende 8 → Abweichung ±2
    ss_a = 4 * ((-2) ** 2 + 0 ** 2 + 2 ** 2)   # Zeilenmittel 4, 6, 8, Gesamtmittel 6
    assert FB == F(ss_b, 1) / F(2) == 24
    assert FA == (F(ss_a) / 2) / F(2) == 8
    assert beta[iB[0]] == F(-(0 + 4 + 8), 3 * 2) == -2 and beta[0] == 6        # −(mittlerer Wochenendeffekt)/2, ungewichtetes Mittel der Zellmittel
    assert abs(stats.f.sf(24, 1, 6) - 0.0027) < 5e-4 and abs(stats.f.sf(8, 2, 6) - 0.0203) < 5e-4


def test_h1_wechselwirkung_ist_in_jeder_codierung_f_8(h1):
    a, b, y = h1
    for kind in ("Referenz", "Summe", "Helmert", "Gewichtet zentriert"):
        Xk, iA, iB, iI = X.design(a, b, 3, 2, kind)
        assert X.f_drop(Xk, y, iI)[0] == 8, kind
    Xs, _, _, iI = X.design(a, b, 3, 2, "Summe")
    Xadd = [row[: len(row) - len(iI)] for row in Xs]                          # additives Modell: ohne die zwei Wechselwirkungsspalten
    beta_add, sse_add = X.ls(Xadd, y)
    _, sse_full = X.ls(Xs, y)
    resid = [F(y[k]) - sum(Xadd[k][c] * beta_add[c] for c in range(len(beta_add))) for k in range(len(y))]
    # je Zelle zwei Beobachtungen m ∓ 1: die Wechselwirkungsreste der Zellmittel sind (2, −2, 0, 0, −2, 2) (Import, Export, Umschlag je Wochentag und Wochenende), SS = 2 · (4 + 4 + 0 + 0 + 4 + 4) = 32
    cell_res = [sum(resid[k] for k in range(len(y)) if (a[k], b[k]) == (i, j)) / 2 for i in range(3) for j in range(2)]
    assert cell_res == [2, -2, 0, 0, -2, 2]
    assert sse_add - sse_full == 2 * (4 + 4 + 0 + 0 + 4 + 4) == 32
    assert (F(32) / 2) / (F(12) / 6) == 8                                    # (32 / 2 Freiheitsgrade) / (12 / 6) = 16 / 2


def test_h1_zweiter_rechenweg_wald_gibt_denselben_bruch(h1):
    a, b, y = h1
    cn, mu = X.cells(a, b, y)
    sse, dfres = F(12), 6
    assert X.wald_f(cn, mu, sse, dfres, X.weekend_hypothesis(3, 2, [F(1, 3)] * 3), 3, 2) == 24
    assert X.wald_f(cn, mu, sse, dfres, X.weekend_hypothesis(3, 2, [1, 0, 0]), 3, 2) == 0
    assert X.wald_f(cn, mu, sse, dfres, X.rows_hypothesis(3, 2, [[F(1, 2), F(1, 2)]] * 3), 3, 2) == 8
    assert X.wald_f(cn, mu, sse, dfres, X.rows_hypothesis(3, 2, [[1, 0]] * 3), 3, 2) == 0


def test_h1_die_app_rechnet_dieselben_zahlen(h1):
    a, b, y = h1
    h = ev.make_hand(tuple(len(H1_VALUES[(i, j)]) for i in range(3) for j in range(2)))
    cmp = ev.compare_codings(h)
    rows = {r["name"]: r for r in cmp["rows"]}
    assert rows["Referenz Import"]["F_B"] == pytest.approx(0, abs=1e-9) and rows["Referenz Import"]["p_B"] == pytest.approx(1.0)
    assert rows["Summe"]["F_B"] == pytest.approx(24, rel=1e-9) and rows["Summe"]["F_A"] == pytest.approx(8, rel=1e-9)
    assert rows["Summe"]["p_B"] == pytest.approx(0.0027, abs=5e-4) and rows["Summe"]["p_A"] == pytest.approx(0.0203, abs=5e-4)
    assert rows["Gewichtet zentriert"]["F_B"] == pytest.approx(24, rel=1e-9)
    assert cmp["F_I"] == pytest.approx(8, rel=1e-9)


# --- H2 --------------------------------------------------------------------------------------------------------------------------------------

def test_h2_koeffizienten_wochenende_0_minus_2_und_24_siebtel(h2):
    a, b, y = h2
    n = len(y)
    assert n == 14
    cn, mu = X.cells(a, b, y)
    assert all(mu[k] == v for k, v in {(0, 0): 4, (1, 0): 4, (2, 0): 4, (0, 1): 4, (1, 1): 8, (2, 1): 12}.items())
    Xr, _, iB, _ = X.design(a, b, 3, 2, "Referenz")
    _, sse = X.ls(Xr, y)
    assert sse == 2 + 4 + 2 + 0 + 2 + 2 == 12 and n - 6 == 8                  # s² = 12 / 8 = 3/2
    assert [F(sum(1 for x in a if x == i), n) for i in range(3)] == [F(3, 7), F(2, 7), F(2, 7)]
    assert X.ls(Xr, y)[0][iB[0]] == 0
    Xs, _, iBs, _ = X.design(a, b, 3, 2, "Summe")
    assert X.ls(Xs, y)[0][iBs[0]] == F(-(0 + 4 + 8), 6)                        # −2
    Xg, _, iBg, _ = X.design(a, b, 3, 2, "Gewichtet zentriert")
    assert X.ls(Xg, y)[0][iBg[0]] == F(3, 7) * 0 + F(2, 7) * 4 + F(2, 7) * 8 == F(24, 7)


def test_h2_f_wochenende_je_codierung(h2):
    a, b, y = h2
    for kind, ref, expected in (("Referenz", 0, F(0)), ("Referenz", 1, F(8)), ("Referenz", 2, F(128, 3)), ("Summe", 0, F(1152, 37)), ("Gewichtet zentriert", 0, F(4608, 193))):
        Xk, _, iB, _ = X.design(a, b, 3, 2, kind, ref)
        assert X.f_drop(Xk, y, iB)[0] == expected, (kind, ref)
    assert float(F(1152, 37)) == pytest.approx(31.135, abs=1e-3) and float(F(4608, 193)) == pytest.approx(23.876, abs=1e-3)
    # von Hand: Var(Summe) = s² · Σ ω² (1/n_We + 1/n_Wt) mit ω = 1/3: (3/2)/9 · (1/2+1/4 + 1/3+1 + 1/2+1/2) = 37/72
    assert F(3, 2) / 9 * (F(1, 2) + F(1, 4) + F(1, 3) + F(1, 1) + F(1, 2) + F(1, 2)) == F(37, 72)
    assert F(4) ** 2 / F(37, 72) == F(1152, 37)                                # Effekt 4 (Mittel von 0, 4, 8), (4)² / (37/72)
    assert F(24, 7) ** 2 / F(193, 392) == F(4608, 193)                         # Var(gewichtet) = 193/392


def test_h2_typ_i_und_typ_iii_art(h2):
    a, b, y = h2
    cn, mu = X.cells(a, b, y)
    sse, dfres = F(12), 8
    w_typ1 = [[F(cn[(i, 0)], cn[(i, 0)] + cn[(i, 1)]), F(cn[(i, 1)], cn[(i, 0)] + cn[(i, 1)])] for i in range(3)]
    assert X.wald_f(cn, mu, sse, dfres, X.rows_hypothesis(3, 2, w_typ1), 3, 2) == F(92, 7)                      # Zeilenmittel 4, 5, 8 (nach Zellgrößen gewichtet)
    assert X.wald_f(cn, mu, sse, dfres, X.rows_hypothesis(3, 2, [[F(1, 2), F(1, 2)]] * 3), 3, 2) == F(1360, 111)  # Zeilenmittel 4, 6, 8 (ungewichtet)
    h = ev.make_hand((2, 4, 3, 1, 2, 2))
    assert h.n == 14
    cmp = ev.compare_codings(h)
    assert cmp["typ1"][0] == pytest.approx(float(F(92, 7)), rel=1e-9) and cmp["typ3"][0] == pytest.approx(float(F(1360, 111)), rel=1e-9)
    assert cmp["typ1"][0] != pytest.approx(cmp["typ3"][0], rel=1e-3)
    rows = {r["name"]: r for r in cmp["rows"]}
    assert rows["Summe"]["F_B"] == pytest.approx(float(F(1152, 37)), rel=1e-9) and rows["Gewichtet zentriert"]["F_B"] == pytest.approx(float(F(4608, 193)), rel=1e-9)
    assert rows["Referenz Umschlag"]["F_B"] == pytest.approx(float(F(128, 3)), rel=1e-9) and rows["Referenz Export"]["F_B"] == pytest.approx(8, rel=1e-9)
    assert stats.f.sf(8, 1, 8) < 0.05 < stats.f.sf(4, 1, 8)


# --- H3: additives Modell ----------------------------------------------------------------------------------------------------------------------

def test_h3_additives_modell_f_gleich_in_jeder_codierung_und_koeffizient_4_beziehungsweise_minus_2(h1):
    a, b, y = h1
    fs, coefs = {}, {}
    for kind in ("Referenz", "Summe", "Helmert", "Gewichtet zentriert"):
        Xk, iA, iB, _ = X.design(a, b, 3, 2, kind, interaction=False)
        fs[kind] = X.f_drop(Xk, y, iB)[0]
        coefs[kind] = X.ls(Xk, y)[0][iB[0]]
    assert len(set(fs.values())) == 1 and fs["Referenz"] == F(96, 11)
    assert coefs["Referenz"] == 4 and coefs["Summe"] == -2                    # Effekt im ausgewogenen Mittel (nicht bei Import); die Hälfte mit anderem Vorzeichen
    h = ev.make_hand((2,) * 6)
    cmp = ev.compare_codings(h, interaction=False)
    assert all(r["F_B"] == pytest.approx(96 / 11, rel=1e-9) for r in cmp["rows"]) and cmp["fb_dev"] < 1e-9


# --- H4: Dummy-Falle ---------------------------------------------------------------------------------------------------------------------------

def test_h4_nullvektor_und_drei_loesungen_mit_derselben_anpassung():
    av, yv = [], []
    for i, m in enumerate((4, 6, 8)):
        for d in (-1, 1):
            av.append(i), yv.append(F(m + d))
    D4 = [[F(1)] + [F(1 if a == j else 0) for j in range(3)] for a in av]
    v = [F(1), F(-1), F(-1), F(-1)]
    assert all(sum(D4[i][c] * v[c] for c in range(4)) == 0 for i in range(6))
    AtA = [[sum(D4[i][r] * D4[i][s] for i in range(6)) for s in range(4)] for r in range(4)]
    assert all(sum(AtA[r][c] * v[c] for c in range(4)) == 0 for r in range(4))
    b_ref, _ = X.ls(D4, yv, [0, 2, 3])                                          # Spalte der ersten Art gestrichen = Referenzcodierung
    assert b_ref == [4, 2, 4]
    b0 = [F(0), F(4), F(6), F(8)]
    t = -sum(b0[c] * v[c] for c in range(4)) / sum(x * x for x in v)
    bmin = [b0[c] + t * v[c] for c in range(4)]
    assert bmin == [F(9, 2), F(-1, 2), F(3, 2), F(7, 2)]                       # Lösung kleinster Norm
    fits = [[sum(D4[i][c] * bb[c] for c in range(4)) for i in range(6)] for bb in (bmin, [F(4), F(0), F(2), F(4)], [F(6), F(-2), F(0), F(2)])]
    assert fits[0] == fits[1] == fits[2] == [F(x) for x in (4, 4, 6, 6, 8, 8)]
    bl = np.linalg.lstsq(np.array([[float(x) for x in r] for r in D4]), np.array([float(x) for x in yv]), rcond=None)[0]
    assert np.max(np.abs(bl - np.array([float(x) for x in bmin]))) < 1e-12


# --- H5: Konventionen --------------------------------------------------------------------------------------------------------------------------

def test_h5_summencodierung_zweier_stufen_und_vertauschen_der_stufen(h1):
    assert X.coding("Summe", 2) == [[1], [-1]]
    a, b, y = h1
    Xa = [[F(1)] + X.coding("Summe", 2)[bi] for bi in b]
    Xb = [[F(1)] + X.coding("Summe", 2)[1 - bi] for bi in b]
    ba, bb = X.ls(Xa, y)[0], X.ls(Xb, y)[0]
    assert ba[1] == -bb[1] and ba[0] == bb[0]
    assert np.array_equal(D.coding_matrix("Summe", 2), np.array([[1.0], [-1.0]]))
