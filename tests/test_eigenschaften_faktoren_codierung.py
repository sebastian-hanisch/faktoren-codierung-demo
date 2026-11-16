"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für Codierungen und F-Tests gelten müssen, ohne dass man das richtige Ergebnis kennt. Je Eigenschaft 40 Zufallsinstanzen mit festem Zufalls-Seed,
auch ungewöhnliche (Zellen mit einer Beobachtung, sehr ungleiche Besetzung). Abweichungen relativ zu max(F, 1); Toleranz 1e-9."""

import numpy as np
import pytest

import fcd_constants as C
import fcd_design as D
import fcd_tests as T

KINDS = C.COD_KEYS
N_INST = 40


def instances(seed, ka=4, kb=2, lo=1, hi=12):
    rng = np.random.default_rng(seed)
    for _ in range(N_INST):
        cnt = rng.integers(lo, hi + 1, size=(ka, kb))
        if rng.random() < 0.3:                                       # sehr ungleich besetzt
            cnt[rng.integers(0, ka), rng.integers(0, kb)] = 1
            cnt[rng.integers(0, ka), rng.integers(0, kb)] = 25
        a = np.concatenate([[i] * cnt[i, j] for i in range(ka) for j in range(kb)])
        b = np.concatenate([[j] * cnt[i, j] for i in range(ka) for j in range(kb)])
        y = rng.normal(size=a.size) * 1.5 + 0.8 * a + 0.9 * b * (a % 2) + rng.normal() * 3
        perm = rng.permutation(a.size)
        yield a[perm], b[perm], y[perm], rng


def rel(x, y):
    return abs(x - y) / max(abs(x), abs(y), 1.0)


def test_dieselbe_anpassung_in_jeder_codierung():
    for a, b, y, rng in instances(1):
        fits = []
        for kind in KINDS:
            ra, rb = (int(rng.integers(0, 4)), int(rng.integers(0, 2))) if kind in ("Referenz", "Gewichtet zentriert") else (0, 0)
            d = D.build_design(a, b, 4, 2, kind, ra, rb)
            if D.full_rank(d.X):
                fits.append(D.fit(d.X, y)[1])
        assert len(fits) >= 3 and all(np.max(np.abs(f - fits[0])) < 1e-9 for f in fits)


def test_test_der_wechselwirkung_ist_in_jeder_codierung_gleich():
    for a, b, y, _ in instances(2):
        vals = []
        for kind in KINDS:
            d = D.build_design(a, b, 4, 2, kind)
            if D.full_rank(d.X) and len(y) > 8:
                vals.append(D.drop_test(d.X, y, d.i_ab)[0])
        assert len(vals) >= 3 and max(rel(v, vals[0]) for v in vals) < 1e-9


def test_summe_und_helmert_geben_denselben_f_wert_fuer_wochenende_und_art():
    for a, b, y, _ in instances(3):
        if len(y) <= 8:
            continue
        ds, dh = D.build_design(a, b, 4, 2, "Summe"), D.build_design(a, b, 4, 2, "Helmert")
        if not (D.full_rank(ds.X) and D.full_rank(dh.X)):
            continue
        assert rel(D.drop_test(ds.X, y, ds.i_b)[0], D.drop_test(dh.X, y, dh.i_b)[0]) < 1e-9
        assert rel(D.drop_test(ds.X, y, ds.i_a)[0], D.drop_test(dh.X, y, dh.i_a)[0]) < 1e-9


def test_additives_modell_f_des_haupteffekts_ist_in_allen_einstellungen_gleich():
    for a, b, y, _ in instances(4):
        fs = []
        for st in T.settings_list(4, C.ART_LABELS):
            d = D.build_design(a, b, 4, 2, st["kind"], st["ref_a"], st["ref_b"], interaction=False)
            if len(y) > d.X.shape[1]:
                fs.append(D.drop_test(d.X, y, d.i_b)[0])
        assert len(fs) == 8 and max(rel(v, fs[0]) for v in fs) < 1e-9


def test_umordnen_der_stufen_aendert_die_anpassung_nicht_und_kehrt_das_vorzeichen_des_summenkoeffizienten_um():
    for a, b, y, _ in instances(5):
        if len(y) <= 8:
            continue
        d1 = D.build_design(a, b, 4, 2, "Summe")
        d2 = D.build_design(a, 1 - b, 4, 2, "Summe")                 # Stufen von B vertauscht
        if not (D.full_rank(d1.X) and D.full_rank(d2.X)):
            continue
        b1, f1, _ = D.fit(d1.X, y)
        b2, f2, _ = D.fit(d2.X, y)
        assert np.max(np.abs(f1 - f2)) < 1e-9
        assert b1[d1.i_b[0]] == pytest.approx(-b2[d2.i_b[0]], abs=1e-9) and b1[0] == pytest.approx(b2[0], abs=1e-9)
        assert rel(D.drop_test(d1.X, y, d1.i_b)[0], D.drop_test(d2.X, y, d2.i_b)[0]) < 1e-9


def test_lineare_umrechnung_des_ausgangs_laesst_alle_f_werte_unveraendert_und_skaliert_die_koeffizienten():
    rng = np.random.default_rng(6)
    for a, b, y, _ in instances(6):
        if len(y) <= 8:
            continue
        c, shift = float(rng.uniform(0.2, 5)), float(rng.normal() * 4)
        for kind in KINDS:
            d = D.build_design(a, b, 4, 2, kind)
            if not D.full_rank(d.X):
                continue
            b1 = D.fit(d.X, y)[0]
            b2 = D.fit(d.X, c * y + shift)[0]
            assert rel(D.drop_test(d.X, y, d.i_b)[0], D.drop_test(d.X, c * y + shift, d.i_b)[0]) < 1e-8
            assert rel(D.drop_test(d.X, y, d.i_ab)[0], D.drop_test(d.X, c * y + shift, d.i_ab)[0]) < 1e-8
            assert np.allclose(b2[1:], c * b1[1:], atol=1e-8)          # Koeffizienten außer dem Achsenabschnitt skalieren mit c
            assert b2[0] == pytest.approx(c * b1[0] + shift, abs=1e-8)  # (die Spalten haben Mittelwert null nur bei Summe und gewichtet, der Achsenabschnitt verschiebt sich um shift)


def test_vertauschen_der_beobachtungen_aendert_nichts():
    rng = np.random.default_rng(7)
    for a, b, y, _ in instances(7):
        p = rng.permutation(len(y))
        d1, d2 = D.build_design(a, b, 4, 2, "Gewichtet zentriert"), D.build_design(a[p], b[p], 4, 2, "Gewichtet zentriert")
        if not D.full_rank(d1.X) or len(y) <= 8:
            continue
        assert np.max(np.abs(D.fit(d1.X, y)[0] - D.fit(d2.X, y[p])[0])) < 1e-9
        assert rel(D.drop_test(d1.X, y, d1.i_b)[0], D.drop_test(d2.X, y[p], d2.i_b)[0]) < 1e-9


def test_referenzcodierung_der_koeffizient_ist_der_effekt_bei_der_referenzart():
    for a, b, y, rng in instances(8):
        n, mean, _ = T.cell_stats(a, b, y, 4, 2)
        if np.any(n == 0):
            continue
        r = int(rng.integers(0, 4))
        d = D.build_design(a, b, 4, 2, "Referenz", r, 0)
        beta = D.fit(d.X, y)[0]
        assert beta[d.i_b[0]] == pytest.approx(mean[r, 1] - mean[r, 0], abs=1e-9)
        assert beta[0] == pytest.approx(mean[r, 0], abs=1e-9)


def test_summe_und_gewichtet_koeffizient_ist_das_ungewichtete_beziehungsweise_gewichtete_mittel_der_effekte():
    for a, b, y, _ in instances(9):
        n, mean, _ = T.cell_stats(a, b, y, 4, 2)
        if np.any(n == 0):
            continue
        delta = mean[:, 1] - mean[:, 0]
        ds = D.build_design(a, b, 4, 2, "Summe")
        assert D.fit(ds.X, y)[0][ds.i_b[0]] == pytest.approx(-0.5 * delta.mean(), abs=1e-9)
        dg = D.build_design(a, b, 4, 2, "Gewichtet zentriert")
        assert D.fit(dg.X, y)[0][dg.i_b[0]] == pytest.approx(float((n.sum(axis=1) / len(y)) @ delta), abs=1e-9)


def test_jede_zelle_mit_gleicher_zahl_beobachtungen_macht_typ_i_und_typ_iii_gleich():
    rng = np.random.default_rng(10)
    for _ in range(N_INST):
        m = int(rng.integers(2, 6))
        a = np.repeat(np.arange(4), 2 * m)
        b = np.tile(np.repeat(np.arange(2), m), 4)
        y = rng.normal(size=a.size) + 0.5 * a
        d = D.build_design(a, b, 4, 2, "Summe")
        assert rel(D.typ1_art_F(a, b, y, 4, 2)[0], D.drop_test(d.X, y, d.i_a)[0]) < 1e-9     # ausgewogen: dieselbe Hypothese


def test_skalierung_aller_zellmittel_und_der_streuung_laesst_die_exakte_teststaerke_unveraendert():
    theta = np.array([[1.4, 1.7], [1.5, 1.5], [0.9, 0.9], [1.9, 2.2]])
    n_cell = np.array([[50.0, 20], [43, 17], [36, 14], [14, 6]])
    for kind in KINDS:
        om = D.hypothesis_weights(kind, 4, 1, n_cell.sum(axis=1) / n_cell.sum())
        p1 = T.power_wochenende(theta, n_cell, 0.7, om)[0]
        p2 = T.power_wochenende(3.0 * theta, n_cell, 2.1, om)[0]
        assert p1 == pytest.approx(p2, rel=1e-9)
