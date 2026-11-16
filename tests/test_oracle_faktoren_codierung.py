"""Orakel-Tests (anderer Rechenweg) für Codierung, F-Tests, Teststärke, Rangmangel, Messreihen und die Sätze zu den Koeffizienten (ORACLE-PLAYBOOK §2, §5, §6).

Orakel: (1) Normalgleichung in Brüchen (`tests/exact_ref.py`) gegen den Löser der App (Entwurfsmatrix und `lstsq`); (2) Wald-Test in Zellmittel-Parametern gegen den Modellvergleichs-F, Bruch gleich Bruch
und auf Zufallsdaten; (3) Typ I und Typ III gegen den Wald-Test der gewichteten und ungewichteten Zeilenmittel; (4) simulierte Häufigkeit gegen die nichtzentrale F-Verteilung und gegen eine eigene
Simulation mit festen Zellgrößen (numpy-Zufall nur hier, im Test); (5) `lstsq`, `pinv`, QR mit Spaltenwahl gegen die aufgezählte Lösungsgerade; (6) Koeffizient gleich Inverse der Zellmatrix mal Zellmittel;
(7) die gestapelte Messreihe gegen Wiederholung für Wiederholung mit der Entwurfsmatrix. Gleitkomma-Vergleiche mit Toleranz (letzte Binärstelle ist plattformabhängig), Brüche exakt."""

from fractions import Fraction as F

import numpy as np
import pytest
from scipy import linalg as sla
from scipy import stats

import exact_ref as X
import fcd_constants as C
import fcd_design as D
import fcd_evaluation as ev
import fcd_generator as G
import fcd_simulation as SIM
import fcd_tests as T

KINDS = ("Referenz", "Summe", "Helmert", "Gewichtet zentriert")


def rand_data(rng, ka, kb, lo=2, hi=6, integer=True):
    """Zufällige unausgewogene Daten: je Zelle lo bis hi Beobachtungen."""
    cnt = rng.integers(lo, hi + 1, size=(ka, kb))
    a = np.concatenate([[i] * cnt[i, j] for i in range(ka) for j in range(kb)])
    b = np.concatenate([[j] * cnt[i, j] for i in range(ka) for j in range(kb)])
    y = rng.integers(0, 20, size=a.size) if integer else rng.normal(size=a.size) * 2 + a * 0.7 + b * 0.5
    perm = rng.permutation(a.size)
    return a[perm], b[perm], y[perm]


# --- (1) Normalgleichung in Brüchen gegen den Löser der App --------------------------------------------------------------------------------------

@pytest.mark.parametrize("ka", [2, 3, 4])
def test_koeffizienten_und_sse_in_bruechen_gegen_lstsq(ka):
    rng = np.random.default_rng(130101 + ka)
    for k in range(10):
        a, b, y = rand_data(rng, ka, 2)
        kind = KINDS[k % 4]
        ref_a, ref_b = (int(rng.integers(0, ka)), int(rng.integers(0, 2))) if kind in ("Referenz", "Gewichtet zentriert") else (0, 0)
        Xf, iA, iB, iI = X.design(list(a), list(b), ka, 2, kind, ref_a, ref_b)
        beta_x, sse_x = X.ls(Xf, [int(v) for v in y])
        d = D.build_design(a, b, ka, 2, kind, ref_a, ref_b)
        beta, _, sse = D.fit(d.X, y.astype(float))
        assert np.max(np.abs(beta - np.array([float(v) for v in beta_x]))) < 1e-9, (ka, kind)
        assert sse == pytest.approx(float(sse_x), rel=1e-9, abs=1e-9)
        FB_x = X.f_drop(Xf, [int(v) for v in y], iB)[0]
        FB, _, q, dfres = D.drop_test(d.X, y.astype(float), d.i_b)
        assert FB == pytest.approx(float(FB_x), rel=1e-9, abs=1e-9) and q == 1 and dfres == len(y) - 2 * ka


# --- (2) Wald-Test in Zellmittel-Parametern gegen den Modellvergleichs-F --------------------------------------------------------------------------

def test_wald_gegen_modellvergleich_bruch_gleich_bruch():
    rng = np.random.default_rng(130201)
    for _ in range(15):
        a, b, y = rand_data(rng, 4, 2)
        al, bl, yl = list(a), list(b), [int(v) for v in y]
        cn, mu = X.cells(al, bl, yl)
        n = len(yl)
        _, sse = X.ls(X.design(al, bl, 4, 2, "Referenz")[0], yl)
        dfres = n - 8
        sa = [F(sum(1 for v in al if v == i), n) for i in range(4)]
        for kind, ref, om_a in (("Referenz", 1, [0, 1, 0, 0]), ("Summe", 0, [F(1, 4)] * 4), ("Gewichtet zentriert", 0, sa)):
            Xk, _, iB, _ = X.design(al, bl, 4, 2, kind, ref)
            assert X.wald_f(cn, mu, sse, dfres, X.weekend_hypothesis(4, 2, om_a), 4, 2) == X.f_drop(Xk, yl, iB)[0], kind
        Xs, iA, _, iI = X.design(al, bl, 4, 2, "Summe")
        assert X.wald_f(cn, mu, sse, dfres, X.rows_hypothesis(4, 2, [[F(1, 2), F(1, 2)]] * 4), 4, 2) == X.f_drop(Xs, yl, iA)[0]
        L = []
        for i in range(1, 4):
            r = [F(0)] * 8
            r[i * 2], r[i * 2 + 1], r[0], r[1] = F(-1), F(1), F(1), F(-1)
            L.append(r)
        assert X.wald_f(cn, mu, sse, dfres, L, 4, 2) == X.f_drop(Xs, yl, iI)[0]


def test_wald_in_fcd_tests_gegen_modellvergleich_der_app_auf_zufallsdaten():
    rng = np.random.default_rng(130202)
    for _ in range(25):
        a, b, y = rand_data(rng, 4, 2, 1, 9, integer=False)
        n, mean, within = T.cell_stats(a, b, y, 4, 2)
        sh_a, sh_b = n.sum(axis=1) / len(y), n.sum(axis=0) / len(y)
        for st in T.settings_list(4, C.ART_LABELS):
            d = D.build_design(a, b, 4, 2, st["kind"], st["ref_a"], st["ref_b"])
            oa, ob = T.setting_weights(st, sh_a, sh_b)
            FB = D.drop_test(d.X, y, d.i_b)[0]
            FA = D.drop_test(d.X, y, d.i_a)[0]
            assert T.wochenende_test(n, mean, within, oa)[0] == pytest.approx(FB, rel=1e-9, abs=1e-12), st["name"]
            assert T.art_test(n, mean, within, ob)[0] == pytest.approx(FA, rel=1e-9, abs=1e-12), st["name"]
        d = D.build_design(a, b, 4, 2, "Summe")
        assert T.interaction_test(n, mean, within)[0] == pytest.approx(D.drop_test(d.X, y, d.i_ab)[0], rel=1e-9, abs=1e-12)


# --- (3) Typ I und Typ III gegen den Wald-Test der gewichteten und ungewichteten Zeilenmittel -----------------------------------------------------------

def test_typ_i_und_typ_iii_gegen_wald_der_zeilenmittel():
    rng = np.random.default_rng(130701)
    differ = 0
    for _ in range(100):
        a, b, y = rand_data(rng, 4, 2, 3, 11, integer=False)
        n, mean, within = T.cell_stats(a, b, y, 4, 2)
        F1 = D.typ1_art_F(a, b, y, 4, 2)[0]
        d = D.build_design(a, b, 4, 2, "Summe")
        F3 = D.drop_test(d.X, y, d.i_a)[0]
        assert T.art_test(n, mean, within, T.typ1_row_weights(n))[0] == pytest.approx(F1, rel=1e-9)
        assert T.art_test(n, mean, within, np.array([0.5, 0.5]))[0] == pytest.approx(F3, rel=1e-9)
        differ += abs(F1 - F3) > 1e-6 * max(F1, F3)
    assert differ == 100


# --- (4) Teststärke ---------------------------------------------------------------------------------------------------------------------------------

def test_nichtzentrale_f_gegen_eigene_simulation_mit_festen_zellgroessen():
    """Feste Zellgrößen (keine Zufallsbesetzung), gleiche Normalstreuung: die Häufigkeit der Verwerfung gegen `power_wochenende`, höchstens 3 Standardfehler; numpy-Zufall nur im Test."""
    rng = np.random.default_rng(130401)
    theta = G.cell_means_exact(C.W_GEN)
    sigma = G.sd_pool_exact()
    n_cell = np.array([[40, 16], [34, 14], [28, 12], [12, 8]], dtype=float)
    cell_a = np.concatenate([[i] * int(n_cell[i, j]) for i in range(4) for j in range(2)])
    cell_b = np.concatenate([[j] * int(n_cell[i, j]) for i in range(4) for j in range(2)])
    reps = 3000
    for kind, ref in (("Referenz", 0), ("Summe", 0), ("Referenz", 3)):
        st = {"kind": kind, "ref_a": ref, "ref_b": 0}
        oa = D.hypothesis_weights(kind, 4, ref)
        exact = T.power_wochenende(theta, n_cell, sigma, oa)[0]
        d = D.build_design(cell_a, cell_b, 4, 2, kind, ref, 0)
        rej = 0
        for _ in range(reps):
            y = theta[cell_a, cell_b] + sigma * rng.standard_normal(cell_a.size)
            rej += D.drop_test(d.X, y, d.i_b)[1] < 0.05
        se = np.sqrt(exact * (1 - exact) / reps)
        assert abs(rej / reps - exact) < 3 * se + 1e-9, (kind, ref, rej / reps, exact)


def test_teststaerke_der_vorgerechneten_zeilen_gegen_die_exakte_formel():
    """Streuung „gleich“: Simulation gegen nichtzentrale F mit erwarteten Zellgrößen. Gemessen: größte Abweichung 2,0 Prozentpunkte, 62 von 72 Zeilen innerhalb 3 Standardfehler
    (die Formel setzt die erwarteten statt der zufälligen Zellgrößen ein und liegt dadurch bei Summe und gewichtet etwas über der Häufigkeit)."""
    inside3 = total = 0
    worst = 0.0
    for r in ev.load_frozen()["zeilen"]:
        if r["mode"] != "gleich":
            continue
        ex = SIM.exact_power_row(r["mode"], r["szenario"], r["n"])
        for k, e in enumerate(ex):
            p, ph = e["power_B"], r["rej_B"][k] / r["valid"]
            se = np.sqrt(max(p * (1 - p), 0.0025) / r["valid"])
            inside3 += abs(ph - p) <= 3 * se
            total += 1
            worst = max(worst, abs(ph - p))
    assert total == 72 and inside3 >= 55 and worst < 0.025


def test_nichtzentrale_f_bei_null_ist_das_niveau_und_stetig():
    assert T.noncentral_power(1, 100, 0.0) == 0.05
    assert stats.f.sf(stats.f.ppf(0.95, 1, 100), 1, 100) == pytest.approx(0.05)
    last = 0.05
    for lam in (1e-9, 1e-6, 1e-3, 0.1, 1.0, 4.0, 10.0, 30.0):
        pw = T.noncentral_power(1, 100, lam)
        assert last - 1e-9 <= pw <= 1.0 and pw >= 0.05 - 1e-9, lam
        last = pw
    assert T.noncentral_power(3, 60, 2.5) == pytest.approx(stats.ncf.sf(stats.f.ppf(0.95, 3, 60), 3, 60, 2.5))


# --- (5) Dummy-Falle: Lösungsgerade -----------------------------------------------------------------------------------------------------------------

def test_lstsq_pinv_qr_und_solve_liegen_auf_der_loesungsgeraden():
    for seed in (7003, 7004, 123456):
        res = ev.dummy_trap(ev.Settings(n=200, seed=seed))
        assert res.rank == 4 and res.columns == 5 and res.smallest_sv < 1e-10
        v = res.null_vector
        base = res.solutions["lstsq (kleinste Norm)"]
        for name, sol in res.solutions.items():
            if sol is None:
                continue
            diff = sol - base
            t = diff @ v / (v @ v)
            assert np.max(np.abs(diff - t * v)) < 1e-6 * max(1.0, np.max(np.abs(sol))), name     # jede Lösung = Lösung kleinster Norm + t · Nullvektor
        assert res.fits_max_dev < 1e-9
        # kleinste Norm: orthogonal zum Nullvektor
        assert abs(base @ v) < 1e-10
        assert np.allclose(res.solutions["pinv (kleinste Norm)"], base, atol=1e-10)
        assert res.solutions["QR mit Spaltenwahl"].tolist().count(0.0) >= 1


def test_dummy_anpassung_ist_das_zellmittel_der_art():
    s = ev.Settings(n=200, seed=7003)
    c = G.draw(s.seed, s.n)
    y = G.outcome(c, s.streuung, C.SZENARIEN[s.szenario])
    res = ev.dummy_trap(s)
    mean = np.array([y[c.art == j].mean() for j in range(4)])
    assert np.allclose(res.cell_means, mean)
    for f in res.fitted.values():
        assert np.max(np.abs(f - mean[c.art])) < 1e-9


# --- (6) Koeffizient gleich Inverse der Zellmatrix mal Zellmittel, und die Sätze zu den Koeffizienten --------------------------------------------------

def test_koeffizienten_sind_inverse_zellmatrix_mal_zellmittel():
    rng = np.random.default_rng(130301)
    for k in range(12):
        a, b, y = rand_data(rng, 3, 2, 1, 6, integer=False)
        n, mean, _ = T.cell_stats(a, b, y, 3, 2)
        kind = KINDS[k % 4]
        ra, rb = int(rng.integers(0, 3)), int(rng.integers(0, 2))
        ra, rb = (ra, rb) if kind in ("Referenz", "Gewichtet zentriert") else (0, 0)
        d = D.build_design(a, b, 3, 2, kind, ra, rb)
        beta = D.fit(d.X, y)[0]
        cd = D.cell_design(3, 2, kind, ra, rb, n.sum(axis=1) / len(y), n.sum(axis=0) / len(y))
        assert np.max(np.abs(beta - D.cell_weights(cd) @ mean.reshape(-1))) < 1e-9, kind


def test_saetze_zum_koeffizienten_geben_denselben_wert_wie_der_loeser():
    """`predicted_coefficients` (Achsenabschnitt, Art, Wochenende aus den Zellmitteln) gegen die Koeffizienten von `lstsq`, alle Codierungen, alle Referenzstufen, ungleiche Zellgrößen."""
    rng = np.random.default_rng(130302)
    for _ in range(30):
        sizes = tuple(int(v) for v in rng.integers(1, 7, size=6))
        for kind in C.COD_KEYS:
            for ra in range(3):
                for rb in range(2):
                    s = ev.Settings(sizes=sizes, cod=kind, ref_art=ra, ref_we=rb)
                    res = ev.analyse_hand(s)
                    h = res.hand
                    ra2, rb2 = ev.refs_for(kind, s)
                    pred = ev.predicted_coefficients(h, kind, ra2, rb2)
                    idx = list(res.design.i_int) + list(res.design.i_a) + list(res.design.i_b)
                    assert len(pred) == len(idx)
                    for (name, value, text), i in zip(pred, idx):
                        assert value == pytest.approx(res.table["Wert"][i], abs=1e-9), (kind, ra, rb, name)
                        assert text


# --- (7) gestapelte Messreihe gegen Wiederholung für Wiederholung mit der Entwurfsmatrix ----------------------------------------------------------

@pytest.mark.parametrize("mode,szen", [("gleich", "Generator"), ("Generator", "Überkreuz"), ("gleich", "additiv")])
def test_gestapelte_simulation_zaehlt_wie_die_entwurfsmatrix_je_wiederholung(mode, szen):
    n, reps = 200, 40
    row = SIM.simulate_row(mode, szen, n, reps=reps, chunk=7)
    seed = SIM.row_seed(mode, szen, n)
    w = C.SZENARIEN[szen]
    rejB = np.zeros(8, dtype=int)
    rejA = np.zeros(8, dtype=int)
    rejI = valid = 0
    for r in range(reps):
        c = G.draw(seed, n, first=r * n)
        y = G.outcome(c, mode, w)
        cn, _, _ = T.cell_stats(c.art, c.we, y, 4, 2)
        if np.any(cn == 0):
            continue
        valid += 1
        for k, st in enumerate(SIM.SETTINGS):
            d = D.build_design(c.art, c.we, 4, 2, st["kind"], st["ref_a"], st["ref_b"])
            rejB[k] += D.drop_test(d.X, y, d.i_b)[1] < 0.05
            rejA[k] += D.drop_test(d.X, y, d.i_a)[1] < 0.05
        d = D.build_design(c.art, c.we, 4, 2, "Summe")
        rejI += D.drop_test(d.X, y, d.i_ab)[1] < 0.05
    assert row["valid"] == valid and row["rej_B"] == rejB.tolist() and row["rej_A"] == rejA.tolist() and row["rej_I"] == rejI


def test_gestapelte_kundenreihe_gegen_einzelrechnung():
    reps, n = 30, C.N_KUNDE_EXP
    out = SIM.simulate_kunde(reps=reps, n=n, chunk=11)
    k = C.K_KUNDEN
    tcrit = stats.t.ppf(0.975, n - k)
    sig = np.zeros(k)
    for r in range(reps):
        c = G.draw_import(C.KUNDE_SEED, n, first=r * n)
        y = G.import_outcome(c, True)
        cnt = np.bincount(c.kunde, minlength=k).astype(float)
        mean = np.bincount(c.kunde, weights=y, minlength=k) / cnt
        s2 = np.sum((y - mean[c.kunde]) ** 2) / (n - k)
        for ref in range(k):
            others = [j for j in range(k) if j != ref]
            t = (mean[others] - mean[ref]) / np.sqrt(s2 * (1 / cnt[others] + 1 / cnt[ref]))
            sig[ref] += np.sum(np.abs(t) > tcrit)
    assert np.allclose(out["sig_sum_by_ref"], sig) and out["valid"] == reps


def test_kundenfaktor_f_test_gegen_scipy_f_oneway():
    s = ev.Settings(seed=7004, ref_kunde=4)
    ks = ev.kunde_sample(s)
    c = G.draw_import(s.seed, C.N_KUNDE_EXP)
    y = G.import_outcome(c, True)
    F_ref, p_ref = stats.f_oneway(*[y[c.kunde == j] for j in range(C.K_KUNDEN)])
    assert ks.F == pytest.approx(F_ref, rel=1e-9) and ks.p == pytest.approx(p_ref, rel=1e-6, abs=1e-300)
    # Einzelkoeffizienten gegen den t-Test mit gemeinsamer Restvarianz (eigene Formel), nicht gegen scipy ttest_ind (anderer Varianzschätzer)
    t = ks.table
    cnt = np.bincount(c.kunde, minlength=12).astype(float)
    mean = np.bincount(c.kunde, weights=y, minlength=12) / cnt
    s2 = np.sum((y - mean[c.kunde]) ** 2) / (len(y) - 12)
    j = 7
    row = t[t["Kunde"] == f"K{j}"].iloc[0]
    assert row["Koeffizient"] == pytest.approx(mean[j] - mean[4]) and row["Standardfehler"] == pytest.approx(np.sqrt(s2 * (1 / cnt[j] + 1 / cnt[4])))


def test_lstsq_orakel_ist_selbst_geprueft_gegen_scipy_lstsq_und_cholesky():
    """Das Orakel (Brüche) gegen ein zweites Verfahren (Cholesky der Normalgleichung in scipy), damit nicht nur zwei Rechnungen mit demselben Fehler übereinstimmen."""
    rng = np.random.default_rng(130303)
    a, b, y = rand_data(rng, 3, 2, 2, 5)
    Xf, *_ = X.design(list(a), list(b), 3, 2, "Summe")
    beta_x, _ = X.ls(Xf, [int(v) for v in y])
    Xn = np.array([[float(v) for v in row] for row in Xf])
    c, low = sla.cho_factor(Xn.T @ Xn)
    beta_c = sla.cho_solve((c, low), Xn.T @ y.astype(float))
    assert np.max(np.abs(beta_c - np.array([float(v) for v in beta_x]))) < 1e-9


# --- (8) Bedeutung des Wochenende-Koeffizienten, exakt in Brüchen (40 Datensätze) --------------------------------------------------------------------

def test_wochenende_koeffizient_ist_in_40_datensaetzen_exakt_der_effekt_an_der_stelle_der_codierung():
    """Referenz: Effekt bei der Referenzart; Summe: −½ × ungewichtetes Mittel der Effekte; Helmert: +½ × dasselbe; gewichtet zentriert: mit den Stichprobenanteilen gewichtetes Mittel. Bruch gleich Bruch."""
    rng = np.random.default_rng(130101)
    ok = {"Referenz": 0, "Summe": 0, "Helmert": 0, "Gewichtet zentriert": 0}
    for _ in range(40):
        a, b, y = rand_data(rng, 4, 2)
        al, bl, yl = list(a), list(b), [int(v) for v in y]
        n = len(yl)
        cn, mu = X.cells(al, bl, yl)
        delta = [mu[(i, 1)] - mu[(i, 0)] for i in range(4)]
        r = int(rng.integers(0, 4))
        Xr, _, iB, _ = X.design(al, bl, 4, 2, "Referenz", r)
        ok["Referenz"] += X.ls(Xr, yl)[0][iB[0]] == delta[r]
        Xs, _, iB, _ = X.design(al, bl, 4, 2, "Summe")
        ok["Summe"] += X.ls(Xs, yl)[0][iB[0]] == -sum(delta) / 8
        Xh, _, iB, _ = X.design(al, bl, 4, 2, "Helmert")
        ok["Helmert"] += X.ls(Xh, yl)[0][iB[0]] == sum(delta) / 8
        Xg, _, iB, _ = X.design(al, bl, 4, 2, "Gewichtet zentriert")
        sa = [F(sum(1 for v in al if v == i), n) for i in range(4)]
        ok["Gewichtet zentriert"] += X.ls(Xg, yl)[0][iB[0]] == sum(sa[i] * delta[i] for i in range(4))
    assert ok == {"Referenz": 40, "Summe": 40, "Helmert": 40, "Gewichtet zentriert": 40}
