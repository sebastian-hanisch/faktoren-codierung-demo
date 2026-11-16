"""Randfälle: leere Zelle, Zelle mit einer Beobachtung, so viele Beobachtungen wie Zellen (Rest ohne Freiheitsgrade), Faktor mit zwei Stufen, Rangmangel, Nichtzentralität null, Formatierung."""

import numpy as np
import pytest

import fcd_constants as C
import fcd_design as D
import fcd_evaluation as ev
import fcd_generator as G
import fcd_tests as T


def test_hand_jede_zelle_eine_beobachtung_hat_keine_freiheitsgrade_und_die_app_meldet_es():
    s = ev.Settings(sizes=(1,) * 6)
    res = ev.analyse_hand(s)
    assert res.hand.n == 6 and res.hand.dfres == 0 and "keine Freiheitsgrade" in res.status
    assert np.isnan(res.table["Standardfehler"]).all() and np.isnan(res.table["p"]).all()
    cmp = ev.compare_codings(res.hand)
    assert cmp["dfres"] == 0 and all(np.isnan(r["F_B"]) and np.isnan(r["p_B"]) for r in cmp["rows"]) and np.isnan(cmp["F_I"])
    assert ev.verdict_hypotheses(cmp)[0] == "warning" and ev.verdict_typ(cmp)[0] == "warning"
    kind, text = ev.verdict_switch(res, cmp)
    assert kind == "info" and "nicht definiert" in text
    assert res.max_dev_fit < 1e-9                                       # die Anpassung ist trotzdem eindeutig (die sechs Beobachtungen selbst)


def test_hand_additives_modell_mit_einer_beobachtung_je_zelle_hat_zwei_freiheitsgrade():
    res = ev.analyse_hand(ev.Settings(sizes=(1,) * 6, modell="additiv"))
    assert res.status == "ok" and res.hand.dfres == 0                    # der Rest des Modells MIT Wechselwirkung hätte keine Freiheitsgrade, das additive hat 6 − 4 = 2
    cmp = ev.compare_codings(res.hand, interaction=False)
    assert cmp["dfres"] == 2 and not np.isnan(cmp["rows"][0]["F_B"])


def test_hand_alle_zellen_mit_acht_beobachtungen_und_gemischte_groessen():
    for sizes in ((8,) * 6, (1, 8, 8, 1, 1, 8), (1, 1, 1, 1, 1, 2)):
        res = ev.analyse_hand(ev.Settings(sizes=sizes))
        assert res.max_dev_fit < 1e-9 and np.isfinite(res.table["Wert"]).all()
        cmp = ev.compare_codings(res.hand)
        assert np.isfinite(cmp["F_I"]) and all(np.isfinite(r["F_B"]) for r in cmp["rows"])


def test_handdaten_haben_in_jeder_zellgroesse_das_vorgegebene_zellmittel():
    for n in range(1, 9):
        for i in range(3):
            for j in range(2):
                sizes = [2] * 6
                sizes[2 * i + j] = n
                a, b, y = ev.hand_data(sizes)
                sel = (a == i) & (b == j)
                assert sel.sum() == n and y[sel].mean() == C.HAND_MU[i][j]
                assert np.sum((y[sel] - y[sel].mean()) ** 2) == 2 * (n // 2)


def test_stichprobe_mit_leerer_zelle_wird_gemeldet_und_nicht_gerechnet():
    seed = next(sd for sd in range(20000) if np.any(T.cell_stats(*(lambda c: (c.art, c.we, c.z))(G.draw(sd, 200)), 4, 2)[0] == 0))
    res = ev.sample_tests(ev.Settings(n=200, seed=seed))
    assert not res.ok and "Zelle leer" in res.reason and res.rows == []
    c = G.draw(seed, 200)
    y = G.outcome(c, "gleich", C.SZENARIEN["Generator"])
    d = D.build_design(c.art, c.we, 4, 2, "Referenz")
    assert not D.full_rank(d.X)
    with pytest.raises(D.RankError):
        D.fit(d.X, y)
    n, mean, within = T.cell_stats(c.art, c.we, y, 4, 2)
    assert np.isnan(T.wochenende_test(n, mean, within, [1, 0, 0, 0])[0]) and np.isnan(T.art_test(n, mean, within, [1, 0])[0]) and np.isnan(T.interaction_test(n, mean, within)[0])


def test_einzelkoeffizient_ohne_freiheitsgrade_oder_mit_leerer_zelle_ist_nan():
    n = np.ones((4, 2))
    assert np.isnan(T.single_coefficient_t(n, np.zeros((4, 2)), 0.0, 3, 0)[0])
    n2 = np.full((4, 2), 3.0)
    n2[3, 0] = 0
    assert np.isnan(T.single_coefficient_t(n2, np.zeros((4, 2)), 5.0, 3, 0)[0])


def test_faktor_mit_zwei_stufen_summe_und_helmert_gleich_bis_aufs_vorzeichen():
    rng = np.random.default_rng(5)
    for _ in range(20):
        a = np.repeat([0, 1], 12)
        b = np.tile(np.repeat([0, 1], 6), 2)
        y = rng.normal(size=24) + 0.7 * a + 0.4 * b
        ds, dh = D.build_design(a, b, 2, 2, "Summe"), D.build_design(a, b, 2, 2, "Helmert")
        bs, bh = D.fit(ds.X, y)[0], D.fit(dh.X, y)[0]
        assert np.allclose(np.abs(bs[1:]), np.abs(bh[1:]), atol=1e-9)                              # gleiche Beträge, Vorzeichen nach Codierung
        assert bs[ds.i_a[0]] == pytest.approx(-bh[dh.i_a[0]], abs=1e-9) and bs[ds.i_b[0]] == pytest.approx(-bh[dh.i_b[0]], abs=1e-9)
        assert D.drop_test(ds.X, y, ds.i_b)[0] == pytest.approx(D.drop_test(dh.X, y, dh.i_b)[0], rel=1e-9)


def test_eine_stufe_nur_in_einer_zelle_ist_rangmangel_im_modell_mit_wechselwirkung():
    a = np.array([0] * 6 + [1] * 6 + [2] * 6)
    b = np.array([0, 1] * 3 + [0, 1] * 3 + [0] * 6)
    d = D.build_design(a, b, 3, 2, "Summe")                 # Art 2 kommt nur am Wochentag vor
    assert not D.full_rank(d.X)
    add = D.build_design(a, b, 3, 2, "Summe", interaction=False)
    assert D.full_rank(add.X)                                # das additive Modell ist schätzbar


def test_zellmittel_der_leeren_zelle_sind_nan_und_ohne_wirkung_auf_die_summe_der_quadrate():
    n, mean, within = T.cell_stats(np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1.0, 2.0, 4.0]), 2, 2)
    assert n.tolist() == [[1, 1], [1, 0]] and np.isnan(mean[1, 1]) and within == 0.0


def test_teststaerke_bei_nichtzentralitaet_null_ist_das_niveau_auch_bei_negativer_rundung():
    assert T.noncentral_power(1, 50, 0.0) == C.ALPHA and T.noncentral_power(1, 50, -1e-15) == C.ALPHA and T.noncentral_power(3, 50, 5e-13) == C.ALPHA


def test_exakte_teststaerke_ist_bei_wahrem_effekt_null_das_niveau():
    theta = G.cell_means_exact(C.SZENARIEN["Überkreuz"])
    n_cell = 600 * np.outer(C.ART_P, [5 / 7, 2 / 7])
    assert T.power_wochenende(theta, n_cell, 0.69, D.hypothesis_weights("Summe", 4))[0] == C.ALPHA          # Summe testet 0 im Überkreuz-Szenario
    assert T.power_wochenende(theta, n_cell, 0.69, D.hypothesis_weights("Referenz", 4, 1))[0] == C.ALPHA   # Referenz Export testet 0


def test_unbekannte_zeile_der_vorgerechneten_reihe_wirft_keyerror():
    with pytest.raises(KeyError):
        ev.frozen_row("gleich", "Generator", 999)


def test_einstellungen_aus_dem_session_state_nehmen_fehlende_schluessel_als_standard():
    s = ev.settings_from_state(lambda k: {"seed": 5, "cell_21": 7}.get(k))
    assert s.seed == 5 and s.sizes == (2, 2, 2, 2, 2, 7) and s.cod == C.COD_REFERENZ and s.n == 600
