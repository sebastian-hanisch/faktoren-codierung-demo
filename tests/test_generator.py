"""Container-Generator: Verteilung, exakte Zellmittel und Streuung gegen eine große Stichprobe (Orakel 6), Stabilität des Stroms."""

import numpy as np
import pytest
from scipy import integrate

import fcd_constants as C
import fcd_generator as G

N_BIG = 400_000


@pytest.fixture(scope="module")
def big():
    c = G.draw(20261116, N_BIG)
    return c, G.outcome(c, "Generator", C.W_GEN)


def test_artanteile_wochenende_und_zoll(big):
    c, _ = big
    for j, p in enumerate(C.ART_P):
        assert abs(np.mean(c.art == j) - p) < 4 * np.sqrt(p * (1 - p) / N_BIG), j
    assert abs(np.mean(c.we == 1) - 2 / 7) < 4 * np.sqrt(2 / 7 * 5 / 7 / N_BIG)
    imp = c.art == 0
    assert c.zoll[~imp].sum() == 0 and abs(np.mean(c.zoll[imp]) - C.P_ZOLL) < 4 * np.sqrt(0.25 * 0.75 / imp.sum())
    assert np.all((c.vorlauf[c.art == 1] >= 1) & (c.vorlauf[c.art == 1] <= 10)) and np.all((c.vorlauf[c.art == 2] >= 0.5) & (c.vorlauf[c.art == 2] <= 6))
    assert np.all(c.vorlauf[(c.art == 0) | (c.art == 3)] == 0)
    assert np.bincount(c.kunde, minlength=12).min() > 0


def test_exakte_zellmittel_gegen_die_grosse_stichprobe(big):
    """Orakel 6: die geschlossenen Formeln der Zellmittel gegen den Mittelwert von 400 000 Containern, höchstens 4 Standardfehler des Mittels."""
    c, y = big
    m = G.cell_means_exact(C.W_GEN)
    worst = 0.0
    for i in range(4):
        for j in range(2):
            sel = (c.art == i) & (c.we == j)
            se = y[sel].std() / np.sqrt(sel.sum())
            worst = max(worst, abs(y[sel].mean() - m[i, j]) / se)
    assert worst < 4.0, worst


def test_wochenendeffekt_des_generators_nur_bei_import_und_leer():
    m = G.cell_means_exact(C.W_GEN)
    assert np.allclose(m[:, 1] - m[:, 0], [0.30, 0.0, 0.0, 0.30], atol=1e-15)
    assert G.cell_means_exact(C.SZENARIEN["additiv"])[:, 1] - G.cell_means_exact(C.SZENARIEN["additiv"])[:, 0] == pytest.approx(0.15, abs=1e-15)


def test_exakte_streuung_je_art_gegen_numerische_quadratur_und_stichprobe(big):
    c, y = big
    for a0, b0, v0, v1 in ((0.6, 0.85, 1.0, 10.0), (0.3, 0.8, 0.5, 6.0)):
        f = lambda v: np.log(a0 + b0 * v)
        m1 = integrate.quad(f, v0, v1)[0] / (v1 - v0)
        m2 = integrate.quad(lambda v: f(v) ** 2, v0, v1)[0] / (v1 - v0)
        assert G.var_log_linear(a0, b0, v0, v1) == pytest.approx(m2 - m1 ** 2, rel=1e-8)
        assert G.e_log_linear(a0, b0, v0, v1) == pytest.approx(m1, rel=1e-9)
    var_art = G.var_within_art()
    for i in range(4):
        sel = c.art == i
        # Varianz innerhalb der Art = Mittel der Zellvarianzen (das Wochenende verschiebt nur den Mittelwert, hier mit dem Effekt des Generators)
        resid = y[sel] - G.cell_means_exact(C.W_GEN)[i][c.we[sel]]
        assert np.var(resid) == pytest.approx(var_art[i], rel=0.03), i
    assert G.sd_pool_exact() == pytest.approx(0.6887, abs=2e-4)


def test_streuung_gleich_hat_in_jeder_zelle_dieselbe_streuung(big):
    c, _ = big
    y = G.outcome(c, "gleich", C.W_GEN)
    sd = G.sd_pool_exact()
    for i in range(4):
        for j in range(2):
            sel = (c.art == i) & (c.we == j)
            assert y[sel].std() == pytest.approx(sd, rel=0.03)


def test_streuung_des_generators_ist_je_zelle_verschieden_im_verhaeltnis_1_7(big):
    c, y = big
    sds = [y[(c.art == i) & (c.we == j)].std() for i in range(4) for j in range(2)]
    assert max(sds) / min(sds) == pytest.approx(1.73, abs=0.05) and min(sds) > 0.55 and max(sds) < 0.98


def test_strom_ist_deterministisch_und_praefixstabil():
    a, b = G.draw(5, 300), G.draw(5, 500)
    assert np.array_equal(a.art, b.art[:300]) and np.array_equal(a.z, b.z[:300]) and np.array_equal(a.vorlauf, b.vorlauf[:300])
    assert not np.array_equal(G.draw(6, 300).art, a.art)


def test_szenarien_wirken_nur_ueber_den_wochenendeffekt():
    c = G.draw(31, 5000)
    y1 = G.outcome(c, "Generator", C.SZENARIEN["Generator"])
    y2 = G.outcome(c, "Generator", C.SZENARIEN["Überkreuz"])
    diff = y2 - y1
    assert np.allclose(diff[c.we == 0], 0) and np.allclose(diff[(c.we == 1) & (c.art == 3)], -0.6) and np.allclose(diff[(c.we == 1) & (c.art == 0)], 0)


def test_import_ausgang_mit_und_ohne_kundeneffekt():
    c = G.draw_import(8, 2000)
    assert np.all(c.art == 0) and abs(np.mean(c.zoll) - 0.25) < 0.04
    d = G.import_outcome(c, True) - G.import_outcome(c, False)
    assert np.allclose(d, np.array(C.UK)[c.kunde])
