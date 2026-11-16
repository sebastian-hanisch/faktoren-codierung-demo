"""Einheiten der Codierung und der Entwurfsmatrix, an kleinen Beispielen von Hand geprüft."""

import numpy as np
import pytest

import fcd_constants as C
import fcd_design as D


def test_referenzcodierung_ist_die_einheitsmatrix_ohne_die_referenzspalte():
    assert np.array_equal(D.coding_matrix("Referenz", 3, 0), [[0, 0], [1, 0], [0, 1]])
    assert np.array_equal(D.coding_matrix("Referenz", 3, 1), [[1, 0], [0, 0], [0, 1]])


def test_summencodierung_laesst_die_letzte_stufe_weg_und_zeilen_summieren_sich_zu_null_spalten_auch():
    m = D.coding_matrix("Summe", 4)
    assert np.array_equal(m, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]])
    assert np.all(m.sum(axis=0) == 0)                 # jede Spalte: +1 und −1


def test_helmert_spalten_sind_stufe_gegen_mittel_der_vorigen_und_orthogonal():
    m = D.coding_matrix("Helmert", 4)
    assert np.array_equal(m, [[-1, -1, -1], [1, -1, -1], [0, 2, -1], [0, 0, 3]])
    assert np.all(m.sum(axis=0) == 0)
    g = m.T @ m
    assert np.array_equal(g, np.diag(np.diag(g)))     # orthogonale Spalten
    assert np.array_equal(D.coding_matrix("Helmert", 2), [[-1], [1]])


def test_gewichtete_codierung_zentriert_jede_spalte_am_stichprobenanteil():
    w = np.array([0.5, 0.3, 0.2])
    m = D.coding_matrix("Gewichtet zentriert", 3, 0, w)
    assert np.allclose(w @ m, 0)                      # mit den Anteilen gewichtet verschwindet jede Spalte
    assert np.allclose(m, [[-0.3, -0.2], [0.7, -0.2], [-0.3, 0.8]])


def test_hypothesengewichte_je_codierung():
    assert np.array_equal(D.hypothesis_weights("Referenz", 4, 2), [0, 0, 1, 0])
    assert np.allclose(D.hypothesis_weights("Summe", 4), 0.25) and np.allclose(D.hypothesis_weights("Helmert", 4), 0.25)
    assert np.allclose(D.hypothesis_weights("Gewichtet zentriert", 3, 0, [0.5, 0.3, 0.2]), [0.5, 0.3, 0.2])


def test_entwurfsmatrix_form_spaltengruppen_und_namen():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([0, 1, 0, 1, 0, 1])
    d = D.build_design(a, b, 3, 2, "Referenz", 0, 0, ("Import", "Export", "Umschlag"))
    assert d.X.shape == (6, 6) and d.i_int == (0,) and d.i_a == (1, 2) and d.i_b == (3,) and d.i_ab == (4, 5)
    assert d.names == ("Achsenabschnitt", "Art Export", "Art Umschlag", "Wochenende", "Art Export × Wochenende", "Art Umschlag × Wochenende")
    assert np.array_equal(d.X, [[1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 0], [1, 0, 1, 1, 0, 1]])
    add = D.build_design(a, b, 3, 2, "Summe", interaction=False)
    assert add.X.shape == (6, 4) and add.i_ab == ()
    assert D.full_rank(d.X) and np.linalg.matrix_rank(d.X) == 6


def test_zellmatrix_mal_gewichte_ist_die_einheitsmatrix_und_koeffizient_ist_gewichtete_summe_der_zellmittel():
    for kind in C.COD_KEYS:
        cd = D.cell_design(3, 2, kind, 1, 1, [0.5, 0.3, 0.2], [0.6, 0.4])
        W = D.cell_weights(cd)
        assert np.allclose(cd.X @ W, np.eye(6)), kind
        theta = np.array([4.0, 4, 4, 8, 4, 12])        # Zellmittel in der Reihenfolge Art mal Wochenende
        beta = W @ theta
        assert np.allclose(cd.X @ beta, theta), kind     # das Modell mit Wechselwirkung passt die Zellmittel an


def test_rangpruefung_leere_zelle_und_dummyfalle():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([0, 1, 0, 1, 0, 0])                    # Zelle (Umschlag, Wochenende) leer
    d = D.build_design(a, b, 3, 2, "Referenz")
    assert not D.full_rank(d.X) and D.smallest_relative_singular_value(d.X) < 1e-10
    with pytest.raises(D.RankError):
        D.fit(d.X, np.arange(6.0))
    dummy = np.column_stack([np.ones(6)] + [(np.array([0, 0, 1, 1, 2, 2]) == j).astype(float) for j in range(3)])
    assert not D.full_rank(dummy) and np.linalg.matrix_rank(dummy) == 3


def test_f_test_ohne_freiheitsgrade_ist_nicht_definiert():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([0, 1, 0, 1, 0, 1])
    d = D.build_design(a, b, 3, 2, "Referenz")
    F, p, q, dfres = D.drop_test(d.X, np.array([1.0, 2, 3, 5, 8, 13]), d.i_b)
    assert np.isnan(F) and np.isnan(p) and q == 1 and dfres == 0


def test_koeffizienten_standardfehler_und_t_gegen_formel_in_einem_einfachen_fall():
    """Zwei Zellen (nur B), Referenzcodierung: Koeffizient = Mittelwertdifferenz, Standardfehler = s·sqrt(1/n0 + 1/n1)."""
    b = np.array([0, 0, 0, 1, 1])
    y = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
    X = np.column_stack([np.ones(5), b])
    d = D.Design(X, ("Achsenabschnitt", "B"), (0,), (), (1,), (), "Referenz", 0, 0)
    beta, se, t, p, dfres = D.coefficient_table(d, y)
    s2 = (2 + 2) / 3                                     # Quadratsumme innerhalb: 1+0+1 und 1+1
    assert beta == pytest.approx([2.0, 4.0]) and dfres == 3
    assert se[1] == pytest.approx(np.sqrt(s2 * (1 / 3 + 1 / 2))) and se[0] == pytest.approx(np.sqrt(s2 / 3))
    assert t[1] == pytest.approx(4.0 / se[1])


def test_spaltennamen_der_vier_codierungen():
    lv = ("Import", "Export", "Umschlag")
    assert D.column_labels("Referenz", lv, 1, "Art") == ["Art Import", "Art Umschlag"]
    assert D.column_labels("Summe", lv, 0, "Art") == ["Art Import (Summe)", "Art Export (Summe)"]
    assert D.column_labels("Helmert", lv, 0, "Art") == ["Art Export gegen Import", "Art Umschlag gegen Mittel von Import, Export"]
    assert D.column_labels("Gewichtet zentriert", lv, 0, "Art") == ["Art Export (zentriert)", "Art Umschlag (zentriert)"]
