"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Zufalls-Seeds. Damit prüft der Test die Grenzen (p < 0,05 ist strikt, Differenz unter 1e-9) und jede genannte Zahl direkt."""

from types import SimpleNamespace

import numpy as np

import fcd_constants as C
import fcd_evaluation as ev
import fcd_simulation as SIM

NAMES = SIM.SETTING_NAMES        # Referenz Import / Export / Umschlag / Leer, Referenz Import / Wochenende, Summe, Helmert, Gewichtet zentriert


def _row(name, F, p, FA=5.0, pA=0.01):
    return {"name": name, "kind": "x", "effect": 0.0, "sd": 1.0, "F_B": F, "p_B": p, "F_A": FA, "p_A": pA}


def _cmp(rows, interaction=True, dfres=6, **kw):
    out = {"rows": rows, "dfres": dfres, "interaction": interaction}
    out.update(kw)
    return out


# --- Formatierung ----------------------------------------------------------------------------------------------------------------------------

def test_zahlenformat_mit_komma_minuszeichen_und_ohne_vorzeichen_bei_null():
    assert ev.de(3.14159, 2) == "3,14" and ev.de(-2.0, 2) == "−2,00" and ev.de(-0.0004, 2) == "0,00" and ev.de(float("nan")) == "–"
    assert ev.pct(0.4949, 1) == "49,5 %" and ev.pct(0.05, 0) == "5 %"
    assert ev.p_text(0.0027) == "0,0027" and ev.p_text(5e-5) == "< 0,0001" and ev.p_text(float("nan")) == "–"
    assert ev.sci(3.6e-15) == "3,6e-15" and ev.sci(2e-15, 0) == "2e-15"


# --- Kernfrage 2 -----------------------------------------------------------------------------------------------------------------------------

def test_hypothesen_verdikt_gemischt_zaehlt_verwerfende_codierungen_und_nennt_jede_zahl():
    rows = [_row("Referenz Import", 0.0, 1.0), _row("Referenz Export", 8.0, 0.0222), _row("Referenz Umschlag", 42.6667, 0.0002), _row("Referenz Import / Wochenende", 0.0, 1.0),
            _row("Summe", 31.1351, 0.0005), _row("Helmert", 31.1351, 0.0005), _row("Gewichtet zentriert", 23.8756, 0.0012)]
    kind, text = ev.verdict_hypotheses(_cmp(rows))
    assert kind == "warning"
    assert "Referenz Import F = 0,00 (p = 1,0000)" in text and "Referenz Export F = 8,00 (p = 0,0222)" in text and "Referenz Umschlag F = 42,67 (p = 0,0002)" in text
    assert "Summe F = 31,14 (p = 0,0005)" in text and "Gewichtet zentriert F = 23,88 (p = 0,0012)" in text
    assert "4 von 5 Codierungen verwerfen" in text                      # Helmert (gleich Summe) und die Zeile „/ Wochenende“ zählen nicht doppelt
    assert "Helmert" not in text.split("verschiedene Antworten:")[1].split("Codierungen verwerfen")[0]


def test_hypothesen_verdikt_schwelle_ist_strikt_bei_p_005():
    rows = [_row("Referenz Import", 3.0, 0.05), _row("Summe", 4.0, 0.0499)]
    kind, text = ev.verdict_hypotheses(_cmp(rows))
    assert kind == "warning" and "1 von 2 Codierungen verwerfen" in text           # p = 0,05 verwirft nicht, p = 0,0499 verwirft


def test_hypothesen_verdikt_gleicher_schluss():
    all_rej = [_row("Referenz Import", 9.0, 0.01), _row("Summe", 12.0, 0.001)]
    assert ev.verdict_hypotheses(_cmp(all_rej)) == ("info", "Referenz Import F = 9,00 (p = 0,0100), Summe F = 12,00 (p = 0,0010). Bei 5 % verwerfen alle der 2 Codierungen (gleicher Schluss in allen).")
    none = [_row("Referenz Import", 0.5, 0.6), _row("Summe", 0.2, 0.7)]
    kind, text = ev.verdict_hypotheses(_cmp(none))
    assert kind == "info" and "verwirft keine der 2 Codierungen" in text


def test_hypothesen_verdikt_ohne_freiheitsgrade_und_additiv():
    assert ev.verdict_hypotheses(_cmp([_row("Summe", float("nan"), float("nan"))], dfres=0))[0] == "warning"
    rows = [_row("Referenz Import", 8.7273, 0.0173), _row("Summe", 8.7273, 0.0173), _row("Gewichtet zentriert", 8.7273, 0.0173)]
    kind, text = ev.verdict_hypotheses(_cmp(rows, interaction=False, fb_dev=1e-14))
    assert kind == "info" and "F = 8,73 (p = 0,0173) bei allen 3 Einstellungen" in text and "verwerfen alle" in text
    kind, text = ev.verdict_hypotheses(_cmp(rows, interaction=False, fb_dev=1e-6))
    assert kind == "warning" and "(relative Abweichung 1,0e-06)" in text


def test_typ_verdikt():
    same = _cmp([], typ1=(8.0, 0.0203), typ3=(8.0, 0.0203))
    assert ev.verdict_typ(same)[0] == "info" and "denselben F-Wert 8,00 (p = 0,0203)" in ev.verdict_typ(same)[1]
    diff = _cmp([], typ1=(13.1429, 0.003), typ3=(12.2523, 0.0037))
    kind, text = ev.verdict_typ(diff)
    assert kind == "warning" and "Typ I (nach Zellgrößen gewichtete Zeilenmittel) F = 13,14 (p = 0,0030)" in text and "Typ III (ungewichtete Zeilenmittel) F = 12,25 (p = 0,0037)" in text
    assert ev.verdict_typ(_cmp([], interaction=False))[0] == "info" and ev.verdict_typ(_cmp([], dfres=0, typ1=(0, 1), typ3=(0, 1)))[0] == "warning"


# --- Kernfrage 3 und Experiment 1 ------------------------------------------------------------------------------------------------------------

def _res(dev=1e-15):
    summ = [{"Codierung": "Referenz", "Achsenabschnitt": 4.0, "Wochenende": 0.0}, {"Codierung": "Summe", "Achsenabschnitt": 6.0, "Wochenende": -2.0},
            {"Codierung": "Helmert", "Achsenabschnitt": 6.0, "Wochenende": 2.0}, {"Codierung": "Gewichtet zentriert", "Achsenabschnitt": 6.0, "Wochenende": 4.0}]
    return SimpleNamespace(summary=summ, max_dev_fit=dev)


def test_anpassungs_verdikt_nennt_achsenabschnitt_und_koeffizient_je_codierung():
    kind, text = ev.verdict_fit(_res())
    assert kind == "info" and "größte Abweichung unter 1e-9" in text
    for part in ("Referenz: Achsenabschnitt 4,00, Wochenende 0,00", "Summe: Achsenabschnitt 6,00, Wochenende −2,00", "Helmert: Achsenabschnitt 6,00, Wochenende 2,00",
                 "Gewichtet zentriert: Achsenabschnitt 6,00, Wochenende 4,00"):
        assert part in text
    assert ev.verdict_fit(_res(1e-9))[0] == "warning" and ev.verdict_fit(_res(9.9e-10))[0] == "info"        # Schwelle 1e-9: gleich ist schon zu viel


def test_wechsel_verdikt_nennt_spannen():
    rows = [_row("Referenz Import", 0.0, 1.0), _row("Referenz Export", 8.0, 0.03), _row("Referenz Import / Wochenende", 0.0, 1.0), _row("Summe", 24.0, 0.0027)]
    kind, text = ev.verdict_switch(_res(), _cmp(rows, F_I=8.0, p_I=0.0203, fi_dev=1e-15))
    assert kind == "info" and "F der Wechselwirkung in allen vier Codierungen 8,00 (p = 0,0203)" in text
    assert "von −2,00 bis 4,00" in text and "Haupteffekt-F von 0,00 bis 24,00" in text
    assert ev.verdict_switch(_res(1e-6), _cmp(rows, F_I=8.0, p_I=0.02, fi_dev=0.0))[0] == "warning"
    k2, t2 = ev.verdict_switch(_res(), _cmp([_row("Referenz Import", 8.73, 0.017)], interaction=False, fb_dev=0.0))
    assert k2 == "info" and "Im additiven Modell" in t2 and "(8,73)" in t2


# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------------

def _rates(b, i, **kw):
    r = {"B": b, "A": [1.0] * 8, "I": i, "li": 0.05, "le": 0.1, "leer": 0, "reps": 10000, "valid": 10000}
    r.update(kw)
    return r


def test_container_verdikt_generator_szenario_nennt_haeufigkeiten_und_wechselwirkung():
    b = [0.80, 0.05, 0.05, 0.33, 0.80, 0.55, 0.55, 0.56]
    s = ev.Settings(n=600, szenario="Generator", streuung="gleich")
    kind, text = ev.verdict_container(s, _rates(b, 0.488), _rates(b, 0.174))
    assert kind == "info"
    assert "Referenz Import 80 %, Referenz Export 5 %, Referenz Umschlag 5 %, Referenz Leer 33 %, Summe 55 % und gewichtet 56 %" in text
    assert "Bei n = 600" in text and "in 49 % der Wiederholungen erkannt (bei n = 200: 17 %)" in text


def test_container_verdikt_additiv_nennt_spanne_und_null_wechselwirkung():
    b = [0.294, 0.266, 0.224, 0.118, 0.294, 0.571, 0.571, 0.669]
    kind, text = ev.verdict_container(ev.Settings(n=600, szenario="additiv"), _rates(b, 0.0493), _rates(b, 0.048))
    assert kind == "info" and "zwischen 12 % und 67 %" in text and "verwirft 4,9 % (wahre Wechselwirkung 0)" in text
    assert "in jeder Codierung denselben F-Wert" in text


def test_container_verdikt_warnt_bei_streuung_des_generators_und_wahrem_effekt_null():
    b = [0.796, 0.016, 0.032, 0.382, 0.796, 0.105, 0.105, 0.23]
    s = ev.Settings(n=600, szenario="Überkreuz", streuung="Generator")
    kind, text = ev.verdict_container(s, _rates(b, 0.687), _rates(b, 0.31))
    assert kind == "warning"
    assert "Referenz Export 1,6 %, Referenz Umschlag 3,2 %, Summe 10,5 %" in text and "nicht bei 5 %" in text
    assert "Gewichtet" not in text.split("liegen die Häufigkeiten bei wahrem Effekt 0")[1]      # gewichtet testet hier +0,075, nicht 0
    k2, t2 = ev.verdict_container(ev.Settings(n=600, szenario="Überkreuz", streuung="gleich"), _rates(b, 0.7), _rates(b, 0.3))
    assert k2 == "info" and "nicht bei 5 %" not in t2


def test_getestete_effekte_je_szenario():
    assert np.allclose(ev.true_effects("Generator"), [0.3, 0, 0, 0.3, 0.3, 0.15, 0.15, 0.135])
    assert np.allclose(ev.true_effects("Überkreuz"), [0.3, 0, 0, -0.3, 0.3, 0.0, 0.0, 0.075])
    assert np.allclose(ev.true_effects("additiv"), [0.15] * 8)


# --- Experimente 4 und 5 ---------------------------------------------------------------------------------------------------------------------

def test_dummy_verdikt():
    sol = {"a": np.array([1.30, 0, 0, 0, 0]), "b": np.array([1.30, 0, 0, 0, 0]), "c": np.array([1.69, 0, 0, 0, 0]), "d": None}
    res = SimpleNamespace(rank=4, columns=5, smallest_sv=2.4e-15, solutions=sol, fits_max_dev=2e-15)
    kind, text = ev.verdict_dummy(res)
    assert kind == "warning" and "Rang 4 von 5 (kleinster singulärer Wert 2,4e-15): 3 Lösungswege, Achsenabschnitte 1,30, 1,30, 1,69" in text and "größte Abweichung 2e-15" in text
    ok = SimpleNamespace(rank=5, columns=5, smallest_sv=0.3, solutions=sol, fits_max_dev=0.0)
    assert ev.verdict_dummy(ok)[0] == "info"


def test_kunde_verdikt_nennt_referenz_spanne_und_ohne_effekt():
    sm = [6.3, 7.1, 5.2, 4.7, 4.8, 5.0, 6.2, 5.7, 5.3, 7.8, 9.6, 6.4]
    fz = {"sig_mean": sm, "f_alt": 1.0, "f_h0": 0.0517, "any_h0": [0.31] * 12, "effects": list(C.UK), "n": 300, "reps": 10000}
    kind, text = ev.verdict_kunde(fz, 10)
    assert kind == "warning"
    assert "Referenz K10: im Mittel 9,6 von 11" in text and "von 4,7 (Referenz K3) bis 9,6 (Referenz K10)" in text
    assert "100,0 % der 10000 Wiederholungen" in text and "in 31 % der Wiederholungen mindestens ein Einzelkoeffizient" in text and "verwirft 5,2 %" in text
