"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler durch seine Werte
und prüft, dass kein Python-Fehler auftritt (insbesondere `StreamlitDuplicateElementId` bei mehreren `st.plotly_chart` mit gleichem Inhalt). Dazu je Preset die erwartete Verdikt-Zeile
(DEMO-PLAYBOOK §11a Punkt 7: kein Preset ohne Effekt meldet einen Effekt)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import fcd_constants as C
import fcd_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))   # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
WIDGETS = ("number_input", "select_slider", "selectbox", "radio")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return [e.value for kind in ("info", "warning", "success", "error") for e in getattr(at, kind)]


def _widget(at, key):
    for kind in WIDGETS:
        try:
            return getattr(at, kind)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} nicht gefunden")


def test_app_loads_without_exception():
    at = _fresh_app()
    assert at.title and not at.error


def test_every_preset_and_the_seed_button_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert len(labels) == len(C.PRESETS) + 1, labels
    for index, label in enumerate(labels):
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_regler_value_does_not_raise():
    """Jeder Regler durch alle seine Werte (Zellgrößen an den Grenzen, jede Auswahl), in einer Sitzung nacheinander."""
    at = _fresh_app()
    for key, spec in P.SETTING_SPECS.items():
        w = _widget(at, key)
        if w.disabled:
            continue
        if spec.choices is not None:
            values = list(spec.choices)
        elif key == "seed":
            values = [spec.lo, spec.hi, 7001]
        elif key.startswith("ref"):
            values = list(range(int(spec.lo), int(spec.hi) + 1))
        else:
            values = [spec.lo, spec.hi, spec.default]
        for v in values:
            _widget(at, key).set_value(v).run(timeout=TIMEOUT)
            assert not at.exception, (key, v, [str(e) for e in at.exception])


def test_codierung_summe_sperrt_die_referenzstufen():
    at = _fresh_app()
    assert not _widget(at, "ref_art").disabled
    at.selectbox(key="cod").set_value(C.COD_SUMME).run(timeout=TIMEOUT)
    assert _widget(at, "ref_art").disabled and _widget(at, "ref_we").disabled and not at.exception
    at.selectbox(key="cod").set_value(C.COD_GEWICHTET).run(timeout=TIMEOUT)
    assert not _widget(at, "ref_art").disabled


def test_additives_modell_laeuft_in_jeder_codierung():
    at = _fresh_app()
    at.radio(key="modell").set_value("additiv").run(timeout=TIMEOUT)
    for cod in C.COD_KEYS:
        at.selectbox(key="cod").set_value(cod).run(timeout=TIMEOUT)
        assert not at.exception, cod
    texts = " ".join(_texts(at))
    assert "Im additiven Modell ohne Wechselwirkung testet jede Codierung dieselbe Hypothese" in texts


# --- je Preset die erwartete Verdikt-Zeile -----------------------------------------------------------------------------------------------------

def _after(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at, " ".join(_texts(at))


def test_preset_von_hand_h1_meldet_f_0_gegen_24():
    at, text = _after("Von Hand (H1)")
    assert "Referenz Import F = 0,00 (p = 1,0000)" in text and "Summe F = 24,00 (p = 0,0027)" in text
    assert "Achsenabschnitt 4,00, Wochenende 0,00" in text and "Summe: Achsenabschnitt 6,00, Wochenende −2,00" in text
    assert "F der Wechselwirkung in allen vier Codierungen 8,00 (p = 0,0203)" in text


def test_preset_unausgewogen_h2_meldet_die_f_werte_je_codierung_und_typ_i_gegen_typ_iii():
    at, text = _after("Unausgewogen (H2)")
    for part in ("Referenz Export F = 8,00 (p = 0,0222)", "Referenz Umschlag F = 42,67 (p = 0,0002)", "Summe F = 31,14 (p = 0,0005)", "Gewichtet zentriert F = 23,88 (p = 0,0012)",
                 "Typ I (nach Zellgrößen gewichtete Zeilenmittel) F = 13,14", "Typ III (ungewichtete Zeilenmittel) F = 12,25"):
        assert part in text, part


def test_preset_container_generator_meldet_die_haeufigkeiten_der_vorgerechneten_zeile():
    at, text = _after("Container Generator")
    assert "Bei n = 600 verwirft der Test „Haupteffekt Wochenende“ in Referenz Import 80 %, Referenz Export 5 %, Referenz Umschlag 5 %, Referenz Leer 33 %, Summe 55 % und gewichtet 56 %" in text
    assert "Die Wechselwirkung wird in 49 % der Wiederholungen erkannt (bei n = 200: 17 %)" in text


def test_preset_additiv_meldet_keine_wechselwirkung_und_keinen_unterschied_im_additiven_modell():
    at, text = _after("Additiv")
    assert "Additives Szenario" in text and "(wahre Wechselwirkung 0)" in text and "denselben F-Wert" in text
    verdikt = next(x for x in _texts(at) if x.startswith("Additives Szenario"))
    assert "verschiedene Antworten" not in verdikt and "Wechselwirkung erkannt" not in verdikt      # das Szenarioverdikt behauptet weder einen Codierungseffekt noch eine Wechselwirkung


def test_preset_ueberkreuz_haelt_das_niveau_bei_gleicher_streuung_und_warnt_mit_der_streuung_des_generators():
    at, text = _after("Überkreuz")
    assert "Summe 5 %" in text and "nicht bei 5 %" not in text
    at2, text2 = _after("Überkreuz, Streuung des Generators")
    assert "Summe 10,5 %" in text2 and "nicht bei 5 %" in text2


def test_preset_dummy_falle_meldet_rang_4_von_5():
    at, text = _after("Dummy-Falle")
    assert "Rang 4 von 5" in text and ("4 Lösungswege" in text or "3 Lösungswege" in text)           # unter manchen Plattformen meldet solve(X'X) einen Fehler


def test_preset_zwoelf_kunden_meldet_die_referenz_k4():
    at, text = _after("Zwölf Kunden")
    assert "Referenz K4: im Mittel 4,8 von 11" in text
