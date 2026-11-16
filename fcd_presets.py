"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Jeder Regler hat einen Eintrag in SETTING_SPECS (auch die Regler im Hauptbereich bei ihren Experimenten): ein Preset legt den Stand ALLER Regler fest (fehlende Schlüssel nehmen den
Standard), und jede Änderung eines Reglers hebt die Hervorhebung des Presets auf (`tests/test_layout.py`)."""

import random
from dataclasses import dataclass

import streamlit as st

import fcd_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    **{f"cell_{i}{j}": SettingSpec(f"c{i}{j}", int, C.DEFAULTS[f"cell_{i}{j}"], C.CELL_MIN, C.CELL_MAX) for i in range(3) for j in range(2)},
    "cod": SettingSpec("cod", str, C.DEFAULTS["cod"], choices=C.COD_KEYS),
    "ref_art": SettingSpec("ra", int, C.DEFAULTS["ref_art"], 0, 2),
    "ref_we": SettingSpec("rw", int, C.DEFAULTS["ref_we"], 0, 1),
    "modell": SettingSpec("mod", str, C.DEFAULTS["modell"], choices=C.MODELL_KEYS),
    "n": SettingSpec("n", int, C.DEFAULTS["n"], choices=C.N_CHOICES),
    "szenario": SettingSpec("szen", str, C.DEFAULTS["szenario"], choices=C.SZENARIO_KEYS),
    "streuung": SettingSpec("str", str, C.DEFAULTS["streuung"], choices=C.STREUUNG_KEYS),
    "seed": SettingSpec("seed", int, C.DEFAULTS["seed"], 0, C.SEED_MAX),
    "ref_kunde": SettingSpec("rk", int, C.DEFAULTS["ref_kunde"], C.KUNDE_MIN, C.KUNDE_MAX),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt; unbekannte Auswahlwerte werden verworfen."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.choices is not None and value not in spec.choices:
                continue
            if spec.lo is not None:
                value = max(spec.lo, value)
            if spec.hi is not None:
                value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Lädt ein Szenario: alle Regler werden gesetzt, die das Preset nicht nennt, auf ihren Standard."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(state_key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden (setzt alle Regler):")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed"] = random.randint(0, C.SEED_MAX)
