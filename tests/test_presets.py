"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jeder Preset-Stand ist eindeutig."""

import fcd_constants as C
import fcd_evaluation as ev
import fcd_presets as P


def _state(name):
    return {k: C.PRESETS[name].get(k, spec.default) for k, spec in P.SETTING_SPECS.items()}


def test_every_preset_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    assert all(C.PRESET_HELP[n] for n in C.PRESETS)
    assert len(C.PRESETS) == 8


def test_preset_keys_are_known_settings():
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name


def test_preset_values_are_inside_the_regler_bounds_or_choices():
    for name in C.PRESETS:
        for key, value in _state(name).items():
            spec = P.SETTING_SPECS[key]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            if spec.lo is not None:
                assert spec.lo <= value <= spec.hi, (name, key)
            assert isinstance(value, spec.caster), (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert C.PRESETS[first] == {} and _state(first) == {k: s.default for k, s in P.SETTING_SPECS.items()}
    assert ev.settings_from_state(lambda k: _state(first)[k]) == ev.Settings()


def test_setting_specs_cover_every_default_and_no_more():
    assert set(P.SETTING_SPECS) == set(C.DEFAULTS)


def test_every_preset_state_is_unique_so_the_highlight_is_unambiguous():
    states = [tuple(sorted(_state(n).items())) for n in C.PRESETS]
    assert len(set(states)) == len(states)


def test_h2_preset_sets_the_cell_sizes_2_4_3_1_2_2():
    s = ev.settings_from_state(lambda k: _state("Unausgewogen (H2)")[k])
    assert s.sizes == (2, 4, 3, 1, 2, 2) == C.HAND_H2 and sum(s.sizes) == 14


def test_scenario_presets_name_their_scenario_and_scatter():
    assert _state("Additiv")["szenario"] == "additiv" and _state("Überkreuz")["szenario"] == "Überkreuz"
    assert _state("Überkreuz, Streuung des Generators")["streuung"] == "Generator" and _state("Überkreuz")["streuung"] == "gleich"
    assert _state("Dummy-Falle")["n"] == 200 and _state("Zwölf Kunden")["ref_kunde"] == 4


def test_every_preset_row_exists_in_the_frozen_series():
    for name in C.PRESETS:
        st = _state(name)
        assert ev.frozen_row(st["streuung"], st["szenario"], st["n"]) and ev.frozen_row(st["streuung"], st["szenario"], 200)
