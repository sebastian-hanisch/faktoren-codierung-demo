"""Zufall: die vektorisierte SplitMix64-Folge (`fcd_sample`) gegen die skalare Referenz (`fcd_rng`, Kopie der geprüften Vorlage) und gegen feste Handwerte.

Gleichverteilte Zahlen sind bitgleich (Ganzzahlarithmetik); Normalzahlen unterscheiden sich vom `decimal`-Wert der Vorlage um wenige Einheiten der letzten Stelle (Grundrechenarten statt 45-stelliger Dezimalrechnung)."""

import re
from pathlib import Path

import numpy as np

import fcd_rng
import fcd_sample as S

ROOT = Path(__file__).resolve().parent.parent


def test_uniform_ist_bitgleich_zur_skalaren_folge():
    for seed in (0, 1, 12345, 2 ** 63 + 5):
        g = fcd_rng.SplitMix64(seed)
        scalar = np.array([g.random() for _ in range(500)])
        assert np.array_equal(S.uniform(seed, 500), scalar), seed


def test_erste_ausgaben_des_generators_mit_seed_0_sind_die_bekannten_splitmix64_werte():
    """SplitMix64 mit Seed 0: erste Ausgabe 0xE220A8397B1DCDAF (Referenzimplementierung von Vigna, public domain)."""
    assert int(S.u64(0, 1)[0]) == 0xE220A8397B1DCDAF
    assert fcd_rng.SplitMix64(0).next_u64() == 0xE220A8397B1DCDAF


def test_normalzahlen_stimmen_mit_der_dezimalfassung_der_vorlage_ueberein():
    g = fcd_rng.SplitMix64(777)
    scalar = np.array([g.normal() for _ in range(2000)])
    u = S.uniform(777, 4000).reshape(2000, 2)
    vec = S.normal_from_uniform(u[:, 0], u[:, 1])
    assert np.max(np.abs(vec - scalar)) < 1e-13


def test_logarithmus_und_kosinus_gegen_die_dezimalfassung():
    x = np.concatenate([np.linspace(1e-12, 1.0, 301), [0.5, 0.25, 0.7071067811865476, 1.0 - 1e-16]])
    assert np.max(np.abs(S.ln_pos(x) - np.array([fcd_rng.exact_log(float(v)) for v in x]))) < 5e-15
    t = np.linspace(0.0, 1.0, 401)
    assert np.max(np.abs(S.cos_2pi(t) - np.array([fcd_rng.exact_cos(2.0 * np.pi * float(v)) for v in t]))) < 5e-15


def test_normalzahlen_haben_mittel_null_und_streuung_eins():
    u = S.uniform(4242, 400000).reshape(200000, 2)
    z = S.normal_from_uniform(u[:, 0], u[:, 1])
    se = 1 / np.sqrt(z.size)
    assert abs(z.mean()) < 4 * se and abs(z.std() - 1) < 4 / np.sqrt(2 * z.size)
    assert abs(np.mean(np.abs(z) < 1.959963984540054) - 0.95) < 4 * np.sqrt(0.95 * 0.05 / z.size)   # 95 %-Quantil der Normalverteilung


def test_zeilenpraefix_ist_stabil():
    a = S.uniform_rows(9, 50, 7)
    b = S.uniform_rows(9, 80, 7)
    assert np.array_equal(a, b[:50])
    assert np.array_equal(S.uniform_rows(9, 30, 7, first_row=20), b[20:50])


def test_kein_libm_und_kein_numpy_random_auf_dem_zufallsweg():
    """Der Strom der App darf weder `numpy.random` noch `math.log`/`math.cos`/`np.log`/`np.cos`/`np.exp` benutzen (Windows und Linux unterscheiden sich dort in der letzten Stelle)."""
    for name in ("fcd_sample.py", "fcd_generator.py", "fcd_simulation.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for verboten in ("default_rng", "numpy.random", "np.random", "math.log", "math.cos", "np.log(", "np.cos(", "np.exp(", "random.random", "hash("):
            assert verboten not in _strip_docstrings(text), (name, verboten)


def _strip_docstrings(text: str) -> str:
    return re.sub(r'""".*?"""', "", text, flags=re.S)
