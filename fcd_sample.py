"""Zufallszahlen aus SplitMix64, vektorisiert (numpy), auf jeder Plattform bitgleich bei den Gleichverteilten und nur durch Grundrechenarten bei den Normalzahlen.

Herkunft: `uniform` und `u64` sind die vektorisierte Fassung der skalaren Folge in `fcd_rng.py` (Kopie der Vorlage `rng_splitmix64_exact.py`); die i-te Ausgabe (i = 1, 2, ...) des
Generators mit Zufalls-Seed s ist mix(s + i · γ) mod 2⁶⁴. Die Zahl ist (Ausgabe >> 11) · 2⁻⁵³: reine Ganzzahlarithmetik, keine C-Bibliothek.

Normalzahlen (Box-Muller) brauchen einen Logarithmus und einen Kosinus. `numpy.log` und `numpy.cos` kommen je nach Plattform und CPU-Befehlssatz aus verschiedenen Bibliotheken und können in der
letzten Binärstelle abweichen. Darum rechnet `normal_from_uniform` beide Funktionen aus Grundrechenarten (+, −, ·, /, √ sind nach IEEE 754 exakt gerundet und auf jeder Plattform gleich): den
Logarithmus über `frexp` und die Potenzreihe für artanh, den Kosinus über eine Taylorreihe auf [0, π/2]. Der Fehler liegt bei wenigen Einheiten der letzten Stelle (gemessen gegen die mit `decimal`
gerechnete skalare Fassung: `tests/test_sample.py`); die Folge ist auf jeder Plattform dieselbe, weil jede Operation einzeln gerundet wird (numpy verschmilzt getrennte ufunc-Aufrufe nicht)."""

import math

import numpy as np

MASK = (1 << 64) - 1
_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def u64(seed, count, offset=0):
    """Die Ausgaben offset + 1 ... offset + count des Generators mit Zufalls-Seed `seed` als uint64-Feld."""
    idx = np.arange(offset + 1, offset + count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed & MASK) + idx * _GAMMA
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        z = z ^ (z >> np.uint64(31))
    return z


def uniform(seed, count, offset=0):
    """Gleichverteilt in [0, 1) mit 53 Bit."""
    return (u64(seed, count, offset) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def uniform_rows(seed, rows, cols, first_row=0):
    """Matrix (rows, cols): Zeile i (von 0 an) nimmt die Ausgaben (first_row + i)·cols + 1 ... (first_row + i + 1)·cols. Das Präfix ist stabil: die ersten k Zeilen hängen nicht von `rows` ab."""
    return uniform(seed, rows * cols, first_row * cols).reshape(rows, cols)


# --- Logarithmus und Kosinus aus Grundrechenarten --------------------------------------------------------------------------------------------
_LN2 = 0.6931471805599453                              # ln 2 als Literal (kein Funktionsaufruf, damit die Konstante nicht von der Plattform abhängt)
_SQRT_HALF = 0.7071067811865476
_ARTANH = [1.0 / (2 * k + 1) for k in range(14)]       # 1, 1/3, 1/5, ...
_COS = [(-1.0) ** k / float(math.factorial(2 * k)) for k in range(14)]
_TWO_PI = 6.283185307179586


def ln_pos(x):
    """Natürlicher Logarithmus für x > 0 (Feld), nur mit Grundrechenarten; Fehler wenige Einheiten der letzten Stelle."""
    m, e = np.frexp(np.asarray(x, dtype=np.float64))   # x = m · 2^e, m in [0,5, 1): beide Ergebnisse sind exakt
    small = m < _SQRT_HALF
    m = np.where(small, m * 2.0, m)                    # nun m in [√½, √2)
    e = np.where(small, e - 1, e).astype(np.float64)
    s = (m - 1.0) / (m + 1.0)                          # |s| <= 0,1716
    s2 = s * s
    p = np.full_like(s, _ARTANH[-1])
    for c in _ARTANH[-2::-1]:
        p = p * s2 + c
    return e * _LN2 + 2.0 * s * p


def cos_2pi(t):
    """cos(2π t) für t in [0, 1] (Feld), nur mit Grundrechenarten."""
    t = np.asarray(t, dtype=np.float64)
    a = np.abs(t - np.rint(t))                         # Abstand zur nächsten ganzen Zahl, in [0, 0,5]
    big = a > 0.25
    a = np.where(big, 0.5 - a, a)                      # in [0, 0,25]; cos(2π a) = −cos(2π (0,5 − a)) für a > 0,25
    phi = _TWO_PI * a
    q = phi * phi
    p = np.full_like(q, _COS[-1])
    for c in _COS[-2::-1]:
        p = p * q + c
    return np.where(big, -p, p)


def normal_from_uniform(u1, u2):
    """Standardnormalverteilt (Box-Muller) aus zwei gleichverteilten Feldern; gleiche Zuordnung wie `fcd_rng.SplitMix64.normal` (u1 zuerst, mit Untergrenze 1e-12)."""
    u1 = np.maximum(u1, 1e-12)
    return np.sqrt(-2.0 * ln_pos(u1)) * cos_2pi(u2)
