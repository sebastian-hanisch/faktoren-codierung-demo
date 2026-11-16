"""Container-Generator: Art, Kunde, Zollhalt, Wochenende, Vorlauf und die log-Verweilzeit; vektorisiert, Zufall nur aus SplitMix64 (`fcd_sample`).

Nachbau (kein Import) der Verteilung aus der `verweilzeit-demo`; der Strom ist nicht bitgleich zu dem dortigen (dort numpy `default_rng`), die Verteilung ist dieselbe:
  Art           Import 35 %, Export 30 %, Umschlag 25 %, Leer 10 %
  Kunde         gleichverteilt auf 12 Kunden mit festen Effekten `UK`, bei Export und Umschlag mit Faktor 0,3
  Zollhalt      bei Import mit Wahrscheinlichkeit 0,25 (verlängert den Median um den Faktor exp(0,7))
  Wochenende    Wahrscheinlichkeit 2/7; bei Import und Leer Median mal exp(0,30)
  Vorlauf       Export gleichverteilt (1, 10), Umschlag gleichverteilt (0,5, 6), sonst 0
  Median        Import 3,5·exp(0,7·Zoll + 0,3·WE), Export 0,6 + 0,85·Vorlauf, Umschlag 0,3 + 0,8·Vorlauf, Leer 7·exp(0,3·WE)
  Verweilzeit   lognormal: log-Verweilzeit = log(Median) + Kundeneffekt + s_Art · z, z ~ N(0, 1)

Der Logarithmus der Mediane kommt aus `fcd_sample.ln_pos` (Grundrechenarten), damit der Strom auf jeder Plattform gleich ist. Für den Auswertungsteil gibt es zwei Arten, den Ausgang zu bilden:
Streuung „gleich“ (Zellmittel exakt, gleiche Normalstreuung σ in allen Zellen) und „Generator“ (Streuung je Art, Kunde, Zoll und Vorlauf bleiben im Rest)."""

from dataclasses import dataclass

import numpy as np

import fcd_constants as C
import fcd_sample as S

N_COLS = 7                                           # je Container 7 Ausgaben des Stroms: Art, Kunde, Zoll, Wochenende, Vorlauf, u1, u2 der Normalzahl
_CUM = np.cumsum(np.array(C.ART_P))
_UK = np.array(C.UK)
_UBAR = float(_UK.mean())
_U_SCALE = np.array(C.U_SCALE)
_S_BASE = np.array(C.S_BASE)
_LN35, _LN7 = float(S.ln_pos(3.5)), float(S.ln_pos(7.0))


@dataclass(frozen=True)
class Containers:
    art: np.ndarray        # 0 Import, 1 Export, 2 Umschlag, 3 Leer
    kunde: np.ndarray      # 0 bis 11
    zoll: np.ndarray       # 0 oder 1 (nur bei Import)
    we: np.ndarray         # 0 Wochentag, 1 Wochenende
    vorlauf: np.ndarray
    z: np.ndarray          # Standardnormalzahl je Container

    @property
    def n(self) -> int:
        return int(self.art.size)


def draw(seed: int, n: int, first: int = 0) -> Containers:
    """n Container aus dem Strom mit Zufalls-Seed `seed` (Container first ... first + n − 1); die ersten k Container hängen nicht von n ab."""
    u = S.uniform_rows(seed, n, N_COLS, first)
    art = np.minimum(np.searchsorted(_CUM, u[:, 0]), 3).astype(int)
    kunde = np.minimum((u[:, 1] * C.K_KUNDEN).astype(int), C.K_KUNDEN - 1)
    zoll = ((art == 0) & (u[:, 2] < C.P_ZOLL)).astype(int)
    we = (u[:, 3] < C.P_WE).astype(int)
    vorlauf = np.where(art == 1, 1.0 + 9.0 * u[:, 4], np.where(art == 2, 0.5 + 5.5 * u[:, 4], 0.0))
    return Containers(art, kunde, zoll, we, vorlauf, S.normal_from_uniform(u[:, 5], u[:, 6]))


def mu_log(c: Containers) -> np.ndarray:
    """Bedingter Mittelwert der log-Verweilzeit gegeben die Merkmale (log-Median plus Kundeneffekt), mit dem Wochenendeffekt 0,30 des Generators bei Import und Leer."""
    med_log = np.select(
        [c.art == 0, c.art == 1, c.art == 2],
        [_LN35 + 0.7 * c.zoll + 0.30 * c.we, S.ln_pos(0.6 + 0.85 * c.vorlauf), S.ln_pos(0.3 + 0.8 * c.vorlauf)],
        default=_LN7 + 0.30 * c.we,
    )
    return med_log + _UK[c.kunde] * _U_SCALE[c.art]


def e_log_linear(a0: float, b0: float, v0: float, v1: float) -> float:
    """E[ln(a0 + b0 v)] für v gleichverteilt auf (v0, v1), geschlossene Form."""
    f = lambda u: u * float(S.ln_pos(u)) - u
    return (f(a0 + b0 * v1) - f(a0 + b0 * v0)) / (b0 * (v1 - v0))


def var_log_linear(a0: float, b0: float, v0: float, v1: float) -> float:
    """Var[ln(a0 + b0 v)] für v gleichverteilt auf (v0, v1), geschlossene Form (E[ln²] = [u ln²u − 2u ln u + 2u]/(b0 (v1 − v0)))."""
    def g(u):
        lu = float(S.ln_pos(u))
        return u * lu * lu - 2.0 * u * lu + 2.0 * u
    e2 = (g(a0 + b0 * v1) - g(a0 + b0 * v0)) / (b0 * (v1 - v0))
    return e2 - e_log_linear(a0, b0, v0, v1) ** 2


def cell_means_exact(w) -> np.ndarray:
    """Wahre Zellmittel der log-Verweilzeit (4 Arten × 2) für den Wochenendeffekt w je Art: Wochentag-Spalte aus den Formeln des Generators, Wochenende plus w."""
    base = np.array([
        _LN35 + 0.7 * C.P_ZOLL + _UBAR * C.U_SCALE[0],
        e_log_linear(0.6, 0.85, 1.0, 10.0) + _UBAR * C.U_SCALE[1],
        e_log_linear(0.3, 0.8, 0.5, 6.0) + _UBAR * C.U_SCALE[2],
        _LN7 + _UBAR * C.U_SCALE[3],
    ])
    m = np.zeros((4, 2))
    m[:, 0] = base
    m[:, 1] = base + np.asarray(w, dtype=float)
    return m


def var_within_art() -> np.ndarray:
    """Varianz der log-Verweilzeit innerhalb einer Art (unabhängig vom Wochenende): s_Art² + Varianz des Kundeneffekts + Varianz des Mediansanteils aus Zoll beziehungsweise Vorlauf."""
    var_u = float(np.var(_UK))                       # 12 gleich wahrscheinliche Kunden: Varianz der Grundgesamtheit
    v = _S_BASE ** 2 + _U_SCALE ** 2 * var_u
    v[0] += 0.7 ** 2 * C.P_ZOLL * (1 - C.P_ZOLL)
    v[1] += var_log_linear(0.6, 0.85, 1.0, 10.0)
    v[2] += var_log_linear(0.3, 0.8, 0.5, 6.0)
    return v


def sd_pool_exact() -> float:
    """Gepoolte Streuung in den Zellen: Wurzel des mit den Artanteilen gewichteten Mittels der Varianzen (das Wochenende ändert die Varianz nicht)."""
    return float(np.sqrt(np.dot(np.array(C.ART_P), var_within_art())))


def outcome(c: Containers, streuung: str, w) -> np.ndarray:
    """Ausgang y (log-Verweilzeit). 'gleich': exakte Zellmittel + gleiche Normalstreuung σ_pool; 'Generator': Streuung je Art, Kunde, Zoll, Vorlauf im Rest."""
    w = np.asarray(w, dtype=float)
    if streuung == "gleich":
        return cell_means_exact(w)[c.art, c.we] + sd_pool_exact() * c.z
    return mu_log(c) + _S_BASE[c.art] * c.z + c.we * (w[c.art] - np.array(C.W_GEN)[c.art])


def draw_import(seed: int, n: int, first: int = 0) -> Containers:
    """Experiment Referenzkunde: n Import-Container (Art 0, Zollhalt mit Wahrscheinlichkeit 0,25, Wochenende mit 2/7, Kunde gleichverteilt)."""
    u = S.uniform_rows(seed, n, N_COLS, first)
    return Containers(np.zeros(n, dtype=int), np.minimum((u[:, 1] * C.K_KUNDEN).astype(int), C.K_KUNDEN - 1), (u[:, 2] < C.P_ZOLL).astype(int),
                      (u[:, 3] < C.P_WE).astype(int), np.zeros(n), S.normal_from_uniform(u[:, 5], u[:, 6]))


def import_outcome(c: Containers, with_kunde: bool = True) -> np.ndarray:
    """Ausgang eines Import-Containers: log 3,5 + 0,7·Zoll + 0,3·Wochenende (+ Kundeneffekt) + 0,55·z."""
    return _LN35 + 0.7 * c.zoll + 0.30 * c.we + (_UK[c.kunde] if with_kunde else 0.0) + _S_BASE[0] * c.z
