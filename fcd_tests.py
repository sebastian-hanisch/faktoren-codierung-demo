"""Zellstatistik, Wald-Tests in Zellmittel-Parametern und exakte Teststärke (nichtzentrale F-Verteilung).

Zweiter Rechenweg zu den F-Tests der Entwurfsmatrix (`fcd_design.drop_test`): Das volle Modell mit Wechselwirkung passt die Zellmittel an; jeder Haupteffekt-Test prüft eine lineare Hypothese über die
Zellmittel θ_ij (Zeile i = Stufe von A, Spalte j = Stufe von B), und der F-Wert ist ihr Wald-Wert geteilt durch die Zahl der Bedingungen:
  Wochenende:  ψ = Σ_i ω_i (θ_i1 − θ_i0)   mit den Gewichten ω der Codierung von A (Referenz: eine Stufe; Summe, Helmert: Mittel; gewichtet: Stichprobenanteile)
  Art:         ψ_i = Σ_j ω_j θ_ij  sollen für alle i gleich sein (ω aus der Codierung von B)
  Wechselwirkung: die Wochenendeffekte θ_i1 − θ_i0 sollen für alle i gleich sein
Die Schätzer der Zeilen sind unabhängig, Varianz s²·Σ_j ω_ij²/n_ij; „alle gleich“ ergibt F = Σ_i p_i (ψ_i − ψ̄_p)² / (k − 1) mit den Genauigkeiten p_i = 1/Var_i und dem
mit ihnen gewichteten Mittel ψ̄_p (Rest-Freiheitsgrade n − k_A·k_B)."""

import numpy as np
from scipy import stats

import fcd_constants as C
from fcd_design import hypothesis_weights


def cell_stats(a, b, y, ka: int, kb: int):
    """Zellgrößen, Zellmittel (nan bei leerer Zelle) und Fehlerquadratsumme innerhalb der Zellen."""
    a, b, y = np.asarray(a, dtype=int), np.asarray(b, dtype=int), np.asarray(y, dtype=float)
    idx = a * kb + b
    n = np.bincount(idx, minlength=ka * kb).astype(float)
    s = np.bincount(idx, weights=y, minlength=ka * kb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.where(n > 0, n, 1.0), np.nan)
    within = float(np.sum((y - np.nan_to_num(mean)[idx]) ** 2))
    return n.reshape(ka, kb), mean.reshape(ka, kb), within


def _equal_rows_F(psi: np.ndarray, var: np.ndarray) -> float:
    """F-Wert der Hypothese „alle ψ_i gleich“ bei bekannten (geschätzten) Varianzen var_i der unabhängigen Schätzer."""
    p = 1.0 / var
    pm = float(np.sum(p * psi) / np.sum(p))
    return float(np.sum(p * (psi - pm) ** 2) / (len(psi) - 1))


def wochenende_test(n, mean, within: float, omega_a):
    """Haupteffekt B bei Wechselwirkung: (F, p, getesteter Effekt, Standardabweichung des Effekts, Freiheitsgrade). Nur bei Zellen ohne Lücke und Rest-Freiheitsgraden."""
    n, mean = np.asarray(n), np.asarray(mean)
    omega_a = np.asarray(omega_a, dtype=float)
    dfres = int(n.sum()) - n.size
    if np.any(n == 0):
        return float("nan"), float("nan"), float("nan"), float("nan"), dfres
    s2 = within / dfres if dfres > 0 else float("nan")
    delta = mean[:, 1] - mean[:, 0]
    d = float(omega_a @ delta)
    sd = float(np.sqrt(s2 * np.sum(omega_a ** 2 * (1.0 / n[:, 0] + 1.0 / n[:, 1])))) if dfres > 0 else float("nan")
    if dfres <= 0 or s2 <= 0:
        return float("nan"), float("nan"), d, sd, dfres
    F = (d / sd) ** 2
    return F, float(stats.f.sf(F, 1, dfres)), d, sd, dfres


def art_test(n, mean, within: float, omega_b):
    """Haupteffekt A bei Wechselwirkung: (F, p, Freiheitsgrade des Zählers, Freiheitsgrade des Rests); omega_b: Vektor (k_B) oder Matrix (k_A × k_B, je Zeile eigene Gewichte)."""
    n, mean = np.asarray(n), np.asarray(mean)
    ka, kb = n.shape
    om = np.broadcast_to(np.asarray(omega_b, dtype=float), (ka, kb))
    dfres = int(n.sum()) - n.size
    if dfres <= 0 or within <= 0 or np.any(n == 0):
        return float("nan"), float("nan"), ka - 1, dfres
    s2 = within / dfres
    psi = np.sum(om * mean, axis=1)
    var = s2 * np.sum(om ** 2 / n, axis=1)
    F = _equal_rows_F(psi, var)
    return F, float(stats.f.sf(F, ka - 1, dfres)), ka - 1, dfres


def interaction_test(n, mean, within: float):
    """Wechselwirkung: die Wochenendeffekte θ_i1 − θ_i0 sind für alle Arten gleich: (F, p, Freiheitsgrade des Zählers, des Rests)."""
    n, mean = np.asarray(n), np.asarray(mean)
    ka = n.shape[0]
    dfres = int(n.sum()) - n.size
    if dfres <= 0 or within <= 0 or np.any(n == 0):
        return float("nan"), float("nan"), ka - 1, dfres
    s2 = within / dfres
    delta = mean[:, 1] - mean[:, 0]
    var = s2 * (1.0 / n[:, 0] + 1.0 / n[:, 1])
    F = _equal_rows_F(delta, var)
    return F, float(stats.f.sf(F, ka - 1, dfres)), ka - 1, dfres


def typ1_row_weights(n) -> np.ndarray:
    """Typ I (A zuerst): je Art die nach Zellgrößen gewichteten Stufen von B (Zeilenmittel der Beobachtungen)."""
    n = np.asarray(n, dtype=float)
    return n / n.sum(axis=1, keepdims=True)


def single_coefficient_t(n, mean, within: float, i: int, ref: int):
    """Einzelkoeffizient „Art i × Wochenende“ bei Referenzstufe ref (Referenzcodierung): Differenz der Wochenendeffekte δ_i − δ_ref, t-Wert und Freiheitsgrade."""
    n, mean = np.asarray(n), np.asarray(mean)
    dfres = int(n.sum()) - n.size
    if dfres <= 0 or within <= 0 or np.any(n[[i, ref]] == 0):
        return float("nan"), float("nan"), dfres
    s2 = within / dfres
    delta = mean[:, 1] - mean[:, 0]
    var = s2 * (1.0 / n[i, 0] + 1.0 / n[i, 1] + 1.0 / n[ref, 0] + 1.0 / n[ref, 1])
    t = (delta[i] - delta[ref]) / np.sqrt(var)
    return float(t), float(2.0 * stats.t.sf(abs(t), dfres)), dfres


# --- exakte Teststärke -------------------------------------------------------------------------------------------------------------------------

def noncentral_power(crit_q: int, dfres: int, lam: float, alpha: float = C.ALPHA) -> float:
    """Teststärke eines F-Tests mit q Zähler- und dfres Nenner-Freiheitsgraden bei Nichtzentralität λ. Bei λ < 1e-12 (Nullhypothese wahr) gilt das Niveau: `scipy.stats.ncf.sf` lieferte bei λ = 0 einen
    falschen Wert (gemessen −0,95 statt 0,05, scipy 1.18.1), darum die zentrale Verteilung."""
    if lam < 1e-12:
        return alpha
    fcrit = stats.f.ppf(1.0 - alpha, crit_q, dfres)
    return float(stats.ncf.sf(fcrit, crit_q, dfres, lam))


def power_wochenende(theta, n_cell, sigma: float, omega_a, alpha: float = C.ALPHA):
    """Exakte Teststärke des Haupteffekt-Tests Wochenende (1 Freiheitsgrad) bei wahren Zellmitteln θ (k_A × 2), erwarteten Zellgrößen n_cell und gleicher Streuung σ; Rückgabe (Teststärke, Nichtzentralität)."""
    theta, n_cell, omega_a = np.asarray(theta), np.asarray(n_cell), np.asarray(omega_a, dtype=float)
    dfres = int(round(float(n_cell.sum()))) - theta.size
    var_cell = sigma ** 2 / n_cell
    d = float(omega_a @ (theta[:, 1] - theta[:, 0]))
    var = float(np.sum(omega_a ** 2 * (var_cell[:, 1] + var_cell[:, 0])))
    lam = d * d / var
    return noncentral_power(1, dfres, lam, alpha), lam


def power_art(theta, n_cell, sigma: float, omega_b, alpha: float = C.ALPHA):
    """Exakte Teststärke des Haupteffekt-Tests Art (k_A − 1 Freiheitsgrade); omega_b: Gewichte der Stufen von B."""
    theta, n_cell, omega_b = np.asarray(theta), np.asarray(n_cell), np.asarray(omega_b, dtype=float)
    ka = theta.shape[0]
    dfres = int(round(float(n_cell.sum()))) - theta.size
    var_cell = sigma ** 2 / n_cell
    psi = theta @ omega_b
    varpsi = (var_cell * omega_b[None, :] ** 2).sum(axis=1)
    L = np.zeros((ka - 1, ka))
    L[:, 0] = -1.0
    L[:, 1:] = np.eye(ka - 1)
    Lt = L @ psi
    lam = float(Lt @ np.linalg.solve(L @ np.diag(varpsi) @ L.T, Lt))
    return noncentral_power(ka - 1, dfres, lam, alpha), lam


# --- die geprüften Einstellungen der Experimente --------------------------------------------------------------------------------------------

def settings_list(ka: int, art_labels) -> list[dict]:
    """Die Codierungs-Einstellungen, die Experiment 2 und die Messreihen nebeneinander zeigen: Referenz je Art (B-Referenz Wochentag), Referenz erste Art mit B-Referenz Wochenende, Summe, Helmert, gewichtet."""
    out = [{"name": f"Referenz {art_labels[r]}", "kind": C.COD_REFERENZ, "ref_a": r, "ref_b": 0} for r in range(ka)]
    out.append({"name": f"Referenz {art_labels[0]} / Wochenende", "kind": C.COD_REFERENZ, "ref_a": 0, "ref_b": 1})
    out.append({"name": "Summe", "kind": C.COD_SUMME, "ref_a": 0, "ref_b": 0})
    out.append({"name": "Helmert", "kind": C.COD_HELMERT, "ref_a": 0, "ref_b": 0})
    out.append({"name": "Gewichtet zentriert", "kind": C.COD_GEWICHTET, "ref_a": 0, "ref_b": 0})
    return out


def setting_weights(setting: dict, shares_a, shares_b):
    """(ω_A, ω_B) einer Einstellung aus `settings_list`: ω_A mittelt die Stufen von A im Wochenende-Test, ω_B die Stufen von B im Art-Test."""
    kind = setting["kind"]
    return (hypothesis_weights(kind, len(shares_a), setting["ref_a"], shares_a), hypothesis_weights(kind, len(shares_b), setting["ref_b"], shares_b))
