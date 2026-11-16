"""Messreihen über viele Wiederholungen (Zähleinheit: Wiederholung = eine Stichprobe von n Containern), über alle Wiederholungen gestapelt gerechnet.

Gerechnet wird mit den Wald-Formeln in Zellmittel-Parametern (`fcd_tests`), nicht mit einer Entwurfsmatrix je Wiederholung: Das volle Modell mit Wechselwirkung passt die Zellmittel an,
und jeder Haupteffekt-Test ist eine lineare Hypothese über sie (`tests/test_oracle_faktoren_codierung.py` vergleicht beide Wege Bruch gegen Bruch und auf Zufallsdaten). Die Messreihen der App
sind vorgerechnet (`data/fcd_frozen.json`, erzeugt von `tools/fcd_make_frozen.py`); dieses Modul ist ihre Quelle und wird von dort und von den Tests aufgerufen.

Zufall: SplitMix64 (`fcd_sample`). Eine Zeile (Streuung, Szenario, n) hat EINEN Strom mit eigenem Zufalls-Seed; Wiederholung r nimmt die Container r·n bis (r + 1)·n − 1 dieses Stroms,
so ändert die Blockgröße der Rechnung nichts am Ergebnis."""

import numpy as np
from scipy import stats

import fcd_constants as C
import fcd_generator as G
import fcd_tests as T

KA, KB = 4, 2
SETTINGS = T.settings_list(KA, C.ART_LABELS)           # 8 Einstellungen: Referenz Import/Export/Umschlag/Leer, Referenz Import / Wochenende, Summe, Helmert, Gewichtet zentriert
SETTING_NAMES = tuple(s["name"] for s in SETTINGS)
GRID = tuple((mode, szen, n) for mode in C.STREUUNG_KEYS for szen in C.SZENARIO_KEYS for n in C.N_CHOICES)


def row_seed(mode: str, szenario: str, n: int) -> int:
    """Zufalls-Seed des Stroms einer Zeile (Streuung, Szenario, n): SIM_BASE_SEED + 1 + laufende Nummer in `GRID`."""
    return C.SIM_BASE_SEED + 1 + GRID.index((mode, szenario, n))


def _weights(setting: dict, sh_a: np.ndarray, sh_b: np.ndarray):
    """ω_A und ω_B je Wiederholung (Zeilen = Wiederholungen). Referenz: eine Stufe; Summe, Helmert: Mittel; Gewichtet: Stichprobenanteile der Wiederholung."""
    kind = setting["kind"]
    reps = sh_a.shape[0]
    if kind == C.COD_REFERENZ:
        oa, ob = np.zeros((reps, KA)), np.zeros((reps, KB))
        oa[:, setting["ref_a"]] = 1.0
        ob[:, setting["ref_b"]] = 1.0
        return oa, ob
    if kind in (C.COD_SUMME, C.COD_HELMERT):
        return np.full((reps, KA), 1.0 / KA), np.full((reps, KB), 1.0 / KB)
    return sh_a, sh_b


def _equal_rows_F(psi: np.ndarray, var: np.ndarray) -> np.ndarray:
    p = 1.0 / var
    pm = np.sum(p * psi, axis=1, keepdims=True) / np.sum(p, axis=1, keepdims=True)
    return np.sum(p * (psi - pm) ** 2, axis=1) / (psi.shape[1] - 1)


def simulate_row(mode: str, szenario: str, n: int, reps: int = C.SIM_REPS, chunk: int = 125) -> dict:
    """Eine Zeile der Messreihe: Häufigkeiten, mit denen die Tests bei Niveau 5 % verwerfen (als Zählungen und Zahl der gültigen Wiederholungen)."""
    w = C.SZENARIEN[szenario]
    seed = row_seed(mode, szenario, n)
    dfres = n - KA * KB
    crit1, crit3 = stats.f.ppf(1 - C.ALPHA, 1, dfres), stats.f.ppf(1 - C.ALPHA, KA - 1, dfres)
    tcrit = stats.t.ppf(1 - C.ALPHA / 2, dfres)
    rej_B, rej_A = np.zeros(len(SETTINGS), dtype=int), np.zeros(len(SETTINGS), dtype=int)
    rej_I = sig_li = sig_le = valid = 0
    for r0 in range(0, reps, chunk):
        m = min(chunk, reps - r0)
        c = G.draw(seed, m * n, first=r0 * n)
        y = G.outcome(c, mode, w).reshape(m, n)
        a, b = c.art.reshape(m, n), c.we.reshape(m, n)
        idx = (np.arange(m)[:, None] * (KA * KB) + a * KB + b).ravel()
        cn = np.bincount(idx, minlength=m * KA * KB).astype(float).reshape(m, KA, KB)
        cs = np.bincount(idx, weights=y.ravel(), minlength=m * KA * KB).reshape(m, KA, KB)
        ok = np.all(cn > 0, axis=(1, 2))                 # leere Zelle: Wechselwirkung nicht schätzbar, Wiederholung wird nicht gezählt
        valid += int(ok.sum())
        if not ok.any():
            continue
        cn, cs, y, a = cn[ok], cs[ok], y[ok], a[ok]
        mean = cs / cn
        within = np.sum(y ** 2, axis=1) - np.sum(cn * mean ** 2, axis=(1, 2))
        s2 = within / dfres
        delta = mean[:, :, 1] - mean[:, :, 0]
        inv0, inv1 = 1.0 / cn[:, :, 0], 1.0 / cn[:, :, 1]
        sh_a = cn.sum(axis=2) / n
        sh_b = cn.sum(axis=1) / n
        for k, st in enumerate(SETTINGS):
            oa, ob = _weights(st, sh_a, sh_b)
            d = np.sum(oa * delta, axis=1)
            var = s2 * np.sum(oa ** 2 * (inv0 + inv1), axis=1)
            rej_B[k] += int(np.sum(d * d / var > crit1))
            psi = np.sum(ob[:, None, :] * mean, axis=2)
            varpsi = s2[:, None] * np.sum(ob[:, None, :] ** 2 / cn, axis=2)
            rej_A[k] += int(np.sum(_equal_rows_F(psi, varpsi) > crit3))
        varI = s2[:, None] * (inv0 + inv1)
        rej_I += int(np.sum(_equal_rows_F(delta, varI) > crit3))
        for ref, name in ((0, "import"), (1, "export")):
            t = (delta[:, 3] - delta[:, ref]) / np.sqrt(s2 * (inv0[:, 3] + inv1[:, 3] + inv0[:, ref] + inv1[:, ref]))
            if name == "import":
                sig_li += int(np.sum(np.abs(t) > tcrit))
            else:
                sig_le += int(np.sum(np.abs(t) > tcrit))
    return {"mode": mode, "szenario": szenario, "n": n, "seed": seed, "reps": reps, "valid": valid, "leer": reps - valid,
            "rej_B": [int(x) for x in rej_B], "rej_A": [int(x) for x in rej_A], "rej_I": rej_I, "sig_leer_import": sig_li, "sig_leer_export": sig_le}


def simulate_kunde(reps: int = C.KUNDE_REPS, n: int = C.N_KUNDE_EXP, chunk: int = 200) -> dict:
    """Experiment Referenzkunde: Import-Container, Kunde als Faktor mit 12 Stufen, Koeffizienten gegen jede mögliche Referenzstufe.
    Gezählt wird je Referenz die mittlere Zahl der 11 Koeffizienten mit |t| > t(0,975), mit den Kundeneffekten des Generators und ohne sie, sowie der F-Test des ganzen Faktors."""
    k = C.K_KUNDEN
    dfres = n - k
    tcrit = stats.t.ppf(1 - C.ALPHA / 2, dfres)
    fcrit = stats.f.ppf(1 - C.ALPHA, k - 1, dfres)
    cnt_alt = np.zeros(k)
    any_h0 = np.zeros(k, dtype=int)
    cnt_h0 = np.zeros(k)
    f_alt = f_h0 = valid = 0
    for r0 in range(0, reps, chunk):
        m = min(chunk, reps - r0)
        c = G.draw_import(C.KUNDE_SEED, m * n, first=r0 * n)
        kunde = c.kunde.reshape(m, n)
        idx = (np.arange(m)[:, None] * k + kunde).ravel()
        cn = np.bincount(idx, minlength=m * k).astype(float).reshape(m, k)
        ok = np.all(cn > 0, axis=1)
        valid += int(ok.sum())
        for with_kunde in (True, False):
            y = G.import_outcome(c, with_kunde).reshape(m, n)
            cs = np.bincount(idx, weights=y.ravel(), minlength=m * k).reshape(m, k)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = cs / cn
                within = np.sum(y ** 2, axis=1) - np.nansum(cn * mean ** 2, axis=1)
                s2 = within / dfres
                gm = y.mean(axis=1, keepdims=True)
                F = np.sum(cn * (mean - gm) ** 2, axis=1) / (k - 1) / s2
                for ref in range(k):
                    others = [j for j in range(k) if j != ref]
                    t = (mean[:, others] - mean[:, [ref]]) / np.sqrt(s2[:, None] * (1.0 / cn[:, others] + 1.0 / cn[:, [ref]]))
                    sig = (np.abs(t) > tcrit) & ok[:, None]
                    if with_kunde:
                        cnt_alt[ref] += sig.sum()
                    else:
                        any_h0[ref] += int(np.sum(sig.any(axis=1)))
                        cnt_h0[ref] += sig.sum()
            if with_kunde:
                f_alt += int(np.sum((F > fcrit) & ok))
            else:
                f_h0 += int(np.sum((F > fcrit) & ok))
    return {"n": n, "reps": reps, "valid": valid, "seed": C.KUNDE_SEED, "kunde_effekte": [float(x) for x in C.UK], "sig_sum_by_ref": [float(x) for x in cnt_alt],
            "f_alt": f_alt, "f_h0": f_h0, "any_h0_by_ref": [int(x) for x in any_h0], "sig_sum_h0_by_ref": [float(x) for x in cnt_h0]}


def exact_power_row(mode: str, szenario: str, n: int):
    """Exakte Teststärke (nichtzentrale F, gleiche Streuung σ_pool, erwartete Zellgrößen) der acht Einstellungen für Wochenende und Art; nur sinnvoll für die Streuung „gleich“."""
    theta = G.cell_means_exact(C.SZENARIEN[szenario])
    sigma = G.sd_pool_exact()
    pa, pb = np.array(C.ART_P), np.array([1 - C.P_WE, C.P_WE])
    ncell = n * np.outer(pa, pb)
    out = []
    for st in SETTINGS:
        oa, ob = T.setting_weights(st, pa, pb)
        pw_b, lam_b = T.power_wochenende(theta, ncell, sigma, oa)
        pw_a, lam_a = T.power_art(theta, ncell, sigma, ob)
        out.append({"name": st["name"], "effect": float(oa @ (theta[:, 1] - theta[:, 0])), "power_B": pw_b, "power_A": pw_a})
    return out


def realised_power_row(mode: str, szenario: str, n: int, reps: int = C.SIM_REPS, chunk: int = 250):
    """Mittlere Teststärke des Haupteffekt-Tests Wochenende über die TATSÄCHLICHEN Zellgrößen der Wiederholungen derselben Zeile (gleicher Strom wie `simulate_row`), nichtzentrale F mit gleicher Streuung σ_pool.
    Die exakte Teststärke in `exact_power_row` setzt dagegen die erwarteten Zellgrößen ein; die Teststärke ist in 1/n konvex, das Mittel über zufällige Zellgrößen liegt darunter.
    Nur sinnvoll für die Streuung „gleich“. Rückgabe: Liste mit dem Mittel je Einstellung (Reihenfolge SETTINGS)."""
    w = np.array(C.SZENARIEN[szenario])
    seed = row_seed(mode, szenario, n)
    theta = G.cell_means_exact(w)
    delta_true = theta[:, 1] - theta[:, 0]
    sigma = G.sd_pool_exact()
    dfres = n - KA * KB
    total = np.zeros(len(SETTINGS))
    valid = 0
    for r0 in range(0, reps, chunk):
        m = min(chunk, reps - r0)
        c = G.draw(seed, m * n, first=r0 * n)
        idx = (np.arange(m)[:, None] * (KA * KB) + c.art.reshape(m, n) * KB + c.we.reshape(m, n)).ravel()
        cn = np.bincount(idx, minlength=m * KA * KB).astype(float).reshape(m, KA, KB)
        ok = np.all(cn > 0, axis=(1, 2))
        cn = cn[ok]
        valid += int(ok.sum())
        sh_a = cn.sum(axis=2) / n
        sh_b = cn.sum(axis=1) / n
        for k, st in enumerate(SETTINGS):
            oa, _ = _weights(st, sh_a, sh_b)
            d = oa @ delta_true
            var = sigma ** 2 * np.sum(oa ** 2 * (1.0 / cn[:, :, 0] + 1.0 / cn[:, :, 1]), axis=1)
            lam = d * d / var
            pw = np.where(lam > 1e-12, 0.0, C.ALPHA)
            big = lam > 1e-12
            if big.any():
                pw[big] = stats.ncf.sf(stats.f.ppf(1 - C.ALPHA, 1, dfres), 1, dfres, lam[big])
            total[k] += pw.sum()
    return [float(x / valid) for x in total]
