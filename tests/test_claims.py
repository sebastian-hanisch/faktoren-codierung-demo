"""Claims-Tests: jede Zahl der Befunde B5 bis B15 der README mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a).

B1 bis B4 und B16 stehen in `test_handbeispiele.py`, `test_oracle_faktoren_codierung.py` und `test_eigenschaften_faktoren_codierung.py`; B17 ist eine Messung der README (`tools/fcd_seed_check.py`), kein Test.

Plattform (Lehre 2026-11-15): Gleitkomma-Werte als Band (relative Toleranz oder absolute Stellen), Zählungen aus Neu-Rechnung ± 2, Anteile ± 0,01; exakt nur Brüche. Die vorgerechnete Messreihe
(`data/fcd_frozen.json`) wird auf Konsistenz mit den README-Zahlen (auf eine Stelle gerundet) und mit einer Neu-Rechnung einiger Zeilen geprüft; mit FCD_FULL=1 rechnet der Test sie ganz neu (rund 70 s)."""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

import fcd_constants as C
import fcd_evaluation as ev
import fcd_generator as G
import fcd_simulation as SIM

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
FROZEN = json.loads((ROOT / "data" / C.FROZEN_FILE).read_text(encoding="utf-8"))
NAMES = list(SIM.SETTING_NAMES)
IDX = {nm: i for i, nm in enumerate(NAMES)}


def row(mode, szen, n):
    return ev.frozen_row(mode, szen, n)


def p1(x):
    """Prozentzahl mit einer Stelle und Dezimalkomma wie in der README."""
    return f"{100 * x:.1f}".replace(".", ",")


def rates(mode, szen, n):
    return ev.rates(row(mode, szen, n))


def b_rate(mode, szen, n, name):
    return rates(mode, szen, n)["B"][IDX[name]]


def readme_has(*parts):
    for part in parts:
        assert part in README, part


# --- B5: Zellmittel und Streuung des Generators ------------------------------------------------------------------------------------------------

def test_b5_exakte_zellmittel_und_streuung():
    m = G.cell_means_exact(C.W_GEN)
    assert np.allclose(m, [[1.400, 1.700], [1.548, 1.548], [0.937, 0.937], [1.918, 2.218]], atol=6e-4)
    readme_has("Import 1,400 / 1,700, Export 1,548 / 1,548, Umschlag 0,937 / 0,937, Leer 1,918 / 2,218")
    assert round(G.sd_pool_exact(), 3) == 0.689
    sd = np.sqrt(G.var_within_art())
    assert round(sd.min(), 2) == 0.56 and sd.argmin() == 1 and round(sd.max(), 2) == 0.97 and sd.argmax() == 3 and round(sd.max() / sd.min(), 2) == 1.73
    readme_has("σ_pool = 0,689", "0,56 (Export) bis 0,97 (Leer), Verhältnis 1,73")
    assert round(m.min(), 1) == 0.9 and round(m.max(), 1) == 2.2                      # „Zellmittel 0,9 bis 2,2“
    readme_has("Zellmittel 0,9 bis 2,2")


# --- B6: Generator-Szenario, gleiche Streuung --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,vals", [
    ("Referenz Import", (36.8, 79.9, 100.0)), ("Referenz Export", (4.7, 5.1, 5.1)), ("Referenz Umschlag", (5.2, 5.2, 4.7)), ("Referenz Leer", (13.8, 32.8, 78.8)),
    ("Summe", (22.1, 55.2, 97.3)), ("Helmert", (22.1, 55.2, 97.3)), ("Gewichtet zentriert", (23.7, 56.3, 97.9)),
])
def test_b6_haeufigkeiten_generator_szenario_gleiche_streuung(name, vals):
    for n, v in zip(C.N_CHOICES, vals):
        assert round(100 * b_rate("gleich", "Generator", n, name), 1) == v, (name, n)
    if name != "Helmert":
        readme_has(" / ".join(f"{v:.1f}".replace(".", ",") for v in vals))


def test_b6_wechselwirkung_exakte_teststaerke_und_getestete_effekte():
    assert [round(100 * rates("gleich", "Generator", n)["I"], 1) for n in C.N_CHOICES] == [17.4, 48.8, 97.3]
    readme_has("Test der Wechselwirkung 17,4 / 48,8 / 97,3")
    exact = {n: {e["name"]: e["power_B"] for e in SIM.exact_power_row("gleich", "Generator", n)} for n in C.N_CHOICES}
    got = {nm: [round(100 * exact[n][nm], 1) for n in C.N_CHOICES] for nm in ("Referenz Import", "Referenz Leer", "Summe", "Gewichtet zentriert")}
    assert got == {"Referenz Import": [37.4, 81.2, 99.9], "Referenz Leer": [14.1, 33.1, 79.4], "Summe": [23.4, 57.2, 97.5], "Gewichtet zentriert": [23.8, 58.1, 97.7]}
    readme_has("(37,4 / 81,2 / 99,9)", "(14,1 / 33,1 / 79,4)", "(23,4 / 57,2 / 97,5)", "(23,8 / 58,1 / 97,7)", "(5,0)")
    assert all(abs(exact[n][nm] - 0.05) < 1e-12 for n in C.N_CHOICES for nm in ("Referenz Export", "Referenz Umschlag"))
    assert np.allclose(ev.true_effects("Generator"), [0.300, 0, 0, 0.300, 0.300, 0.150, 0.150, 0.135])
    readme_has("Getestete Effekte: +0,300 / 0 / 0 / +0,300 / +0,150 / +0,135")


def test_b6_intro_und_kernzahlen_stehen_in_der_readme():
    assert [round(100 * b_rate("gleich", "Generator", 600, nm), 1) for nm in ("Referenz Import", "Referenz Export", "Summe")] == [79.9, 5.1, 55.2]
    readme_has("in 79,9 % der Wiederholungen bei Referenz Import, in 5,1 % bei Referenz Export und in 55,2 % bei Summencodierung")


# --- B7, B8: andere Szenarien, Streuung des Generators ------------------------------------------------------------------------------------------

def test_b7_additiv_und_ueberkreuz_n600_gleiche_streuung():
    add = rates("gleich", "additiv", 600)
    assert [round(100 * x, 1) for x in add["B"][:4]] == [29.4, 26.6, 22.4, 11.8] and round(100 * add["B"][IDX["Summe"]], 1) == 57.1 and round(100 * add["B"][IDX["Gewichtet zentriert"]], 1) == 66.9
    assert round(100 * add["I"], 1) == 5.0
    readme_has("Referenz Import 29,4, Export 26,6, Umschlag 22,4, Leer 11,8, Summe 57,1, gewichtet 66,9, Wechselwirkung 5,0")
    kr = rates("gleich", "Überkreuz", 600)
    assert [round(100 * x, 1) for x in kr["B"][:4]] == [80.8, 4.8, 5.1, 32.3] and round(100 * kr["B"][IDX["Summe"]], 1) == 5.4 and round(100 * kr["B"][IDX["Gewichtet zentriert"]], 1) == 22.7
    assert round(100 * kr["I"], 1) == 69.0
    readme_has("Referenz Import 80,8, Export 4,8, Umschlag 5,1, Leer 32,3, **Summe 5,4**, gewichtet 22,7, Wechselwirkung 69,0")


def test_b8_streuung_des_generators_haelt_das_niveau_nicht():
    g = rates("Generator", "Generator", 600)
    assert [round(100 * x, 1) for x in g["B"][:4]] == [78.7, 1.7, 3.3, 38.6] and round(100 * g["B"][IDX["Summe"]], 1) == 55.7 and round(100 * g["B"][IDX["Gewichtet zentriert"]], 1) == 58.1
    assert round(100 * g["I"], 1) == 52.4
    readme_has("Referenz Import 78,7, Export 1,7, Umschlag 3,3, Leer 38,6, Summe 55,7, gewichtet 58,1, Wechselwirkung 52,4")
    sums = [round(100 * b_rate("Generator", "Überkreuz", n, "Summe"), 1) for n in C.N_CHOICES]
    assert sums == [10.0, 10.5, 9.9] and min(sums) > 9 and C.SZENARIEN["Überkreuz"][0] + C.SZENARIEN["Überkreuz"][3] == 0           # wahrer Effekt der Summe: Mittel 0
    readme_has("Summe 10,0 / 10,5 / 9,9 % bei n = 200 / 600 / 2000 statt 5 %")
    inter = [round(100 * rates("Generator", "additiv", n)["I"], 1) for n in C.N_CHOICES]
    assert inter == [9.6, 9.6, 9.9]
    readme_has("Test der Wechselwirkung 9,6 / 9,6 / 9,9 % statt 5 %")
    exp_rates = [100 * b_rate("Generator", s, n, "Referenz Export") for s in ("Generator", "Überkreuz") for n in C.N_CHOICES]
    um_rates = [100 * b_rate("Generator", s, n, "Referenz Umschlag") for s in ("Generator", "Überkreuz") for n in C.N_CHOICES]
    assert (round(min(exp_rates), 1), round(max(exp_rates), 1)) == (1.5, 1.8) and (round(min(um_rates), 1), round(max(um_rates), 1)) == (3.0, 3.7)
    readme_has("nur in 1,5 bis 1,8 %", "Referenz Umschlag in 3,0 bis 3,7 %")
    # Gegenprobe (11a Punkt 3): mit gleicher Streuung hält dieselbe Summe das Niveau
    assert all(abs(b_rate("gleich", "Überkreuz", n, "Summe") - 0.05) < 0.011 for n in C.N_CHOICES)
    assert all(abs(b_rate("gleich", s, n, nm) - 0.05) < 0.011 for s in ("Generator", "Überkreuz") for n in C.N_CHOICES for nm in ("Referenz Export", "Referenz Umschlag"))


# --- B9: Gegenprobe Simulation gegen exakte Teststärke -----------------------------------------------------------------------------------------

def test_b9_gegenprobe_zaehlung_und_groesste_abweichung():
    inside2 = inside3 = total = 0
    worst = 0.0
    for r in FROZEN["zeilen"]:
        if r["mode"] != "gleich":
            continue
        for k, e in enumerate(SIM.exact_power_row(r["mode"], r["szenario"], r["n"])):
            p, ph = e["power_B"], r["rej_B"][k] / r["valid"]
            se = np.sqrt(max(p * (1 - p), 0.0025) / r["valid"])
            inside2 += abs(ph - p) <= 2 * se
            inside3 += abs(ph - p) <= 3 * se
            total += 1
            worst = max(worst, abs(ph - p))
    assert (total, inside2, inside3) == (72, 59, 62) and round(100 * worst, 1) == 2.0
    readme_has("Größte Abweichung 2,0 Prozentpunkte; 59 von 72 Zeilen innerhalb zweier, 62 innerhalb dreier Standardfehler")


def test_b9_mittlere_teststaerke_ueber_die_tatsaechlichen_zellgroessen():
    """n = 600, Generator-Szenario, gleiche Streuung: Summe 56,1 % (erwartete Zellgrößen: 57,2 %, simuliert 55,2 %). Dieselben 10 000 Wiederholungen wie die vorgerechnete Zeile."""
    real = SIM.realised_power_row("gleich", "Generator", 600)
    exact = {e["name"]: e["power_B"] for e in SIM.exact_power_row("gleich", "Generator", 600)}
    assert round(100 * real[IDX["Summe"]], 1) == pytest.approx(56.1, abs=0.15)
    assert real[IDX["Summe"]] < exact["Summe"] - 0.008                       # darunter, wie die Konvexität in 1/n vorhersagt
    assert round(100 * exact["Summe"], 1) == 57.2 and round(100 * b_rate("gleich", "Generator", 600, "Summe"), 1) == 55.2
    readme_has("n = 600: 56,1 % statt 57,2 %, simuliert 55,2 %")
    for k in (IDX["Referenz Export"], IDX["Referenz Umschlag"]):
        assert real[k] == pytest.approx(0.05, abs=1e-12)                      # wahrer Effekt 0: kein Unterschied


# --- B10: Negativbefund Art --------------------------------------------------------------------------------------------------------------------

def test_b10_haupteffekt_art_verwirft_fast_immer():
    low = []
    imp_we = {200: [], 600: [], 2000: []}
    for r in FROZEN["zeilen"]:
        q = ev.rates(r)
        for k, x in enumerate(q["A"]):
            if k == IDX["Referenz Import / Wochenende"]:
                imp_we[r["n"]].append(100 * x)
            else:
                low.append(100 * x)
    assert len(FROZEN["zeilen"]) == 18 and round(min(low), 1) == 99.0 and round(max(low), 1) == 100.0
    assert (round(min(imp_we[200]), 1), round(max(imp_we[200]), 1)) == (75.5, 92.8)
    assert round(min(imp_we[600] + imp_we[2000]), 1) == 99.8 and round(max(imp_we[600] + imp_we[2000]), 1) == 100.0
    readme_has("Verwirft in 99,0 bis 100,0 % der Wiederholungen in allen 18 Zeilen", "n = 200: 75,5 bis 92,8 %; n = 600 und 2000: 99,8 bis 100,0 %")


# --- B11: Einzelkoeffizient Leer × Wochenende -------------------------------------------------------------------------------------------------

def test_b11_einzelkoeffizient_leer_mal_wochenende():
    vals = {n: (ev.rates(row("gleich", "Generator", n))["li"], ev.rates(row("gleich", "Generator", n))["le"]) for n in (600, 2000)}
    assert (p1(vals[600][0]), p1(vals[600][1]), p1(vals[2000][0]), p1(vals[2000][1])) == ("5,3", "25,5", "5,4", "67,3")
    readme_has("n = 600: bei Referenz Import 5,3 %, bei Referenz Export 25,5 %; n = 2000: 5,4 % gegen 67,3 %")
    assert abs(C.W_GEN[3] - C.W_GEN[0]) == 0 and abs(C.W_GEN[3] - C.W_GEN[1]) == 0.30      # wahre Differenz der Wochenendeffekte: Leer − Import 0, Leer − Export 0,30


# --- B12: Dummy-Falle ------------------------------------------------------------------------------------------------------------------------------

def test_b12_dummy_falle_rang_achsenabschnitte_und_anpassung():
    res = ev.dummy_trap(ev.Settings(n=200, seed=7003))
    assert (res.rank, res.columns) == (4, 5) and res.smallest_sv < 1e-10 and res.fits_max_dev < 1e-9
    assert np.array_equal(res.null_vector, [1, -1, -1, -1, -1])
    assert res.solutions["lstsq (kleinste Norm)"][0] == pytest.approx(1.22, abs=0.005) and res.solutions["pinv (kleinste Norm)"][0] == pytest.approx(1.22, abs=0.005)
    assert res.solutions["QR mit Spaltenwahl"][0] == pytest.approx(0.92, abs=0.005)
    sol = res.solutions["Normalgleichung solve(X'X)"]
    if sol is not None:                                                         # plattformabhängig (Rundungsfehler): nur „liegt auf der Lösungsgeraden“ wird geprüft, nicht die Zahl
        base, v = res.solutions["lstsq (kleinste Norm)"], res.null_vector
        t = (sol - base) @ v / (v @ v)
        assert np.max(np.abs(sol - base - t * v)) < 1e-6 * max(1.0, np.max(np.abs(sol)))
    readme_has("`lstsq` und `pinv` geben den Achsenabschnitt 1,22, QR mit Spaltenwahl 0,92", "(Achsenabschnitt 1,43)", "n = 200, Zufalls-Seed 7003")


# --- B13: Referenzkunde ----------------------------------------------------------------------------------------------------------------------------

def test_b13_referenzkunde():
    k = ev.kunde_frozen()
    sm = np.array(k["sig_mean"])
    eff = np.array(k["effects"])
    assert (round(sm.min(), 1), int(sm.argmin()), round(eff[sm.argmin()], 2)) == (4.7, 3, -0.08)
    assert (round(sm.max(), 1), int(sm.argmax()), round(eff[sm.argmax()], 2)) == (9.7, 10, 0.70)
    assert round(100 * k["f_alt"], 1) == 100.0 and round(100 * k["f_h0"], 1) == 5.2
    any_h0 = np.array(k["any_h0"])
    assert (round(100 * any_h0.min(), 1), round(100 * any_h0.max(), 1), round(100 * any_h0.mean(), 1)) == (30.4, 31.7, 30.8)
    assert round(100 * (1 - 0.95 ** 11), 1) == 43.1 and round(eff.min(), 2) == -0.59 and round(eff.max(), 2) == 0.70
    readme_has("im Mittel 4,7 (Referenz K3, Effekt −0,08) bis 9,7 (Referenz K10, Effekt +0,70)", "verwirft in 100,0 %", "Ohne Kundeneffekt verwirft er in 5,2 %", "30,4 bis 31,7 %", "(Mittel 30,8 %", "43,1 %")
    assert k["n"] == 300 and k["reps"] == 10000 and len(eff) == 12


# --- B15: leere Zelle ------------------------------------------------------------------------------------------------------------------------------

def test_b15_leere_zellen_bei_n_200():
    leer = {(r["mode"], r["szenario"]): r["leer"] for r in FROZEN["zeilen"] if r["n"] == 200}
    assert [leer[("gleich", s)] for s in C.SZENARIO_KEYS] == [31, 41, 24] and [leer[("Generator", s)] for s in C.SZENARIO_KEYS] == [31, 33, 35]
    assert all(r["leer"] == 0 for r in FROZEN["zeilen"] if r["n"] >= 600)
    assert min(leer.values()) == 24 and max(leer.values()) == 41
    readme_has("24 bis 41 von 10 000 Wiederholungen", "31, 41, 24 in den Szenarien Generator, additiv, Überkreuz", "31, 33, 35")
    assert all(r["valid"] + r["leer"] == r["reps"] == 10000 for r in FROZEN["zeilen"])


# --- Nachrechnen der vorgerechneten Messreihe ----------------------------------------------------------------------------------------------------

def test_vorgerechnete_messreihe_stimmt_mit_der_neu_gerechneten_ueberein():
    """Zeilen mit n = 200 neu rechnen (Zählungen ± 2 gegen die Datei); mit FCD_FULL=1 alle 18 Zeilen und die Messreihe zur Referenzkunde (`tools/fcd_make_frozen.py --check`)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import fcd_make_frozen as MF
    if os.environ.get("FCD_FULL"):
        assert MF.differences(FROZEN, MF.compute()) == []
        return
    rows = [g for g in SIM.GRID if g[2] == 200]
    new = {"zeilen": [SIM.simulate_row(m, s, n, C.SIM_REPS) for m, s, n in rows], "kunde": FROZEN["kunde"]}
    bad = MF.differences(FROZEN, new)
    assert bad == [], bad


def test_datei_hat_alle_18_zeilen_mit_den_vorgesehenen_zufalls_seeds():
    assert len(FROZEN["zeilen"]) == 18 and len(SIM.GRID) == 18
    for r in FROZEN["zeilen"]:
        assert r["seed"] == SIM.row_seed(r["mode"], r["szenario"], r["n"]) and r["reps"] == C.SIM_REPS
    assert sorted(r["seed"] for r in FROZEN["zeilen"]) == list(range(C.SIM_BASE_SEED + 1, C.SIM_BASE_SEED + 19))
    assert FROZEN["kunde"]["seed"] == C.KUNDE_SEED and FROZEN["ziel"]["settings"] == NAMES


# --- Texte der App gegen Messung -------------------------------------------------------------------------------------------------------------

def test_zahlen_in_den_grenzen_und_hilfetexten_der_app():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "9,9 bis 10,5 %" in app and "1,5 bis 1,8 %" in app and "24 bis 41 von 10 000" in app and "rund 31 %" in app and "rund 5 %" in app
    sums = [100 * b_rate("Generator", "Überkreuz", n, "Summe") for n in C.N_CHOICES]
    assert (round(min(sums), 1), round(max(sums), 1)) == (9.9, 10.5)
    exp = [100 * b_rate("Generator", s, n, "Referenz Export") for s in ("Generator", "Überkreuz") for n in C.N_CHOICES]
    assert (round(min(exp), 1), round(max(exp), 1)) == (1.5, 1.8)                  # Referenz Export testet in beiden Szenarien den Effekt 0
    assert abs(ev.true_effects("Generator")[IDX["Referenz Export"]]) == 0 and abs(ev.true_effects("Überkreuz")[IDX["Referenz Export"]]) == 0
    k = ev.kunde_frozen()
    assert round(100 * float(np.mean(k["any_h0"]))) == 31 and round(100 * k["f_h0"]) == 5
    assert "0,689" in app and "0,689" in C.STREUUNG_HELP["gleich"] and "0,56 bis 0,97" in app and round(G.sd_pool_exact(), 3) == 0.689
    assert "n = 600 in rund 80 %" in app and "rund 5 %" in app
    assert round(100 * b_rate("gleich", "Generator", 600, "Referenz Import")) == 80 and round(100 * b_rate("gleich", "Generator", 600, "Referenz Export")) == 5


def test_presets_help_texte_nennen_nur_gemessene_zahlen():
    assert "nur bei Import und Leer" in C.PRESET_HELP["Container Generator"] and C.W_GEN[1] == C.W_GEN[2] == 0.0 and C.W_GEN[0] == C.W_GEN[3] == 0.30
    assert "n = 600" in C.PRESET_HELP["Container Generator"] and C.DEFAULTS["n"] == 600
    assert "+0,3 bei Import, −0,3 bei Leer" in C.PRESET_HELP["Überkreuz"] and C.SZENARIEN["Überkreuz"] == (0.30, 0.0, 0.0, -0.30)
    assert "n = 200" in C.PRESET_HELP["Dummy-Falle"] and "Rang 4 von 5" in C.PRESET_HELP["Dummy-Falle"] and C.PRESETS["Dummy-Falle"]["n"] == 200
    assert "K4" in C.PRESET_HELP["Zwölf Kunden"] and C.PRESETS["Zwölf Kunden"]["ref_kunde"] == 4
    assert "2 / 4, 3 / 1, 2 / 2" in C.PRESET_HELP["Unausgewogen (H2)"] and C.HAND_H2 == (2, 4, 3, 1, 2, 2)


def test_testzahl_der_readme_stimmt_mit_der_suite_ueberein():
    import subprocess
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"], cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    m = re.search(r"(\d+)(?:/\d+)? tests? collected", out)
    n = int(m.group(1)) if m else len([ln for ln in out.splitlines() if "::" in ln])
    mention = re.search(r"Gesamtzahl: (\d+) Tests", README)
    assert mention and int(mention.group(1)) == n
