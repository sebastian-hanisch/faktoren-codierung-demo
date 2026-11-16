"""Auswertungen der fünf Experimente, ihre Koeffizienten-Sätze und die Verdikte (reine Funktionen: Art und Text, die App ruft `st.<Art>(Text)`).

Zahlen in Texten haben das Dezimalkomma (`de`); gerechnet wird ungerundet."""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import fcd_constants as C
import fcd_design as D
import fcd_generator as G
import fcd_simulation as SIM
import fcd_tests as T

ROOT = Path(__file__).resolve().parent
KA_HAND, KB = 3, 2


def de(x: float, nd: int = 2) -> str:
    """Zahl mit Dezimalkomma und echtem Minuszeichen; ein Wert, der auf null rundet, steht ohne Vorzeichen."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    r = round(float(x), nd)
    s = f"{abs(r):.{nd}f}".replace(".", ",")
    return ("−" + s) if r < 0 else s


def sci(x: float, nd: int = 1) -> str:
    """Zahl in wissenschaftlicher Schreibweise mit Dezimalkomma (zum Beispiel 3,6e-15)."""
    return f"{x:.{nd}e}".replace(".", ",")


def pct(x: float, nd: int = 1) -> str:
    return de(100.0 * x, nd) + " %"


def p_text(p: float) -> str:
    """p-Wert mit vier Stellen, sehr kleine als „< 0,0001“."""
    if p is None or np.isnan(p):
        return "–"
    return "< 0,0001" if p < 1e-4 else de(p, 4)


@dataclass(frozen=True)
class Settings:
    sizes: tuple = (2, 2, 2, 2, 2, 2)
    cod: str = C.DEFAULTS["cod"]
    ref_art: int = C.DEFAULTS["ref_art"]
    ref_we: int = C.DEFAULTS["ref_we"]
    modell: str = C.DEFAULTS["modell"]
    n: int = C.DEFAULTS["n"]
    szenario: str = C.DEFAULTS["szenario"]
    streuung: str = C.DEFAULTS["streuung"]
    seed: int = C.DEFAULTS["seed"]
    ref_kunde: int = C.DEFAULTS["ref_kunde"]

    @property
    def interaction(self) -> bool:
        return self.modell == C.MODELL_KEYS[0]


def settings_from_state(get) -> Settings:
    """`get(key)` liefert den Wert eines Reglers (Session State); fehlende Schlüssel nehmen den Standard."""
    g = lambda k: get(k) if get(k) is not None else C.DEFAULTS[k]
    return Settings(tuple(int(g(f"cell_{i}{j}")) for i in range(3) for j in range(2)), g("cod"), int(g("ref_art")), int(g("ref_we")), g("modell"),
                    int(g("n")), g("szenario"), g("streuung"), int(g("seed")), int(g("ref_kunde")))


# ======================================================================================================================================================
# Handbeispiel (Experimente 1 und 2)
# ======================================================================================================================================================

def hand_data(sizes):
    """Erfundene ganze Tage: Zelle (Art i, Wochentag/Wochenende j) mit Mittel `HAND_MU[i][j]` und n_ij Beobachtungen m − 1, m + 1 (paarweise) und bei ungeradem n zusätzlich m. Rückgabe a, b, y."""
    a, b, y = [], [], []
    for i in range(KA_HAND):
        for j in range(KB):
            n = int(sizes[2 * i + j])
            devs = ([0] if n % 2 else []) + [-1, 1] * (n // 2)
            for d in devs:
                a.append(i), b.append(j), y.append(C.HAND_MU[i][j] + d)
    return np.array(a), np.array(b), np.array(y, dtype=float)


@dataclass(frozen=True)
class Hand:
    a: np.ndarray
    b: np.ndarray
    y: np.ndarray
    n_cell: np.ndarray
    mean: np.ndarray
    within: float
    dfres: int
    shares_a: np.ndarray
    shares_b: np.ndarray

    @property
    def n(self) -> int:
        return int(self.y.size)


def make_hand(sizes) -> Hand:
    a, b, y = hand_data(sizes)
    n_cell, mean, within = T.cell_stats(a, b, y, KA_HAND, KB)
    n = y.size
    return Hand(a, b, y, n_cell, mean, within, n - KA_HAND * KB, np.bincount(a, minlength=KA_HAND) / n, np.bincount(b, minlength=KB) / n)


def refs_for(kind: str, s: Settings) -> tuple[int, int]:
    """Referenzstufen (A, B) der gewählten Codierung: Referenz und gewichtet lassen die gewählte Stufe aus; Summe und Helmert kennen keine Referenz (Regler gesperrt)."""
    return (s.ref_art, s.ref_we) if kind in (C.COD_REFERENZ, C.COD_GEWICHTET) else (0, 0)


def hand_design(h: Hand, kind: str, s: Settings) -> D.Design:
    ra, rb = refs_for(kind, s)
    return D.build_design(h.a, h.b, KA_HAND, KB, kind, ra, rb, C.HAND_ARTEN, C.WE_LABELS, s.interaction)


def predicted_coefficients(h: Hand, kind: str, ra: int, rb: int):
    """Koeffizienten des Modells MIT Wechselwirkung für Achsenabschnitt, Art-Spalten und Wochenende-Spalte, aus den Zellmitteln allein, mit dem Satz dazu (zweiter Rechenweg zum Löser).
    Rückgabe: Liste (Name, Wert, Satz)."""
    th = h.mean
    ka = th.shape[0]
    delta = th[:, 1] - th[:, 0]
    om_b = D.hypothesis_weights(kind, KB, rb, h.shares_b)
    om_a = D.hypothesis_weights(kind, ka, ra, h.shares_a)
    psi = th @ om_b
    out = []
    if kind == C.COD_REFERENZ:
        out.append(("Achsenabschnitt", float(th[ra, rb]), f"Zellmittel von {C.HAND_ARTEN[ra]} / {C.WE_LABELS[rb]} (der Referenzzelle)"))
    elif kind == C.COD_GEWICHTET:
        out.append(("Achsenabschnitt", float(h.shares_a @ th @ h.shares_b), "Mittel aller Zellmittel mit den Gewichten Anteil der Art mal Anteil der Wochenendstufe (Stichprobe)"))
    else:
        out.append(("Achsenabschnitt", float(th.mean()), "ungewichtetes Mittel aller Zellmittel"))
    labels = D.column_labels(kind, C.HAND_ARTEN, ra, "Art")
    if kind in (C.COD_REFERENZ, C.COD_GEWICHTET):
        ref_txt = f"bei {C.WE_LABELS[rb]}" if kind == C.COD_REFERENZ else "gemittelt über Wochentag und Wochenende mit den Stichprobenanteilen"
        for j, lab in zip([x for x in range(ka) if x != ra], labels):
            out.append((lab, float(psi[j] - psi[ra]), f"Mittel der Art {C.HAND_ARTEN[j]} minus Mittel der Art {C.HAND_ARTEN[ra]}, {ref_txt}"))
    elif kind == C.COD_SUMME:
        for j, lab in enumerate(labels):
            out.append((lab, float(psi[j] - psi.mean()), f"Mittel der Art {C.HAND_ARTEN[j]} minus ungewichtetes Mittel der Arten (beide über Wochentag und Wochenende ungewichtet gemittelt)"))
    else:
        for j, lab in zip(range(1, ka), labels):
            prev = ", ".join(C.HAND_ARTEN[:j])
            out.append((lab, float((psi[j] - psi[:j].mean()) / (j + 1)), f"(Mittel der Art {C.HAND_ARTEN[j]} minus Mittel von {prev}) geteilt durch {j + 1} (Spalte unskaliert)"))
    d = float(om_a @ delta)
    b_label = D.column_labels(kind, C.WE_LABELS, rb, "")[0].strip()
    if kind == C.COD_REFERENZ:
        txt = (f"Wochenendeffekt (Wochenende minus Wochentag) bei der Art {C.HAND_ARTEN[ra]}" if rb == 0
               else f"Wochentag minus Wochenende bei der Art {C.HAND_ARTEN[ra]} (der Wochenendeffekt mit umgekehrtem Vorzeichen)")
        out.append((b_label, d if rb == 0 else -d, txt))
    elif kind == C.COD_GEWICHTET:
        txt = "mit den Stichprobenanteilen der Arten gewichtetes Mittel der Wochenendeffekte" + ("" if rb == 0 else " (mit umgekehrtem Vorzeichen: Wochentag minus Wochenende)")
        out.append((b_label, d if rb == 0 else -d, txt))
    elif kind == C.COD_SUMME:
        out.append((b_label, -0.5 * d, "−½ × ungewichtetes Mittel der Wochenendeffekte über die Arten (Wochentag mit +1, Wochenende mit −1 codiert)"))
    else:
        out.append((b_label, 0.5 * d, "+½ × ungewichtetes Mittel der Wochenendeffekte über die Arten (Wochentag mit −1, Wochenende mit +1 codiert)"))
    return out


def format_weight(x: float) -> str:
    """Gewicht als einfacher Bruch (Nenner bis 1000), sonst als Dezimalzahl."""
    if abs(x) < 1e-12:
        return "0"
    from fractions import Fraction
    f = Fraction(x).limit_denominator(1000)
    if abs(float(f) - x) < 1e-9:
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}".replace("-", "−")
    return de(x, 3)


@dataclass(frozen=True)
class HandResult:
    hand: Hand
    kind: str
    design: D.Design
    table: pd.DataFrame            # Koeffizient, Wert, Standardfehler, t, p, Satz
    cell_matrix: pd.DataFrame      # Entwurfsmatrix, eine Zeile je Zelle
    weights: pd.DataFrame | None   # Gewicht jedes Zellmittels je Koeffizient (nur mit Wechselwirkung)
    fitted_cells: np.ndarray       # angepasste Zellmittel (Art × Wochenende)
    sse: float
    summary: list                  # je Codierung: Name, Achsenabschnitt, Wochenende-Koeffizient
    max_dev_fit: float             # größte Abweichung der angepassten Werte zwischen den vier Codierungen
    status: str                    # "ok" oder Grund, warum nichts gerechnet wurde


def analyse_hand(s: Settings) -> HandResult:
    h = make_hand(s.sizes)
    d = hand_design(h, s.cod, s)
    cell_names = [f"{C.HAND_ARTEN[i]} / {C.WE_LABELS[j]}" for i in range(KA_HAND) for j in range(KB)]
    ra, rb = refs_for(s.cod, s)
    # Entwurfsmatrix der Zellen (jede Beobachtung einer Zelle hat dieselbe Zeile)
    cd = D.cell_design(KA_HAND, KB, s.cod, ra, rb, h.shares_a, h.shares_b, C.HAND_ARTEN)
    Xc = cd.X if s.interaction else cd.X[:, [*cd.i_int, *cd.i_a, *cd.i_b]]
    names = list(cd.names) if s.interaction else [cd.names[i] for i in (*cd.i_int, *cd.i_a, *cd.i_b)]
    cell_matrix = pd.DataFrame(np.round(Xc, 4), index=cell_names, columns=names)
    status = "ok"
    if h.dfres <= 0 and not (h.dfres == 0 and not s.interaction):
        status = "keine Freiheitsgrade im Rest (n gleich Zahl der Zellen): Tests sind nicht definiert"
    beta, se, t, pv, dfres = D.coefficient_table(d, h.y)
    sentences = [""] * len(names)
    if s.interaction:
        pred = predicted_coefficients(h, s.cod, ra, rb)
        for i, (_, _, txt) in zip((*d.i_int, *d.i_a, *d.i_b), pred):
            sentences[i] = txt
        for i in d.i_ab:
            sentences[i] = "Unterschied der Wochenendeffekte (siehe Gewichte der Zellmittel unten)"
    else:
        for i in d.i_int:
            sentences[i] = "Kleinste Quadrate; ohne Wechselwirkung gilt ein Wochenendeffekt für alle Arten"
        for i in d.i_a:
            sentences[i] = "Artunterschied, derselbe an beiden Wochenendstufen (Kleinste Quadrate, nach Zellgrößen gewichtet)"
        for i in d.i_b:
            sentences[i] = "Wochenendeffekt, derselbe bei jeder Art (Kleinste Quadrate, nach Zellgrößen gewichtet; die Codierung ändert Maß und Vorzeichen)"
    table = pd.DataFrame({"Koeffizient": list(d.names), "Wert": beta, "Standardfehler": se, "t": t, "p": pv, "Bedeutung": sentences})
    weights = None
    if s.interaction:
        W = D.cell_weights(cd)
        weights = pd.DataFrame([[format_weight(x) for x in row] for row in W], index=list(cd.names), columns=cell_names)
    fitted = (d.X @ beta)
    _, fitted_cells_m, _ = T.cell_stats(h.a, h.b, fitted, KA_HAND, KB)
    # Zusammenfassung über alle vier Codierungen (gleiche Anpassung, andere Koeffizienten)
    summary, ref_fit = [], None
    max_dev = 0.0
    for kind in C.COD_KEYS:
        dk = hand_design(h, kind, s)
        bk, fk, _ = D.fit(dk.X, h.y)
        ref_fit = fk if ref_fit is None else ref_fit
        max_dev = max(max_dev, float(np.max(np.abs(fk - ref_fit))))
        summary.append({"Codierung": kind, "Achsenabschnitt": float(bk[0]), "Wochenende": float(bk[dk.i_b[0]]), "Name": dk.names[dk.i_b[0]]})
    return HandResult(h, s.cod, d, table, cell_matrix, weights, fitted_cells_m, float(np.sum((h.y - fitted) ** 2)), summary, max_dev, status)


def verdict_fit(res: HandResult) -> tuple[str, str]:
    """Verdikt Experiment 1: dieselbe Anpassung, andere Koeffizienten."""
    parts = "; ".join(f"{r['Codierung']}: Achsenabschnitt {de(r['Achsenabschnitt'], 2)}, Wochenende {de(r['Wochenende'], 2)}" for r in res.summary)
    same = res.max_dev_fit < 1e-9
    if not same:
        return "warning", f"Die angepassten Werte unterscheiden sich zwischen den Codierungen (größte Abweichung {sci(res.max_dev_fit)}); das darf nicht vorkommen."
    return "info", (f"Die angepassten Werte sind in allen vier Codierungen gleich (größte Abweichung unter 1e-9). Die Koeffizienten nicht: {parts}.")


def verdict_meaning(res: HandResult, s: Settings) -> tuple[str, str]:
    """Kernfrage 1: Welche Zahl bedeutet welcher Koeffizient? Der Satz zum Wochenende-Koeffizienten der gewählten Codierung."""
    if not s.interaction:
        return "info", ("Im additiven Modell (ohne Wechselwirkung) gilt ein Wochenendeffekt für alle Arten; der Koeffizient ist ein nach Zellgrößen gewichtetes Mittel (Kleinste Quadrate), "
                        "und die Codierung ändert nur sein Maß und Vorzeichen, nicht den Test.")
    d = res.design
    row = res.table.iloc[d.i_b[0]]
    ref = f" (Referenzstufe von Art: {C.HAND_ARTEN[s.ref_art]}, von Wochenende: {C.WE_LABELS[s.ref_we]})" if s.cod in (C.COD_REFERENZ, C.COD_GEWICHTET) else ""
    return "info", f"Codierung {s.cod}{ref}: der Koeffizient „{row['Koeffizient']}“ ist {de(row['Wert'], 2)}. Er ist {row['Bedeutung']}."


def verdict_switch(res: HandResult, cmp: dict) -> tuple[str, str]:
    """Kernfrage 3: Was ändert sich, wenn man nur die Codierung wechselt?"""
    coefs = [r["Wochenende"] for r in res.summary]
    rows = [r for r in cmp["rows"] if "/ Wochenende" not in r["name"]]
    fs = [r["F_B"] for r in rows if not np.isnan(r["F_B"])]
    if res.max_dev_fit >= 1e-9:
        return "warning", f"Die angepassten Werte unterscheiden sich (größte Abweichung {sci(res.max_dev_fit)}); das darf nicht vorkommen."
    if not cmp["interaction"]:
        zusatz = f" ({de(fs[0], 2)})" if fs else ""
        return "info", (f"Im additiven Modell bleiben die angepassten Werte (größte Abweichung unter 1e-9) und der Haupteffekt-F-Wert gleich{zusatz}; es ändern sich Achsenabschnitt und Koeffizienten, "
                        f"der Wochenende-Koeffizient von {de(min(coefs), 2)} bis {de(max(coefs), 2)} bei den vier Codierungen (Maß und Vorzeichen).")
    fa = f"Haupteffekt-F von {de(min(fs), 2)} bis {de(max(fs), 2)}" if fs else "Haupteffekt-F nicht definiert (Rest ohne Freiheitsgrade)"
    same_i = (not np.isnan(cmp["fi_dev"])) and cmp["fi_dev"] < 1e-9
    wi = (f"F der Wechselwirkung in allen vier Codierungen {de(cmp['F_I'], 2)} (p = {p_text(cmp['p_I'])})" if same_i else "F der Wechselwirkung nicht definiert (Rest ohne Freiheitsgrade)")
    return "info", (f"Gleich bleiben die angepassten Werte (größte Abweichung unter 1e-9) und das {wi}. Es ändern sich der Achsenabschnitt, der Wochenende-Koeffizient "
                    f"(von {de(min(coefs), 2)} bis {de(max(coefs), 2)} bei den vier Codierungen) und die Haupteffekt-Tests ({fa}, je nach Referenzstufe oder Mittelung).")


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Wo wird der Haupteffekt gemessen?
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def compare_codings(h: Hand, interaction: bool = True) -> dict:
    """Haupteffekt-Tests Wochenende und Art je Codierung (Streichung der Haupteffektspalten bei erhaltener Wechselwirkung beziehungsweise im additiven Modell), Wechselwirkung, Typ I und Typ III der Art.
    Im additiven Modell ist „getesteter Effekt“ der Koeffizient des Wochenendes mit seinem Standardfehler; Wechselwirkung, Typ I und Typ III entfallen."""
    rows = []
    for st in T.settings_list(KA_HAND, C.HAND_ARTEN):
        d = D.build_design(h.a, h.b, KA_HAND, KB, st["kind"], st["ref_a"], st["ref_b"], C.HAND_ARTEN, C.WE_LABELS, interaction)
        FB, pB, _, dfres = D.drop_test(d.X, h.y, d.i_b)
        FA, pA, qa, _ = D.drop_test(d.X, h.y, d.i_a)
        if interaction:
            oa, ob = T.setting_weights(st, h.shares_a, h.shares_b)
            _, _, eff, sd, _ = T.wochenende_test(h.n_cell, h.mean, h.within, oa)
        else:
            beta, se, _, _, _ = D.coefficient_table(d, h.y)
            eff, sd = float(beta[d.i_b[0]]), float(se[d.i_b[0]])
        rows.append({"name": st["name"], "kind": st["kind"], "effect": eff, "sd": sd, "F_B": FB, "p_B": pB, "F_A": FA, "p_A": pA})
    out = {"rows": rows, "dfres": h.n - (KA_HAND * KB if interaction else KA_HAND + KB - 1), "interaction": interaction}
    if not interaction:
        fb = [r["F_B"] for r in rows if not np.isnan(r["F_B"])]
        out["fb_dev"] = (max(fb) - min(fb)) / max(max(abs(x) for x in fb), 1.0) if fb else float("nan")
        return out
    fis = []
    for kind in C.COD_KEYS:
        d = D.build_design(h.a, h.b, KA_HAND, KB, kind, 0, 0, C.HAND_ARTEN)
        fis.append(D.drop_test(d.X, h.y, d.i_ab)[:2])
    FI, pI = fis[0]
    fi_dev = max(abs(f - FI) / max(abs(FI), 1.0) for f, _ in fis if not np.isnan(f)) if not np.isnan(FI) else float("nan")
    F1, p1 = D.typ1_art_F(h.a, h.b, h.y, KA_HAND, KB)
    out.update({"F_I": FI, "p_I": pI, "fi_dev": fi_dev, "typ1": (F1, p1), "typ3": next((r["F_A"], r["p_A"]) for r in rows if r["name"] == "Summe")})
    return out


def verdict_hypotheses(cmp: dict) -> tuple[str, str]:
    """Kernfrage 2: Gibt es einen Wochenendeffekt? Dieselbe Frage, je Codierung eine andere Antwort (mit Wechselwirkung); im additiven Modell überall dieselbe."""
    if cmp["dfres"] <= 0:
        return "warning", "Der Rest hat keine Freiheitsgrade (so viele Beobachtungen wie Parameter): die F-Tests sind nicht definiert."
    rows = [r for r in cmp["rows"] if r["name"] not in ("Helmert",) and "/ Wochenende" not in r["name"]]
    if not cmp["interaction"]:
        r0 = rows[0]
        same = cmp["fb_dev"] < 1e-9
        if not same:
            return "warning", f"Im additiven Modell unterscheiden sich die F-Werte zwischen den Codierungen (relative Abweichung {sci(cmp['fb_dev'])}); das darf nicht vorkommen."
        verdict = "verwerfen alle" if r0["p_B"] < C.ALPHA else "verwirft keine"
        return "info", (f"Im additiven Modell ohne Wechselwirkung testet jede Codierung dieselbe Hypothese: F = {de(r0['F_B'], 2)} (p = {p_text(r0['p_B'])}) bei allen {len(rows)} Einstellungen "
                        f"(größte relative Abweichung unter 1e-9); bei 5 % {verdict} der Codierungen. Nur Maß und Vorzeichen des Koeffizienten ändern sich.")
    reject = [r for r in rows if r["p_B"] < C.ALPHA]
    parts = ", ".join(f"{r['name']} F = {de(r['F_B'], 2)} (p = {p_text(r['p_B'])})" for r in rows)
    if reject and len(reject) < len(rows):
        return "warning", (f"Dieselben Daten, dieselbe Frage „Haupteffekt Wochenende“, verschiedene Antworten: {parts}. {len(reject)} von {len(rows)} Codierungen verwerfen bei 5 %; "
                           "jede testet einen anderen Effekt (Effekt bei einer Referenzart, ungewichtetes Mittel der Wochenendeffekte, mit Stichprobenanteilen gewichtetes Mittel).")
    word = "verwerfen alle" if reject else "verwirft keine"
    return "info", f"{parts}. Bei 5 % {word} der {len(rows)} Codierungen (gleicher Schluss in allen)."


def verdict_typ(cmp: dict) -> tuple[str, str]:
    """Verdikt Experiment 2: Typ I gegen Typ III für den Haupteffekt Art (nur mit Wechselwirkung definiert)."""
    if not cmp["interaction"]:
        return "info", "Typ I und Typ III für Art sind hier nicht gezeigt (nur im Modell mit Wechselwirkung)."
    if cmp["dfres"] <= 0:
        return "warning", "Der Rest hat keine Freiheitsgrade: Typ I und Typ III sind nicht definiert."
    F1, p1 = cmp["typ1"]
    F3, p3 = cmp["typ3"]
    if abs(F1 - F3) < 1e-9 * max(abs(F1), 1.0):
        return "info", (f"Typ I (Art zuerst) und Typ III (Summencodierung) geben für den Haupteffekt Art denselben F-Wert {de(F1, 2)} (p = {p_text(p1)}): bei gleichen Zeilengrößen mitteln beide "
                        "die Stufen des Wochenendes mit denselben Gewichten.")
    return "warning", (f"Für den Haupteffekt Art gibt Typ I (nach Zellgrößen gewichtete Zeilenmittel) F = {de(F1, 2)} (p = {p_text(p1)}), Typ III (ungewichtete Zeilenmittel) F = {de(F3, 2)} (p = {p_text(p3)}): "
                       "bei ungleich besetzten Zellen mitteln beide anders und prüfen eine andere Hypothese.")


# ======================================================================================================================================================
# Container-Experimente (3 bis 5)
# ======================================================================================================================================================

@lru_cache(maxsize=1)
def load_frozen() -> dict:
    return json.loads((ROOT / "data" / C.FROZEN_FILE).read_text(encoding="utf-8"))


def frozen_row(mode: str, szenario: str, n: int) -> dict:
    for r in load_frozen()["zeilen"]:
        if (r["mode"], r["szenario"], r["n"]) == (mode, szenario, n):
            return r
    raise KeyError((mode, szenario, n))


def rates(row: dict) -> dict:
    """Verwerfungshäufigkeiten einer vorgerechneten Zeile (Anteile 0 bis 1, bezogen auf die gültigen Wiederholungen)."""
    v = row["valid"]
    return {"B": [x / v for x in row["rej_B"]], "A": [x / v for x in row["rej_A"]], "I": row["rej_I"] / v,
            "li": row["sig_leer_import"] / v, "le": row["sig_leer_export"] / v, "leer": row["leer"], "reps": row["reps"], "valid": v}


def mc_se(p: float, valid: int) -> float:
    """Standardfehler einer Häufigkeit aus `valid` unabhängigen Wiederholungen."""
    return float(np.sqrt(max(p * (1 - p), 0.0) / valid))


def true_effects(szenario: str) -> list[float]:
    """Der getestete Effekt je Einstellung (ω_A · wahre Wochenendeffekte), mit den erwarteten Artanteilen als Gewichte der gewichteten Codierung."""
    w = np.array(C.SZENARIEN[szenario])
    pa, pb = np.array(C.ART_P), np.array([1 - C.P_WE, C.P_WE])
    return [float(T.setting_weights(st, pa, pb)[0] @ w) for st in SIM.SETTINGS]


@dataclass(frozen=True)
class SampleResult:
    ok: bool
    reason: str
    rows: list = field(default_factory=list)
    F_I: float = float("nan")
    p_I: float = float("nan")
    n_cell: np.ndarray | None = None


def sample_tests(s: Settings) -> SampleResult:
    """Eine Stichprobe von n Containern (Zufalls-Seed der Seitenleiste): die Tests je Codierung über die Entwurfsmatrix (`fcd_design`), wie sie ein Programm rechnet."""
    c = G.draw(s.seed, s.n)
    y = G.outcome(c, s.streuung, C.SZENARIEN[s.szenario])
    n_cell, mean, within = T.cell_stats(c.art, c.we, y, 4, 2)
    if np.any(n_cell == 0):
        return SampleResult(False, "Zelle leer: Art und Wochenende kommen nicht in allen acht Kombinationen vor, die Wechselwirkung ist nicht schätzbar (anderen Zufalls-Seed oder größeres n wählen).", n_cell=n_cell)
    rows = []
    for st in SIM.SETTINGS:
        d = D.build_design(c.art, c.we, 4, 2, st["kind"], st["ref_a"], st["ref_b"], C.ART_LABELS)
        FB, pB, _, _ = D.drop_test(d.X, y, d.i_b)
        FA, pA, _, _ = D.drop_test(d.X, y, d.i_a)
        rows.append({"name": st["name"], "F_B": FB, "p_B": pB, "F_A": FA, "p_A": pA})
    d = D.build_design(c.art, c.we, 4, 2, C.COD_SUMME, 0, 0, C.ART_LABELS)
    FI, pI, _, _ = D.drop_test(d.X, y, d.i_ab)
    return SampleResult(True, "", rows, FI, pI, n_cell)


def exact_powers(s: Settings) -> list[dict]:
    return SIM.exact_power_row(s.streuung, s.szenario, s.n)


def verdict_container(s: Settings, r: dict, r200: dict) -> tuple[str, str]:
    """Verdikt Experiment 3 aus der vorgerechneten Zeile r (n, Szenario, Streuung der Seitenleiste) und der Zeile r200 (n = 200, sonst gleich) für die Wechselwirkung."""
    names = SIM.SETTING_NAMES
    pos = {nm: i for i, nm in enumerate(names)}
    eff = dict(zip(names, true_effects(s.szenario)))
    def rb(nm):
        return r["B"][pos[nm]]
    ref_names = [f"Referenz {a}" for a in C.ART_LABELS]
    ref_txt = ", ".join(f"{nm.replace('Referenz ', 'Referenz ')} {pct(rb(nm), 0)}" for nm in ref_names)
    head = f"Bei n = {s.n} verwirft der Test „Haupteffekt Wochenende“ in {ref_txt}, Summe {pct(rb('Summe'), 0)} und gewichtet {pct(rb('Gewichtet zentriert'), 0)} der Wiederholungen: dieselben Daten, je nach Codierung eine andere Antwort. "
    inter = f"Die Wechselwirkung wird in {pct(r['I'], 0)} der Wiederholungen erkannt (bei n = 200: {pct(r200['I'], 0)})"
    if s.szenario == "additiv":
        vals = r["B"][:4] + [rb("Summe"), rb("Gewichtet zentriert")]
        text = (f"Additives Szenario (derselbe Effekt 0,15 bei jeder Art, n = {s.n}): der getestete Effekt ist in jeder Codierung derselbe, die Häufigkeit, mit der der Haupteffekt-Test im Modell mit "
                f"Wechselwirkung verwirft, liegt trotzdem zwischen {pct(min(vals), 0)} und {pct(max(vals), 0)}, weil seine Varianz an den Gewichten der Codierung hängt (Referenz Leer mittelt nur die kleine Art). "
                f"Der Test der Wechselwirkung selbst verwirft {pct(r['I'], 1)} (wahre Wechselwirkung 0); das additive Modell ohne Wechselwirkungsspalten hat in jeder Codierung denselben F-Wert (Experiment 1 und Tests).")
        return "info", text
    text = head + inter + "."
    if s.streuung == "Generator":
        zero = [nm for nm in (*ref_names, "Summe", "Gewichtet zentriert") if abs(eff[nm]) < 1e-12]
        if zero:
            text += (" Mit der Streuung des Generators (Zellstreuungen im Verhältnis 1,7 : 1) liegen die Häufigkeiten bei wahrem Effekt 0 nicht bei 5 %: "
                     + ", ".join(f"{nm} {pct(rb(nm), 1)}" for nm in zero) + ".")
        return "warning", text
    return "info", text


# ======================================================================================================================================================
# Experiment 4: Dummy-Falle
# ======================================================================================================================================================

@dataclass(frozen=True)
class DummyResult:
    n: int
    rank: int
    columns: int
    smallest_sv: float
    null_vector: np.ndarray
    solutions: dict            # Name -> Koeffizientenvektor oder None
    notes: dict                # Name -> Bemerkung
    fits_max_dev: float
    cell_means: np.ndarray
    fitted: dict


def dummy_trap(s: Settings) -> DummyResult:
    """Achsenabschnitt plus vier Artspalten (redundante Codierung) an einer Stichprobe von n Containern: Rang, Nullvektor, vier Lösungswege und ihre Anpassungen."""
    from scipy.linalg import qr as sp_qr
    c = G.draw(s.seed, s.n)
    y = G.outcome(c, s.streuung, C.SZENARIEN[s.szenario])
    X = np.column_stack([np.ones(s.n)] + [(c.art == j).astype(float) for j in range(4)])
    sv = np.linalg.svd(X, compute_uv=False)
    rank = int(np.sum(sv > C.SING_TOL * sv[0]))
    null = np.array([1.0, -1, -1, -1, -1])
    sols, notes = {}, {}
    sols["lstsq (kleinste Norm)"] = np.linalg.lstsq(X, y, rcond=None)[0]
    notes["lstsq (kleinste Norm)"] = "Lösung kleinster Länge"
    sols["pinv (kleinste Norm)"] = np.linalg.pinv(X) @ y
    notes["pinv (kleinste Norm)"] = "Lösung kleinster Länge"
    Q, R, piv = sp_qr(X, mode="economic", pivoting=True)
    r_eff = int(np.sum(np.abs(np.diag(R)) > C.SING_TOL * abs(R[0, 0])))
    bq = np.zeros(X.shape[1])
    bq[piv[:r_eff]] = np.linalg.solve(R[:r_eff, :r_eff], (Q.T @ y)[:r_eff])
    sols["QR mit Spaltenwahl"] = bq
    notes["QR mit Spaltenwahl"] = f"eine Spalte ({int(piv[-1])}) fällt aus der Anpassung und erhält 0"
    try:
        sols["Normalgleichung solve(X'X)"] = np.linalg.solve(X.T @ X, X.T @ y)
        notes["Normalgleichung solve(X'X)"] = "meldet hier keinen Fehler; die Zahl hängt am Rundungsfehler (andere Plattformen melden einen Fehler oder eine andere Zahl)"
    except np.linalg.LinAlgError as e:
        sols["Normalgleichung solve(X'X)"] = None
        notes["Normalgleichung solve(X'X)"] = f"Fehlermeldung: {e}"
    fitted = {k: X @ v for k, v in sols.items() if v is not None}
    base = next(iter(fitted.values()))
    dev = max(float(np.max(np.abs(f - base))) for f in fitted.values())
    zm = np.array([y[c.art == j].mean() for j in range(4)])
    return DummyResult(s.n, rank, X.shape[1], float(sv[-1]), null, sols, notes, dev, zm, fitted)


def verdict_dummy(res: DummyResult) -> tuple[str, str]:
    """Verdikt Experiment 4: Rang, Zahl der Lösungswege, Zahl der angepassten Anpassungen."""
    k = sum(v is not None for v in res.solutions.values())
    if res.rank == res.columns:
        return "info", f"Rang {res.rank} von {res.columns}: die Spalten sind unabhängig, die Lösung ist eindeutig."
    a0 = [v[0] for v in res.solutions.values() if v is not None]
    return "warning", (f"Rang {res.rank} von {res.columns} (kleinster singulärer Wert {sci(res.smallest_sv)}): {k} Lösungswege, Achsenabschnitte {', '.join(de(x, 2) for x in a0)}; "
                       f"die angepassten Werte sind dieselben (größte Abweichung {sci(res.fits_max_dev, 0)}) und gleich den Zellmitteln der Art.")


# ======================================================================================================================================================
# Experiment 5: Referenzkunde
# ======================================================================================================================================================

def kunde_frozen() -> dict:
    k = load_frozen()["kunde"]
    reps, v = k["reps"], k["valid"]
    return {"sig_mean": [x / v for x in k["sig_sum_by_ref"]], "f_alt": k["f_alt"] / v, "f_h0": k["f_h0"] / v, "any_h0": [x / v for x in k["any_h0_by_ref"]],
            "effects": k["kunde_effekte"], "n": k["n"], "reps": reps}


@dataclass(frozen=True)
class KundeSample:
    ok: bool
    table: pd.DataFrame | None
    n_sig: int
    F: float
    p: float
    counts: np.ndarray | None


def kunde_sample(s: Settings) -> KundeSample:
    """Eine Stichprobe von 300 Import-Containern, Kunde als Faktor mit 12 Stufen: die 11 Koeffizienten gegen die gewählte Referenzstufe (t-Tests) und der F-Test des ganzen Faktors."""
    k = C.K_KUNDEN
    c = G.draw_import(s.seed, C.N_KUNDE_EXP)
    y = G.import_outcome(c, True)
    cnt = np.bincount(c.kunde, minlength=k).astype(float)
    if np.any(cnt == 0):
        return KundeSample(False, None, 0, float("nan"), float("nan"), cnt)
    mean = np.bincount(c.kunde, weights=y, minlength=k) / cnt
    within = float(np.sum((y - mean[c.kunde]) ** 2))
    dfres = C.N_KUNDE_EXP - k
    s2 = within / dfres
    ref = s.ref_kunde
    others = [j for j in range(k) if j != ref]
    diff = mean[others] - mean[ref]
    se = np.sqrt(s2 * (1.0 / cnt[others] + 1.0 / cnt[ref]))
    t = diff / se
    p = 2 * stats.t.sf(np.abs(t), dfres)
    F = float(np.sum(cnt * (mean - y.mean()) ** 2) / (k - 1) / s2)
    table = pd.DataFrame({"Kunde": [f"K{j}" for j in others], "Koeffizient": diff, "Standardfehler": se, "t": t, "p": p, "wahrer Effekt gegen Referenz": [C.UK[j] - C.UK[ref] for j in others],
                          "Container": cnt[others]})
    return KundeSample(True, table, int(np.sum(p < C.ALPHA)), F, float(stats.f.sf(F, k - 1, dfres)), cnt)


def verdict_kunde(frozen: dict, ref: int) -> tuple[str, str]:
    """Verdikt Experiment 5 aus der vorgerechneten Messreihe (Referenz `ref` und die Spannweite über alle Referenzen)."""
    sm = frozen["sig_mean"]
    lo, hi = int(np.argmin(sm)), int(np.argmax(sm))
    return "warning", (f"Referenz K{ref}: im Mittel {de(sm[ref], 1)} von 11 Koeffizienten „signifikant“ (|t| > t(0,975)); über alle Referenzen von {de(sm[lo], 1)} (Referenz K{lo}) bis {de(sm[hi], 1)} (Referenz K{hi}); "
                       f"der F-Test des ganzen Faktors verwirft in {pct(frozen['f_alt'], 1)} der {frozen['reps']} Wiederholungen, unabhängig von der Referenz. Ohne Kundeneffekt ist in {pct(float(np.mean(frozen['any_h0'])), 0)} "
                       f"der Wiederholungen mindestens ein Einzelkoeffizient „signifikant“ (Mittel über die Referenzen), der F-Test verwirft {pct(frozen['f_h0'], 1)}.")
