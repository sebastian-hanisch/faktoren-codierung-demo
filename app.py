"""Faktoren, Wechselwirkung, Codierung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Dieselbe Anpassung, andere Koeffizienten, andere Tests: Handbeispiel (Experimente 1 und 2), Container-Strom mit vorgerechneten Messreihen (Experimente 3 und 5) und die Dummy-Falle (Experiment 4).
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import fcd_constants as C
import fcd_evaluation as ev
import fcd_simulation as SIM
from fcd_layout import dunkles_design, kopf_css
from fcd_presets import (
    SETTING_SPECS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from fcd_visualization import build_coef_bars, build_dummy, build_f_bars, build_fit, build_kunde, build_rates

st.set_page_config(page_title="Faktoren, Wechselwirkung, Codierung – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _hand(s):
    res = ev.analyse_hand(s)
    return res, ev.compare_codings(res.hand, s.interaction)


@st.cache_data(show_spinner=False)
def _sample(s):
    return ev.sample_tests(s)


@st.cache_data(show_spinner=False)
def _dummy(s):
    return ev.dummy_trap(s)


@st.cache_data(show_spinner=False)
def _kunde(s):
    return ev.kunde_sample(s)


@st.cache_data(show_spinner=False)
def _exact(s):
    return ev.exact_powers(s)


def _table(df: pd.DataFrame, key: str):
    st.dataframe(df, width="stretch", hide_index=True, key=key)


def _show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in fcd_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧮 Faktoren, Wechselwirkung, Codierung")
st.markdown(
    """
**Ein Faktor mit Wechselwirkung hat keinen „Haupteffekt“ an sich.** Wer die Art eines Containers (Import, Export, Umschlag, Leer) und das Wochenende in ein lineares Modell steckt, muss die Stufen in Zahlen
**codieren**. Die angepassten Werte bleiben bei jeder Codierung dieselben, die **Koeffizienten** und die **Tests der Haupteffekte** beantworten aber je nach Codierung eine andere Frage: den Effekt bei einer
Referenzstufe, das ungewichtete Mittel der Effekte oder das mit den Stichprobenanteilen gewichtete Mittel. Die Demo zeigt das an einem Handbeispiel von Hand (Experimente 1 und 2) und am Container-Strom mit
Verweilzeiten (Experimente 3 bis 5).

Weiter unten: die **Kernfragen** mit ihren Antworten, die fünf **🔬 Experimente**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Codierung", expanded=True):
    st.markdown(
        """
1. **Ein Faktor mit k Stufen braucht k − 1 Spalten** neben dem Achsenabschnitt; *welche* Spalten, legt die **Codierung** fest. Alle vier gleich gut passen dieselben Werte an, nur die Zahlen in der Koeffiziententabelle ändern sich.
2. **Mit Wechselwirkung** (Produktspalten der beiden Faktoren) ist der Koeffizient eines Haupteffekts sein Effekt **an der Stelle, an der alle Spalten des anderen Faktors null sind**. Die Codierung wählt diese Stelle.
3. **Der Haupteffekt-Test** streicht die Haupteffektspalten bei erhaltenen Wechselwirkungsspalten (F-Test): Er prüft genau den Effekt an dieser Stelle, also bei jeder Codierung eine andere Hypothese.

| Codierung | Spalten eines Faktors mit k Stufen | Der Koeffizient des Wochenendes ist … |
|---|---|---|
| **Referenz** | k − 1 Punktvariablen ohne die Referenzstufe | der Wochenendeffekt bei der Referenzart |
| **Summe** | Zeilen summieren sich zu null, die **letzte** Stufe mit −1 | −½ × das ungewichtete Mittel der Wochenendeffekte (Wochentag +1, Wochenende −1) |
| **Helmert** | Stufe j gegen das Mittel der vorigen (nicht skaliert) | +½ × das ungewichtete Mittel der Wochenendeffekte (Wochentag −1, Wochenende +1) |
| **Gewichtet zentriert** | Referenz-Punktvariablen minus ihr Stichprobenanteil | das mit den Stichprobenanteilen der Arten gewichtete Mittel der Wochenendeffekte |

Die Namen und Vorzeichen der Codierungen in Programmen sind hier nicht geprüft; „gewichtet zentriert“ ist eine eigene Bezeichnung dieser Demo.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Container-Strom (Experimente 3 bis 5)**")
    st.select_slider("Container n", options=C.N_CHOICES, key="n", help="Stichprobengröße; die vorgerechneten Messreihen gibt es für 200, 600 und 2000 Container.")
    st.selectbox("Szenario (wahrer Wochenendeffekt je Art)", C.SZENARIO_KEYS, key="szenario",
                 help="Generator: +0,30 bei Import und Leer, sonst 0 (log Tage). Additiv: +0,15 bei jeder Art. Überkreuz: +0,30 bei Import, −0,30 bei Leer, sonst 0.")
    st.radio("Streuung in den Zellen", C.STREUUNG_KEYS, key="streuung", format_func=lambda k: C.STREUUNG_HELP[k],
             help="Gleich: in jeder der acht Zellen dieselbe Normalstreuung 0,689. Generator: die Streuung des Container-Generators (0,56 bis 0,97 je nach Zelle).")
    st.number_input("Zufalls-Seed", *bounds("seed"), key="seed", step=1, help="Bestimmt die eine Stichprobe in den Experimenten 3 bis 5 (Handbeispiel und vorgerechnete Messreihen haben feste Zufalls-Seeds).")
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)

sync_query_params({k: st.session_state[k] for k in SETTING_SPECS})
s = ev.settings_from_state(st.session_state.get)
res, cmp = _hand(s)

# --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Handbeispiel: dieselbe Anpassung, andere Koeffizienten")
st.caption(
    "Erfundene ganze Tage im Gewand von Art und Wochenende (kein Container-Strom): Zellmittel 4 / 4 / 4 am Wochentag und 4 / 8 / 12 am Wochenende für Import, Export, Umschlag; "
    "je Zelle Beobachtungen im Abstand ±1 um das Zellmittel (bei ungerader Zahl auch das Mittel selbst). Der Wochenendeffekt ist also 0, 4 und 8 Tage."
)
st.markdown("**Beobachtungen je Zelle** (Preset „Von Hand (H1)“: überall 2, „Unausgewogen (H2)“: 2 / 4, 3 / 1, 2 / 2)")
cols = st.columns(3)
for i, art in enumerate(C.HAND_ARTEN):
    with cols[i]:
        st.number_input(f"{art}, Wochentag", C.CELL_MIN, C.CELL_MAX, key=f"cell_{i}0", step=1)
        st.number_input(f"{art}, Wochenende", C.CELL_MIN, C.CELL_MAX, key=f"cell_{i}1", step=1)
c1, c2, c3, c4 = st.columns(4)
c1.selectbox("Codierung", C.COD_KEYS, key="cod", help="Referenz, Summe (letzte Stufe −1), Helmert (Stufe gegen Mittel der vorigen) oder gewichtet zentriert (am Stichprobenanteil zentriert).")
c2.radio("Modell", C.MODELL_KEYS, key="modell", help="Mit Wechselwirkung (Produktspalten) oder additiv (ein Wochenendeffekt für alle Arten).")
ref_off = s.cod in (C.COD_SUMME, C.COD_HELMERT)
c3.selectbox("Referenzstufe Art", [0, 1, 2], format_func=lambda i: C.HAND_ARTEN[i], key="ref_art", disabled=ref_off,
             help="Welche Art fehlt in den Spalten (Referenz, gewichtet zentriert). Bei Summe und Helmert gibt es keine Referenzstufe: die Summe lässt die letzte Stufe weg, Helmert vergleicht Stufe für Stufe.")
c4.selectbox("Referenzstufe Wochenende", [0, 1], format_func=lambda i: C.WE_LABELS[i], key="ref_we", disabled=ref_off, help="Welche Stufe fehlt (Referenz, gewichtet zentriert).")

if res.status != "ok":
    st.warning(res.status)
d = res.design
m1, m2, m3 = st.columns(3)
m1.metric("Achsenabschnitt", ev.de(float(res.table["Wert"][0]), 2), help=f"Codierung {s.cod}")
m2.metric(f"Koeffizient „{d.names[d.i_b[0]]}“", ev.de(float(res.table["Wert"][d.i_b[0]]), 2), help=f"Codierung {s.cod}")
m3.metric("Fehlerquadratsumme (Anpassung)", ev.de(res.sse, 2), delta=f"{res.hand.dfres} Freiheitsgrade im Rest", delta_color="off")
g1, g2 = st.columns(2)
with g1:
    st.markdown("**Die Anpassung: dieselbe in jeder Codierung** (Rauten: angepasste Zellmittel, Punkte: Beobachtungen)")
    st.plotly_chart(build_fit(res.hand, res.fitted_cells), width="stretch", key="fit_chart")
with g2:
    st.markdown("**Die Koeffizienten: je Codierung andere Zahlen**")
    st.plotly_chart(build_coef_bars(res.summary), width="stretch", key="coef_chart")
_show(ev.verdict_fit(res))

disp = pd.DataFrame({"Koeffizient": res.table["Koeffizient"], "Wert": [ev.de(x, 3) for x in res.table["Wert"]], "Standardfehler": [ev.de(x, 3) for x in res.table["Standardfehler"]],
                     "t": [ev.de(x, 2) for x in res.table["t"]], "p": [ev.p_text(x) for x in res.table["p"]], "Bedeutung": res.table["Bedeutung"]})
st.markdown(f"**Koeffizienten der Codierung „{s.cod}“** (Modell: {s.modell})")
_table(disp, "coef_table")
with st.expander("Entwurfsmatrix (eine Zeile je Zelle) und Gewichte der Zellmittel je Koeffizient"):
    st.markdown("Jede Beobachtung einer Zelle hat dieselbe Zeile der Entwurfsmatrix. Bei Gewichtet zentriert hängt sie von den Stichprobenanteilen ab.")
    st.dataframe(res.cell_matrix.map(lambda x: ev.de(x, 3)), width="stretch", key="cell_matrix")
    if res.weights is not None:
        st.markdown("**Jeder Koeffizient ist eine feste Summe der Zellmittel** (Gewichte; Zeile = Koeffizient, Spalte = Zelle). Das ist die genaue Aussage hinter den Sätzen oben.")
        st.dataframe(res.weights, width="stretch", key="weights_table")

# --- Kernfragen ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Welche Zahl bedeutet welcher Koeffizient?**")
_show(ev.verdict_meaning(res, s))
st.markdown("**2. Gibt es einen Wochenendeffekt?**")
_show(ev.verdict_hypotheses(cmp))
st.markdown("**3. Was ändert sich, wenn man nur die Codierung wechselt?**")
_show(ev.verdict_switch(res, cmp))

st.markdown("---")

# --- Experiment 2 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wo wird der Haupteffekt gemessen?")
st.caption("Dasselbe Handbeispiel (Zellgrößen oben): F-Test der Streichung der Haupteffektspalten bei erhaltener Wechselwirkung, je Codierung. „Getesteter Effekt“: der Effekt, den der Test prüft (Wochenendeffekt, gemittelt mit den Gewichten der Codierung), ± seine Standardabweichung.")
rows = []
for r in cmp["rows"]:
    rows.append({"Codierung": r["name"], "getesteter Effekt (Wochenende)": f"{ev.de(r['effect'], 2)} ± {ev.de(r['sd'], 2)}", "F Wochenende": ev.de(r["F_B"], 2), "p Wochenende": ev.p_text(r["p_B"]),
                 "F Art": ev.de(r["F_A"], 2), "p Art": ev.p_text(r["p_A"])})
_table(pd.DataFrame(rows), "cmp_table")
if cmp["interaction"]:
    st.markdown(
        f"**Typ I gegen Typ III für den Haupteffekt Art** (Art zuerst im Modell: Vergleich der nach Zellgrößen gewichteten Zeilenmittel; Typ III: Summencodierung, Vergleich der ungewichteten Zeilenmittel): "
        f"Typ I F = {ev.de(cmp['typ1'][0], 2)} (p = {ev.p_text(cmp['typ1'][1])}), Typ III F = {ev.de(cmp['typ3'][0], 2)} (p = {ev.p_text(cmp['typ3'][1])}). "
        f"**Wechselwirkung** (in jeder Codierung dasselbe): F = {ev.de(cmp['F_I'], 2)} (p = {ev.p_text(cmp['p_I'])})."
    )
st.plotly_chart(build_f_bars(cmp), width="stretch", key="f_chart")
_show(ev.verdict_typ(cmp))

st.markdown("---")

# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wie oft verwirft der Test? (Container-Strom, vorgerechnet)")
row = ev.frozen_row(s.streuung, s.szenario, s.n)
row200 = ev.frozen_row(s.streuung, s.szenario, 200)
rt = ev.rates(row)
frozen_note = (f"{rt['reps']:,} Wiederholungen je Zeile".replace(",", ".") + f" (Zähleinheit: Wiederholung = eine Stichprobe von {s.n} Containern), vorgerechnet mit `tools/fcd_make_frozen.py` (`data/fcd_frozen.json`); "
               f"{rt['leer']} Wiederholungen mit leerer Zelle sind nicht gezählt. Die Raute ist die exakte Teststärke (nichtzentrale F-Verteilung, erwartete Zellgrößen, gleiche Streuung); Fehlerbalken: 2 Standardfehler.")
st.caption(f"Einstellungen aus der Seitenleiste: n = {s.n}, Szenario {s.szenario}, Streuung {s.streuung}. " + frozen_note)
names = list(SIM.SETTING_NAMES)
exact = [e["power_B"] for e in _exact(s)] if s.streuung == "gleich" else None
if exact is None:
    st.caption("Mit der Streuung des Generators gibt es keine exakte Teststärke in einer Formel (die Zellstreuungen sind verschieden); die Rauten entfallen.")
st.plotly_chart(build_rates(names, rt["B"], rt["valid"], exact, "Test „Haupteffekt Wochenende“ verwirft (%)"), width="stretch", key="rate_chart")
eff = ev.true_effects(s.szenario)
tab = pd.DataFrame({"Codierung": names, "getesteter Effekt": [ev.de(x, 3) for x in eff], "verwirft Wochenende": [ev.pct(x) for x in rt["B"]],
                    "exakt (gleiche Streuung)": [ev.pct(e) for e in exact] if exact is not None else ["–"] * len(names), "verwirft Art": [ev.pct(x) for x in rt["A"]]})
_table(tab, "rate_table")
st.markdown(f"Test der **Wechselwirkung** (in jeder Codierung derselbe): verwirft in {ev.pct(rt['I'])} der Wiederholungen. Einzelkoeffizient „Leer × Wochenende“ (t-Test): bei Referenz Import {ev.pct(rt['li'])}, bei Referenz Export {ev.pct(rt['le'])}.")
_show(ev.verdict_container(s, rt, ev.rates(row200)))

st.markdown(f"**Eine Stichprobe live** (n = {s.n}, Zufalls-Seed {s.seed}): dieselben Tests, wie ein Programm sie über die Entwurfsmatrix rechnet.")
smp = _sample(s)
if not smp.ok:
    st.warning(smp.reason)
else:
    st.dataframe(pd.DataFrame({"Codierung": [r["name"] for r in smp.rows], "F Wochenende": [ev.de(r["F_B"], 2) for r in smp.rows], "p Wochenende": [ev.p_text(r["p_B"]) for r in smp.rows],
                               "verwirft (5 %)": ["ja" if r["p_B"] < C.ALPHA else "nein" for r in smp.rows], "F Art": [ev.de(r["F_A"], 2) for r in smp.rows], "p Art": [ev.p_text(r["p_A"]) for r in smp.rows]}),
                 width="stretch", hide_index=True, key="sample_table")
    st.caption(f"Wechselwirkung in dieser Stichprobe: F = {ev.de(smp.F_I, 2)} (p = {ev.p_text(smp.p_I)}). Die Spalte „verwirft“ ist ein Einzelergebnis; die Häufigkeit über viele Wiederholungen steht oben.")

st.markdown("---")

# --- Experiment 4 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Die Dummy-Falle")
st.caption(f"Achsenabschnitt plus vier Spalten für die vier Arten: die vier Spalten summieren sich zum Achsenabschnitt. Stichprobe von n = {s.n} Containern, Zufalls-Seed {s.seed}, Szenario {s.szenario}, Streuung {s.streuung}.")
dm = _dummy(s)
_show(ev.verdict_dummy(dm))
sol = pd.DataFrame({"": ["Achsenabschnitt", *[f"Art {a}" for a in C.ART_LABELS]]})
for name, v in dm.solutions.items():
    sol[name] = [ev.de(x, 3) for x in v] if v is not None else ["Fehler"] * 5
st.dataframe(sol, width="stretch", hide_index=True, key="dummy_table")
st.plotly_chart(build_dummy(dm), width="stretch", key="dummy_chart")
st.markdown("Bemerkungen zu den Wegen: " + "; ".join(f"**{k}**: {v}" for k, v in dm.notes.items()) + f". Nullvektor der Spalten: (1, −1, −1, −1, −1), `X·v = 0`.")

st.markdown("---")

# --- Experiment 5 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Referenzkunde (Kunde als Faktor mit zwölf Stufen)")
kf = ev.kunde_frozen()
st.caption(f"Import-Container, n = {kf['n']}, Kunde als Faktor mit 12 Stufen und den echten Kundeneffekten des Generators; {kf['reps']:,} Wiederholungen".replace(",", ".")
           + " vorgerechnet (`data/fcd_frozen.json`). Gezählt wird je Referenzkunde, wie viele der 11 Koeffizienten „signifikant“ sind (|t| > t(0,975)).")
st.selectbox("Referenzkunde", list(range(C.K_KUNDEN)), key="ref_kunde", format_func=lambda j: f"K{j} (Effekt {ev.de(C.UK[j], 2)})")
st.plotly_chart(build_kunde(kf, s.ref_kunde), width="stretch", key="kunde_chart")
_show(ev.verdict_kunde(kf, s.ref_kunde))
ks = _kunde(s)
st.markdown(f"**Eine Stichprobe live** (300 Import-Container, Zufalls-Seed {s.seed}, Referenz K{s.ref_kunde}):")
if not ks.ok:
    st.warning("Ein Kunde kommt in der Stichprobe nicht vor: anderen Zufalls-Seed wählen.")
else:
    t = ks.table
    st.dataframe(pd.DataFrame({"Kunde": t["Kunde"], "Koeffizient": [ev.de(x, 3) for x in t["Koeffizient"]], "t": [ev.de(x, 2) for x in t["t"]], "p": [ev.p_text(x) for x in t["p"]],
                               "„signifikant“": ["ja" if x < C.ALPHA else "nein" for x in t["p"]], "wahrer Effekt gegen Referenz": [ev.de(x, 2) for x in t["wahrer Effekt gegen Referenz"]],
                               "Container": [int(x) for x in t["Container"]]}), width="stretch", hide_index=True, key="kunde_table")
    st.caption(f"{ks.n_sig} von 11 Koeffizienten „signifikant“. F-Test des ganzen Faktors (hängt nicht an der Referenz): F = {ev.de(ks.F, 2)}, p = {ev.p_text(ks.p)}.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wo gemessen |
|---|---|---|
| **Gleiche Streuung in allen Zellen** (eine gemeinsame Restvarianz im F-Test) | Mit der Streuung des Generators (Zellstreuungen 0,56 bis 0,97, Verhältnis 1,7) verwirft die Summencodierung bei wahrem Effekt 0 (Überkreuz) in 9,9 bis 10,5 % statt 5 % der Wiederholungen (n = 200 bis 2000); Referenz Export (wahrer Effekt 0 im Generator- und im Überkreuz-Szenario) verwirft nur in 1,5 bis 1,8 %. | Experiment 3, Streuung „Generator“ |
| **Die Referenzstufe ist eine Wahl, keine Eigenschaft der Daten** | Wer nur die Referenzstufe wechselt, wechselt die Frage: Referenz Import verwirft bei n = 600 in rund 80 %, Referenz Export in rund 5 % der Wiederholungen. | Experiment 3 |
| **Volle Spaltenzahl ohne Redundanz** (Rang p) | Achsenabschnitt plus alle Artspalten hat Rang 4 von 5: die Anpassung ist eindeutig, die Koeffizienten nicht, und `solve(X'X)` meldet hier keinen Fehler. | Experiment 4 |
| **Jede Zelle besetzt** | Bei n = 200 fehlt in 24 bis 41 von 10 000 Wiederholungen eine der acht Zellen; die Wechselwirkung ist dann nicht schätzbar (die App meldet „Zelle leer“). | Experiment 3 |
| **Ein Test für den ganzen Faktor statt vieler Einzelvergleiche** | Zwölf Kunden ohne Kundeneffekt: ein Einzelkoeffizient ist in rund 31 % der Wiederholungen „signifikant“, der F-Test des Faktors in rund 5 %. | Experiment 5 |
| **Nur dieses Modell** | Zwei Faktoren, Ausgang log-Verweilzeit, Modell mit Wechselwirkung. Drei und mehr Faktoren, geordnete Faktoren, Typ II, selbst gewählte Kontraste und Mehrfachvergleiche sind nicht gezeigt. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Zellmittelmodell.** Faktor A (Art) hat $k_A$, Faktor B (Wochenende) hat $k_B$ Stufen; $\theta_{ij}$ ist der Erwartungswert der Zelle $(i, j)$, $n_{ij}$ ihre Besetzung, $\sigma^2$ die Restvarianz:
$$y_{ijl} = \theta_{ij} + \varepsilon_{ijl}, \qquad \varepsilon_{ijl} \sim N(0, \sigma^2).$$

**Entwurfsmatrix.** Zu jeder Codierung gehört eine $k \times (k-1)$-Matrix $C$ (Zeile = Stufe). Spalten des Modells mit Wechselwirkung: Achsenabschnitt, $C_A$, $C_B$ und alle Produkte $C_A \odot C_B$,
insgesamt $p = k_A k_B$ Spalten. Die Zeile der Zelle $(i,j)$ sei $x_{ij}$, $X_c$ die Matrix aller Zellzeilen (quadratisch, invertierbar bei voller Besetzung). Das Modell passt die Zellmittel an, und der Koeffizientenvektor folgt allein aus ihnen:
$$\hat\beta = X_c^{-1}\,\hat\theta, \qquad \hat\theta_{ij} = \bar y_{ij}.$$
Jeder Koeffizient ist eine feste Summe der Zellmittel; die Gewichte (Zeile von $X_c^{-1}$) zeigt der Expander unter der Koeffiziententabelle.

**Wo der Koeffizient eines Haupteffekts misst.** Die Zellmittel an der Stelle, an der alle Spalten des anderen Faktors null sind, sind $\sum_{i,j} \omega_i^{A}\omega_j^{B}\,\theta_{ij}$ mit Gewichten, die die Codierung festlegt:
Referenz $\omega = e_r$ (die Referenzstufe allein), Summe und Helmert $\omega = (1/k, \dots, 1/k)$, gewichtet zentriert $\omega$ = Stichprobenanteile. Für den Koeffizienten von B gilt bei Referenz und gewichtet zentriert
$\hat\beta_B = \sum_i \omega_i^{A}(\bar y_{i1} - \bar y_{i0})$, bei Summe $-\tfrac12$ mal, bei Helmert $+\tfrac12$ mal dasselbe Mittel der Wochenendeffekte.

**Haupteffekt-Test.** Streichung der Haupteffektspalten bei erhaltenen Wechselwirkungsspalten: $F = \dfrac{(\mathrm{SSE}_0 - \mathrm{SSE}_1)/q}{\mathrm{SSE}_1/(n - k_A k_B)}$. Das ist dieselbe Zahl wie der Wald-Test der Hypothese
$$H_B:\ \sum_i \omega_i^{A}\,(\theta_{i1} - \theta_{i0}) = 0, \qquad F = \frac{\hat d^{\,2}}{\hat s^2 \sum_i (\omega_i^{A})^2 \left(\frac{1}{n_{i0}} + \frac{1}{n_{i1}}\right)}, \quad \hat d = \sum_i \omega_i^{A}(\bar y_{i1} - \bar y_{i0}),$$
mit $1$ und $n - k_A k_B$ Freiheitsgraden. Für Art: $\psi_i = \sum_j \omega_j^{B}\theta_{ij}$ sollen für alle $i$ gleich sein, $F = \tfrac{1}{k_A - 1}\sum_i p_i(\hat\psi_i - \bar\psi_p)^2$ mit $p_i = 1/\widehat{\mathrm{Var}}(\hat\psi_i)$.
Wechselwirkung: die Wochenendeffekte $\theta_{i1} - \theta_{i0}$ sollen für alle $i$ gleich sein.

**Typ I und Typ III für Art.** Typ I (Art zuerst) vergleicht die nach Zellgrößen gewichteten Zeilenmittel $\sum_j n_{ij}\bar y_{ij}/n_{i\cdot}$; Typ III in der Summencodierung die ungewichteten Zeilenmittel $\tfrac1{k_B}\sum_j \bar y_{ij}$.
Beide stimmen nur bei gleichen Zellgrößen überein. Typ II ist nicht nachgerechnet.

**Exakte Teststärke.** Bei wahren Zellmitteln $\theta$ und gleicher Streuung ist $F$ nichtzentral $F(q,\, n - k_A k_B,\, \lambda)$ mit $\lambda = d^2/\mathrm{Var}(\hat d)$ (Wochenende, $q = 1$); die Teststärke ist
$P(F' > F_{0{,}95;\,q,\,n - k_A k_B})$. Hier mit den erwarteten Zellgrößen $n\,P(\mathrm{Art})\,P(\mathrm{Wochenende})$ gerechnet; bei $\lambda = 0$ gilt das Niveau 5 %.

**Rangmangel.** Bei Achsenabschnitt plus allen $k$ Stufenspalten gilt $X v = 0$ für $v = (1, -1, \dots, -1)$. Die Anpassung $X\hat\beta$ ist eindeutig, $\hat\beta$ nur bis auf $\hat\beta + t v$; `lstsq` und `pinv` wählen die Lösung kleinster Länge.

**Literatur.** Hastie, Tibshirani, Friedman: *The Elements of Statistical Learning*, 2. Aufl., Abschn. 3.2, S. 46 (Seitenzahl nach Quellenbericht, nicht im Original erneut nachgeschlagen): bei nicht vollem Rang (zum Beispiel durch redundant codierte qualitative Eingaben) sind die Koeffizienten nicht
eindeutig, die angepassten Werte aber schon. Kontraste, Typ-III-Quadratsummen und die Namen der Codierungen in Programmen sind hier nicht an einer Quelle geprüft.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Regression und GLM: wann die Gleichung trügt](https://sebastianhanisch.net/konzepte-regression.html)."
)
