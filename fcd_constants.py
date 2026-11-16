"""Konstanten: Container-Vehikel (Generator), Handbeispiel, Codierungen, Regler-Grenzen, Farben und Presets.

Der Container-Generator ist ein Nachbau (kein Import) der Merkmale und Verweilzeit-Verteilung aus der `verweilzeit-demo`: Art, Kunde, Zollhalt, Wochenende, Vorlauf; Ausgang ist die
log-Verweilzeit (in log Tagen). Die zwölf Kundeneffekte sind die dort mit numpy `default_rng(12345).normal(0, 0,30, 12)` gezogenen Werte, hier einmal ausgelesen und als Literale
eingetragen (so hängt die Demo nicht an numpy.random). Alle Merkmale und Größenordnungen sind erfunden (synthetisch, kein Echtdatenbezug)."""

# --- Container-Vehikel -------------------------------------------------------------------------------------------------------------------------
ART_LABELS = ("Import", "Export", "Umschlag", "Leer")
ART_P = (0.35, 0.30, 0.25, 0.10)                    # Anteile der Containerarten (unausgewogen)
K_KUNDEN = 12
UK = (-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
      -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771)
U_SCALE = (1.0, 0.3, 0.3, 1.0)                      # Kundeneffekt je Art (bei Export und Umschlag ein Drittel)
S_BASE = (0.55, 0.25, 0.35, 0.90)                   # Streuung der log-Verweilzeit je Art um den Median (nur bei der Streuung „wie im Generator“)
P_ZOLL = 0.25                                       # Zollhalt bei Import
P_WE = 2.0 / 7.0                                    # Anteil der Ankünfte am Wochenende
W_GEN = (0.30, 0.0, 0.0, 0.30)                      # Wochenendeffekt je Art im Generator (log): nur bei Import und Leer
WE_LABELS = ("Wochentag", "Wochenende")
SZENARIEN = {
    "Generator": W_GEN,                              # Wochenendeffekt 0,30 / 0 / 0 / 0,30 (so im Generator)
    "additiv": (0.15, 0.15, 0.15, 0.15),             # derselbe Effekt bei jeder Art
    "Überkreuz": (0.30, 0.0, 0.0, -0.30),            # Effekte mit verschiedenem Vorzeichen
}
SZENARIO_KEYS = tuple(SZENARIEN)
STREUUNG_KEYS = ("gleich", "Generator")             # gleiche Normalstreuung in allen Zellen oder die Streuung des Generators
STREUUNG_HELP = {"gleich": "gleiche Streuung in allen Zellen (gepoolt, 0,689)", "Generator": "Streuung des Generators (je Art verschieden, Kunde, Zoll und Vorlauf im Rest)"}

# --- Handbeispiel (erfundene ganze Tage im Gewand von Art und Wochenende, nicht aus dem Generator) ------------------------------------------------
HAND_ARTEN = ("Import", "Export", "Umschlag")
HAND_MU = ((4, 4), (4, 8), (4, 12))                 # Zellmittel (Art, Wochentag / Wochenende)
HAND_H1 = (2, 2, 2, 2, 2, 2)                        # Beobachtungen je Zelle, Reihenfolge (Import WT, Import WE, Export WT, Export WE, Umschlag WT, Umschlag WE)
HAND_H2 = (2, 4, 3, 1, 2, 2)
CELL_MIN, CELL_MAX = 1, 8

# --- Codierungen -----------------------------------------------------------------------------------------------------------------------------
COD_REFERENZ, COD_SUMME, COD_HELMERT, COD_GEWICHTET = "Referenz", "Summe", "Helmert", "Gewichtet zentriert"
COD_KEYS = (COD_REFERENZ, COD_SUMME, COD_HELMERT, COD_GEWICHTET)
MODELL_KEYS = ("mit Wechselwirkung", "additiv")
ALPHA = 0.05
SING_TOL = 1e-10                                    # Rangentscheidung: kleinster singulärer Wert relativ zum größten

# --- Regler ----------------------------------------------------------------------------------------------------------------------------------
N_CHOICES = (200, 600, 2000)
SEED_MAX = 999999
KUNDE_MIN, KUNDE_MAX = 0, K_KUNDEN - 1
N_KUNDE_EXP = 300                                   # Import-Container im Experiment Referenzkunde

# --- vorgerechnete Messreihen -----------------------------------------------------------------------------------------------------------------
FROZEN_FILE = "fcd_frozen.json"
SIM_REPS = 10000
KUNDE_REPS = 10000
SIM_BASE_SEED = 130400                              # Zufalls-Seed der Zeile j: SIM_BASE_SEED + j
KUNDE_SEED = 130601
DUMMY_N = 200

# --- Farben ----------------------------------------------------------------------------------------------------------------------------------
COLOR_BAR = "#2a6fb0"
COLOR_BAD = "#c0504d"
COLOR_GOOD = "#2e7d4f"
COLOR_LINE = "#c77700"
COLOR_GREY = "rgba(150,150,150,0.6)"

# --- Presets: jeder legt den Stand ALLER Regler fest (fehlende Schlüssel = Standard) -----------------------------------------------------------
DEFAULTS = {
    "cell_00": 2, "cell_01": 2, "cell_10": 2, "cell_11": 2, "cell_20": 2, "cell_21": 2,
    "cod": COD_REFERENZ, "ref_art": 0, "ref_we": 0, "modell": MODELL_KEYS[0],
    "n": 600, "szenario": "Generator", "streuung": "gleich", "seed": 7001, "ref_kunde": 10,
}


def _cells(sizes):
    return {f"cell_{i}{j}": sizes[2 * i + j] for i in range(3) for j in range(2)}


PRESETS = {
    "Von Hand (H1)": {},
    "Unausgewogen (H2)": _cells(HAND_H2),
    "Container Generator": {"seed": 7002},
    "Additiv": {"szenario": "additiv", "seed": 7002},
    "Überkreuz": {"szenario": "Überkreuz", "seed": 7002},
    "Überkreuz, Streuung des Generators": {"szenario": "Überkreuz", "streuung": "Generator", "seed": 7002},
    "Dummy-Falle": {"n": 200, "seed": 7003},
    "Zwölf Kunden": {"ref_kunde": 4, "seed": 7004},
}
PRESET_HELP = {
    "Von Hand (H1)": "Sechs Zellen mit je zwei Beobachtungen, Zellmittel 4 / 4 / 4 am Wochentag und 4 / 8 / 12 am Wochenende: Experimente 1 und 2.",
    "Unausgewogen (H2)": "Dieselben Zellmittel, aber ungleich besetzte Zellen (2 / 4, 3 / 1, 2 / 2): Experiment 2 zeigt, dass jede Codierung eine andere Frage testet.",
    "Container Generator": "Container-Strom des Generators, Wochenendeffekt nur bei Import und Leer, n = 600, gleiche Streuung: Experiment 3.",
    "Additiv": "Derselbe Wochenendeffekt bei jeder Art: die Codierung ändert den Test des additiven Modells nicht, im Modell mit Wechselwirkung schon (Experiment 3).",
    "Überkreuz": "Wochenendeffekt +0,3 bei Import, −0,3 bei Leer, sonst 0: im Mittel über die Arten kein Effekt (Experiment 3).",
    "Überkreuz, Streuung des Generators": "Wie Überkreuz, aber mit der Streuung des Generators: die Summencodierung verwirft dann häufiger als das Niveau 5 % (Experiment 3).",
    "Dummy-Falle": "n = 200 Container, Achsenabschnitt plus vier Artspalten: Rang 4 von 5 (Experiment 4).",
    "Zwölf Kunden": "Zwölf Kunden als Faktor, Referenzkunde K4 (Experiment 5).",
}
