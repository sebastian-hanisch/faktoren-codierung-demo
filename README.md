# Faktoren, Wechselwirkung, Codierung – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-faktoren-codierung-demo.streamlit.app/)** -->

Konzepte-Demo zu **Regression und verallgemeinerten linearen Modellen**. Sie folgt auf [`kollinearitaet-auswahl-demo`](https://github.com/sebastian-hanisch/kollinearitaet-auswahl-demo), in der ein Merkmal eine Spalte ist und sein Koeffizient eine feste Bedeutung hat („Änderung des Ausgangs je Einheit bei sonst Gleichem“), die bei **fast** abhängigen Spalten schlecht bestimmt ist. Hier fällt diese Annahme an zwei Stellen: Ein Faktor mit k Stufen braucht k − 1 Spalten und eine **Codierung**, und bei **Wechselwirkung** hängt der Effekt eines Merkmals vom Wert des anderen ab, sodass der „Haupteffekt“ der Effekt an *einer* Stelle ist, die die Codierung wählt. Der Zusammenhang zur Projektion auf den Spaltenraum (gleicher Spaltenraum, andere Basis) steht in [`ols-projektion-demo`](https://github.com/sebastian-hanisch/ols-projektion-demo). Die Verweilzeit-Verteilung des Container-Vehikels ist ein Nachbau der in [`verweilzeit-demo`](https://github.com/sebastian-hanisch/verweilzeit-demo) benutzten (kein Import).

## Warum dieses Problem

Wer „Art“ (Import, Export, Umschlag, Leer) und „Wochenende“ in ein lineares Modell steckt und die Wechselwirkung zulässt, bekommt je nach Codierung andere Koeffizienten und andere Tests der Haupteffekte, obwohl die angepassten Werte dieselben bleiben. Im Container-Modell dieser Demo (Wochenendeffekt +0,30 log Tage bei Import und Leer, sonst 0) verwirft der Test „Haupteffekt Wochenende“ bei n = 600 Containern, gleicher Streuung in den Zellen und 10 000 Wiederholungen in 79,9 % der Wiederholungen bei Referenz Import, in 5,1 % bei Referenz Export und in 55,2 % bei Summencodierung: die Tests beantworten je nach Codierung eine andere Frage (Effekt bei Export 0, bei Import +0,30).

## Kernfragen

1. **Welche Zahl bedeutet welcher Koeffizient?** Der Koeffizient eines Haupteffekts ist sein Effekt an der Stelle, an der alle Spalten des anderen Faktors null sind; die Codierung legt die Stelle fest (Referenzstufe, ungewichtetes Mittel, gewichtetes Mittel).
2. **Gibt es einen Wochenendeffekt?** Der Haupteffekt-Test prüft genau den Effekt an dieser Stelle, also je Codierung eine andere Hypothese.
3. **Was ändert sich, wenn man nur die Codierung wechselt?** Gleich bleiben die angepassten Werte und der Test der Wechselwirkung; es ändern sich Achsenabschnitt, Koeffizienten und die Haupteffekt-Tests.

## Modell

Zwei Faktoren: A (Art, vier Stufen im Container-Strom, drei im Handbeispiel) und B (Wochenende, zwei Stufen); Ausgang y = log-Verweilzeit (log Tage) im Container-Strom, ganze Tage im Handbeispiel. Volles Modell mit Wechselwirkung: p = k_A · k_B Spalten **einschließlich Achsenabschnitt** (im Container-Strom 8). „Haupteffekt-Test“ heißt hier immer: **F-Test der Streichung der Haupteffektspalten bei erhaltenen Wechselwirkungsspalten**. Vier Codierungen, alle mit vollem Rang:

| Codierung | Spalten eines Faktors mit k Stufen | Der Koeffizient des Wochenendes (mit Wechselwirkung) ist … |
|---|---|---|
| **Referenz** (Referenzstufe r) | k − 1 Punktvariablen ohne Stufe r | der Wochenendeffekt bei Art r |
| **Summe** (letzte Stufe −1) | k − 1 Spalten, Zeilen summieren sich zu null | −½ × das **ungewichtete** Mittel der Wochenendeffekte (Wochentag +1, Wochenende −1 codiert) |
| **Helmert** (Stufe j gegen Mittel der vorigen, nicht skaliert) | k − 1 Spalten, Zeilen summieren sich zu null | +½ × dasselbe Mittel (Wochentag −1, Wochenende +1 codiert) |
| **Gewichtet zentriert** (eigene Bezeichnung, Referenzspalten am Stichprobenanteil zentriert) | k − 1 Punktvariablen minus ihr Anteil | das mit den **Stichprobenanteilen der Arten** gewichtete Mittel der Wochenendeffekte |

Allgemein: Im Modell mit Wechselwirkung passt die Anpassung die Zellmittel an, und jeder Koeffizient ist eine feste Summe der Zellmittel (Zeile der Inversen der Zell-Entwurfsmatrix); die App zeigt diese Gewichte. Konventionen, die man kennen muss: Die Summencodierung lässt die **letzte** Stufe weg, Umordnen der Stufen kehrt das Vorzeichen um. Wie Programme die Codierungen nennen, welche Stufe sie als Referenz nehmen und in welche Richtung sie Helmert rechnen, ist hier **nicht geprüft**; die Namen „Typ I, II, III“ der Quadratsummen stammen aus der Software-Praxis und sind hier nicht an einer Quelle geprüft; nachgerechnet sind nur Typ I (Art zuerst) und Typ III (Summencodierung), Typ II **nicht**. Die Formeln stehen im 📐-Expander der App.

**Quelle.** Hastie, Tibshirani, Friedman: *The Elements of Statistical Learning*, 2. Aufl., Abschn. 3.2, S. 46 (nach Quellenbericht gelesen, hier nicht im Original erneut nachgeschlagen): bei nicht vollem Rang (zum Beispiel durch redundant codierte qualitative Eingaben) sind die Koeffizienten nicht eindeutig, die angepassten Werte aber schon. Das ist die einzige hier gelesene Quelle zum Thema; alle Zahlen und Formeln dieser README sind eigene Nachrechnung in diesem Modell, bei diesen Parametern. Eine Neuheit wird nicht behauptet (die Falle ist bekannt); neu ist nur, dass man sie mit Zähleinheiten und Wiederholungen vorführen kann.

## Methodik

- **Rechenweg der App** (`fcd_design.py`): Entwurfsmatrix je Codierung, Kleinste Quadrate per `numpy.linalg.lstsq`, F-Test der Streichung von Spalten (Modellvergleich). Bei Rangmangel (zum Beispiel einer leeren Zelle im Modell mit Wechselwirkung) rechnet die App **nicht**, sondern meldet „Zelle leer“.
- **Zweiter Rechenweg** (`fcd_tests.py`): Wald-Tests in Zellmittel-Parametern; dieselben F-Werte (Orakel in `tests/test_oracle_faktoren_codierung.py`). Die vorgerechneten Messreihen benutzen ihn, weil er über alle Wiederholungen gestapelt rechnet.
- **Experiment 1 und 2: Handbeispiel** mit erfundenen ganzen Tagen (kein Container-Strom): Zellmittel Wochentag 4 / 4 / 4, Wochenende 4 / 8 / 12 für Import, Export, Umschlag, je Zelle Beobachtungen im Abstand ±1 um das Zellmittel. H1: je Zelle zwei Beobachtungen; H2: Zellgrößen Import 2 / 4, Export 3 / 1, Umschlag 2 / 2 (n = 14).
- **Experiment 3: Häufigkeit über Wiederholungen** (Zähleinheit: Wiederholung = eine Stichprobe von n Containern): vorgerechnet (`data/fcd_frozen.json`, erzeugt von `tools/fcd_make_frozen.py`), 18 Zeilen (Streuung gleich oder wie im Generator, drei Szenarien, n = 200, 600, 2000) zu je 10 000 Wiederholungen, je Zeile ein Strom mit eigenem Zufalls-Seed; Wiederholungen mit leerer Zelle zählen nicht. Daneben die exakte Teststärke (nichtzentrale F-Verteilung, erwartete Zellgrößen, gleiche Streuung) und eine Stichprobe live.
- **Experiment 4: Dummy-Falle** (Achsenabschnitt plus vier Artspalten): Rang, Nullvektor, vier Lösungswege und ihre Anpassungen.
- **Experiment 5: Referenzkunde** (Import-Container, Kunde als Faktor mit zwölf Stufen, n = 300): vorgerechnet (10 000 Wiederholungen) und eine Stichprobe live.
- **Zufall:** nur ganzzahliger SplitMix64 (`fcd_sample.py`, vektorisiert, bitgleich zur skalaren Folge in `fcd_rng.py`, einer Kopie der geprüften Vorlage des Website-Repos). Logarithmus und Kosinus der Normalzahlen und der Mediane laufen über Grundrechenarten (`ln_pos`, `cos_2pi`), nicht über die C-Bibliothek des Systems; das Ergebnis ist auf jeder Plattform dasselbe. Der Strom ist **nicht bitgleich** zu dem der `verweilzeit-demo` (dort `numpy.random`), die Verteilung ist dieselbe.
- **Container-Generator** (`fcd_generator.py`): Art 35 / 30 / 25 / 10 %, zwölf Kunden mit festen Effekten (bei Export und Umschlag mit Faktor 0,3), Zollhalt bei Import (Wahrscheinlichkeit 0,25), Wochenende (Wahrscheinlichkeit 2/7), Vorlauf; Median Import 3,5·exp(0,7·Zoll + 0,3·Wochenende), Export 0,6 + 0,85·Vorlauf, Umschlag 0,3 + 0,8·Vorlauf, Leer 7·exp(0,3·Wochenende); lognormale Verweilzeit. Alle Merkmale und Größenordnungen sind erfunden (synthetisch, kein Echtdatenbezug). Die Streuung „gleich“ nimmt in jeder Zelle dieselbe Normalstreuung σ_pool; „Generator“ nimmt die Streuung je Art (Kunde, Zoll und Vorlauf bleiben im Rest).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Container-Strom dieser Demo, Modell Art × Wochenende mit Wechselwirkung, Ausgang log-Verweilzeit, Wochenendeffekt je Art wie angegeben (Generator-Szenario: 0,30 / 0 / 0 / 0,30), Niveau 5 %, **gleiche Streuung** in den Zellen, wenn nichts anderes steht; 10 000 Wiederholungen je Zeile (Monte-Carlo-Fehler höchstens ±1,0 Prozentpunkte bei zwei Standardfehlern). Die Zähleinheit ist die Wiederholung (eine Stichprobe von n Containern); Datei der Messung: `data/fcd_frozen.json`, Quelle `fcd_simulation.py`.

| Nr. | Frage | Befund |
|---|---|---|
| B1 | Wo misst der Koeffizient des Wochenendes? | Im Modell mit Wechselwirkung ist er bei Referenz der Effekt bei der Referenzart, bei Summe −½ × ungewichtetes Mittel der vier Artwirkungen, bei Helmert +½ × dasselbe Mittel, bei gewichtet zentriert das mit den Stichprobenanteilen gewichtete Mittel: Bruch gleich Bruch in 40 von 40 Zufalls-Datensätzen (4 × 2 Zellen, je 2 bis 6 Beobachtungen, ganzzahlige Werte). |
| B2 | Handbeispiel H1 (je Zelle zwei Beobachtungen) | SSE 12 bei 6 Freiheitsgraden. Referenz: F Wochenende = 0 und F Art = 0 (p = 1). Summe: F Wochenende = 24 (= 48 / 2, p = 0,0027), F Art = 8 (p = 0,0203). Wechselwirkung: F = 8 in jeder Codierung. Koeffizienten: Achsenabschnitt 4 (Referenz) gegen 6 (Summe), Wochenende 0 gegen −2. |
| B3 | Handbeispiel H2 (Zellgrößen 2 / 4, 3 / 1, 2 / 2) | SSE 12 bei 8 Freiheitsgraden. F Wochenende: Referenz Import 0, Referenz Export 8 (p = 0,0222), Referenz Umschlag 128/3 = 42,67, Summe 1152/37 = 31,14, gewichtet zentriert 4608/193 = 23,88. Wochenende-Koeffizient 0, −2 und 24/7. Art: Typ I F = 92/7 = 13,14, Typ III F = 1360/111 = 12,25. |
| B4 | Additives Modell (ohne Wechselwirkung), H1 | F Wochenende = 96/11 = 8,73 in jeder Codierung; Koeffizient 4 (Referenz) gegen −2 (Summe): Maß und Vorzeichen ändern sich, der Test nicht. |
| B5 | Zellmittel und Streuung des Generators | Exakte Zellmittel (Wochentag / Wochenende): Import 1,400 / 1,700, Export 1,548 / 1,548, Umschlag 0,937 / 0,937, Leer 1,918 / 2,218 (log Tage); gepoolte Streuung σ_pool = 0,689; Zellstreuung 0,56 (Export) bis 0,97 (Leer), Verhältnis 1,73. Gegen 400 000 simulierte Container: größte Abweichung unter 4 Standardfehlern. |
| B6 | Generator-Szenario, n = 200 / 600 / 2000: Test „Haupteffekt Wochenende“ verwirft in … % (in Klammern die exakte Teststärke) | Referenz Import 36,8 / 79,9 / 100,0 (37,4 / 81,2 / 99,9); Referenz Export 4,7 / 5,1 / 5,1 (5,0); Referenz Umschlag 5,2 / 5,2 / 4,7 (5,0); Referenz Leer 13,8 / 32,8 / 78,8 (14,1 / 33,1 / 79,4); Summe = Helmert 22,1 / 55,2 / 97,3 (23,4 / 57,2 / 97,5); gewichtet zentriert 23,7 / 56,3 / 97,9 (23,8 / 58,1 / 97,7); Test der Wechselwirkung 17,4 / 48,8 / 97,3. Getestete Effekte: +0,300 / 0 / 0 / +0,300 / +0,150 / +0,135. |
| B7 | n = 600, Streuung gleich, andere Szenarien | Additiv (0,15 überall): Referenz Import 29,4, Export 26,6, Umschlag 22,4, Leer 11,8, Summe 57,1, gewichtet 66,9, Wechselwirkung 5,0 (Einzelmessung mit Monte-Carlo-Fehler von etwa ±1,0 Punkt; bei einer anderen Wiederholung können Summe und gewichtet 1 bis 1,5 Punkte tiefer liegen, B17). Überkreuz (0,30 / 0 / 0 / −0,30): Referenz Import 80,8, Export 4,8, Umschlag 5,1, Leer 32,3, **Summe 5,4**, gewichtet 22,7, Wechselwirkung 69,0. |
| B8 | n = 600, Streuung des Generators | Generator-Szenario: Referenz Import 78,7, Export 1,7, Umschlag 3,3, Leer 38,6, Summe 55,7, gewichtet 58,1, Wechselwirkung 52,4. **Überkreuz (wahrer Effekt der Summe 0): Summe 10,0 / 10,5 / 9,9 % bei n = 200 / 600 / 2000 statt 5 %.** Additiv (wahre Wechselwirkung 0): Test der Wechselwirkung 9,6 / 9,6 / 9,9 % statt 5 %. Referenz Export (wahrer Effekt 0) verwirft nur in 1,5 bis 1,8 %, Referenz Umschlag in 3,0 bis 3,7 %. |
| B9 | Simulation gegen exakte Teststärke (Streuung gleich, 72 Zeilen: 3 Szenarien × 3 Größen × 8 Einstellungen) | Größte Abweichung 2,0 Prozentpunkte; 59 von 72 Zeilen innerhalb zweier, 62 innerhalb dreier Standardfehler. Die Formel setzt die **erwarteten** Zellgrößen ein; das Mittel der exakten Teststärke über die **tatsächlichen** Zellgrößen der Wiederholungen liegt bei Summe darunter (n = 600: 56,1 % statt 57,2 %, simuliert 55,2 %). |
| B10 | Negativbefund: der Haupteffekt-Test für Art | Verwirft in 99,0 bis 100,0 % der Wiederholungen in allen 18 Zeilen und allen Einstellungen außer Referenz Import / Wochenende (n = 200: 75,5 bis 92,8 %; n = 600 und 2000: 99,8 bis 100,0 %). Die Unterschiede zwischen den Arten sind so groß (Zellmittel 0,9 bis 2,2), dass die Codierung beim Art-Test kaum sichtbar wird; die Falle zeigt sich nur dort, wo der Effekt je Stufe verschieden und klein genug ist, um am Niveau zu kippen. |
| B11 | Einzelkoeffizient „Leer × Wochenende“ (t-Test), Generator-Szenario | n = 600: bei Referenz Import 5,3 %, bei Referenz Export 25,5 %; n = 2000: 5,4 % gegen 67,3 %. Derselbe Datensatz, dieselbe Anpassung, andere Antwort auf „ist die Wechselwirkung Leer × Wochenende signifikant?“ (wahre Differenz der Effekte 0 gegen 0,30). |
| B12 | Dummy-Falle (n = 200, Zufalls-Seed 7003, Generator-Szenario, gleiche Streuung) | Rang 4 von 5, kleinster singulärer Wert unter 1e-10, Nullvektor (1, −1, −1, −1, −1). `lstsq` und `pinv` geben den Achsenabschnitt 1,22, QR mit Spaltenwahl 0,92; die angepassten Werte sind in allen Wegen gleich (Abweichung unter 1e-9) und gleich den Zellmitteln der Art. `solve(X'X)` meldete auf der Messplattform (Windows) keinen Fehler und lieferte eine Lösung auf der Lösungsgeraden (Achsenabschnitt 1,43); diese Zahl hängt am Rundungsfehler der Plattform und ist nicht Gegenstand eines Tests (die Tests prüfen „liegt auf der Lösungsgeraden“). |
| B13 | Referenzkunde (Import-Container, n = 300, zwölf Kunden, 10 000 Wiederholungen) | Mit den echten Kundeneffekten (−0,59 bis +0,70) sind je nach Referenz im Mittel 4,7 (Referenz K3, Effekt −0,08) bis 9,7 (Referenz K10, Effekt +0,70) der 11 Koeffizienten „signifikant“ (\|t\| > t(0,975)); der F-Test des ganzen Faktors verwirft in 100,0 %, unabhängig von der Referenz. Ohne Kundeneffekt verwirft er in 5,2 %, **mindestens ein Einzelkoeffizient** ist dann in 30,4 bis 31,7 % der Wiederholungen „signifikant“ (Mittel 30,8 %; bei 11 unabhängigen Tests wären es 43,1 %, die Tests teilen aber die Referenz). |
| B14 | Typ I gegen Typ III für Art | Typ I ist der Wald-Test der nach Zellgrößen gewichteten Zeilenmittel, Typ III (Summencodierung) der der ungewichteten (Abweichung unter 1e-9); in 100 von 100 Zufallsdatensätzen (Zellgrößen 3 bis 11) sind beide F-Werte verschieden, bei gleichen Zellgrößen gleich. |
| B15 | Leere Zelle | Bei n = 200 fehlt in 24 bis 41 von 10 000 Wiederholungen eine der acht Zellen (Streuung gleich: 31, 41, 24 in den Szenarien Generator, additiv, Überkreuz; Streuung des Generators: 31, 33, 35); bei n = 600 und 2000 in keiner. Die Wechselwirkung ist dann nicht schätzbar, die App meldet „Zelle leer“. |
| B16 | Invarianten (je 40 Zufallsinstanzen, auch mit Zellen mit einer Beobachtung und sehr ungleicher Besetzung) | Dieselbe Anpassung in allen Codierungen (Abweichung unter 1e-9); Test der Wechselwirkung gleich in allen vier Codierungen; Summe und Helmert geben denselben F-Wert für Wochenende und Art; im additiven Modell ist F des Haupteffekts in allen acht Einstellungen gleich; Umordnen der Stufen kehrt das Vorzeichen des Summenkoeffizienten um; lineare Umrechnung des Ausgangs lässt alle F-Werte unverändert. |
| B17 | Stabilität der Messreihen (Messung dieser README, nicht im Test) | Mit anderen Zufalls-Seeds (`python tools/fcd_seed_check.py`, Basis-Seed 990000, Kunden-Seed 990001) weichen in diesen Läufen alle Häufigkeiten der 18 Zeilen um höchstens 1,5 Prozentpunkte von der Datei ab; bei der Referenzkunde die mittlere Zahl signifikanter Koeffizienten um höchstens 0,03, die Häufigkeit „mindestens ein Einzelkoeffizient“ um höchstens 2,3 Prozentpunkte. Das sind beobachtete Spannen in wenigen Läufen, keine Schranken (bei 18 Zeilen und mehreren Vergleichen ist 1,5 Punkte knapp bemessen). |

**Wie man diese Zahlen liest.** Die Häufigkeit „verwirft“ ist keine Wahrscheinlichkeit für „es gibt einen Effekt“: bei Referenz Export im Generator-Szenario testet der Haupteffekt-Test einen Effekt, der 0 ist, und verwirft zu Recht nur in 5 % der Wiederholungen; bei Referenz Import testet er +0,30 und verwirft in 80 %. Beide Tests sind **in Ordnung**, sie beantworten verschiedene Fragen. Der Satz „es gibt keinen Wochenendeffekt“ folgt aus keinem von ihnen allein.

### Zahl → Test

| Nr. | Test (Band oder exakt) |
|---|---|
| B1 | `tests/test_oracle_faktoren_codierung.py::test_wochenende_koeffizient_ist_in_40_datensaetzen_exakt_der_effekt_an_der_stelle_der_codierung` (exakt); `test_eigenschaften_faktoren_codierung.py` (Float) |
| B2, B3, B4 | `tests/test_handbeispiele.py` (Brüche; die App-Routine gegen dieselben Brüche mit relativer Toleranz 1e-9) und `tests/test_app.py` (Verdikt-Zeilen je Preset) |
| B5 | `tests/test_generator.py`, `tests/test_claims.py::test_b5_*` |
| B6 bis B8, B10, B11, B15 | `tests/test_claims.py::test_b6_*` bis `test_b15_*` (die Datei `data/fcd_frozen.json` gegen die README-Zahlen, auf eine Stelle gerundet; `tests/test_claims.py::test_vorgerechnete_messreihe_stimmt_mit_der_neu_gerechneten_ueberein` rechnet Zeilen neu, Zählungen ± 2, mit `FCD_FULL=1` alle) |
| B9 | `tests/test_claims.py::test_b9_*`, `tests/test_oracle_faktoren_codierung.py::test_teststaerke_der_vorgerechneten_zeilen_gegen_die_exakte_formel` |
| B12 | `tests/test_claims.py::test_b12_*` (Rang, Achsenabschnitte 1,22 und 0,92 als Band), `tests/test_oracle_faktoren_codierung.py::test_lstsq_pinv_qr_und_solve_liegen_auf_der_loesungsgeraden` |
| B13 | `tests/test_claims.py::test_b13_*` |
| B14 | `tests/test_oracle_faktoren_codierung.py::test_typ_i_und_typ_iii_gegen_wald_der_zeilenmittel` |
| B16 | `tests/test_eigenschaften_faktoren_codierung.py` |
| B17 | Messung dieser README, nicht im Test (`tools/fcd_seed_check.py`) |

## Ehrliche Grenzen

- **Nur dieses Modell:** zwei Faktoren (Art, Wochenende), Ausgang log-Verweilzeit, gleiche Streuung der Zellen als Standard. Die Aussagen über die Codierung gelten bei diesen Parametern, bei n = 200 bis 2000 und für die vier gezeigten Codierungen; Häufigkeiten in anderen Modellen werden nicht abgeleitet.
- **Die Häufigkeiten gelten für die gemessene Streuung.** Mit gleicher Streuung halten Summe und Referenz Export bei wahrem Effekt 0 das Niveau (Summe im Überkreuz-Szenario 5,4 % bei n = 600), mit der Streuung des Generators nicht (B8). Warum, ist nicht untersucht; vermutlich liegt es am Streuungsverhältnis 1,73 der Zellen (nicht gemessen).
- **Der F-Test benutzt eine gemeinsame Restvarianz.** Die Streuung zu korrigieren (robuste Standardfehler, gewichtete Kleinste Quadrate) zeigt die Demo nicht, siehe [`streuung-wls-demo`](https://github.com/sebastian-hanisch/streuung-wls-demo).
- **Die exakte Teststärke ist eine Näherung** (erwartete statt zufälliger Zellgrößen); sie liegt in der Simulation bis 2,0 Prozentpunkte über der Häufigkeit (B9).
- **`gewichtet zentriert` ist eine eigene Bezeichnung.** Es wird nicht behauptet, sie sei gleich einem Quadratsummen-Typ oder einer Codierung eines Programms.
- **Handbeispiel und Container-Strom sind verschiedene Vehikel:** das Handbeispiel hat ganze Zahlen, damit man von Hand rechnen kann; der Generator liefert keine ganzen Zahlen.
- **Die Lösung von `solve(X'X)` bei Rangmangel ist plattformabhängig:** die App zeigt eine Fehlermeldung, wenn die Plattform eine meldet.
- **Ursachen:** Aussagen mit „weil“ in der App (zum Beispiel zur Varianz der Codierungen im additiven Szenario) folgen aus den Formeln des Wald-Tests, nicht aus einer eigenen Messung der Ursache; die Lücke zwischen Simulation und exakter Formel erklärt die Mittelung über zufällige Zellgrößen nur teilweise (B9).

## Befunde und Korrekturen gegenüber dem Plan

- **Wiederholungen:** 10 000 statt 2 500 je Zeile (die gestapelte Rechnung mit Wald-Formeln braucht für alle Zeilen etwa 70 Sekunden); die Häufigkeiten sind daher mit ±1,0 statt ±2 Prozentpunkten Monte-Carlo-Fehler angegeben.
- **Gegenprobe:** Die frühere Messung fand 66 von 72 Zeilen innerhalb zweier Standardfehler der exakten Teststärke und nannte das unauffällig. Mit 10 000 Wiederholungen sind es 59 von 72 (62 innerhalb dreier), größte Abweichung 2,0 Prozentpunkte: Die Formel liegt bei Summe und gewichtet etwas über der Simulation. Die Abweichung ist teilweise erklärt (B9); der Rest liegt innerhalb zweier Standardfehler.
- **σ_pool exakt:** 0,689 aus geschlossenen Formeln (Varianz der Art plus Varianz des Kundeneffekts plus Varianz aus Zoll und Vorlauf) statt 0,688 aus einer Monte-Carlo-Stichprobe; die exakten Teststärken weichen deshalb um höchstens 0,1 Prozentpunkte von den früheren ab (zum Beispiel Referenz Import bei n = 600: 81,2 statt 81,3 %).
- **Kernzahlen neu gemessen** mit dem Strom dieser Demo (SplitMix64 statt numpy): bei n = 600, Generator-Szenario, gleiche Streuung verwirft Referenz Import in 79,9 % (früher 81,0), Referenz Export in 5,1 % (5,2), Summe in 55,2 % (56,2). Die Aussagen bleiben, die Zahlen verschieben sich um bis 1,1 Prozentpunkte, innerhalb des Monte-Carlo-Fehlers beider Messungen.
- **Experiment 5 (Referenzkunde) ist enthalten** (im Entwurf war es als letztes Experiment vorgesehen und streichbar); Typ I und Typ III für Art stehen als Zeilen in Experiment 2.
- **Zusätzlich gegenüber dem Entwurf:** ein achtes Preset „Überkreuz, Streuung des Generators“ (der Negativbefund B8 hat ein eigenes Szenario), ein Modell-Schalter „additiv“ im Handbeispiel und die Gewichte der Zellmittel je Koeffizient (Zeile der Inversen der Zellmatrix).
- **Sätze zu den Koeffizienten sind geprüft:** die Sätze („Mittel Import / Wochentag“, „−½ × ungewichtetes Mittel“, …) werden gegen die Koeffizienten von `lstsq` gehalten (alle Codierungen, alle Referenzstufen, 30 Zufallsgrößen der Zellen); der Entwurf nannte sie nur.
- **Port:** 9965 (der Entwurf nannte 9964, der später gesetzte Eintrag in `.claude/launch.json` gilt).

## Verifikation

- **Handwerte** (`tests/test_handbeispiele.py`): H1 bis H5 in Brüchen, mit der Rechnung im Test (`48 / 2`), und dieselben Zahlen aus der App-Routine.
- **Orakel** (`tests/test_oracle_faktoren_codierung.py`): Normalgleichung in Brüchen gegen den Löser der App; Wald-Test gegen Modellvergleich (Bruch gleich Bruch und auf Zufallsdaten); Typ I und Typ III gegen Wald der gewichteten und ungewichteten Zeilenmittel; nichtzentrale F gegen eine eigene Simulation mit festen Zellgrößen und gegen die vorgerechneten Zeilen; `lstsq`, `pinv`, QR mit Spaltenwahl und `solve` gegen die Lösungsgerade; Koeffizient gleich Inverse der Zellmatrix mal Zellmittel; die Sätze zu den Koeffizienten; die gestapelte Messreihe gegen Wiederholung für Wiederholung mit der Entwurfsmatrix; der Kunden-F-Test gegen `scipy.stats.f_oneway`; der Bruch-Löser selbst gegen eine Cholesky-Zerlegung.
- **Eigenschaften** (`tests/test_eigenschaften_faktoren_codierung.py`), **Zufall** (`tests/test_sample.py`: vektorisierte gegen skalare SplitMix64-Folge, kein `numpy.random` und keine C-Bibliotheksfunktion auf dem Zufallsweg), **Generator** (`tests/test_generator.py`), **Randfälle** (`tests/test_edge_cases.py`: leere Zelle, Zelle mit einer Beobachtung, Rest ohne Freiheitsgrade, zwei Stufen, Nichtzentralität 0), **Verdikte** (`tests/test_verdicts.py`).
- **Claims** (`tests/test_claims.py`): jede Zahl der Befunde B5 bis B15 aus `data/fcd_frozen.json` und aus Neu-Rechnung. „Neu gerechnet, exakt verglichen“ gibt es nicht: Zählungen aus Neu-Rechnung als Band (± 2), Anteile ± 0,01, Gleitkomma relativ 1e-9, exakt nur Brüche.
- **Oberfläche:** AppTest-Rauchtest (jedes Preset, jeder Regler durch alle Werte, additives Modell in jeder Codierung), je Preset die erwartete Verdikt-Zeile, Footer-Test, Laufzeit-Importe.
- **Layout:** Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben; jede Änderung eines Reglers (auch der Regler im Hauptbereich) hebt die Markierung auf (`tests/test_layout.py`).
- **Plattform:** Linux-Prüfung mit `tools/demo_linux_check.py` (Docker) vor dem ersten Push.

Gesamtzahl: 173 Tests.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Handbeispiel (Experiment 1), Kernfragen, Experimente 2 bis 5, Grenzen, Formeln |
| `data/fcd_frozen.json` | vorgerechnete Messreihen (Experimente 3 und 5) |
| `fcd_constants.py` | Generator-Parameter, Handbeispiel, Codierungen, Regler, Presets |
| `fcd_design.py` | Codierungen, Entwurfsmatrix, Kleinste Quadrate, F-Test der Streichung von Spalten, Gewichte der Zellmittel |
| `fcd_evaluation.py` | Auswertungen der fünf Experimente, Sätze zu den Koeffizienten, Verdikte |
| `fcd_generator.py` | Container-Generator, exakte Zellmittel und Streuungen |
| `fcd_layout.py` | Fester Kopf (Stylesheet) |
| `fcd_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `fcd_rng.py` | skalare SplitMix64-Folge (Kopie der geprüften Vorlage des Website-Repos), nur Referenz in den Tests |
| `fcd_sample.py` | vektorisierte SplitMix64-Folge, Logarithmus und Kosinus aus Grundrechenarten |
| `fcd_simulation.py` | gestapelte Messreihen über Wiederholungen (Wald-Formeln), exakte Teststärke |
| `fcd_tests.py` | Zellstatistik, Wald-Tests in Zellmittel-Parametern, nichtzentrale F |
| `fcd_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tools/fcd_make_frozen.py` | erzeugt `data/fcd_frozen.json` (`--check` rechnet neu und vergleicht) |
| `tools/fcd_seed_check.py` | Stabilitätsprobe mit anderen Zufalls-Seeds (B17) |
| `tests/` | Handwerte, Orakel, Eigenschaften, Zufall, Generator, Randfälle, Verdikte, Claims, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Typ II** der Quadratsummen, **drei und mehr Faktoren**, geordnete Faktoren (Polynomkontraste), selbst gewählte Kontraste und **Mehrfachvergleiche** (Tukey, Bonferroni): nicht nachgerechnet und nicht gezeigt.
- **Vortest-Strategie** („erst die Wechselwirkung testen, dann das Modell wählen“): sie ist eine Modellwahl mit Folgen für die Tests danach, siehe [`modellwahl-demo`](https://github.com/sebastian-hanisch/modellwahl-demo); hier nicht gemessen.
- **Ridge und die Wahl von λ** bei fast abhängigen Spalten: nur als Verweis auf [`tikhonov-demo`](https://github.com/sebastian-hanisch/tikhonov-demo) und [`parameterwahl-demo`](https://github.com/sebastian-hanisch/parameterwahl-demo); die fast abhängigen Spalten selbst behandelt [`kollinearitaet-auswahl-demo`](https://github.com/sebastian-hanisch/kollinearitaet-auswahl-demo).
- **Kunde als Zufallseffekt** (Schrumpfung statt zwölf fester Koeffizienten): [`cluster-gemischt-demo`](https://github.com/sebastian-hanisch/cluster-gemischt-demo); hier nur als fester Faktor mit Referenzkunde.
- **Rohskala (Tage)** des Ausgangs: nicht gemessen; gerechnet wird auf der log-Verweilzeit.
- **Zentrieren stetiger Merkmale** als Gegenstück der Summencodierung (dieselbe Algebra): nicht gemessen; Wechselwirkung als Regressionsterm bei der Wirkungsschätzung steht in [`schaetzen-demo`](https://github.com/sebastian-hanisch/schaetzen-demo) und [`wen-behandeln-demo`](https://github.com/sebastian-hanisch/wen-behandeln-demo).
- **PDF-Export** und eine Live-Simulation über viele Wiederholungen: die Messreihen sind vorgerechnet (Rechenzeit der gestapelten Messreihen etwa 70 Sekunden), in der App läuft nur eine Stichprobe live.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die vorgerechneten Messreihen neu erzeugen oder prüfen: `python tools/fcd_make_frozen.py` bzw. `python tools/fcd_make_frozen.py --check`.

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
