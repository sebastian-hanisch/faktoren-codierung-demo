"""Stabilitätsprobe der vorgerechneten Messreihen: dieselben 18 Zeilen und die Messreihe zur Referenzkunde mit anderen Zufalls-Seeds noch einmal rechnen und gegen `data/fcd_frozen.json` halten.

Aufruf (aus dem Repo-Ordner):  python tools/fcd_seed_check.py [Basis-Seed der Zeilen] [Zufalls-Seed der Referenzkunde]      (Standard 990000 und 990001; etwa 1 Minute)
Ausgabe: die größte Abweichung einer Häufigkeit in Prozentpunkten (alle Tests und Zeilen) und der Kundenzahlen. Schreibt nichts."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fcd_constants as C  # noqa: E402

C.SIM_BASE_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 990000
C.KUNDE_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 990001

import fcd_simulation as SIM  # noqa: E402

if __name__ == "__main__":
    old = json.loads((ROOT / "data" / C.FROZEN_FILE).read_text(encoding="utf-8"))
    worst, where = 0.0, None
    for r in old["zeilen"]:
        new = SIM.simulate_row(r["mode"], r["szenario"], r["n"], C.SIM_REPS)
        for key in ("rej_B", "rej_A", "rej_I", "sig_leer_import", "sig_leer_export"):
            a = r[key] if isinstance(r[key], list) else [r[key]]
            b = new[key] if isinstance(new[key], list) else [new[key]]
            for k, (x, y) in enumerate(zip(a, b)):
                d = abs(x / r["valid"] - y / new["valid"])
                if d > worst:
                    worst, where = d, (r["mode"], r["szenario"], r["n"], key, k)
    print(f"Zeilen: größte Abweichung einer Häufigkeit {100 * worst:.2f} Prozentpunkte bei {where}")
    k1, k2 = old["kunde"], SIM.simulate_kunde(C.KUNDE_REPS)
    print("Referenzkunde: größte Abweichung der mittleren Zahl signifikanter Koeffizienten",
          f"{max(abs(a - b) for a, b in zip(k1['sig_sum_by_ref'], k2['sig_sum_by_ref'])) / C.KUNDE_REPS:.3f}",
          "; mindestens ein Einzelkoeffizient ohne Kundeneffekt (Prozentpunkte)", f"{100 * max(abs(a - b) for a, b in zip(k1['any_h0_by_ref'], k2['any_h0_by_ref'])) / C.KUNDE_REPS:.2f}")
