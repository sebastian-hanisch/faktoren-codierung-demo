"""Erzeugt data/fcd_frozen.json: die vorgerechneten Messreihen der App (Experimente 3 und 5).

Aufruf (aus dem Repo-Ordner):
  python tools/fcd_make_frozen.py            schreibt die Datei neu (18 Zeilen zu je 10 000 Wiederholungen und die Messreihe zur Referenzkunde; etwa 1 bis 2 Minuten)
  python tools/fcd_make_frozen.py --check    rechnet alles neu und vergleicht mit der Datei (Zählungen ± 2, sonst Abbruch mit Fehlercode 1); schreibt nichts

Der Zufall kommt aus SplitMix64 (`fcd_sample`); die Rechnung läuft mit Grundrechenarten und `scipy.stats` für die kritischen Werte, Zählungen sind daher auf jeder Plattform dieselben
bis auf Wiederholungen, deren Teststatistik auf der Grenze liegt (Band ± 2 in `tests/test_claims.py`)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fcd_constants as C  # noqa: E402
import fcd_simulation as SIM  # noqa: E402

PATH = ROOT / "data" / C.FROZEN_FILE
TOL = 2


def compute(rows=None) -> dict:
    grid = SIM.GRID if rows is None else rows
    return {
        "_hinweis": "erzeugt von tools/fcd_make_frozen.py (SplitMix64, fcd_simulation.py); je Zeile ein eigener Strom mit eigenem Zufalls-Seed, Wiederholungen r nehmen Container r·n bis (r+1)·n−1",
        "ziel": {"niveau": C.ALPHA, "settings": list(SIM.SETTING_NAMES)},
        "zeilen": [SIM.simulate_row(mode, szen, n, C.SIM_REPS) for mode, szen, n in grid],
        "kunde": SIM.simulate_kunde(C.KUNDE_REPS),
    }


def _flat(row):
    out = {}
    for key, val in row.items():
        if isinstance(val, list):
            for i, x in enumerate(val):
                out[f"{key}[{i}]"] = x
        elif isinstance(val, (int, float)):
            out[key] = val
    return out


def differences(old: dict, new: dict) -> list[str]:
    """Abweichungen jenseits der Toleranz (Zählungen ± TOL, Gleitkommazahlen relativ 1e-9)."""
    bad = []
    old_rows = {(r["mode"], r["szenario"], r["n"]): r for r in old["zeilen"]}
    for r in new["zeilen"]:
        key = (r["mode"], r["szenario"], r["n"])
        o = old_rows.get(key)
        if o is None:
            bad.append(f"Zeile {key} fehlt in der Datei")
            continue
        fo, fn = _flat(o), _flat(r)
        for k, v in fn.items():
            if k in ("seed", "reps", "n"):
                if fo.get(k) != v:
                    bad.append(f"{key} {k}: Datei {fo.get(k)} gegen neu {v}")
            elif abs(fo.get(k, 1e99) - v) > TOL:
                bad.append(f"{key} {k}: Datei {fo.get(k)} gegen neu {v}")
    fo, fn = _flat(old["kunde"]), _flat(new["kunde"])
    for k, v in fn.items():
        tol = 1e-12 if k.startswith("kunde_effekte") else TOL
        if k in ("n", "reps", "seed", "valid"):
            tol = 0
        if abs(fo.get(k, 1e99) - v) > tol:
            bad.append(f"kunde {k}: Datei {fo.get(k)} gegen neu {v}")
    return bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        new = compute()
        old = json.loads(PATH.read_text(encoding="utf-8"))
        bad = differences(old, new)
        for line in bad:
            print(line)
        print("Messreihe stimmt mit der Datei überein" if not bad else f"{len(bad)} Abweichungen")
        sys.exit(1 if bad else 0)
    out = compute()
    PATH.parent.mkdir(exist_ok=True)
    PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{PATH}: {len(out['zeilen'])} Zeilen, Messreihe zur Referenzkunde mit {out['kunde']['reps']} Wiederholungen")
