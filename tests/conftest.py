"""Gemeinsame Fixtures: die Handbeispiele, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b), und der Importpfad."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fcd_constants as C  # noqa: E402

# H1: je Zelle zwei Beobachtungen m − 1 und m + 1, Zellmittel Wochentag 4 / 4 / 4, Wochenende 4 / 8 / 12 (Import / Export / Umschlag)
H1_VALUES = {(0, 0): [3, 5], (0, 1): [3, 5], (1, 0): [3, 5], (1, 1): [7, 9], (2, 0): [3, 5], (2, 1): [11, 13]}
# H2: dieselben Zellmittel, Zellgrößen Import 2 / 4, Export 3 / 1, Umschlag 2 / 2
H2_VALUES = {(0, 0): [3, 5], (0, 1): [3, 5, 3, 5], (1, 0): [3, 4, 5], (1, 1): [8], (2, 0): [3, 5], (2, 1): [11, 13]}


def flatten(values):
    a, b, y = [], [], []
    for (i, j), vs in values.items():
        for v in vs:
            a.append(i), b.append(j), y.append(v)
    return a, b, y


@pytest.fixture
def h1():
    return flatten(H1_VALUES)


@pytest.fixture
def h2():
    return flatten(H2_VALUES)
