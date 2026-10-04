"""Random text for training labels.

Real CW traffic is highly structured (callsigns, RST reports, Q-codes,
contest exchanges), but a model trained only on that would learn a language
prior and fail on random characters. So we mix four sources:

    qso      - ham-radio conversation templates
    contest  - short contest exchanges
    words    - random English / ham abbreviations
    random   - uniformly random characters (forces acoustic decoding)
"""
from __future__ import annotations

import numpy as np

from .alphabet import MORSE

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
ALL_CHARS = [c for c in MORSE.keys()]

PREFIXES = [
    "VU2", "VU3", "W", "K", "N", "AA", "KA", "WA", "VE", "VA", "G", "M", "2E",
    "DL", "DK", "DJ", "F", "EA", "I", "IK", "JA", "JH", "JR", "HL", "BY", "BV",
    "VK", "ZL", "PY", "LU", "UA", "RA", "SP", "OK", "OM", "HA", "YO", "LZ",
    "SM", "OH", "LA", "OZ", "PA", "ON", "HB9", "OE", "EI", "4X", "A7", "9V",
    "S5", "9A", "YB", "DU", "HS", "ZS", "5B", "CT", "EA8", "KH6", "KL7",
]
NAMES = ["SAM", "RAJ", "JOHN", "ANU", "BOB", "HANS", "YUKI", "LUCA", "OM",
         "PETE", "RAVI", "MIKE", "TOM", "ALEX", "KEN", "DAVE", "IVAN", "JOSE"]
CITIES = ["CHENNAI", "DELHI", "MUMBAI", "PUNE", "TOKYO", "BERLIN", "PARIS",
          "LONDON", "BOSTON", "DALLAS", "ROME", "OSLO", "SYDNEY", "MOSCOW",
          "MADRAS", "BANGALORE", "KOLKATA", "MADRID", "VIENNA", "PRAGUE"]
RIGS = ["IC7300", "FT991", "K3", "TS590", "FT817", "KX3", "IC705", "FTDX10"]
ANTS = ["DIPOLE", "YAGI", "VERTICAL", "EFHW", "LOOP", "GP", "INV V", "BEAM"]
WORDS = (
    "THE OF AND TO IN IS YOU THAT IT HE WAS FOR ON ARE AS WITH HIS THEY AT BE "
    "THIS HAVE FROM OR ONE HAD BY WORD BUT NOT WHAT ALL WERE WE WHEN YOUR CAN "
    "SAID THERE USE AN EACH WHICH SHE DO HOW THEIR IF WILL UP OTHER ABOUT OUT "
    "MANY THEN THEM THESE SO SOME HER WOULD MAKE LIKE HIM INTO TIME HAS LOOK "
    "TWO MORE WRITE GO SEE NUMBER NO WAY COULD PEOPLE MY THAN FIRST WATER BEEN "
    "CALL WHO OIL ITS NOW FIND LONG DOWN DAY DID GET COME MADE MAY PART OVER "
    "NEW SOUND TAKE ONLY LITTLE WORK KNOW PLACE YEAR LIVE ME BACK GIVE MOST "
    "VERY AFTER THING OUR JUST NAME GOOD SENTENCE MAN THINK SAY GREAT WHERE "
    "HELP THROUGH MUCH BEFORE LINE RIGHT TOO MEAN OLD ANY SAME TELL BOY FOLLOW "
    "CAME WANT SHOW ALSO AROUND FORM THREE SMALL SET PUT END DOES ANOTHER WELL "
    "LARGE MUST BIG EVEN SUCH BECAUSE TURN HERE WHY ASK WENT MEN READ NEED LAND "
    "RADIO ANTENNA SIGNAL POWER BAND WEATHER SUNNY RAIN COLD WARM TEMP "
    "CQ DE K KN BK R TU TNX TKS FB OM YL XYL HR UR RST QTH QSL QRZ QRM QRN QSB "
    "QRS QRQ QRP QRO WX ES FER GE GM GA GN HW CPY PSE AGN SRI BCNU CUL GL 73 88 "
    "RIG ANT PWR W ABT NR 5NN 599 579 559 449 TEST ENN TTT"
).split()


def callsign(rng: np.random.Generator) -> str:
    pre = PREFIXES[rng.integers(len(PREFIXES))]
    if not pre[-1].isdigit():
        pre += DIGITS[rng.integers(10)]
    suffix = "".join(LETTERS[i] for i in rng.integers(0, 26, size=rng.integers(1, 4)))
    call = pre + suffix
    if rng.random() < 0.05:
        call += "/P" if rng.random() < 0.5 else "/" + DIGITS[rng.integers(10)]
    return call


def rst(rng: np.random.Generator) -> str:
    if rng.random() < 0.4:
        return "5NN"
    return f"5{rng.integers(1, 10)}9"


def qso_text(rng: np.random.Generator) -> str:
    a, b = callsign(rng), callsign(rng)
    templates = [
        lambda: f"CQ CQ CQ DE {a} {a} K",
        lambda: f"CQ DE {a} {a} {a} PSE K",
        lambda: f"{b} DE {a} {a} K",
        lambda: f"{b} DE {a} GM OM TNX FER CALL = UR RST {rst(rng)} {rst(rng)} = NAME {NAMES[rng.integers(len(NAMES))]} = QTH {CITIES[rng.integers(len(CITIES))]} = HW CPY? {b} DE {a} KN",
        lambda: f"R R TNX FB REPORT = RIG {RIGS[rng.integers(len(RIGS))]} PWR {rng.integers(5, 1001)}W ANT {ANTS[rng.integers(len(ANTS))]} = WX {['SUNNY', 'RAIN', 'CLOUDY', 'HOT', 'COLD'][rng.integers(5)]} TEMP {rng.integers(-10, 45)}C",
        lambda: f"TNX FER QSO {NAMES[rng.integers(len(NAMES))]} 73 ES GL + {b} DE {a} TU",
        lambda: f"QRZ? DE {a} K",
        lambda: f"{a} DE {b} R R 73",
        lambda: f"VVV VVV DE {a} {a} TEST",
    ]
    return templates[rng.integers(len(templates))]()


def contest_text(rng: np.random.Generator) -> str:
    a, b = callsign(rng), callsign(rng)
    nr = rng.integers(1, 2000)
    nr_s = str(nr).replace("0", "T") if rng.random() < 0.3 else str(nr)
    templates = [
        f"TEST {a} {a}",
        f"{b} 5NN {nr_s}",
        f"TU {a}",
        f"{b} 5NN {rng.integers(1, 41)}",
        f"{a} 5NN {['VU', 'W', 'DL', 'JA', 'G'][rng.integers(5)]}",
        f"AGN? {nr_s} {nr_s}",
    ]
    return templates[rng.integers(len(templates))]


def word_text(rng: np.random.Generator, n_words: int | None = None) -> str:
    n = n_words or int(rng.integers(1, 9))
    return " ".join(WORDS[i] for i in rng.integers(0, len(WORDS), size=n))


def random_text(rng: np.random.Generator) -> str:
    n_groups = int(rng.integers(1, 7))
    groups = []
    for _ in range(n_groups):
        k = int(rng.integers(1, 7))
        groups.append("".join(ALL_CHARS[i] for i in rng.integers(0, len(ALL_CHARS), size=k)))
    return " ".join(groups)


DEFAULT_MIX = {"qso": 0.3, "contest": 0.15, "words": 0.25, "random": 0.3}


def sample_text(rng: np.random.Generator, mix: dict | None = None) -> str:
    mix = mix or DEFAULT_MIX
    kinds = list(mix.keys())
    p = np.array([mix[k] for k in kinds], dtype=float)
    kind = kinds[rng.choice(len(kinds), p=p / p.sum())]
    if kind == "qso":
        return qso_text(rng)
    if kind == "contest":
        return contest_text(rng)
    if kind == "callsign":
        return " ".join(callsign(rng) for _ in range(int(rng.integers(1, 4))))
    if kind == "words":
        return word_text(rng)
    return random_text(rng)
