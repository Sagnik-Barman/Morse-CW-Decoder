"""Morse code table and the CTC output vocabulary.

Index 0 of the vocabulary is the CTC blank. Prosigns are mapped to single
characters so every label is one token:
    '+' = AR (.-.-.)   end of message
    '=' = BT (-...-)   break / new paragraph
"""

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "/": "-..-.",
    "=": "-...-", "+": ".-.-.", "-": "-....-",
}

REVERSE = {v: k for k, v in MORSE.items()}

BLANK = "_"  # CTC blank symbol (never appears in text)
VOCAB = [BLANK, " "] + sorted(MORSE.keys(), key=lambda c: (not c.isalpha(), not c.isdigit(), c))
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR = {i: c for i, c in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB)


def normalize_text(text: str) -> str:
    """Upper-case, drop unsupported characters and collapse whitespace."""
    text = text.upper()
    text = "".join(c if (c in MORSE or c == " ") else " " for c in text)
    return " ".join(text.split())


def encode(text: str) -> list[int]:
    return [CHAR2IDX[c] for c in normalize_text(text)]


def decode_ids(ids) -> str:
    return "".join(IDX2CHAR[int(i)] for i in ids if int(i) != 0)


def text_to_morse(text: str) -> str:
    """Human-readable Morse: letters separated by ' ', words by ' / '."""
    words = normalize_text(text).split(" ")
    return " / ".join(" ".join(MORSE[c] for c in w) for w in words)
