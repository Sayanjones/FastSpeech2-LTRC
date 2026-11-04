"""
Telugu Number, Date, and Time Expansion Module.

Converts non-standard words (digits, years, ordinals, clock times)
into spoken Telugu orthographic words.
"""

import re
from typing import Match

DIGITS = {
    0: "సున్నా",
    1: "ఒకటి",
    2: "రెండు",
    3: "మూడు",
    4: "నాలుగు",
    5: "ఐదు",
    6: "ఆరు",
    7: "ఏడు",
    8: "ఎనిమిది",
    9: "తొమ్మిది",
}

TEENS = {
    10: "పది",
    11: "పదకొండు",
    12: "పన్నెండు",
    13: "పదమూడు",
    14: "పద్నాలుగు",
    15: "పదిహేను",
    16: "పదహారు",
    17: "పదిహేడు",
    18: "పద్దెనిమిది",
    19: "పంతొమ్మిది",
}

TENS = {
    20: "ఇరవై",
    30: "ముప్పై",
    40: "నలభై",
    50: "యాభై",
    60: "అరవై",
    70: "డెబ్బై",
    80: "ఎనభై",
    90: "తొంభై",
}

ORDINALS = {
    1: "ఒకటవ",
    2: "రెండవ",
    3: "మూడవ",
    4: "నాల్గవ",
    5: "ఐదవ",
    6: "ఆరవ",
    7: "ఏడవ",
    8: "ఎనిమిదవ",
    9: "తొమ్మిదవ",
    10: "పదవ",
    11: "పదకొండవ",
    12: "పన్నెండవ",
    13: "పదమూడవ",
    14: "పద్నాలుగవ",
    15: "పదిహేనవ",
    16: "పదహారవ",
    17: "పదిహేడవ",
    18: "పద్దెనిమిదవ",
    19: "పంతొమ్మిదవ",
    20: "ఇరవయ్యవ",
    30: "ముప్పయ్యవ",
    31: "ముప్పై ఒకటవ",
}


def num_to_telugu(n: int) -> str:
    """Recursively converts an integer (0-99999) into spoken Telugu words."""
    if n < 0:
        return "మైనస్ " + num_to_telugu(abs(n))
    if n < 10:
        return DIGITS[n]
    if n < 20:
        return TEENS[n]
    if n < 100:
        ten, rem = divmod(n, 10)
        base = TENS[ten * 10]
        return f"{base} {DIGITS[rem]}" if rem else base
    if n < 1000:
        hundred, rem = divmod(n, 100)
        h_word = "వంద" if hundred == 1 else f"{DIGITS[hundred]} వందలు"
        return f"{h_word} {num_to_telugu(rem)}" if rem else h_word
    if n < 100000:
        thousand, rem = divmod(n, 1000)
        t_word = "వెయ్యి" if thousand == 1 else f"{num_to_telugu(thousand)} వేల"
        if rem == 0:
            return "రెండు వేలు" if thousand == 2 else f"{num_to_telugu(thousand)} వేలు"
        return f"{t_word} {num_to_telugu(rem)}"
    return str(n)


def ordinal_to_telugu(n: int) -> str:
    """Converts a number with ordinal suffix (e.g. 12 -> పన్నెండవ)."""
    if n in ORDINALS:
        return ORDINALS[n]
    cardinal = num_to_telugu(n)
    # Generic Telugu ordinal suffix is 'వ'
    return f"{cardinal}వ"


# Regex patterns
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_ORDINAL_RE = re.compile(r"\b(\d+)\s*వ\b")
_NUMBER_RE = re.compile(r"\b\d+\b")


def _expand_time(match: Match) -> str:
    hours = int(match.group(1))
    minutes = int(match.group(2))
    h_str = num_to_telugu(hours)
    m_str = num_to_telugu(minutes) if minutes > 0 else ""
    if m_str:
        return f"{h_str} {m_str}"
    return h_str


def _expand_ordinal(match: Match) -> str:
    num = int(match.group(1))
    return ordinal_to_telugu(num)


def _expand_number(match: Match) -> str:
    num = int(match.group(0))
    return num_to_telugu(num)


def normalize_telugu_numbers(text: str) -> str:
    """
    Expands all numbers, times, and ordinals in Telugu text.
    Handles T4 examples:
      '2026' -> 'రెండు వేల ఇరవై ఆరు'
      '12వ' -> 'పన్నెండవ'
      '10:45' -> 'పది నలభై ఐదు'
    """
    # 1. Expand clock times (e.g. 10:45)
    text = _TIME_RE.sub(_expand_time, text)
    # 2. Expand ordinals (e.g. 12వ)
    text = _ORDINAL_RE.sub(_expand_ordinal, text)
    # 3. Expand remaining cardinals (e.g. 2026)
    text = _NUMBER_RE.sub(_expand_number, text)
    return text

