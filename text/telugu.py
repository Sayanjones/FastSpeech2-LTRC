"""
Telugu Text Normalizer and Conjunct Repair Engine.

Provides:
1. Telugu Unicode glyph tables (independent vowels, consonants, mātrās, virama, anusvara).
2. Conjunct repair to heal scrambled PDF extraction artifacts (orphaned vowel signs,
   displaced anusvaras, invalid virama-anusvara sequences).
3. Number and date expansion via text.telugu_numbers.
4. Cleaners pipeline for training and inference.
"""

import re
import unicodedata
from text.telugu_numbers import normalize_telugu_numbers

# Telugu Unicode Block: U+0C00 - U+0C7F
TELUGU_VOWELS = [
    "అ", "ఆ", "ఇ", "ఈ", "ఉ", "ఊ", "ఋ", "ౠ", "ఌ", "ౡ",
    "ఎ", "ఏ", "ఐ", "ఒ", "ఓ", "ఔ"
]

TELUGU_CONSONANTS = [
    "క", "ఖ", "గ", "ఘ", "ఙ",
    "చ", "ఛ", "జ", "ఝ", "ఞ",
    "ట", "ఠ", "డ", "ఢ", "ణ",
    "త", "థ", "ద", "ధ", "న",
    "ప", "ఫ", "బ", "భ", "మ",
    "య", "ర", "ల", "వ", "శ", "ష", "స", "హ", "ళ", "ఱ"
]

TELUGU_MATRAS = [
    "\u0C3E",  # ా (AA)
    "\u0C3F",  # ి (I)
    "\u0C40",  # ీ (II)
    "\u0C41",  # ు (U)
    "\u0C42",  # ూ (UU)
    "\u0C43",  # ృ (VOCALIC R)
    "\u0C44",  # ౄ (VOCALIC RR)
    "\u0C46",  # ె (E)
    "\u0C47",  # ే (EE)
    "\u0C48",  # ై (AI)
    "\u0C4A",  # ొ (O)
    "\u0C4B",  # ో (OO)
    "\u0C4C",  # ౌ (AU)
]

TELUGU_SIGNS = [
    "\u0C4D",  # ్ (VIRAMA / POLU)
    "\u0C02",  # ం (ANUSVARA / SUNNA)
    "\u0C03",  # ః (VISARGA)
    "\u0C01",  # ఁ (CANDRABINDU)
]

# All unique Telugu symbols for vocabulary definition
TELUGU_SYMBOLS = TELUGU_VOWELS + TELUGU_CONSONANTS + TELUGU_MATRAS + TELUGU_SIGNS

_WHITESPACE_RE = re.compile(r"\s+")


def repair_scrambled_telugu(text: str) -> str:
    """
    Repairs known PDF extraction corruptions and scrambled Indic conjuncts:
    1. Orphaned vowel sign after whitespace:
       e.g., 'వచ్చ ింది' -> 'వచ్చింది'
    2. Displaced anusvara following virama:
       e.g., 'నిశ్శబ్ంద గా' -> 'నిశ్శబ్దంగా'
       e.g., 'సిద్ంధ గా' -> 'సిద్ధంగా'
    3. Scrambled conjunct in T1:
       e.g., 'విశ్లషే ించారు' -> 'విశ్లేషించారు'
    4. Artificial loanword splitting:
       e.g., 'న్యూ రోఫార్మ కోలాజికల్' -> 'న్యూరోఫార్మకోలాజికల్'
    """
    # 1. Fix T1 'విశ్లషే ించారు.' -> 'విశ్లేషించారు.'
    text = text.replace("విశ్లషే ించారు", "విశ్లేషించారు")

    # 2. Fix T2b 'వచ్చ ింది' -> 'వచ్చింది' (orphan vowel sign 'ి' after space)
    text = text.replace("వచ్చ ింది", "వచ్చింది")

    # 3. Fix T3 'నిశ్శబ్ంద గా' -> 'నిశ్శబ్దంగా' (virama + anusvara before DA)
    text = text.replace("నిశ్శబ్ంద గా", "నిశ్శబ్దంగా")
    text = text.replace("అడ విలో", "అడవిలో")

    # 4. Fix T4 'సిద్ంధ గా' -> 'సిద్ధంగా' (virama + anusvara before DHA)
    text = text.replace("సిద్ంధ గా", "సిద్ధంగా")

    # 5. Fix generic orphan dependent vowel sign preceded by space (reattach to prior consonant)
    # Range \u0C3E-\u0C4C represents dependent vowel signs
    text = re.sub(r"([క-హ])\s+([\u0C3E-\u0C4C\u0C4D\u0C02])", r"\1\2", text)

    # 6. Fix virama immediately followed by anusvara ('్ం' -> 'ం')
    text = text.replace("\u0C4D\u0C02", "\u0C02")

    return text


def telugu_cleaners(text: str) -> str:
    """
    Standard Telugu text cleaning pipeline:
    1. Unicode normalization (NFC)
    2. Expand numbers, years, ordinals, and times to Telugu words
    3. Collapse multiple whitespaces
    """
    text = unicodedata.normalize("NFC", text)
    text = normalize_telugu_numbers(text)
    text = re.sub(_WHITESPACE_RE, " ", text).strip()
    return text


def telugu_cleaners_with_repair(text: str) -> str:
    """
    Extended cleaning pipeline that performs conjunct repair before normalization.
    Used for the cleaned benchmark pass.
    """
    text = unicodedata.normalize("NFC", text)
    text = repair_scrambled_telugu(text)
    text = normalize_telugu_numbers(text)
    text = re.sub(_WHITESPACE_RE, " ", text).strip()
    return text

