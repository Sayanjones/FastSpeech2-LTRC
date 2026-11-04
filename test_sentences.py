"""
Test Sentences from Assignment Specification (Verbatim).

HARD RULE 4:
Copy the 10 test sentences from the PDF verbatim, including odd spacing and punctuation
(e.g. "rhyth m"). Check Telugu Unicode renders correctly, since PDF extraction can
scramble conjuncts. Flag anything suspicious.
"""

ENGLISH_TEST_SENTENCES = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
}

TELUGU_TEST_SENTENCES = {
    "T1": "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలోనూతన ఔషధాల ప్రభావాన్ని విశ్లషే ించారు.",
    "T2a": "తాడుపై అతని పట్టుచాలా బలంగా ఉంది.",
    "T2b": "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చ ింది.",
    "T3": "పులి అడ విలో నిశ్శబ్ంద గా నడిచింది.",
    "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ంధ గా ఉంచండి.",
}

# Detailed character and anomaly audits for pedagogical study and frontend handling:
SUSPICIOUS_ANOMALIES = {
    "E3": {
        "text": "The rhyth m of the orchestra changed unexpectedly.",
        "issue": "Odd spacing: 'rhyth m' contains an artificial whitespace splitting the word 'rhythm'.",
        "tts_challenge": "Phonemizer may pronounce 'rhyth' and 'm' as two separate words / isolated letter."
    },
    "E4": {
        "text": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
        "issue": "Code-mixing and numeric formats: 'andaroo' is Telugu transliterated in Latin script ('everyone'); contains time ('10:45 AM') and date ('12 November 2026').",
        "tts_challenge": "English G2P rule-based engines must expand numbers/times correctly, and will attempt English grapheme-to-phoneme rules on the Telugu loanword 'andaroo'."
    },
    "T1": {
        "text": "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలోనూతన ఔషధాల ప్రభావాన్ని విశ్లషే ించారు.",
        "issue": (
            "1. Artificial spacing in loan translation: 'న్యూ రోఫార్మ కోలాజికల్' (split into 3 words).\n"
            "2. Missing space in compound: 'పరిశోధనలోనూతన' (merges 'పరిశోధనలో' and 'నూతన').\n"
            "3. Scrambled conjunct in 'విశ్లషే ించారు.': Vowel sign EE (U+0C47) placed on SSA (ష) "
            "followed by space and orphan vowel sign I (U+0C3F) before anusvara (ం). Intended word is 'విశ్లేషించారు'."
        ),
        "tts_challenge": "Orphan vowel sign after space breaks standard Unicode syllable parsing."
    },
    "T2a": {
        "text": "తాడుపై అతని పట్టుచాలా బలంగా ఉంది.",
        "issue": "Missing word boundary whitespace in 'పట్టుచాలా' (merging 'పట్టు' [grip] and 'చాలా' [very]).",
        "tts_challenge": "Compound concatenation affects prosodic phrasing and stress patterns."
    },
    "T2b": {
        "text": "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చ ింది.",
        "issue": "Artificial space in 'వచ్చ ింది.' separating geminate consonant 'వచ్చ' from vowel-inflected verb ending 'ింది'.",
        "tts_challenge": "The space causes 'ింది' to start with a dependent vowel sign (U+0C3F), invalid in strict Unicode."
    },
    "T3": {
        "text": "పులి అడ విలో నిశ్శబ్ంద గా నడిచింది.",
        "issue": (
            "1. Artificial space in 'అడ విలో' splitting 'అడవిలో' (in the forest).\n"
            "2. Scrambled conjunct in 'నిశ్శబ్ంద గా': virama (U+0C4D) followed directly by anusvara (U+0C02) "
            "before DA (ద), instead of anusvara after DA ('నిశ్శబ్దంగా'). Also trailing space before 'గా'."
        ),
        "tts_challenge": "Virama + anusvara is an invalid conjunct sequence in Indic text processors."
    },
    "T4": {
        "text": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ంధ గా ఉంచండి.",
        "issue": (
            "1. Scrambled conjunct in 'సిద్ంధ గా': virama (U+0C4D) + anusvara (U+0C02) before DHA (ధ). "
            "Intended word is 'సిద్ధంగా' (siddhaṅgā).\n"
            "2. Complex non-standard numbers/dates/times: '2026', '12వ తేదీ', '10:45 గంటలకు'."
        ),
        "tts_challenge": "Requires number-to-words expansion in Telugu ('రెండు వేల ఇరవై ఆరు', 'పన్నెండవ', 'పది నలభై ఐదు')."
    }
}

