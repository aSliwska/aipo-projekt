"""
language_detection.py

Detects language (e.g. en, pl, ru)
for a list of phrases, and maps languages to countries.
"""

from langdetect import DetectorFactory, detect_langs, LangDetectException
import unicodedata

from logic.lang_detect.language_countries import language_countries
from logic.osm.countries import all_countries

# Make langdetect deterministic
DetectorFactory.seed = 0

def detect_language(phrase: str, top_n: int = 1):
    """
    Uses langdetect to return the top_n probable languages for the phrase.
    Returns a list of (lang_code, probability).
    """
    try:
        langs = detect_langs(phrase)
    except LangDetectException:
        return []
    return [(l.lang, l.prob) for l in langs[:top_n]]

def get_countries_for_language(lang_code: str, countries: list) -> list:
    """
    Given a language code (e.g. 'en', 'pl', 'ru'),
    return the subset of `countries` where that language is commonly used.
    """
    # Look up our predefined mapping
    mapped = language_countries.get(lang_code, [])
    # Only return those actually present in the all_countries list
    return [c for c in countries if c in mapped]

def analyze_phrases(data: dict, all_countries: list) -> dict:
    """
    For every phrase in the data, detect language with >90% confidence,
    and map each language to a deduplicated list of countries.
    Returns a dictionary: {language_code: [countries]}.
    """
    lang_to_countries = {}

    for phrases in data.values():
        for phrase in phrases:
            lang_probs = detect_language(phrase, top_n=3)
            for lang_code, prob in lang_probs:
                if prob < 0.9:
                    continue  # skip low-confidence results

                countries = get_countries_for_language(lang_code, all_countries)
                if not countries:
                    continue

                if lang_code not in lang_to_countries:
                    lang_to_countries[lang_code] = set()

                lang_to_countries[lang_code].update(countries)

    # Convert sets to sorted lists
    return {lang: sorted(list(countries)) for lang, countries in lang_to_countries.items()}

if __name__ == "__main__":
    data = {
        "Latin": [
            "Stalpol 4.0 km",
            "McD0nalds 1,2 km ahe4d",
            "Skręt w prawo 2.5 km",
            "Curve  3.0 km",
            "Visit Sta!Bud",
            "Shell station 1.5 mils"
        ],
        "Cyrillic": [
            "Макдоналдс 2 км",
            "СТАЛП0Л 4 км",
            "АЗС Shell 1.2 ми"
        ]
    }
    import pprint
    pprint.pprint(analyze_phrases(data, all_countries))