import time
from typing import List, Dict
from deep_translator import GoogleTranslator


def translate_phrases(
    data: Dict[str, List[str]],
    delay_per_request: float = 0.5,  # Delay in seconds to not overload the API
    source_language: str = "auto",
    target_language: str = "en"
) -> List[str]:
    """
    Translate phrases to English using Google Translate with rate-limit protection.
    Skips translation if rate limit is exceeded or any error occurs.

    :param data: Dictionary with phrases under any key.
    :param delay_per_request: Delay between translation requests in seconds.
    :param source_language: Language of input phrases ('auto' detects automatically).
    :param target_language: Target translation language (default: 'en').
    :return: List of translated English phrases or empty strings if failed.
    """
    all_phrases = []
    for key, phrases in data.items():
        if isinstance(phrases, list):
            all_phrases.extend(phrases)

    translator = GoogleTranslator(source=source_language, target=target_language)
    translated_phrases = []

    for phrase in all_phrases:
        if not phrase.strip():
            translated_phrases.append("")
            continue

        try:
            translated = translator.translate(phrase)
            translated_phrases.append(translated)
        except Exception as e:
            # Suppress error and return empty string for this phrase
            print(f"Skipping translation due to error: {e}")
            translated_phrases.append("")

        time.sleep(delay_per_request)  # prevent triggering Google Translate limits

    return translated_phrases


if __name__ == "__main__":
    data = {
        "MixedLanguages": [
            "pasukti į kairę",
            "マクドナルドは1.2km先です",
            "اتجه يمينا ٢ كم",
            "ঘুরে বাঁ দিকে যান",
            "เลี้ยวซ้าย",
            "Поверните направо",
            "McDonalds 1.2 km ahead"
        ]
    }

    translated = translate_phrases(data)
    print("Translated Phrases:")
    for phrase in translated:
        print(f"- {phrase}")
