import spacy
import re
from collections import defaultdict
import unicodedata
import yake
from collections import defaultdict, Counter
from rapidfuzz import process, fuzz

#=========================INSTALACJE=================
#!pip install -U pip setuptools wheel
#!pip install -U spacy
#!python -m spacy download en_core_web_sm
#!pip install yake
#!pip install rapidfuzz
#====================================================

# Główna funkcja extract_named_entities_with_distance(translated_texts, ocr_dict)

# =========================INPUT==================================== POSTAC: translated_texts: lista stringow;  ocr_dict: dict "alphabet" : lista stringow

#=========================OUTPUT====================================: dict "slowo_kluczowe" : "odleglosc" (0 jeśli brak)

nlp = spacy.load("en_core_web_sm")

# Czyszczenie nazw z literówek i znaków specjalnych
def normalize_name(name):
    name = "".join(
        c for c in name
        if unicodedata.category(c)[0] in {"L", "N"} or c == " "
    )
    name = name.replace("0", "O").replace("1", "I").replace("4", "A")
    return re.sub(r"\s+", " ", name).strip()

# Detekcja języka do YAKE
def detect_yake_language(text):
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    elif re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    elif re.search(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]", text):
        return "ko"
    elif re.search(r"[\u0E00-\u0E7F]", text):
        return "th"
    elif re.search(r"[\u1700-\u171F\u1720-\u173F\u1740-\u175F]", text):
        return "tl"
    elif re.search(r"[\u0400-\u04FF]", text):
        return "ru"
    return "en"

# Ekstrakcja odległości
def extract_distance(text):
    match = re.search(r"(\d+(?:[\.,]\d+)?)\s*(km|kilometers|miles|mi|m|meters|км|метров|公里|米|マイル|キロ|킬로미터|마일|กม|ไมล์|ми|миль|ก\.ม\.?)", text, re.IGNORECASE)
    if match:
        value = float(match.group(1).replace(",", "."))
        unit = match.group(2).lower()
        if unit in ["m", "meters", "метров", "米", "ม.", "m先"]:
            return value / 1000
        elif unit in ["miles", "mi", "마일", "ไมล์", "マイル", "ми", "миль"]:
            return value * 1.60934
        return value
    return 0.0

# Ekstrakcja z YAKE
def extract_keywords_with_yake(text):
    lang = detect_yake_language(text)
    extractor = yake.KeywordExtractor(lan=lang, n=1, dedupLim=0.9, top=20, features=None)
    keywords = extractor.extract_keywords(text)
    banned_keywords = {
        "visit", "exit", "turn", "ahead", "in",
        "miles", "km", "kilometers", "meters", "mi", "m",
        "км", "метров", "公里", "米",
        "マイル", "キロ", "킬로미터", "마일", "กม", "ไมล์"
    }

    names = set()
    for kw, _ in keywords:
        kw_clean = kw.strip()
        if len(kw_clean) <= 30 and kw_clean.lower() not in banned_keywords:
            if not kw_clean.islower():
                names.add(kw_clean)
    return names

# Grupowanie nazw po fuzzy match
def deduplicate_names(name_dict, threshold=65):
    names = list(name_dict.keys()) #lista wszystkich kluczy ze słownika wejściowego
    merged = {}
    used = set()

    for name in names:
        if name in used:
            continue
        group = [name]
        norm_name = normalize_name(name)
        for other in names:
            if other == name or other in used:
                continue
            norm_other = normalize_name(other)
            if fuzz.ratio(norm_name, norm_other) >= threshold:
                group.append(other)
                used.add(other)
        merged_name = max(group, key=lambda x: len(x)) #Wynik w końcowej liście to będzie ten z najdłuższą nazwą
        avg_distance = sum(name_dict[n] for n in group) / len(group) #Średni dystans
        merged[merged_name] = round(avg_distance, 2) #zaokrąglenie do 2 cyfr po przecinku
        used.update(group)
    return merged

# Główna funkcja
def extract_named_entities_with_distance(translated_texts, ocr_dict):
    result = defaultdict(float)
    all_texts = translated_texts + [text for texts in ocr_dict.values() for text in (texts if isinstance(texts, list) else [texts])]

    for text in all_texts:
        doc = nlp(text)
        distance = extract_distance(text)
        candidates = set()

        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "ORG", "FAC"]:
                candidates.add(ent.text.strip())

        yake_keywords = extract_keywords_with_yake(text)
        candidates.update(yake_keywords)

        for name in candidates:
            name_clean = normalize_name(name)
            if name_clean:
                if name_clean not in result or distance > 0:
                    result[name_clean] = max(result[name_clean], distance)

    return deduplicate_names(result)

# Przykład użycia
if __name__ == "__main__":
    translated_texts = [
        "McDonalds 1.2 km ahead",
        "Next turn: KFC 2.5 km",
        "KFC niedaleko! 2.6km",
        "Visit StalBud",
        "Exit to BurgerKing in 3 km",
        "Shell station 1.5 miles",
        "Warszawa 30 km",
        "Poznań 120"
    ]

    ocr_dict = {
        "Latin": [
            "Stalpol 4.0 km",
            "McD0nalds 1,2 km ahe4d",
            "KFC 2.5 km",
            "BurgerKing in 3.0 km",
            "Visit Sta!Bud",
            "Shell station 1.5 mils"
        ],
        "Cyrillic": [
            "Макдоналдс 2 км",
            "СТАЛП0Л 4 км",
            "АЗС Shell 1.2 ми"
        ],
        "Chinese Simplified": [
            "星巴克 5 km",
            "肯德基 2.5 公里",
            "麦当劳 3公里"
        ],
        "Japanese": [
            "空港中央 Haneda Airport 6km",
            "大黒ふ頭 Daikoku futo 22km",
            "スターバックス 3キロ先",
            "ケンタッキー 2.5キロ",
            "マクドナルド 1.8km"
        ],
        "Korean": [
            "롯데마트 5킬로미터",
            "맥도날드 1.2 마일",
            "버거킹 3 km"
        ],
        "Thai": [
            "เซเว่น 2 กม.",
            "แมคโดนัลด์ 1.5 ไมล์",
            "เบอร์เกอร์คิง 3 ก.ม."
        ],
        "Filipino": [
            "Jollibee 4 km",
            "Mang Inasal 2 miles",
            "Chowking 3.2 km"
        ]
    }

    output = extract_named_entities_with_distance(translated_texts, ocr_dict)
    print(output)

