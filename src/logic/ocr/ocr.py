import easyocr
import numpy as np
import cv2


GROUPS = {
    # “Generation 1” models (only compatible with English + their own script group)
    'thai'         : ['th'],                       # thai_g1
    'chinese_tra'  : ['ch_tra'],                   # zh_tra_g1
    #'tamil'        : ['ta', 'en'],                       # tamil_g1
    'bengali'      : ['bn', 'as'],                 # bengali_g1
    'arabic'       : ['ar', 'fa', 'ur', 'ug'],      # arabic_g1
    'devanagari'   : ['hi', 'mr', 'ne'],           # devanagari_g1

    # “Generation 2” models (only compatible with English + their own script group)
    'chinese_sim'  : ['ch_sim'],                   # zh_sim_g2
    'japanese'     : ['ja'],                       # japanese_g2
    'korean'       : ['ko'],                       # korean_g2
    'telugu'       : ['te'],                       # telugu_g2
    'kannada'      : ['kn'],                       # kannada_g2
    'cyrillic'     : ['ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn'],  # cyrillic_g2

    # Fallback “latin” model (Generation 2)
    # – this is the catch-all Latin‐script reader (incl. English)
    'latin'        : [
        'en', 'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', #'be', 'bg',
        'es', 'et', 'fr', 'ga', 'hr', 'hu', 'id', 'is', 'it', 'la',
        'lt', 'lv', 'mi', 'ms', 'mt', 'nl', 'no', 'oc', 'pl', 'pt',
        'ro', 'sk', 'sl', 'sq', 'sv', 'sw', 'tl', 'tr', 'uz', 'vi'
    ]
}


readers = {
    name: easyocr.Reader(langs, gpu=False)
    for name, langs in GROUPS.items()
}

def ocr_multigroup(data: dict, min_conf: float = 0.4) -> dict:
    """
    Perform OCR and structure results as:
        {
            (x1, y1, x2, y2, x3, y3, x4, y4): {
                'group1': ['word1', 'word2', ...],
                'group2': [...]
            },
            ...
        }

    Returns dict with:
        - 'image': annotated image
        - 'boxes': nested dict as above
    """
    image: np.ndarray = data['image']
    img_out = image.copy()
    seen = set()

    # Nested dict: key = bbox tuple, value = dict(group → list of words)
    boxes = {}

    for grp, reader in readers.items():
        detections = reader.readtext(image, detail=1)
        for bbox, text, conf in detections:
            if conf < min_conf:
                continue

            # Convert bbox to an 8-tuple of ints
            pts = tuple(int(coord) for point in bbox for coord in point)
            if pts in seen:
                # still record text under group if needed
                group_dict = boxes[pts]
                group_dict.setdefault(grp, []).append(text)
                continue

            seen.add(pts)

            # Initialize group dict for this bbox
            boxes[pts] = {grp: [text]}

            # Draw bounding box
            poly = np.array(bbox, dtype=np.int32)
            cv2.polylines(img_out, [poly], True, (0, 255, 0), 2)
            # Label with first detected group/text
            x, y = poly[0]
            cv2.putText(img_out, f"{grp}: {text}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return {'image': img_out, 'boxes': boxes}