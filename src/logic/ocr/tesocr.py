import pytesseract
import numpy as np
import cv2

GROUPS = {
    'thai'         : ['tha'],
    'chinese_tra'  : ['chi_tra'],
    'bengali'      : ['ben', 'asm'],
    'arabic'       : ['ara', 'fas', 'urd', 'uig'],
    'devanagari'   : ['hin', 'mar', 'nep'],
    'chinese_sim'  : ['chi_sim'],
    'japanese'     : ['jpn'],
    'korean'       : ['kor'],
    'telugu'       : ['tel'],
    'kannada'      : ['kan'],
    'cyrillic'     : ['rus', 'srp', 'bel', 'bul', 'ukr', 'mon'],
    'latin'        : [
        'eng', 'afr', 'aze', 'bos', 'ces', 'cym', 'dan', 'deu',
        'spa', 'est', 'fra', 'gle', 'hrv', 'hun', 'ind', 'isl',
        'ita', 'lat', 'lit', 'lav', 'mri', 'msa', 'mlt', 'nld',
        'nor', 'oci', 'pol', 'por', 'ron', 'slk', 'slv', 'sqi',
        'swe', 'swa', 'tur', 'uzb', 'vie' #, 'tgl'
    ]
}

def ocr_multigroup(data: dict, min_conf: float = 0.6, debug: bool = False, lang_groups: dict = GROUPS) -> dict:
    """
    Perform OCR with pytesseract across multiple language groups.
    Args:
      data: {'image': numpy array}
      min_conf: minimum confidence threshold
      debug: if True, returns a debug image with drawn boxes
    Returns:
      'image': original annotated image
      'raw': {bbox: {group: [(text, confidence), ...]}}
      'output': {group: 'full string of found text in the image'}
      'debug': debug image (if debug=True)
    """
    image: np.ndarray = data['image']
    img_deb = image.copy() if debug else None
    if debug and (img_deb.ndim == 2 or img_deb.shape[2] == 1):
        img_deb = cv2.cvtColor(img_deb, cv2.COLOR_GRAY2BGR)

    seen = set()
    boxes = {}
    output_raw = {grp: [] for grp in lang_groups.keys()}  # To store (text, conf) tuples
    output_full_string = {grp: [] for grp in lang_groups.keys()}  # To store concatenated strings

    # Convert to RGB for pytesseract
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for grp, langs in lang_groups.items():
        print("Processing", grp)
        for lang in langs:
            # Use image_to_string for the full text output
            full_text = pytesseract.image_to_string(rgb, lang=lang, config='--oem 3 --psm 6')
            output_full_string[grp].append(full_text.strip()) # Append and add newline for readability

            # Use image_to_data for bounding box and confidence information
            ocr_data = pytesseract.image_to_data(
                rgb, config=f'--oem 3 --psm 11 -l {lang}', output_type=pytesseract.Output.DICT #--psm 6 for regular
            )

            for i, raw_text in enumerate(ocr_data['text']):
                text = raw_text.strip()
                if not text:
                    continue
                try:
                    conf = float(ocr_data['conf'][i]) / 100.
                except (ValueError, TypeError):
                    print(f"Error in tesseract confidence for text: '{raw_text}'")
                    continue
                if conf < min_conf:
                    continue

                output_raw[grp].append((text, conf)) # Still store for potential future use or debugging

                x, y, w, h = (
                    ocr_data['left'][i], ocr_data['top'][i],
                    ocr_data['width'][i], ocr_data['height'][i]
                )
                bbox = (x, y, x + w, y + h)

                if bbox not in seen:
                    seen.add(bbox)
                    boxes[bbox] = {grp: [(text, conf)]}
                    if debug:
                        cv2.rectangle(img_deb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            img_deb,
                            f"{grp}: {text} ({conf:.1f})",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
                else:
                    boxes[bbox].setdefault(grp, []).append((text, conf))

    # Clean up the full string output by removing trailing newlines
    # for grp in output_full_string:
    #     output_full_string[grp] = output_full_string[grp].strip()

    result = {'image': image, 'raw': boxes, 'output': output_full_string}
    if debug:
        result['debug'] = img_deb
    return result
