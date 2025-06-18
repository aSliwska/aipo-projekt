import cv2
import numpy as np
from matplotlib import pyplot as plt

import perspective_correction as pc
import binarization as bn
import perspective_corr_box as pcbox

# import ocr
import tesocr as ocr

from utils import *

import functools
import os


def savepath(mode):
    assert mode in {'in', 'out'}, "Mode must be 'in' or 'out'"

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not isinstance(FILENAME, str):
                raise TypeError("Global FILENAME must be a string")

            name, ext = FILENAME.rsplit('.', 1)
            filename = FILENAME if mode == 'in' else f"{name}_out.{ext}"
            full_path = os.path.join(PATH, filename)

            return func(*args, path=full_path, **kwargs)
        return wrapper
    return decorator












@savepath('in')
def read_img(data : dict, path):
    return {"image" : cv2.imread(path)}

def upscale(data, factor = 1.3):
    image = data["image"]
    return {"image" : cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LANCZOS4)}


def blur(data : dict, sigma = 1.):
    image = data["image"]
    return {"image" : cv2.GaussianBlur(image, (3, 3), sigmaX=sigma)}

def mix_blur(data : dict, strength = 0.6):
    image = data["image"]
    return {"image" : cv2.addWeighted(image, 1 + strength, blur(data)["image"], - strength, 0)}

def invert(data : dict):
    image = data["image"]
    return { "image" : cv2.bitwise_not(image)}

@savepath('out')
def save(data, path):
    image = data["image"]
    cv2.imwrite(path, image)
    return data




def process_single(data : dict, just_text : bool = False, runocr : bool = True):
    """
    Process a single image.

    This function applies a series of image processing techniques to improve text 
    visibility and readability, particularly for OCR tasks. Optional perspective 
    correction and OCR recognition can be applied based on the input flags.

    Args:
        data (dict): A dictionary containing data for processing. Store the image 
                              as 'image' in the dictionary.
        just_text (bool, optional): If True, attempts to apply horizontal perspective 
                                    correction assuming the image region contains only text with some background (!) as a reference for correction.
                                    Defaults to False.
        runocr (bool, optional): If True, performs OCR using a multi-group recognizer. 
                                    Defaults to True.

    Returns:
        dict or processed image data: The processed data, which includes 
        'image' (post-processed image), 'boxes' (dictionary 'boxes' : {'group' : ['text', 'conf']}), 'output' (dictionary 'group' : 'text'). 
                                      
    """
    data = upscale(data,1.4)
    data = mix_blur(data, 0.6)
    if just_text: #try correcting only if the input is a text region; otherwise do not even attempt
        try:
            #raise(RuntimeError())
            data = pc.correct_horizontal_perspective(data, debug=True) #good enough, reliable & generally sufficient
            pass
        except RuntimeError as e:
            print(e)
            try:
                data = pcbox.auto_perspective_correct_plane(data) #fallback, much less reliable (rounded edges) but sometimes produces slightly better results
            except RuntimeError as e:
                print(e) 
                pass
    data = upscale(data,1.4) #small resolution of text on images is assumed
    data = mix_blur(data, 0.4)
    try:
        #data = pc.correct_horizontal_perspective(data, debug=True) 
        pass
    except RuntimeError as e:
        print(e)
        pass
    data = bn.otsu_binarize(data) #bn.adaptive_binarize(data)
    
    
    if runocr:
        data = ocr.ocr_multigroup(data, debug=True)
        plt.imshow(data["debug"])
        plt.show()
        print(data["boxes"])
        print(data["output"])
    return data


if __name__ == "__main__":
    FILENAME = "test6.png"
    PATH = "."
    runocr = True
    
    data = read_img(None)
    data = process_single(data, runocr=runocr)
    data = save(data)