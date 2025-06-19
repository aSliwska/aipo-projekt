import cv2
import numpy as np

def adaptive_binarize(data: dict,
                      block_size: int = 35,
                      c: int = 10,
                      bg_blur_size: int = 51) -> dict:
    """
    Perform adaptive binarization of an image, taking uneven lighting into account.

    Args:
        block_size (int): Size of the neighborhood to calculate the threshold (must be odd and >1).
        c (int): Constant subtracted from the mean or weighted sum.
        bg_blur_size (int): Kernel size for Gaussian blur to estimate background illumination (must be odd).

    Returns:
        np.ndarray: Binary image (uint8) with values 0 or 255.
    """

    
    image: np.ndarray = data["image"]
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Estimate background illumination by blurring
    if bg_blur_size > 1:
        if bg_blur_size % 2 == 0:
            bg_blur_size += 1  # ensure odd
        background = cv2.GaussianBlur(gray, (bg_blur_size, bg_blur_size), 0)
        # Avoid division by zero
        background = np.where(background == 0, 1, background)
        # Normalize image by its background
        norm = cv2.divide(gray, background, scale=255)
    else:
        norm = gray

    # Ensure block_size is odd and >1
    if block_size % 2 == 0:
        block_size += 1
    if block_size <= 1:
        block_size = 3

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        norm,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c
    )

    return {"image" : binary}



def otsu_binarize(data: dict,
                      block_size: int = 35,
                      c: int = 10,
                      bg_blur_size: int = 51) -> dict:
    """
    Perform binarization of an image using Otsu's method, optionally normalizing
    for uneven lighting by dividing by a blurred background.

    Args:
        block_size (int): Unused in Otsu's method, retained for interface compatibility.
        c (int): Unused in Otsu's method, retained for interface compatibility.
        bg_blur_size (int): Kernel size for Gaussian blur to estimate background illumination (must be odd).

    Returns:
        dict: Dictionary with binarized image under key "image".
    """

    image: np.ndarray = data["image"]
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Estimate background illumination by blurring
    if bg_blur_size > 1:
        if bg_blur_size % 2 == 0:
            bg_blur_size += 1  # ensure odd
        background = cv2.GaussianBlur(gray, (bg_blur_size, bg_blur_size), 0)
        background = np.where(background == 0, 1, background)  # avoid division by zero
        norm = cv2.divide(gray, background, scale=255)
    else:
        norm = gray

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {"image": binary}