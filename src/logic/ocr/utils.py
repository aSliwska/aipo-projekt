import functools
import inspect
import cv2


def inject_settings(settings: dict):
    """
    Decorator to inject default keyword arguments into a function,
    allowing explicit args and kwargs to override injected settings.
    """
    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Bind provided args/kwargs to parameters, ignoring missing ones
            bound = sig.bind_partial(*args, **kwargs)
            # Inject settings only for parameters in function signature
            for name, param in sig.parameters.items():
                if name in settings and name not in bound.arguments:
                    bound.arguments[name] = settings[name]
            # Apply defaults for any remaining parameters
            bound.apply_defaults()
            # Call the original function with all resolved arguments
            return func(**bound.arguments)

        return wrapper
    return decorator







def fade_edges(data: dict,
               border_width: float = 0.03,
               blur_ksize: int = 55) -> dict:
    """
    Apply a smooth fade-to-black on the edges of an image.

    Args:
        border_width (float): Width of the fading border as a fraction of the smaller image dimension (e.g., 0.1 = 10%).
        blur_ksize (int): Kernel size for Gaussian blur of the mask (must be odd).

    Returns:
        np.ndarray: Image with edges faded to black.
    """
    img : np.ndarray = data["image"]

    h, w = img.shape[:2]
    # Calculate pixel border width from fraction
    # Use the smaller dimension to keep border proportional
    max_border = min(h, w)  
    bw_px = int(border_width * max_border) if 0 < border_width < 1 else int(border_width)
    bw_px = max(0, min(bw_px, max_border // 2))

    # 1) Create floating-point mask with 1 in center, 0 at borders
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.rectangle(mask,
                  (bw_px, bw_px),
                  (w - bw_px, h - bw_px),
                  color=1.0,
                  thickness=-1)

    # 2) Blur the mask to create a smooth transition
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    mask_blurred = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)

    # 3) Broadcast mask to match image channels if needed
    if img.ndim == 3 and img.shape[2] == 3:
        mask_blurred = mask_blurred[:, :, np.newaxis]

    # 4) Multiply image by mask (fade to black at edges)
    faded = (img.astype(np.float32) * mask_blurred).astype(img.dtype)
    return {"image" : faded}