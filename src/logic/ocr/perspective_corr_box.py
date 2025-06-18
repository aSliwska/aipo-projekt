import cv2
import numpy as np

def preprocess(img):
    # Convert to grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to binarize
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 4
    )

    # Morphological closing to connect text regions
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed

def find_text_quadrilaterals(img):
    preprocessed = preprocess(img)
    contours, _ = cv2.findContours(
        preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    quads = []
    for contour in contours:
        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Check for convex quadrilateral
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 1000:  # Filter small regions
                quads.append(approx.reshape(-1, 2))
    return quads

def select_best_quad(quads, img_area):
    best_quad = None
    max_priority = -np.inf

    for quad in quads:
        # Compute bounding rectangle properties
        rect = cv2.minAreaRect(quad)
        (w, h) = rect[1]
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

        # Heuristic: prioritize large, near-rectangular regions
        area = cv2.contourArea(quad)
        if aspect_ratio < 3 and area > 0.01 * img_area:
            priority = area * (1 / aspect_ratio)
            if priority > max_priority:
                max_priority = priority
                best_quad = quad
    return best_quad


def dltnorm(src_pts, target_pts):
    # Compute a similarity trasformation T and T_prime to normalize src_pts and target_pts
    normalized_src_pts, T = normalization(src_pts)
    normalized_target_pts, T_prime = normalization(target_pts)

    # Construct A Matrix from pairs a and b
    A = []
    for i in range(0, len(normalized_src_pts)):
        ax, ay = normalized_src_pts[i][0], normalized_src_pts[i][1]
        bx, by = normalized_target_pts[i][0], normalized_target_pts[i][1]
        A.append([-ax, -ay, -1, 0, 0, 0, bx*ax, bx*ay, bx])
        A.append([0, 0, 0, -ax, -ay, -1, by*ax, by*ay, by])

    # Compute SVD for A
    A = np.asarray(A)
    U, S, V = np.linalg.svd(A)

    # The solution is the last column of V (9 x 1) Vector
    L = V[-1, :]

    # Divide by last element as we estimate the homography up to a scale
    L = L/V[-1, -1]
    H_tilde = L.reshape(3, 3)

    # Denormalization: denormalize the homography back
    H = np.dot(np.dot(np.linalg.pinv(T_prime), H_tilde), T)
    H = H/H[-1, -1]

    return H

def normalization(pts):
    N = len(pts)
    mean = np.mean(pts, 0)
    s = np.linalg.norm((pts-mean), axis=1).sum() / (N * np.sqrt(2))

    # Compute a similarity transformation T, moves original points to
    # new set of points, such that the new centroid is the origin,
    # and the average distance from origin is square root of 2
    T = np.array([[s, 0, mean[0]],
                  [0, s, mean[1]],
                  [0, 0, 1]])
    T = np.linalg.inv(T)
    pts = np.dot(T, np.concatenate((pts.T, np.ones((1, pts.shape[0])))))
    pts = pts[0:2].T
    return pts, T


def dlt(src_pts, target_pts):
    # Construct A Matrix from pairs a and b
    A = []
    for i in range(0, len(src_pts)):
        ax, ay = src_pts[i][0], src_pts[i][1]
        bx, by = target_pts[i][0], target_pts[i][1]
        A.append([-ax, -ay, -1, 0, 0, 0, bx*ax, bx*ay, bx])
        A.append([0, 0, 0, -ax, -ay, -1, by*ax, by*ay, by])

    # Compute SVD for A
    A = np.asarray(A)
    U, S, V = np.linalg.svd(A)

    # The solution is the last column of V (9 x 1) Vector
    L = V[-1, :]

    # Divide by last element as we estimate the homography up to a scale
    L = L/V[-1, -1]
    H = L.reshape(3, 3)

    return H

def order_points(pts):
    # pts: Nx2 array of (x,y) corner points (in any order)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest diff
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest diff
    return rect

def compute_destination(rect):
    # rect: ordered corners [tl, tr, br, bl]
    (tl, tr, br, bl) = rect
    # compute widths
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    maxW    = max(int(widthA), int(widthB))
    # compute heights
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH    = max(int(heightA), int(heightB))
    # destination corners in perfect rectangle
    dst = np.array([
        [0,       0      ],
        [maxW - 1, 0     ],
        [maxW - 1, maxH - 1],
        [0,       maxH - 1]
    ], dtype="float32")
    return dst, (maxW, maxH)

def perspective_correct_text(img, pts, use_normalized_dlt=False):
    """
    img:       input BGR image (or ROI) containing text
    pts:       list or array of four (x,y) corners around the text region
    use_normalized_dlt: if True, use dltnorm; else use plain dlt
    returns:   warped, fronto-parallel image of the text region
    """
    # 1) Order source points
    src = order_points(np.array(pts, dtype="float32"))

    # 2) Compute destination rectangle
    dst, (w, h) = compute_destination(src)

    # 3) Estimate homography via DLT
    if use_normalized_dlt:
        H = dltnorm(src, dst)
    else:
        H = dlt(src, dst)

    # 4) Warp the image
    warped = cv2.warpPerspective(
        img, H, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped

def auto_perspective_correct(data):
    img = data["image"]
    h, w = img.shape[:2]
    quads = find_text_quadrilaterals(img)
    if not quads:
        raise RuntimeError("No text quad detected.")

    best_quad = select_best_quad(quads, img_area=h*w)
    if best_quad is None:
        raise RuntimeError("No valid quad found.")

    # Order corners and compute destination
    ordered_src = order_points(best_quad)
    dst, _ = compute_destination(ordered_src)

    # Use normalized DLT for robustness
    H = dltnorm(ordered_src, dst)
    warped = cv2.warpPerspective(
        img, H, (int(dst[2][0])+1, int(dst[2][1])+1),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return {"image" : warped}



#
#
#
#
#

def preprocess_for_plane_detection(img):
    """
    Preprocesses the image to enhance edges and make larger structural elements stand out,
    suitable for finding the dominant planar surface.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and help Canny find better edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    # Tuned thresholds for broader edge detection, as we're looking for larger shapes
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Morphological operations to close gaps in edges and make contours more continuous
    # This is crucial for connecting broken lines that form the perimeter of the plane
    kernel_close = np.ones((7, 7), np.uint8) # Larger kernel to bridge bigger gaps
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Optional: Dilate to ensure thicker contours, which can help with contour finding
    dilated_edges = cv2.dilate(closed_edges, np.ones((3,3), np.uint8), iterations=1)

    return dilated_edges

def find_plane_quadrilaterals(img):
    """
    Finds potential quadrilaterals that could represent the main plane in the image.
    Prioritizes larger and more rectangular-like shapes.
    """
    preprocessed = preprocess_for_plane_detection(img)
    contours, _ = cv2.findContours(
        preprocessed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE # RETR_LIST or RETR_EXTERNAL for outer contours
    )

    quads = []
    
    # Sort contours by area in descending order to process larger ones first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_height, img_width = img.shape[:2]
    img_area = img_height * img_width

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter out very small or excessively large contours that are unlikely to be the plane
        # A good plane usually takes up a significant portion of the image.
        if area < 0.05 * img_area or area > 0.99 * img_area:
            continue

        # Approximate contour to polygon
        # Increased epsilon to allow for more "forgiving" approximation, as real-world
        # planes might not have perfectly sharp corners in the image.
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True) # Epsilon increased from 0.02 to 0.03

        # Check if the approximated contour has 4 vertices and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            
            # Get the bounding box to check aspect ratio
            # Use minAreaRect for better handling of rotated rectangles
            rect_params = cv2.minAreaRect(approx)
            (width, height) = rect_params[1]

            if min(width, height) == 0:
                continue

            aspect_ratio = max(width, height) / min(width, height)

            # Filter by aspect ratio: a plane should typically be somewhat rectangular.
            # Adjust these bounds based on the expected shapes of your planes.
            # A common document could be 1.0 to ~2.0 for A4/Letter. A very wide screen could be higher.
            if 0.5 < aspect_ratio < 5.0: # Wider range than for text to accommodate various plane shapes
                quads.append(approx.reshape(-1, 2))
                
    return quads

def select_best_plane_quad(quads, img_area):
    """
    Selects the best quadrilateral from a list, prioritizing based on properties
    that indicate it's the main planar surface.
    """
    if not quads:
        return None

    best_quad = None
    max_score = -np.inf

    for quad in quads:
        area = cv2.contourArea(quad)
        
        # Calculate bounding box for aspect ratio and dimensions
        rect_params = cv2.minAreaRect(quad)
        (width, height) = rect_params[1]

        if min(width, height) == 0:
            continue
        
        aspect_ratio = max(width, height) / min(width, height)

        # Calculate a "fullness" or "coverage" score: how much of the image area it covers
        # The ideal plane should cover a significant portion, but not necessarily 100%
        coverage_score = area / img_area

        # Aspect ratio score: closer to a common document/plane aspect ratio is better.
        # This penalizes highly skewed or extremely thin shapes.
        # Example: if typical documents are ~1.4 (A4) or ~1.29 (Letter), you could
        # tailor this. Here, we generalize for typical rectangular planes.
        aspect_score = 1.0 / (abs(aspect_ratio - 1.5) + 1.0) # Prioritize aspect ratio around 1.5 (e.g., A4)
        if aspect_ratio < 0.8 or aspect_ratio > 3.0: # Penalize extreme aspect ratios more heavily
             aspect_score *= 0.5 # Reduce score if too far from common rectangularity


        # Calculate a combined score
        # Give higher weight to coverage, as the goal is the *whole* plane.
        # Also consider how "rectangular" it is (its aspect ratio).
        current_score = (coverage_score * 0.6) + (aspect_score * 0.4) 
        
        # Additional checks:
        # 1. Ensure the quad is reasonably large (e.g., at least 10% of image area)
        #    This is already handled by filtering in find_plane_quadrilaterals, but can be a final check.
        if area < 0.1 * img_area: # Re-emphasize minimum area for final selection
            continue

        # 2. Check if the quad is reasonably well-formed (e.g., not extremely distorted)
        #    This is partially handled by aspect ratio, but you could add a 'straightness' check
        #    if needed, though approxPolyDP and isContourConvex usually handle this.

        if current_score > max_score:
            max_score = current_score
            best_quad = quad
            
    return best_quad

# Re-using previous helper functions (dltnorm, normalization, dlt, order_points, compute_destination)
# as they are robust and general.
# The `perspective_correct_text` function should be renamed and slightly adjusted
# to reflect the goal of correcting the whole plane without unwanted scaling.


# def normalization(pts):
#     N = len(pts)
#     mean = np.mean(pts, 0)
#     std_dev_sum = np.linalg.norm((pts - mean), axis=1).sum()
#     s = np.sqrt(2) * N / std_dev_sum if std_dev_sum > 1e-6 else 1.0 # Handle case where all points are identical

#     T = np.array([[s, 0, -s * mean[0]],
#                   [0, s, -s * mean[1]],
#                   [0, 0, 1]])
    
#     pts_homogeneous = np.concatenate((pts.T, np.ones((1, pts.shape[0]))), axis=0)
#     normalized_pts_homogeneous = np.dot(T, pts_homogeneous)
#     normalized_pts = normalized_pts_homogeneous[0:2].T
    
#     return normalized_pts, T





def compute_destination_for_plane(rect):
    """
    Computes the destination rectangle for perspective correction,
    aiming to preserve the aspect ratio of the detected plane.
    """
    (tl, tr, br, bl) = rect
    
    # Calculate widths (top and bottom)
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    
    # Calculate heights (left and right)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    # Use the maximum of opposing sides to determine the output dimensions.
    # This helps in preventing downscaling if one side is slightly shorter due to perspective,
    # and keeps the original 'resolution' of the longest dimension.
    output_width = int(max(width_top, width_bottom))
    output_height = int(max(height_left, height_right))

    # Ensure minimum dimensions to prevent issues with very small or degenerate detections
    output_width = max(output_width, 50) # Increased min size
    output_height = max(output_height, 50) # Increased min size

    dst = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype="float32")
    
    return dst, (output_width, output_height)

def perspective_correct_plane(img, pts, use_normalized_dlt=True):
    """
    Applies perspective correction to the entire plane represented by `pts`.
    The output image will be largely rotation-corrected without significant scaling.
    """
    # 1) Order source points
    src = order_points(np.array(pts, dtype="float32"))

    # 2) Compute destination rectangle based on the dimensions of the detected plane
    dst, (w, h) = compute_destination_for_plane(src)

    # 3) Estimate homography via DLT
    if use_normalized_dlt:
        H = dltnorm(src, dst)
    else:
        H = dlt(src, dst)

    # 4) Warp the image
    warped = cv2.warpPerspective(
        img, H, (w, h),
        flags=cv2.INTER_CUBIC, # INTER_CUBIC is good for quality
        borderMode=cv2.BORDER_REPLICATE # Replicate border pixels to avoid black edges
    )
    return warped

def auto_perspective_correct_plane(data):
    """
    Automatically detects the main planar surface in the image and performs perspective correction.
    """
    img = data["image"]
    h, w = img.shape[:2]
    img_area = h * w

    # Find potential quadrilaterals representing the plane
    quads = find_plane_quadrilaterals(img)
    
    best_quad = None
    if quads:
        best_quad = select_best_plane_quad(quads, img_area)

    # Fallback: If no suitable quad is found, use the entire image as the plane.
    # This assumes the whole image is the plane if detection fails.
    if best_quad is None:
        print("Warning: No dominant plane quad detected. Using full image as quad for correction.")
        best_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

    # Order corners and compute destination
    ordered_src = order_points(best_quad)
    
    # Perform perspective correction
    warped = perspective_correct_plane(img, ordered_src, use_normalized_dlt=True)
    
    return {"image": warped}