import cv2
import numpy as np
import functools
from utils import *
from matplotlib import pyplot as plt

H_CORR_SETTINGS = dict()



@inject_settings(H_CORR_SETTINGS)
def correct_horizontal_perspective(
    data,
    canny_low=None,#50,
    canny_high=None,#150,
    hough_rho=1,
    hough_theta=np.pi / 180,
    hough_threshold=160,
    horizontal_angle_thresh=np.pi / 3,
    vp_center_thresh=0.1,
    debug=False
):
    """
    Load an image, detect near-horizontal text lines, pick the best horizontal vanishing point (VP),
    apply a horizontal-only perspective correction, and finally rotate so text lines are truly horizontal.

    Args:
        image_path (str): Path to the input image.
        canny_low (int): Lower threshold for Canny edge detector.
        canny_high (int): Upper threshold for Canny edge detector.
        hough_rho (float): Distance resolution (pixels) for HoughLines.
        hough_theta (float): Angle resolution (radians) for HoughLines.
        hough_threshold (int): Minimum number of intersections to detect a line.
        horizontal_angle_thresh (float): ± threshold around π/2 to keep near-horizontal lines.
        vp_center_thresh (float): Threshold on |x_vp|/width to reject VPs "too close" to image center.
        debug (bool): If True, show intermediate debugging windows and draw detected lines.

    Returns:
        final_img (numpy.ndarray): The corrected (perspective + rotation) image.
        H_persp_adj (numpy.ndarray): 3×3 homography (with centering) for the horizontal perspective correction.
        rotation_deg (float): The final rotation angle (in degrees) that was applied.
    """
    img = data["image"]
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = None
    if canny_high is not None and canny_high is not None:
        edges = cv2.Canny(gray, canny_low, canny_high)
    else:
        med_val = np.median(img) 
        lower = int(max(0 ,0.7*med_val))
        upper = int(min(255,1.3*med_val))
        edges = cv2.Canny(gray, lower, upper)

    if debug:
        plt.imshow(edges, 'grey')
        plt.title('Edges')
        plt.show()

    # 2. HoughLines: find all lines in polar form (rho, theta)
    lines_raw = cv2.HoughLines(edges, hough_rho, hough_theta, hough_threshold)
    if lines_raw is None:
        raise RuntimeError("No lines detected by HoughLines.")

    # 3. Filter "near-horizontal" lines: theta ∈ [π/2 - horizontal_angle_thresh, π/2 + horizontal_angle_thresh]
    horizontal_lines = []  # store homogeneous [a,b,c]
    raw_hough_lines = []   # store (rho,theta) for drawing
    for line in lines_raw:
        rho, theta = line[0]
        if abs(theta - np.pi/2) <= horizontal_angle_thresh:
            a = np.cos(theta)
            b = np.sin(theta)
            c = -rho
            horizontal_lines.append(np.array([a, b, c], dtype=np.float64))
            raw_hough_lines.append((rho, theta))

    if len(horizontal_lines) < 2:
        raise RuntimeError("Not enough near-horizontal lines detected.")

    # If debug: draw detected Hough lines on a copy of original
    if debug:
        debug_img = img.copy()
        for (rho, theta) in raw_hough_lines:
            x0 = rho * np.cos(theta)
            y0 = rho * np.sin(theta)
            dx = -np.sin(theta)
            dy = np.cos(theta)
            pt1 = (int(x0 + 1000 * dx), int(y0 + 1000 * dy))
            pt2 = (int(x0 - 1000 * dx), int(y0 - 1000 * dy))
            cv2.line(debug_img, pt1, pt2, (0, 0, 255), 2)
        
        plt.imshow(debug_img, 'grey')
        plt.title('Detected near-horizontal Hough Lines')
        plt.show()

    # 4. Build all pairwise intersections of those lines -> candidate VPs
    candidates = []
    for i in range(len(horizontal_lines)):
        for j in range(i + 1, len(horizontal_lines)):
            l1 = horizontal_lines[i]
            l2 = horizontal_lines[j]
            vp = np.cross(l1, l2)
            if abs(vp[2]) < 1e-6:
                continue
            x_vp = vp[0] / vp[2]
            y_vp = vp[1] / vp[2]
            if 0 <= x_vp <= w and 0 <= y_vp <= h:
                continue
            if abs(x_vp) / float(w) < vp_center_thresh:
                continue
            candidates.append((x_vp, y_vp))

    if not candidates:
        raise RuntimeError("No valid vanishing-point candidates remained after filtering.")

    # 5. For each candidate VP, build H_persp using l_inf = VP × (0,1,0), then compute stddev of corrected line angles
    best_sigma = float("inf")
    best_angles = None
    best_Hp = None

    for (x_vp, y_vp) in candidates:
        VP = np.array([x_vp, y_vp, 1.0])
        Vy = np.array([0.0, 1.0, 0.0])
        l_inf = np.cross(VP, Vy)
        if abs(l_inf[2]) < 1e-6:
            continue
        l_inf = l_inf / l_inf[2]
        a_norm, b_norm = l_inf[0], l_inf[1]

        # H_persp: [[1,0,0],[0,1,0],[a_norm,b_norm,1]]
        Hp = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [a_norm, b_norm, 1.0]
        ], dtype=np.float64)

        H_inv_T = np.linalg.inv(Hp).T
        angles = []
        for l in horizontal_lines:
            lp = H_inv_T @ l
            a_p, b_p = lp[0], lp[1]
            if abs(b_p) < 1e-8:
                angle = np.pi / 2
            else:
                angle = np.arctan2(-a_p, b_p)
            angles.append(angle)

        sigma = float(np.std(angles))
        if sigma < best_sigma:
            best_sigma = sigma
            best_angles = angles
            best_Hp = Hp.copy()

    if best_Hp is None:
        raise RuntimeError("Failed to find a good vanishing point.")

    # 6. To avoid shearing and keep rotation calculation correct, apply centering transforms: shift origin to image center
    cx, cy = w / 2.0, h / 2.0
    T = np.array([
        [1.0, 0.0, -cx],
        [0.0, 1.0, -cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    T_inv = np.array([
        [1.0, 0.0, cx],
        [0.0, 1.0, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Adjust homography by centering
    H_persp_adj = T_inv @ best_Hp @ T
    warped = cv2.warpPerspective(img, H_persp_adj, (w, h), flags=cv2.INTER_LINEAR)

    if debug:
        plt.imshow(warped, 'grey')
        plt.title('Warped (perspective only)')
        plt.show()

    # 7. Recompute line angles after full adjusted perspective warp
    H_inv_T_adj = np.linalg.inv(H_persp_adj).T
    angles_adj = []
    for l in horizontal_lines:
        lp = H_inv_T_adj @ l
        a_p, b_p = lp[0], lp[1]
        if abs(b_p) < 1e-8:
            angle = np.pi / 2
        else:
            angle = np.arctan2(-a_p, b_p)
        angles_adj.append(angle)

    # Compute mean angle of adjusted lines and rotate to make horizontal exactly
    mean_angle = float(np.mean(angles_adj))
    rotation_deg = mean_angle * 180.0 / np.pi
    center = (w / 2, h / 2)
    M_rot = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    final_img = cv2.warpAffine(warped, M_rot, (w, h), flags=cv2.INTER_LINEAR)

    if debug:
        plt.imshow(final_img, 'grey')
        plt.title("Final (rotated)")
        plt.show()

    return { 'image' : final_img, 'H' : H_persp_adj, 'rotation' : rotation_deg}



if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Horizontal-only perspective correction for text images"
    )
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to save corrected image")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show intermediate windows (edges, detected lines, warped, final)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: '{args.input}' does not exist.")
        exit(1)

    # 1. Load and preprocess
    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f"Could not read '{args.input}'")

    resized = cv2.resize(image, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_LANCZOS4)

    

    corrected, H, rot = correct_horizontal_perspective(
       resized , debug=args.debug
    )

    blurred = cv2.GaussianBlur(corrected, (0, 0), sigmaX=1.0)

    # Unsharp mask: original + weighted difference
    sharpened = cv2.addWeighted(corrected, 1.6, blurred, -0.6, 0)

    cv2.imwrite(args.output, sharpened)
    print(f"Saved corrected image to '{args.output}'")
    print("Perspective homography (with centering):\n", H)
    print("Rotation angle (deg):", rot)
