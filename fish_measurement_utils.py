

import numpy as np
import cv2
import matplotlib.pyplot as plt
from fishidentification.fish_segmentation import segment_fish
from scipy.ndimage import distance_transform_edt


from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist

def polygon_to_mask(poly, img_shape):
    """
    poly: (N, 2) array of (x, y) vertices
    img_shape: (H, W) or (H, W, C) of the original image
    returns: uint8 mask with 255 on fish, 0 background
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    poly_int = poly.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [poly_int], 255)
    return mask


def smooth_fish_mask(mask, sigma=5.0, kernel_wrap=True):
    """
    Smooth a fish-shaped binary mask while preserving its length
    (tip-to-tail max distance).

    Parameters
    ----------
    mask : np.ndarray, 2D
        Binary mask (0/1 or 0/255) of the fish shape.
    sigma : float
        Smoothing amount for the contour (higher = smoother).
    kernel_wrap : bool
        If True, treat contour as circular when smoothing.

    Returns
    -------
    smoothed_mask : np.ndarray, 2D uint8
        New binary mask with smoothed contour and preserved length.
    """
    # Ensure mask is uint8 0/255
    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
    else:
        mask_u8 = (mask > 0).astype(np.uint8) * 255

    # Find the outer contour
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise ValueError("No contour found in mask.")

    # Use the largest contour
    cnt = max(contours, key=cv2.contourArea)  # (N, 1, 2)
    pts = cnt[:, 0, :]                        # (N, 2) -> [x, y]

    # Original length = max pairwise distance among contour points
    # (this is tip-to-tail distance for elongated shapes)
    L0 = pdist(pts.astype(float)).max()

    # Smooth contour coordinates along the index dimension
    x = pts[:, 0].astype(float)
    y = pts[:, 1].astype(float)

    if kernel_wrap:
        # wrap so start/end meet smoothly (closed contour)
        x_smooth = gaussian_filter1d(x, sigma=sigma, mode="wrap")
        y_smooth = gaussian_filter1d(y, sigma=sigma, mode="wrap")
    else:
        x_smooth = gaussian_filter1d(x, sigma=sigma)
        y_smooth = gaussian_filter1d(y, sigma=sigma)

    smooth_pts = np.stack([x_smooth, y_smooth], axis=1)  # (N, 2)

    # Compute new length after smoothing
    L1 = pdist(smooth_pts).max()
    if L1 == 0:
        raise ValueError("Smoothed contour collapsed (L1=0). Try smaller sigma.")

    # Scale smoothed contour so L1 -> L0 (preserve length)
    scale = L0 / L1

    # Scale around the shape centroid to avoid drift
    centroid = smooth_pts.mean(axis=0, keepdims=True)
    smooth_pts_centered = smooth_pts - centroid
    smooth_pts_scaled = smooth_pts_centered * scale + centroid

    # Rasterize back to a mask
    h, w = mask_u8.shape[:2]
    smoothed_mask = np.zeros((h, w), dtype=np.uint8)
    # cv2.fillPoly expects int32
    poly = smooth_pts_scaled.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(smoothed_mask, [poly], 255)

    return smoothed_mask



def skeleton_from_mask(mask):
    """
    mask: uint8 0/255
    returns: skeleton as a boolean array, True on centerline pixels
    """
    skel = skeletonize(mask > 0)  # skel is bool array
    return skel

from collections import deque

def moving_average_smooth(path_coords, k=5):
    pts = np.array(path_coords, dtype=np.float32)
    smoothed = []
    for i in range(len(pts)):
        start = max(0, i - k)
        end = min(len(pts), i + k + 1)
        window = pts[start:end]
        smoothed.append(window.mean(axis=0))
    return [tuple(p) for p in smoothed]


from collections import deque
import numpy as np

def moving_average_smooth(path_coords, k=5):
    pts = np.array(path_coords, dtype=np.float32)
    smoothed = []
    for i in range(len(pts)):
        start = max(0, i - k)
        end = min(len(pts), i + k + 1)
        window = pts[start:end]
        smoothed.append(window.mean(axis=0))
    return [tuple(p) for p in smoothed]



def centerline_from_skeleton(skel, mask=None, smooth_k=5, extend_to_ends=True):
    """
    skel: bool array, True on skeleton pixels
    mask: uint8 fish mask (0/255), used to extend endpoints to fish boundary
    smooth_k: window size for moving-average smoothing
    extend_to_ends: if True and mask is given, push endpoints out to mask boundary

    returns:
        path_coords: list of (x, y) points along the centerline, in order
        length_px: curved length in pixels (sum of segment lengths)
    """
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        raise RuntimeError("Skeleton is empty.")

    coords = list(zip(xs, ys))           # (x, y)
    n = len(coords)
    idx_of = {c: i for i, c in enumerate(coords)}
    coord_set = set(coords)

    # 8-connected neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]

    adj = [[] for _ in range(n)]
    for i, (x, y) in enumerate(coords):
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if (nx, ny) in coord_set:
                adj[i].append(idx_of[(nx, ny)])

    def bfs(start):
        dist = [-1] * n
        prev = [-1] * n
        dist[start] = 0
        dq = deque([start])
        while dq:
            v = dq.popleft()
            for nb in adj[v]:
                if dist[nb] == -1:
                    dist[nb] = dist[v] + 1
                    prev[nb] = v
                    dq.append(nb)
        farthest = max(range(n), key=lambda i: dist[i])
        return farthest, dist, prev

    # pick any skeleton pixel as start, find farthest A
    start = 0
    a, _, _ = bfs(start)
    # from A, find farthest B and predecessor chain
    b, _, prev = bfs(a)

    # reconstruct path A -> B
    path_idx = []
    cur = b
    while cur != -1:
        path_idx.append(cur)
        if cur == a:
            break
        cur = prev[cur]
    path_idx = path_idx[::-1]  # A -> B

    # original skeleton path coordinates (curvy)
    path_coords = [coords[i] for i in path_idx]  # (x, y)

    # --- Smooth the path (still curved) ---
    smooth_coords = moving_average_smooth(path_coords, k=smooth_k)

    # --- Optionally extend endpoints to the mask boundary ---
    if extend_to_ends and (mask is not None):
        h, w = mask.shape[:2]

        def normalize(v):
            v = np.asarray(v, dtype=np.float32)
            nrm = np.linalg.norm(v)
            if nrm < 1e-6:
                return None
            return v / nrm

        def extend_point(p0, p1, direction_sign=1.0, step=1.0, max_steps=5000):
            """
            p0: endpoint (x,y) we want to move outward
            p1: neighbor (x,y) to estimate direction
            direction_sign: +1 or -1; chooses outward direction
            """
            x0, y0 = map(float, p0)
            x1, y1 = map(float, p1)
            # local tangent: p0 - p1 or p1 - p0 depending on sign
            v = np.array([x0 - x1, y0 - y1], dtype=np.float32) * direction_sign
            v = normalize(v)
            if v is None:
                return p0  # can't extend

            x, y = x0, y0
            last_inside = (x, y)
            for _ in range(max_steps):
                x += v[0] * step
                y += v[1] * step
                xi, yi = int(round(x)), int(round(y))
                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    break
                if mask[yi, xi] == 0:
                    break
                last_inside = (x, y)
            return last_inside

        # start endpoint
        if len(smooth_coords) >= 2:
            p0 = smooth_coords[0]
            p1 = smooth_coords[1]
            new_start = extend_point(p0, p1, direction_sign=1.0)
        else:
            new_start = smooth_coords[0]

        # end endpoint
        if len(smooth_coords) >= 2:
            q0 = smooth_coords[-1]
            q1 = smooth_coords[-2]
            new_end = extend_point(q0, q1, direction_sign=1.0)
        else:
            new_end = smooth_coords[-1]

        smooth_coords[0]  = new_start
        smooth_coords[-1] = new_end

    # --- Curved length along the smoothed (and possibly extended) path ---
    length_px = 0.0
    for (x1, y1), (x2, y2) in zip(smooth_coords[:-1], smooth_coords[1:]):
        length_px += float(np.hypot(x2 - x1, y2 - y1))

    return smooth_coords, length_px

def mm_to_inches(length):
    return (length * 0.0394)

def measure_fish(img, show=False):
    """Measure the fish in the image using a curved centerline (in millimeters)."""
    from bigfish import FacialReference, get_font_scale
    img_bgr = cv2.imread(img)
    if img_bgr is None:
        raise ValueError(f"Could not read image from '{img}'")

    # get mm per pixel from face
    facial_ref = FacialReference(show=False)
    mm_per_pixel = facial_ref.estimate(img_bgr)

    # Segment fish -> polygon vertices
    poly = segment_fish(img)
    poly = np.asarray(poly, dtype=np.float32)

    # polygon -> mask -> skeleton -> centerline
    mask = polygon_to_mask(poly, img_bgr.shape)
    mask_smoothed = smooth_fish_mask(mask, sigma=5.0, kernel_wrap=True)
    skel = skeleton_from_mask(mask_smoothed)
    centerline_pts, length_px = centerline_from_skeleton(
        skel,
        mask=mask_smoothed,
        smooth_k=47,        # tweak: 5,7,9 etc.
        extend_to_ends=True,
    )

    for (x1, y1), (x2, y2) in zip(centerline_pts[:-1], centerline_pts[1:]):
        length_px += float(np.hypot(x2 - x1, y2 - y1))
    # pixel length -> mm
    length_mm = float(length_px * mm_per_pixel)

    length_inches = mm_to_inches(length_mm)

    print(f"Curved fish length: {length_px:.2f} px, {length_inches:.2f} inches")

    if show:
        vis = img_bgr.copy()

        # --- Overlay smoothed mask ---
        mask_color = np.zeros_like(vis)
        mask_color[mask_smoothed > 0] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 1.0, mask_color, 0.3, 0)

        # --- Draw mask contour ---
        contours, _ = cv2.findContours(mask_smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

        # --- Draw skeleton (you were missing these two variables!) ---
        ys, xs = np.nonzero(skel)
        for x, y in zip(xs, ys):
            cv2.circle(vis, (int(x), int(y)), 1, (255, 0, 0), -1)

        # --- Draw straightened centerline ---
        centerline_array = np.array(
            [[int(x), int(y)] for (x, y) in centerline_pts],
            dtype=np.int32,
        ).reshape(-1, 1, 2)

        cv2.polylines(vis, [centerline_array], isClosed=False, color=(0, 255, 255), thickness=2)

        # --- Label ---
        x0, y0 = centerline_pts[0]
        text = f"{length_inches:.1f} inches"
        font_scale = get_font_scale(text, vis.shape[1] // 3)
        cv2.putText(
            vis,
            text,
            (int(x0), int(y0) - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        plt.imshow(vis_rgb)
        plt.title("Fish Curved-Length Measurement")
        plt.axis("off")
        plt.show()

    return length_inches