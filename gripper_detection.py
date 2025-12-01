import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import os

def remove_white_background(img: np.ndarray) -> np.ndarray:
    """
    Removes white background
    Keeps object intact
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    mask = np.zeros((h + 2, w + 2), np.uint8)
    flooded = gray.copy()
    fill_color = 0

    cv2.floodFill(flooded, mask, (0, 0), fill_color, loDiff=25, upDiff=25)
    cv2.floodFill(flooded, mask, (w-1, 0), fill_color, loDiff=25, upDiff=25)
    cv2.floodFill(flooded, mask, (0, h-1), fill_color, loDiff=25, upDiff=25)
    cv2.floodFill(flooded, mask, (w-1, h-1), fill_color, loDiff=25, upDiff=25)

    _, object_mask = cv2.threshold(flooded, 1, 255, cv2.THRESH_BINARY_INV)
    object_only = cv2.bitwise_and(img, img, mask=object_mask)
    """
    plt.figure(figsize=(10,5))
    plt.title("Background Removed")
    plt.imshow(object_only)
    plt.axis("off")
    plt.show()
    """
    return object_only

def crop_bottom_half_of_object(img: np.ndarray) -> np.ndarray:
    """
    Takes image with background removed. Finds object bounding box.
    Returns bottom half of that bounding box.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    #ys, xs = np.where(img > 0) # for edges

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    mid_y = (y_min + y_max) // 2
    bottom_half = img[mid_y:y_max, x_min:x_max]
    
    plt.figure(figsize=(10,5))
    plt.title("Bottom Half")
    plt.imshow(bottom_half)
    plt.axis("off")
    plt.show()
    cv2.imwrite("cropped.png", bottom_half)
    
    return bottom_half

def sift_matches(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes SIFT correspondences between two images.
    Returns Nx2 arrays of matched points.
    """

    # ensure uint8
    if img1.dtype != np.uint8:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = (img2 * 255).astype(np.uint8)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.92 * n.distance:
            good.append([m])

    #src_pts = np.asarray([kp1[m[0].queryIdx].pt for m in good])
    #dest_pts = np.asarray([kp2[m[0].trainIdx].pt for m in good])
    src_pts = np.asarray([kp1[good[i][0].queryIdx].pt for i in range(len(good))])
    dest_pts = np.asarray([kp2[good[i][0].trainIdx].pt for i in range(len(good))])

    out1 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(8,8))
    plt.title("Template SIFT")
    #plt.title("Image1 SIFT")
    plt.imshow(cv2.cvtColor(out1, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    out2 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(8,8))
    plt.title("Scene SIFT")
    #plt.title("Image2 SIFT")
    plt.imshow(cv2.cvtColor(out2, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    match_vis = cv2.drawMatchesKnn(
        cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR),
        #img1,
        kp1,
        cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR),
        #img2,
        kp2,
        good,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(14, 8))
    plt.title("SIFT Matches")
    plt.imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    cv2.imwrite("sift_matches.png", match_vis)

    return src_pts, dest_pts, good, kp1, kp2


def find_gripper_location(src_pts: np.ndarray, dst_pts: np.ndarray,
                          template_shape: Tuple[int, int],
                          scene_img: np.ndarray,
                          good_matches, kp1, kp2, img1, img2):
    """
    Compute homography from template -> scene and draw box on scene.
    Returns (output_image, pixel_length) where pixel_length = height in pixels.
    """

    if len(src_pts) < 10:
        return scene_img, None

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i] == 1]

    # ---- visualize inliers ----
    vis = cv2.drawMatchesKnn(
        img1, kp1,
        img2, kp2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(14, 7))
    plt.title("RANSAC Inlier Matches")
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    cv2.imwrite("sift_matches_ransac.png", vis)

    if H is None:
        return scene_img, None

    h, w = template_shape

    # template corner box
    box = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(box, H)
    projected_int = np.int32(projected)

    # draw polygon
    out = scene_img.copy()
    cv2.polylines(out, [projected_int], True, (0, 255, 0), 3)

    # compute pixel length (height)
    top = projected[0][0]   # (0,0)
    bottom = projected[2][0]  # (w,h)

    pixel_length = np.linalg.norm(bottom - top)

    return out, pixel_length

def detect_gripper(template_img: np.ndarray, scene_img: np.ndarray, edges: bool):
    """
    Full pipeline:
    - Remove background from template
    - Crop bottom half
    - Extract SIFT correspondences
    - Compute homography
    - Draw bounding box on scene
    - Return pixel length + output image
    """
    # --- template preprocessing ---
    #no_bg = remove_white_background(template_img)
    #no_bg = template_img
    cropped = crop_bottom_half_of_object(template_img)
    #cropped = template_img

    # grayscale for SIFT
    if edges == False:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        scene_img = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)

    # feature match
    src_pts, dst_pts, good_matches, kp1, kp2 = sift_matches(cropped, scene_img)

    # detect location
    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    result_img, pixel_length = find_gripper_location(
        src_pts,
        dst_pts,
        template_img.shape,
        scene_img,
        good_matches, kp1, kp2, cropped, scene_img
    )

    return pixel_length, result_img


def show_sift_keypoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)
    out = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.axis('off')

if __name__ == "__main__":
    template = cv2.imread("lip-grippers/BogaGrip130-10.75.jpg")
    scene = cv2.imread("imgs/inshore-adventures-charter-fishing-st-augustine-redfish-red-drum-20.jpg")
    #scene = cv2.imread("imgs/inshore-adventures-charter-fishing-st-augustine-redfish-red-drum-24.jpg")
    #scene = cv2.imread("imgs/inshore-adventures-charter-fishing-st-augustine-redfish-red-drum-25.jpg")

    #res = remove_white_background(template)
    #show_sift_keypoints(res)
    
    template_blur = cv2.GaussianBlur(template, (5, 5), 0)
    #scene_blur = cv2.GaussianBlur(scene, (1, 1), 0)
    template_edges = cv2.Canny(cv2.cvtColor(template_blur, cv2.COLOR_BGR2GRAY), 100, 150)
    scene_edges = cv2.Canny(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), 150, 250)
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Template Edges")
    plt.imshow(template_edges, cmap='gray')
    plt.axis('off')
    #cv2.imwrite("template_edges.jpg", template_edges)

    plt.subplot(1,2,2)
    plt.title("Scene Edges")
    plt.imshow(scene_edges, cmap='gray')
    plt.axis('off')
    #cv2.imwrite("scene_edges.jpg", scene_edges)

    plt.show()
    
    
    length_px, output_img = detect_gripper(template, scene, False)
    # length_px, output_img = detect_gripper(template_edges, scene_edges, True) # for edges
    
    print("Detected gripper length (pixels):", length_px)
    plt.figure(figsize=(10,6))
    plt.title("Detected")
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    #cv2.imwrite("detected_output.jpg", output_img)

    """
    # trial
    img_1 = cv2.imread("../../Assignment/Homework 6 Programming/data/face-left.png")
    img_2 = cv2.imread("../../Assignment/Homework 6 Programming/data/face-right.png")
    #src_pts, dst_pts, good_matches, kp1, kp2 = sift_matches(img_1, img_2)
    length_px, output_img = detect_gripper(img_1, img_2, False)
    """
