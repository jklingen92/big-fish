

import argparse
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

SUNGLASSES_HATS_MODEL = YOLO("yolov8s-world.pt")
SUNGLASSES_HATS_MODEL.set_classes(["sunglasses", "hat"])
from fishidentification.fish_segmentation import segment_fish
from skimage.morphology import skeletonize


def id_fish(img, show=False):
    """Identify the species of fish in the image."""
    pass

def id_objects(img, show=False):
    """Identify known objects within the picture from a library of objects."""
    pass

def get_font_scale(text, width):
    for scale in range(59, -1, 1):
        text_size = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = text_size[0][0]
        if (new_width <= width):
            return scale/10
    return 1

def resize(img, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = img.shap[:2]
    if width is None and height is None:
        return img
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(img, dim, interpolation=inter)

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

class Reference:
    """
    Reference is a class that implements an estimate function. 
    """

    def estimate(self, img):
        """Take an image and return an estimate of the scale of 1 pixel."""
        raise NotImplemented()

class Reconciler(Reference):
    """
    Reconciler is a class that reconciles a number of different reference
    points based on weights. 
    """

    references = None

    def estimate(self, img):
        return sum(ref.estimate(img) * weight for ref, weight in self.references)


class FacialReference(Reference):
    """
    FacialReference will identify faces in an image along with their 
    predicted age and gender.
    
    The code in this file is drawn from Gil Levi and Tal Hassner's 
    repository: https://github.com/GilLevi/AgeGenderDeepLearning
    """

    MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    GENDER_COMPONENTS = ["Male", "Female"]
    AGE_INTERVALS = [
        (0, 2), (4, 6), (8, 12), (15, 20),
        (25, 32), (38, 43), (48, 53), (60, 100)
    ]

    def __init__(self, show=False):
        face_proto = "weights/deploy.prototxt"
        gender_proto = "weights/deploy_gender.prototxt"
        age_proto = "weights/deploy_age.prototxt"
        
        face_model = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        gender_model = "weights/gender_net.caffemodel"
        age_model = "weights/age_net.caffemodel"
        
        self.face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
        self.gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
        self.age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)

        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel("weights/lbfmodel.yaml")

        self.show = show

    def estimate(self, img):
        faces = self.get_faces(img)
        success, landmarks = self.facemark.fit(img, np.array([(f[0], f[1], f[2] - f[0], f[3] - f[1]) for f in faces]))
        if not success:
            raise Exception("Failed to identify facial landmarks.")

        # To Do select a face smarter
        face_coords = faces[0]
        face_landmarks = landmarks[0]
        start_x, start_y, end_x, end_y = face_coords

        left_brow = face_landmarks[0][21]
        right_brow = face_landmarks[0][22]
        middle_brow = midpoint(left_brow, right_brow)
    
        top_nose = face_landmarks[0][27]
        middle_nose = midpoint(middle_brow, top_nose)
        bottom_chin = face_landmarks[0][8]

        if self.show:
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
            cv2.line(img, (int(middle_nose[0]), int(middle_nose[1])), (int(bottom_chin[0]), int(bottom_chin[1])),(0, 255, 0), 2)
            plt.imshow(img)
            plt.show()

        face = img[start_y:end_y, start_x:end_x]
        expected_value = self.get_expected_facial_height(face)  # in mm
        actual_value = np.linalg.norm(middle_nose - bottom_chin)  # in px
        print(f"Reference scale: {actual_value}px = {expected_value}mm > {expected_value / actual_value}mm per pixel")
        return expected_value / actual_value

    def get_expected_facial_height(self, face):
        gender_preds = self.get_gender_predictions(face)
        age_preds = self.get_age_predictions(face)
        expected_value = 0
        with open("data/male_facial_height.json", "r") as f:
            male_data = json.load(f)

        with open("data/female_facial_height.json", "r") as f:
            female_data = json.load(f)
        
        for i, gender_confidence in enumerate(gender_preds[0]):
            gender_component = self.GENDER_COMPONENTS[i]
            if gender_component == "Male":
                age_data = male_data
            else:
                age_data = female_data
            for j, age_confidence in enumerate(age_preds[0]):
                age_range = self.AGE_INTERVALS[j]
                age_key = f"({age_range[0]} - {age_range[1]})"
                expected_value += age_data[age_key]["mean"] * age_confidence * gender_confidence
        
        return expected_value
            
    def get_faces(self, frame, threshold=0.5):
        """Detect faces in an image with threshold confidence."""
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        output = np.squeeze(self.face_net.forward())
        faces = []
        for i in range(0, output.shape[0]):
            confidence = output[i, 2]
            if confidence > threshold:
                box = output[i, 3:7] * np.array([
                    frame.shape[1],
                    frame.shape[0],
                    frame.shape[1],
                    frame.shape[0],
                ])
                start_x, start_y, end_x, end_y = box.astype(int)
                start_x = 0 if start_x < 0 else start_x
                start_y = 0 if start_y < 0 else start_y
                end_x = 0 if end_x < 0 else end_x
                end_y = 0 if end_y < 0 else end_y
                faces.append((start_x, start_y, end_x, end_y))
        return faces

    def get_gender_predictions(self, face):
        
        blob = cv2.dnn.blobFromImage(
            image=face, 
            scalefactor=1.0, 
            size=(227, 227), 
            mean=self.MEAN_VALUES, 
            swapRB=False, 
            crop=False
        )
        self.gender_net.setInput(blob)

        gender_preds = self.gender_net.forward()
        return gender_preds
    
    def get_age_predictions(self, face):
        blob = cv2.dnn.blobFromImage(
            image=face, 
            scalefactor=1.0, 
            size=(227, 227),
            mean=self.MEAN_VALUES,
            swapRB=False
        )
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        return age_preds


def id_face(img_location, show=False):
    """Identify the face in the image."""
    img = cv2.imread(img_location)
    fd = FacialReference(show=show)
    fd.estimate(img)

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


def centerline_from_skeleton(skel):
    """
    skel: bool array, True on skeleton pixels
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

    # 8 connected neighbors
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
    path_idx = path_idx[::-1]  # reverse to get A -> B

    path_coords = [coords[i] for i in path_idx]

    # curved length = sum of distances between consecutive points
    length_px = 0.0
    for (x1, y1), (x2, y2) in zip(path_coords[:-1], path_coords[1:]):
        length_px += float(np.hypot(x2 - x1, y2 - y1))

    smooth_coords = moving_average_smooth(path_coords, k=20)

    return smooth_coords, length_px



def measure_fish(img, show=False):
    """Measure the fish in the image using a curved centerline (in millimeters)."""

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
    skel = skeleton_from_mask(mask)
    centerline_pts, length_px = centerline_from_skeleton(skel)

    # pixel length -> mm
    length_mm = float(length_px * mm_per_pixel)

    print(f"Curved fish length: {length_px:.2f} px, {length_mm/10.0:.2f} cm")

    if show:
        vis = img_bgr.copy()

        # draw centerline as polyline
        pts_array = np.array(
            [[int(x), int(y)] for (x, y) in centerline_pts],
            dtype=np.int32,
        ).reshape(-1, 1, 2)

        cv2.polylines(vis, [pts_array], isClosed=False, color=(0, 255, 255), thickness=2)

        # Label near the first point on the centerline
        x0, y0 = centerline_pts[0]
        text = f"{length_mm / 10.0:.1f} cm"
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

    return length_mm



def detect_sunglasses_hats(img_path, show=False):
    """
    Detect sunglasses + hats using YOLO and print measurements.
    """
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image not found: {img_path}")

    results = SUNGLASSES_HATS_MODEL.predict(image, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = x2 - x1
            height = y2 - y1

            print("---Measurement---")
            print(f"{label}: width={width}px, height={height}px, conf={conf:.2f}")
            print("-----------------")

            detections.append({
                "label": label,
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "width": width,
                "height": height
            })

            if show:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(image, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    if show:
        plt.figure(figsize=(12,8))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
        plt.axis("off")
        plt.show()

    return detections


    

if __name__ == "__main__":
    REGISTRY = {
        "fish": id_fish,
        "objects": id_objects,
        "face": id_face,
        "measurement": measure_fish,
        "sunglasses_hats": detect_sunglasses_hats,
    }

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "img",
        type=str,
        help=f"The location of an image file to analyze",
    )
    parser.add_argument(
        "--run_comp",
        type=str,
        nargs="+",
        choices=list(REGISTRY.keys()),
        help=f"The name of the component that you would like to execute. Allowed values: {', '.join(REGISTRY)}",
        metavar="",
    )
    parser.add_argument(
        "-s", "--show",
        action="store_true"
    )
    args = parser.parse_args()

    if args.run_comp is None:
        fns = list(REGISTRY.items())
    else:
        # I got AttributeError: 'Namespace' object has no attribute 'fun_comp'. Did you mean: 'run_comp'?
        #fns = [(fn_name, REGISTRY[fn_name]) for fn_name in args.fun_comp]
        fns = [(fn_name, REGISTRY[fn_name]) for fn_name in args.run_comp]
        fns = [(fn_name, REGISTRY[fn_name]) for fn_name in args.run_comp]

    for fn_name, fn_callable in fns:
        print(f"Running {fn_name}...")
        print(fn_callable(args.img, show=args.show))
        print()