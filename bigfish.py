

import argparse
import json
from pprint import pprint
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

SUNGLASSES_HATS_MODEL = YOLO("yolov8s-world.pt")
SUNGLASSES_HATS_MODEL.set_classes(["sunglasses", "hat"])
from fish_measurement_utils import measure_fish


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


class PoseReference(Reference):
    """
    PoseReference will perform the following operations to construct 
    its estimate:
        1. Detects the person in the image
        2. Identify the pose of the person
        3. Attempt to measure known lengths on the person
        4. Attempt to determine depth difference from fish to person
    """

    def __init__(self):
        pose_proto = "weights/pose_deploy_linevec_faster_4_stages.prototxt"
        pose_model = "weights/pose_iter_160000.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(pose_proto, pose_model)

    def estimate(self, img):
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()
        h, w = img.shape[:2]
        points = []
        for i in range(15):
            prob_map = output[0, i, :, :]
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
            x = (w * point[0]) / output.shape[3]
            y = (h * point[1]) / output.shape[2]
            if prob > 0.1:
                points.append((int(x), int(y)))
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            else:
                points.append(None)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
        print(points)
        return 1.0  # Placeholder value
        

class FacialReference(Reference):
    """
    FacialReference will performs the following operations to construct
    its estimate:
        1. Finds faces in the image
        2. Predicts the age and gender of the faces
        3. Identifies landmarks on the faces
        4. Measures the morphological facial height
        5. Compares the facial height to a lookup table to determine a scale
    
    Face, age, and gender determination are drawn from Gil Levi and 
    Tal Hassner's work: https://github.com/GilLevi/AgeGenderDeepLearning

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

        if len(faces) == 0:
            raise Exception("No faces detected.")
        if not success:
            raise Exception("Failed to identify facial landmarks.")

        landmark_img = img.copy()
        measure_img = img.copy()

        references = []
        # Create a reference point for each face
        for i, face in enumerate(faces):
            try:
                face_landmarks = landmarks[i]
            except IndexError:
                print("Failed to get landmarks for face:", i)
                continue

            start_x, start_y, end_x, end_y = face

            left_brow = face_landmarks[0][21]
            right_brow = face_landmarks[0][22]
            middle_brow = midpoint(left_brow, right_brow)
    
            top_nose = face_landmarks[0][27]
            middle_nose = midpoint(middle_brow, top_nose)
            lip = face_landmarks[0][62]

            if self.show:
                cv2.rectangle(landmark_img, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
                cv2.rectangle(measure_img, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
                for (x, y) in face_landmarks[0]:
                    cv2.circle(landmark_img, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                cv2.line(measure_img, (int(left_brow[0]), int(left_brow[1])), (int(right_brow[0]), int(right_brow[1])),(0, 0, 255), 2)
                cv2.line(measure_img, (int(left_brow[0]), int(left_brow[1])), (int(top_nose[0]), int(top_nose[1])),(0, 0, 255), 2)
                cv2.line(measure_img, (int(right_brow[0]), int(right_brow[1])), (int(top_nose[0]), int(top_nose[1])),(0, 0, 255), 2)
                cv2.line(measure_img, (int(middle_nose[0]), int(middle_nose[1])), (int(lip[0]), int(lip[1])),(0, 255, 0), 2)

            face = img[start_y:end_y, start_x:end_x]
            expected_value, gender_str, age_str = self.get_expected_facial_height(face)  # in mm
            actual_value = np.linalg.norm(middle_nose - lip)  # in px
            if self.show:
                cv2.putText(
                    measure_img,
                    gender_str,
                    (end_x + 10, start_y + 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    measure_img,
                    age_str,
                    (end_x + 10, start_y + 60),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    measure_img,
                    f"Expected facial height: {expected_value:.1f}mm",
                    (end_x + 10, start_y + 90),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    measure_img,
                    f"Measured facial height: {actual_value:.1f}px",
                    (end_x + 10, start_y + 120),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    measure_img,
                    f"Scale: {expected_value / actual_value:.3f}mm per pixel",
                    (end_x + 10, start_y + 150),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            references.append(expected_value / actual_value)
        
        if self.show:
            plt.imshow(cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB))
            plt.show()

            plt.imshow(cv2.cvtColor(measure_img, cv2.COLOR_BGR2RGB))
            plt.show()
        return np.mean(references)

    def get_expected_facial_height(self, face):
        gender_preds = self.get_gender_predictions(face)
        age_preds = self.get_age_predictions(face)
        expected_value = 0
        with open("data/male_facial_height.json", "r") as f:
            male_data = json.load(f)

        with open("data/female_facial_height.json", "r") as f:
            female_data = json.load(f)
        
        gender_pred = np.argmax(gender_preds[0])
        gender_str = f"{self.GENDER_COMPONENTS[gender_pred]} ({gender_preds[0][gender_pred]*100:.2f}%)"
        age_pred = np.argmax(age_preds[0])
        age_range = self.AGE_INTERVALS[age_pred]
        age_str = f"Age Range: {age_range[0]} - {age_range[1]} ({age_preds[0][age_pred]*100:.2f}%)"

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
        
        return expected_value, gender_str, age_str
            
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


def id_pose(img_location, show=False):
    """Identify the face in the image."""
    img = cv2.imread(img_location)
    fd = PoseReference()
    fd.estimate(img)


if __name__ == "__main__":
    REGISTRY = {
        "fish": id_fish,
        "objects": id_objects,
        "face": id_face,
        "pose": id_pose,
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
        fns = [(fn_name, REGISTRY[fn_name]) for fn_name in args.run_comp]

    for fn_name, fn_callable in fns:
        print(f"Running {fn_name}...")
        print(fn_callable(args.img, show=args.show))
        print()