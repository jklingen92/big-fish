

import argparse
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


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


def measure_fish(img, show=False):
    """Measure the fish in the image."""
    pass


if __name__ == "__main__":
    REGISTRY = {
        "fish": id_fish,
        "objects": id_objects,
        "face": id_face,
        "measurement": measure_fish,
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
        fns = [(fn_name, REGISTRY[fn_name]) for fn_name in args.fun_comp]

    for fn_name, fn_callable in fns:
        print(f"Running {fn_name}...")
        print(fn_callable(args.img, show=args.show))
        print()