

import argparse
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

class FaceDetector:
    """
    FaceDetector will identify faces in an image along with their 
    predicted age and gender.
    
    The code in this file is drawn from Gil Levi and Tal Hassner's 
    repository: https://github.com/GilLevi/AgeGenderDeepLearning
    """

    def __init__(self):
        face_proto = "weights/deploy.prototxt"
        gender_proto = "weights/deploy_gender.prototxt"
        age_proto = "weights/deploy_age.prototxt"
        face_model = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        gender_model = "weights/gender_net.caffemodel"
        age_model = "weights/age_net.caffemodel"
        
        self.mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        self.gender_list = ["Male", "Female"]
        self.age_interval_list = [
            (0, 2), (4, 6), (8, 12), (15, 20),
            (25, 32), (38, 43), (48, 53), (60, 100)
        ]
        self.face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
        self.gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
        self.age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)

    def identify(self, img):
        faces = self.get_faces(img)
        for face_coords in faces:
            start_x, start_y, end_x, end_y = face_coords
            face = img[start_y:end_y, start_x:end_x]
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
            gender, gender_confidence = self.get_gender(face)
            age, age_confidence = self.get_age(face)
            print(f"Found a {gender} ({gender_confidence * 100:.2f}%) between {age[0]} and {age[1]} years old ({age_confidence * 100:.2f}%) ")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
            
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

    def get_gender(self, face):
        
        blob = cv2.dnn.blobFromImage(
            image=face, 
            scalefactor=1.0, 
            size=(227, 227), 
            mean=self.mean_values, 
            swapRB=False, 
            crop=False
        )
        self.gender_net.setInput(blob)

        gender_preds = self.gender_net.forward()
        i = gender_preds[0].argmax()
        gender = self.gender_list[i]
        gender_confidence = gender_preds[0][i]
        return gender, gender_confidence
    
    def get_age(self, face):
        blob = cv2.dnn.blobFromImage(
            image=face, 
            scalefactor=1.0, 
            size=(227, 227),
            mean=self.mean_values,
            swapRB=False
        )
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        i = age_preds[0].argmax()
        age = self.age_interval_list[i]
        age_confidence = age_preds[0][i]
        return age, age_confidence


def id_face(img_location, show=False):
    """Identify the face in the image."""
    img = cv2.imread(img_location)
    fd = FaceDetector()
    fd.identify(img)


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