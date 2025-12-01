import cv2
import matplotlib.pyplot as plt
import os
# Run the following commands to install
# pip install ultralytics opencv-python
# pip install git+https://github.com/ultralytics/CLIP.git
from ultralytics import YOLO



def load_model(model_name, class_list):
    """
    Load the YOLOv8-world model
    """
    print("Loading model...")
    model = YOLO(model_name)
    print(f"Setting target classes: {class_list}")
    model.set_classes(class_list)
    return model



def detect(model, image_path):
    """
    Run detection on a single image and annotate results
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    # Run detection
    print(f"Running detection on {image_path}...")
    results = model.predict(image, verbose=False)

    # Annotate
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1

            print("---Measurement---")
            print(f"Detected {label}: width={width}px, height={height}px, confidence={conf:.2f}")
            print("-----------------")

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    output_path = "output_" + image_path.split("/")[-1]
    cv2.imwrite(output_path, image)



if __name__ == "__main__":
    target_classes = ["sunglasses", "hat"]
    model = load_model("yolov8s-world.pt", target_classes)

    img_1 = "imgs/inshore-adventures-charter-fishing-st-augustine-redfish-red-drum-20.jpg"
    img_2 = "imgs/inshore-adventures-charter-fishing-st-augustine-redfish-red-drum-24.jpg"

    print("***Image 1***")
    detect(model, img_1)
    print("***Image 2***")
    detect(model, img_2)
