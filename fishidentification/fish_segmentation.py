import os
import sys
import cv2
import numpy as np
from .models.segmentator_fpn_res18_416_1.inference import Inference

# Base directory of this file (fishidentification/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to the model.ts file
DEFAULT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "segmentator_fpn_res18_416_1",
    "model.ts",
)

_SEGMENTOR = None


def _get_segmentor(model_path: str = DEFAULT_MODEL_PATH) -> Inference:
    """
    Lazily create and cache the Inference object.
    """
    global _SEGMENTOR
    if _SEGMENTOR is None:
        print(f"[fish_segmentation] Loading model from {model_path}...")
        _SEGMENTOR = Inference(model_path=model_path, image_size=416, threshold=0.5)
    return _SEGMENTOR


def segment_fish(image_or_path, model_path: str = DEFAULT_MODEL_PATH):
    """
    Run segmentation on an image and return the polygon points.

    Args:
        image_or_path: Either:
            - a NumPy image array (H, W, 3), or
            - a string path to the image.
        model_path (str): Path to the TorchScript model.

    Returns:
        list[tuple[int, int]] | None:
            A list of (x, y) points defining the fish polygon,
            or None if no polygon was found.
    """
    seg = _get_segmentor(model_path)

    # Handle both array input and path input
    if isinstance(image_or_path, np.ndarray):
        img = image_or_path
    else:
        img = cv2.imread(image_or_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_or_path}")

    polygons = seg.predict(img)  # list of FishialPolygon

    if not polygons:
        return None

    poly = polygons[0]
    return poly.points



def main():
    """
    CLI entrypoint: python fish_segmentation.py <image_path> [<model_path>]
    """
    if len(sys.argv) not in (2, 3):
        print("Usage: python fish_segmentation.py <image_path> [<model_path>]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) == 3 else DEFAULT_MODEL_PATH

    try:
        points = segment_fish(image_path, model_path=model_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    if points is None:
        print("No polygon returned.")
        sys.exit(0)

    print("Polygon points (first 10):", points[:10])
    print(f"Total points: {len(points)}")


if __name__ == "__main__":
    main()




