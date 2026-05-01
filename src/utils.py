import os
import cv2


def imread(path: str, flags: int = cv2.IMREAD_COLOR_BGR) -> cv2.typing.MatLike:
    """Reads an image from the given path and returns a matrix."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not exist: {path}")
    img = cv2.imread(path, flags=flags)
    if img is None:
        raise ValueError(f"Failed to read image file: {path}")
    return img
