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


class TemplateMatch:
    def __init__(self, image: cv2.typing.MatLike, template: cv2.typing.MatLike, method: int = cv2.TM_CCOEFF_NORMED):
        res = cv2.matchTemplate(image, template, method)
        loc = cv2.minMaxLoc(res)
        self.conf = float(loc[1])
        self.x = int(loc[3][0])
        self.y = int(loc[3][1])
        self.x_center = int(self.x + template.shape[1] / 2)
        self.y_center = int(self.y + template.shape[0] / 2)
