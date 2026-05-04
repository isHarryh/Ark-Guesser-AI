import os
import cv2
from typing import Generic, TypeVar


def imread(
    path: str,
    flags: int = cv2.IMREAD_COLOR_BGR,
    *,
    crop: tuple[int, int, int, int] | None = None,
    resize: tuple[int, int] | None = None,
) -> cv2.typing.MatLike:
    """Reads an image from the given path and returns a matrix.
    The image is read as a color image by default, but you can use the flags param to specify other reading modes.

    Optionally, the image can be cropped and resized before returning.
    The crop param is a 4-int tuple (up, right, down, left) representing the inserts.
    The resize param is a 2-int tuple (width, height) representing the target size.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not exist: {path}")
    img = cv2.imread(path, flags=flags)
    if img is None:
        raise ValueError(f"Failed to read image file: {path}")

    if crop is not None:
        up, right, down, left = crop
        if up < 0 or right < 0 or down < 0 or left < 0:
            raise ValueError(f"Crop values must be non-negative: {crop}")
        h, w = img.shape[:2]
        img = img[up : h - down, left : w - right]

    if resize is not None:
        width, height = resize
        if width <= 0 or height <= 0:
            raise ValueError(f"Resize values must be positive: {resize}")
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
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


K = TypeVar("K")


class BatchTemplateMatch(Generic[TypeVar("K")]):
    def __init__(
        self,
        image: cv2.typing.MatLike,
        templates: dict[K, cv2.typing.MatLike] | list[tuple[K, cv2.typing.MatLike]],
        method: int = cv2.TM_CCOEFF_NORMED,
        *,
        size_fit: bool = False,
    ):
        if not templates:
            raise ValueError("No templates provided for matching")
        if isinstance(templates, dict):
            templates_items = list(templates.items())
        else:
            templates_items = templates

        self.min_match_key = None
        self.max_match_key = None
        self.min_match = None
        self.max_match = None

        for key, template in templates_items:
            if size_fit:
                resized = cv2.resize(
                    image,
                    (template.shape[1], template.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
                match = TemplateMatch(resized, template, method)
            else:
                match = TemplateMatch(image, template, method)

            if self.min_match is None or match.conf < self.min_match.conf:
                self.min_match_key = key
                self.min_match = match
            if self.max_match is None or match.conf > self.max_match.conf:
                self.max_match_key = key
                self.max_match = match
