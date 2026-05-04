import random

import cv2
from PIL import Image

from maa.context import Context


class ContextToolkit:
    @staticmethod
    def post_jittered_click(context: Context, x: int, y: int, *, jitter: int) -> None:
        x += random.randint(-jitter, jitter)
        y += random.randint(-jitter, jitter)
        context.tasker.controller.post_click(x, y).wait()

    @staticmethod
    def get_screenshot(context: Context, *, force_refresh: bool = False) -> Image.Image:
        if force_refresh:
            context.tasker.controller.post_screencap().wait()
        raw_image: cv2.typing.MatLike = context.tasker.controller.cached_image
        rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image, mode="RGB")
