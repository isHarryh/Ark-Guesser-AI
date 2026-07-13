import os
import time
from typing import NamedTuple

import cv2
import torch

from src.dataset import RawArkGuesserDataset
from src.dataset_generator import GameRoundRecognizer
from src.model import ArkGuesserModelV1, ArkGuesserModelV2
from src.utils import imread


class InferResult(NamedTuple):
    prob: float
    infer_elapsed_s: float


class InferService:
    def __init__(self, dataset_path: str, model_path: str, model_version: str = "v1"):
        self._dataset = RawArkGuesserDataset(dataset_path)
        if model_version == "v2":
            self._model = ArkGuesserModelV2(num_classes=self._dataset.num_classes)
        else:
            self._model = ArkGuesserModelV1(num_classes=self._dataset.num_classes)
        self._model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self._model.eval()

    def _build_input(self, image: cv2.typing.MatLike) -> torch.Tensor:
        recognizer = GameRoundRecognizer(image)
        group1, group2 = recognizer.get_member_data()
        if not group1 and not group2:
            raise ValueError("No member data recognized from image")

        x = torch.zeros((1, 2, self._dataset.num_classes), dtype=torch.float32)
        for name, quantity in group1.items():
            if name in self._dataset.names:
                x[0, 0, self._dataset.names[name]] = float(quantity)  # type: ignore
        for name, quantity in group2.items():
            if name in self._dataset.names:
                x[0, 1, self._dataset.names[name]] = float(quantity)  # type: ignore
        return x

    def infer(self, image: cv2.typing.MatLike) -> InferResult:
        start = time.perf_counter()
        x = self._build_input(image)
        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        elapsed = time.perf_counter() - start
        return InferResult(prob=float(probs[0]), infer_elapsed_s=elapsed)


def _collect_images(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Image path not found: {path}")

    exts = {".png", ".jpg", ".jpeg"}
    files: list[str] = []
    for name in os.listdir(path):
        file_path = os.path.join(path, name)
        if os.path.isfile(file_path) and os.path.splitext(name)[1].lower() in exts:
            files.append(file_path)
    return sorted(files)


def main(dataset_path: str, model_path: str, image_path: str, model_version: str = "v1") -> None:
    service = InferService(dataset_path, model_path, model_version=model_version)
    images = _collect_images(image_path)
    if not images:
        raise FileNotFoundError("No images found")

    for path in images:
        image = imread(path)
        result = service.infer(image)
        print(f"{os.path.basename(path)} {result.prob:.4f}")
