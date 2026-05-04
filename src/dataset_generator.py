import json
import math
import os
import re
import time
import queue
import multiprocessing as mp

import cv2
import numpy as np

from src.utils import BatchTemplateMatch, imread


def raise_for_ratio(image: cv2.typing.MatLike, target_ratio: float = 16 / 9, *, tolerance: float = 0.01):
    """Raises an error if the image aspect ratio differs from the target ratio (with tolerance)."""
    actual_ratio = image.shape[1] / image.shape[0]
    if abs(actual_ratio - target_ratio) / target_ratio > tolerance:
        raise ValueError("Unexpected image ratio")


def debug_show_image(image: cv2.typing.MatLike, title: str = "Image", *, scale: int = 1):
    """Shows the image in a window for debugging purposes (with optional scaling)."""
    cv2.imshow(
        title,
        cv2.resize(
            image,
            (image.shape[1] * scale, image.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        ),
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compute_inscribed_rect(size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    side = int(round(min(width, height) / math.sqrt(2)))
    left = (width - side) // 2
    right = width - side - left
    up = (height - side) // 2
    down = height - side - up
    return (up, right, down, left)


def crop_to_inscribed_rect(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    up, right, down, left = AVATAR_INSCRIBED_RECT_CROP
    h, w = image.shape[:2]
    if up + down >= h or left + right >= w:
        return image
    return image[up : h - down, left : w - right]


AVATARS_DIR = "assets/avatars"
AVATAR_MARGIN_CROP = (9, 9, 9, 9)
AVATAR_RESIZE = (102, 102)
AVATAR_INSCRIBED_RECT_CROP = compute_inscribed_rect(AVATAR_RESIZE)

AVATARS: dict[str, cv2.typing.MatLike] = {
    os.path.splitext(filename)[0]: imread(
        os.path.join(AVATARS_DIR, filename),
        crop=AVATAR_MARGIN_CROP,
        resize=AVATAR_RESIZE,
    )
    for filename in os.listdir(AVATARS_DIR)
    if filename.endswith(".png")
}
"""
To load the correct avatar images, we did the following preprocessing steps:
1. Crop the avatar image with a fixed margin to remove the existing box-shadow.
2. Resize the avatar image to the size we observed in the game screenshots.
"""

AVATARS_FOR_MARCHING: dict[str, cv2.typing.MatLike] = {
    name: cv2.GaussianBlur(crop_to_inscribed_rect(img), (3, 3), 0.0) for name, img in AVATARS.items()
}
"""
To make the loaded avatar images suitable for template matching, we did the following preprocessing steps:
1. Crop the avatar image into an inscribed square to remove the irrelevant background.
2. Apply gaussian blur to reduce noises.
"""

NUMBERS_DIR = "assets/numbers"

NUMBERS = {i: imread(os.path.join(NUMBERS_DIR, f"number_{i}.png"), flags=cv2.IMREAD_GRAYSCALE) for i in range(10)}

NUMBER_LUT = np.array(
    [0 if v < 192 else 255 if v >= 224 else (v - 192) * 8 for v in range(256)],
    dtype=np.uint8,
)

ROUNDS_DIR = "assets/rounds"

ROUNDS_MAX = 10

ROUNDS = {
    i: imread(os.path.join(ROUNDS_DIR, f"round_{i}.png"), flags=cv2.IMREAD_GRAYSCALE) for i in range(1, ROUNDS_MAX + 1)
}

SCORES_DIR = "assets/scores"

SCORES = [
    ("increase", imread(os.path.join(SCORES_DIR, "score_increase.png"))),  # Player score increased
    ("decrease", imread(os.path.join(SCORES_DIR, "score_decrease1.png"))),  # Player score decreased but not zero
    ("decrease", imread(os.path.join(SCORES_DIR, "score_decrease2.png"))),  # Player score decreased to zero and failed
    ("same", imread(os.path.join(SCORES_DIR, "score_same.png"))),  # Player wait-and-see with no score change
    ("skip", imread(os.path.join(SCORES_DIR, "score_skip.png"))),  # Player already failed
]


class GameRoundRecognizer:
    WORKING_HEIGHT = 1080
    WORKING_RATIO = 16 / 9

    AVATAR_H = 0.095  # Old value: 0.105
    AVATAR_W = 0.053  # Old value: 0.059
    AVATAR_Y = 0.858  # Old value: 0.843
    AVATAR_GAP = 0.057  # Old value: 0.063

    NUMBER_H = 0.020  # Old value: 0.026
    NUMBER_W = 0.047  # Old value: 0.057
    NUMBER_Y = 0.936  # Old value: 0.929

    GROUP_X_1 = 0.271  # Old value: 0.246
    GROUP_X_2 = 0.562  # Old value: 0.569

    def __init__(self, screen: cv2.typing.MatLike):
        raise_for_ratio(screen, target_ratio=self.WORKING_RATIO)
        self._height = round(self.WORKING_HEIGHT)
        self._width = round(self.WORKING_HEIGHT * self.WORKING_RATIO)
        self._screen = cv2.resize(screen, (self._width, self._height))

    @staticmethod
    def _recognize_avatar(image: cv2.typing.MatLike, min_conf: float = 0.5):
        if image is None or image.size == 0:
            return None

        avatar_img = cv2.resize(image, AVATAR_RESIZE, interpolation=cv2.INTER_AREA)
        avatar_img = crop_to_inscribed_rect(avatar_img)

        batch = BatchTemplateMatch(avatar_img, AVATARS_FOR_MARCHING, cv2.TM_SQDIFF_NORMED)
        assert batch.min_match is not None and batch.min_match_key is not None
        if 1.0 - batch.min_match.conf >= min_conf:
            # print(f"Best match {batch.min_match.key} with confidence {best_conf:.2f}")
            return str(batch.min_match_key)
        return None

    @staticmethod
    def _recognize_number(image: cv2.typing.MatLike, min_similarity: float = 0.5, ignore_first_char: bool = True):
        if image is None or image.size == 0:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.LUT(gray, NUMBER_LUT)

        trim_threshold = 1.5
        min_width = 4

        row_sums = np.sum(gray / 255.0, axis=1, dtype=np.float32)
        non_empty_rows = np.where(row_sums > trim_threshold)[0]
        if len(non_empty_rows) == 0:
            return None
        gray = gray[non_empty_rows[0] : non_empty_rows[-1] + 1, :]

        column_sums = np.sum(gray / 255.0, axis=0, dtype=np.float32)
        binary_hist = column_sums > trim_threshold

        char_bounds = []
        in_char = False
        start = 0
        for x, val in enumerate(np.append(binary_hist, 0.0)):
            if val and not in_char:
                start = x
                in_char = True
            elif not val and in_char:
                end = x
                if end - start >= min_width:
                    char_bounds.append((start, end))
                in_char = False

        if char_bounds and ignore_first_char:
            char_bounds = char_bounds[1:]

        recognized_digits = []
        for start, end in char_bounds:
            char_img = gray[:, start:end]
            batch = BatchTemplateMatch(char_img, NUMBERS, size_fit=True)
            assert batch.max_match is not None and batch.max_match_key is not None
            # debug_show_image(char_img, title=f"{match.max_match.key}: {match.max_match.conf:.4f}", scale=32)
            if batch.max_match.conf >= min_similarity:
                recognized_digits.append(str(batch.max_match_key))

        if not recognized_digits:
            return None
        return int("".join(recognized_digits))

    def _get_cropped_screen(self, x: float, y: float, w: float, h: float):
        x = round(self._width * x)
        y = round(self._height * y)
        w = round(self._width * w)
        h = round(self._height * h)
        return self._screen[y : y + h, x : x + w]

    def get_member_data(self):
        # returns {member_name -> member_quantity}
        result = [{}, {}]
        for group_i, group_x in enumerate([self.GROUP_X_1, self.GROUP_X_2]):
            for member_i in range(3):
                avatar_img = self._get_cropped_screen(
                    group_x + member_i * self.AVATAR_GAP,
                    self.AVATAR_Y,
                    self.AVATAR_W,
                    self.AVATAR_H,
                )
                member_name = self._recognize_avatar(avatar_img)
                if member_name is None:
                    continue

                number_img = self._get_cropped_screen(
                    (group_x + member_i * self.AVATAR_GAP) + self.AVATAR_W * (-0.5 if group_i % 2 else 0.5),
                    self.NUMBER_Y,
                    self.NUMBER_W,
                    self.NUMBER_H,
                )
                number_value = self._recognize_number(number_img)
                if number_value is None:
                    continue

                result[group_i][member_name] = number_value
        return tuple(result)

    def debug_get_visualized_member_data(self):
        canvas_height = 600
        canvas_width = 1200
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Constants for layout
        col_width = 200
        row_height = 100
        text_height = 50
        img_size = (100, 100)

        for group_i, group_x in enumerate([self.GROUP_X_1, self.GROUP_X_2]):
            x_offset = 0 if group_i == 0 else 600
            for member_i in range(3):
                avatar_img = self._get_cropped_screen(
                    group_x + member_i * self.AVATAR_GAP,
                    self.AVATAR_Y,
                    self.AVATAR_W,
                    self.AVATAR_H,
                )

                number_img = self._get_cropped_screen(
                    (group_x + member_i * self.AVATAR_GAP) + self.AVATAR_W * (-0.5 if group_i % 2 else 0.5),
                    self.NUMBER_Y,
                    self.NUMBER_W,
                    self.NUMBER_H,
                )

                member_name = self._recognize_avatar(avatar_img)
                number_number = self._recognize_number(number_img)

                y = 0
                col_x = x_offset + member_i * col_width

                # First row: original avatar and OCR image
                member_resized = cv2.resize(avatar_img, img_size)
                number_resized = cv2.resize(number_img, img_size)
                canvas[y : y + row_height, col_x : col_x + img_size[1]] = member_resized
                canvas[y : y + row_height, col_x + img_size[1] : col_x + col_width] = number_resized
                y += row_height

                if member_name is None:
                    continue

                # Second row: recognized avatar from file
                avatar_path = os.path.join(AVATARS_DIR, member_name + ".png")
                if os.path.exists(avatar_path):
                    avatar_img = imread(avatar_path)
                    avatar_resized = cv2.resize(avatar_img, img_size)
                    canvas[y : y + row_height, col_x : col_x + img_size[1]] = avatar_resized
                y += row_height

                # Third row: recognized ID and number as text
                text = f"{member_name}: {number_number}"
                cv2.putText(canvas, text, (col_x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y += text_height

        return canvas


class GameRankRecognizer:
    WORKING_HEIGHT = 1080
    WORKING_RATIO = 16 / 9

    SCORE_AREA_X = 0.486
    SCORE_AREA_Y = 0.129  # Old value: 0.144
    SCORE_AREA_W = 0.036
    SCORE_AREA_H = 0.066  # Old value: 0.072

    ROUND_AREA_X = 0.515
    ROUND_AREA_Y = 0.019
    ROUND_AREA_W = 0.091
    ROUND_AREA_H = 0.051

    MAX_RANK = 8
    MAX_ROUND = 10

    def __init__(self, screen: cv2.typing.MatLike):
        raise_for_ratio(screen, target_ratio=self.WORKING_RATIO)
        self._height = round(self.WORKING_HEIGHT)
        self._width = round(self.WORKING_HEIGHT * self.WORKING_RATIO)
        self._screen = cv2.resize(screen, (self._width, self._height))

    def _get_cropped_screen(self, x: float, y: float, w: float, h: float):
        x = round(self._width * x)
        y = round(self._height * y)
        w = round(self._width * w)
        h = round(self._height * h)
        return self._screen[y : y + h, x : x + w]

    def get_rank_data(self):
        result = {
            "game_round": 0,
            "human_correct": 0,
            "human_wrong": 0,
            "human_neutral": 0,
        }

        round_img = self._get_cropped_screen(
            self.ROUND_AREA_X,
            self.ROUND_AREA_Y,
            self.ROUND_AREA_W,
            self.ROUND_AREA_H,
        )
        round_img = cv2.cvtColor(round_img, cv2.COLOR_BGR2GRAY)
        batch = BatchTemplateMatch(round_img, ROUNDS)
        assert batch.max_match is not None and isinstance(batch.max_match_key, int)
        best_round = batch.max_match_key
        if not (0 < best_round <= self.MAX_ROUND):
            raise ValueError("Failed to recognize game round")

        result["game_round"] = best_round

        my_key = ""
        my_sat = None
        for r in range(1, self.MAX_RANK + 1):
            score_img = self._get_cropped_screen(
                self.SCORE_AREA_X,
                self.SCORE_AREA_Y + (r - 1) * self.SCORE_AREA_H,
                self.SCORE_AREA_W,
                self.SCORE_AREA_H,
            )

            batch = BatchTemplateMatch(score_img, SCORES)
            assert batch.max_match_key is not None
            score_delta_key = batch.max_match_key

            key = ""
            if score_delta_key == "increase":
                key = "human_correct"  # Player correct
            elif score_delta_key == "decrease":
                key = "human_wrong"  # Player wrong
            elif score_delta_key == "same":
                key = "human_neutral"  # Player wait-and-see
            else:
                pass  # Player already failed or unrecognized

            # debug_show_image(score_img, title=f"Rank {r}: {score_delta_key}", scale=8)

            saturation = cv2.cvtColor(score_img, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
            if my_sat is None or saturation > my_sat:
                # Highest saturation is my rank, other ranks are opponent players
                if my_key:
                    # Previously recorded is opponent player rank
                    result[my_key] += 1
                # My rank should not be recorded in the result, so keep it
                my_sat = saturation
                my_key = key
            else:
                # Surely opponent player rank
                if key:
                    result[key] += 1

        return result


class DatasetGenerator:
    VERSION_NAME = "IvyVine"

    def __init__(self, image_dir: str, *, include_ranking: bool = False, num_processes: int = 1):
        self._image_dir = image_dir
        self._include_ranking = include_ranking
        self._num_processes = max(1, num_processes)
        self._build_map()

    def _build_map(self):
        classes = sorted(list(AVATARS.keys()))
        self._id_map = {classname: i for i, classname in enumerate(classes)}
        self._name_map = {i: classname for classname, i in enumerate(classes)}

    def _classname_to_id(self, classname: str):
        if classname not in self._id_map:
            raise ValueError(f"Unknown classname: {classname}")
        return self._id_map[classname]

    def parse(self, image_path: str):
        lower_basename = os.path.basename(image_path).lower()
        result = {}
        if lower_basename.startswith("l_"):
            winner = 0
        elif lower_basename.startswith("r_"):
            winner = 1
        else:
            raise ValueError("Unknown filename format")
        result.update(self._parse_common(image_path, winner))
        if self._include_ranking:
            eval_image_path = os.path.join(self._image_dir, "Rank_" + os.path.basename(image_path))
            if os.path.isfile(eval_image_path):
                result.update(self._parse_eval(eval_image_path))
            else:
                raise FileNotFoundError(f"Eval image not found: {eval_image_path}")
        return result

    def _parse_common(self, image_path: str, winner: int):
        recognizer = GameRoundRecognizer(imread(image_path))
        member_data = recognizer.get_member_data()

        if len(member_data[0]) == 0 or len(member_data[1]) == 0:
            debug_show_image(
                recognizer.debug_get_visualized_member_data(),
                title=os.path.basename(image_path),
            )
            raise ValueError("Failed to recognize member data from one or both groups")

        return {
            "groups": [
                {str(self._classname_to_id(name)): quantity for name, quantity in member_data[0].items()},
                {str(self._classname_to_id(name)): quantity for name, quantity in member_data[1].items()},
            ],
            "winner": winner,
        }

    def _parse_eval(self, image_path: str):
        recognizer = GameRankRecognizer(imread(image_path))
        rank_data = recognizer.get_rank_data()
        if rank_data["human_correct"] + rank_data["human_wrong"] + rank_data["human_neutral"] == 0:
            raise ValueError("No human player rank data recognized")
        return rank_data

    def _mp_worker(self, task_queue: mp.Queue, result_queue: mp.Queue):
        generator = DatasetGenerator(self._image_dir, include_ranking=self._include_ranking)
        while True:
            filename = task_queue.get()
            if filename is None:
                break
            try:
                image_path = os.path.join(self._image_dir, filename)
                entry = generator.parse(image_path)
                result_queue.put((filename, entry, None))
            except Exception as e:
                result_queue.put((filename, None, str(e)))

    def generate(self):
        dataset = {
            "version": self.VERSION_NAME,
            "names": self._name_map,
            "data": [],
        }

        filelist = os.listdir(self._image_dir)
        filelist = filter(lambda x: re.match(r"^[lr]_.+\.(png|jpg|jpeg)$", x, re.IGNORECASE), filelist)
        filelist = sorted(filelist, key=lambda x: "".join(reversed(x)))  # Pseudo random order

        task_queue = mp.Queue()
        result_queue = mp.Queue()
        process_list: list[mp.Process] = []
        parsed_entries: dict[str, dict] = {}

        for _ in range(self._num_processes):
            p = mp.Process(target=self._mp_worker, args=(task_queue, result_queue))
            p.start()
            process_list.append(p)

        for filename in filelist:
            task_queue.put(filename)

        for _ in range(self._num_processes):
            task_queue.put(None)

        force_terminated = False
        for _ in filelist:
            while not force_terminated:
                try:
                    filename, entry, err = result_queue.get(timeout=0.1)
                    if err is None:
                        print(f"Parsed {filename}: {entry}")
                        parsed_entries[filename] = entry
                    else:
                        print(f"Failed to parse {filename}: {err}")
                    break
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    print("Interrupted by user, terminating...")
                    for p in process_list:
                        p.terminate()
                    task_queue.cancel_join_thread()
                    result_queue.cancel_join_thread()
                    force_terminated = True

        if not force_terminated:
            for p in process_list:
                p.join()

        dataset["data"] = [parsed_entries[name] for name in filelist if name in parsed_entries]

        return dataset


def main(image_dir: str, output_path: str | None = None, *, include_ranking: bool = False, num_processes: int = 1):
    if not os.path.isdir(image_dir):
        raise ValueError("Given image_dir not found!")

    print("Start generating...")
    print(f"- Contain eval data: {include_ranking}")
    print(f"- Num processes: {num_processes}")
    t0 = time.perf_counter()
    generator = DatasetGenerator(image_dir, include_ranking=include_ranking, num_processes=num_processes)
    dataset = generator.generate()
    t1 = time.perf_counter()
    print(f"Parsed {len(dataset.get('data', []))} entries in {t1 - t0:.1f} seconds")

    if output_path is None:
        os.makedirs("dataset", exist_ok=True)
        output_path = f"dataset/dataset_{int(time.time())}.json"

    print(f"Writing to file... ({output_path})")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=None, separators=(",", ":"))
    print("Done!")
    return dataset
