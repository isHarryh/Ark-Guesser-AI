import json
import os
import time
import queue
import multiprocessing as mp

import cv2
import numpy as np


def raise_for_ratio(image: cv2.Mat, target_ratio: float = 16 / 9, *, tolerance: float = 0.01):
    """Raises an error if the image aspect ratio differs from the target ratio (with tolerance)."""
    actual_ratio = image.shape[1] / image.shape[0]
    if abs(actual_ratio - target_ratio) / target_ratio > tolerance:
        raise ValueError("Unexpected image ratio")


def debug_show_image(image: cv2.Mat, title: str = "Image", *, scale: int = 1):
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


class AvatarImageFeature:
    DETECTER_INSTANCE = cv2.ORB.create(nfeatures=300, edgeThreshold=0, fastThreshold=0)
    MARCHER_INSTANCE = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def __init__(self, image: cv2.Mat):
        self.kp, self.des = self.DETECTER_INSTANCE.detectAndCompute(image, None)

    def is_empty(self):
        return self.des is None or len(self.des) == 0

    def get_matches(self, other: "AvatarImageFeature"):
        return self.MARCHER_INSTANCE.match(self.des, other.des)

    def get_matches_confidence(self, other: "AvatarImageFeature", max_distance: float = 50.0):
        matches = self.get_matches(other)
        if not matches:
            return 0.0
        good_matches = [m for m in matches if m.distance <= max_distance]
        return len(good_matches) / len(matches) if matches else 0.0


AVATARS_DIR = "assets/avatars"

AVATARS: dict[str, AvatarImageFeature] = {
    os.path.splitext(filename)[0]: AvatarImageFeature(cv2.imread(os.path.join(AVATARS_DIR, filename), cv2.IMREAD_COLOR))
    for filename in os.listdir(AVATARS_DIR)
    if filename.endswith(".png")
}


class NumberImageHash:
    def __init__(self, gray: cv2.Mat):
        self.hash = self._average_hash(gray)

    @staticmethod
    def _average_hash(gray: cv2.Mat, hash_size: int = 32):
        resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        avg = np.mean(resized)
        bits = "".join(["1" if pixel > avg else "0" for pixel in resized.flatten()])
        hex_hash = "{:0{}x}".format(int(bits, 2), len(bits) // 4)
        return hex_hash

    @staticmethod
    def _perceptual_hash(gray: cv2.Mat, hash_size: int = 8, high_freq_factor: int = 4):
        resized = cv2.resize(gray, (hash_size * high_freq_factor, hash_size * high_freq_factor))
        dct = cv2.dct(np.float32(resized))
        dct_low_freq = dct[:hash_size, :hash_size]
        dct_flatten = dct_low_freq.flatten()
        diff = dct_low_freq > np.mean(dct_flatten[1:])
        bits = "".join(["1" if v else "0" for v in diff.flatten()])
        hex_hash = "{:0{}x}".format(int(bits, 2), len(bits) // 4)
        return hex_hash

    def get_hamming_distance(self, other: "NumberImageHash"):
        return sum(c1 != c2 for c1, c2 in zip(self.hash, other.hash))

    def get_similarity(self, other: "NumberImageHash"):
        return 1 - self.get_hamming_distance(other) / max(len(self.hash), len(other.hash))


NUMBERS_DIR = "assets/numbers"

NUMBERS = {
    i: NumberImageHash(cv2.imread(os.path.join(NUMBERS_DIR, f"number_{i}.png"), cv2.IMREAD_GRAYSCALE))
    for i in range(10)
}

NUMBER_LUT = np.array(
    [0 if v < 192 else 255 if v >= 224 else (v - 192) * 8 for v in range(256)],
    dtype=np.uint8,
)


class GameRoundRecognizer:
    WORKING_HEIGHT = 1080
    WORKING_RATIO = 16 / 9

    AVATAR_H = 0.1046
    AVATAR_W = 0.0589
    AVATAR_Y = 0.8426
    AVATAR_GAP = 0.0625

    NUMBER_H = 0.0259
    NUMBER_W = 0.0573
    NUMBER_Y = 0.9287

    GROUP_X_1 = 0.2464
    GROUP_X_2 = 0.5693

    def __init__(self, screen: cv2.Mat):
        raise_for_ratio(screen, target_ratio=self.WORKING_RATIO)
        self._height = round(self.WORKING_HEIGHT)
        self._width = round(self.WORKING_HEIGHT * self.WORKING_RATIO)
        self._screen = cv2.resize(screen, (self._width, self._height))

    @staticmethod
    def _recognize_avatar(image: cv2.Mat, min_conf: float = 0.1):
        if image is None or image.size == 0:
            return None

        feat_this = AvatarImageFeature(image)
        if feat_this.is_empty():
            return None

        best_match = None
        best_conf = 0.0
        for name, feat_template in AVATARS.items():
            conf = feat_this.get_matches_confidence(feat_template)
            if conf > best_conf:
                best_conf = conf
                best_match = name

        if best_conf >= min_conf:
            # print(f"Best match {best_match} with confidence {best_conf:.2f}")
            return best_match
        return None

    @staticmethod
    def _recognize_number(image: cv2.Mat, min_similarity: float = 0.5, ignore_first_char: bool = True):
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
            char_hash = NumberImageHash(char_img)

            best_digit = None
            best_sim = 0.0
            for digit, num_hash in NUMBERS.items():
                sim = char_hash.get_similarity(num_hash)
                if sim > best_sim:
                    best_sim = sim
                    best_digit = digit

            # debug_show_image(char_img, title=f"{best_digit}: {best_sim:.4f}", scale=32)
            if best_sim >= min_similarity:
                recognized_digits.append(str(best_digit))

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
                    avatar_img = cv2.imread(avatar_path)
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

    OCR_AREA_X = 0.5000
    OCR_AREA_Y = 0.1444
    OCR_AREA_W = 0.3177
    OCR_AREA_H = 0.0722

    ROUND_OCR_AREA_X = 0.5385
    ROUND_OCR_AREA_Y = 0.0333
    ROUND_OCR_AREA_W = 0.2083
    ROUND_OCR_AREA_H = 0.0426

    MAX_RANK = 8
    MAX_ROUND = 10

    def __init__(self, screen: cv2.Mat):
        raise_for_ratio(screen, target_ratio=self.WORKING_RATIO)
        self._height = round(self.WORKING_HEIGHT)
        self._width = round(self.WORKING_HEIGHT * self.WORKING_RATIO)
        self._screen = cv2.resize(screen, (self._width, self._height))

    @staticmethod
    def _ocr(image: cv2.Mat):
        import pytesseract

        text: str = pytesseract.image_to_string(
            image, lang="eng", config="--psm 7 -c tessedit_char_whitelist=0123456789+-/"
        )
        return text.strip()

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

        round_ocr_img = self._get_cropped_screen(
            self.ROUND_OCR_AREA_X,
            self.ROUND_OCR_AREA_Y,
            self.ROUND_OCR_AREA_W,
            self.ROUND_OCR_AREA_H,
        )
        round_ocr_img = cv2.convertScaleAbs(round_ocr_img, alpha=1.5, beta=0)
        round_text = self._ocr(round_ocr_img).replace("+", "").replace("-", "").split("/")[0]
        # print(f"Detected round text: '{round_text}'")
        if round_text.isdigit():
            result["game_round"] = int(round_text)
            if not (1 <= result["game_round"] <= self.MAX_ROUND):
                raise ValueError("Unexpected OCR result, round number out of range")
        else:
            raise ValueError("Unexpected OCR result, round number not recognized")

        my_key = ""
        my_hue = None
        for r in range(1, self.MAX_RANK + 1):
            ocr_img = self._get_cropped_screen(
                self.OCR_AREA_X,
                self.OCR_AREA_Y + (r - 1) * self.OCR_AREA_H,
                self.OCR_AREA_W,
                self.OCR_AREA_H,
            )
            ocr_img = cv2.convertScaleAbs(ocr_img, alpha=1.5, beta=-64)
            left_ocr_img = ocr_img[:, : ocr_img.shape[1] // 2]
            right_ocr_img = ocr_img[:, ocr_img.shape[1] // 2 :]
            # debug_show_image(ocr_img, title=f"Rank {r} OCR", scale=4)

            score_delta_text, score_total_text = self._ocr(left_ocr_img), self._ocr(right_ocr_img)
            score_total = int(score_total_text) if score_total_text.isdigit() else -1

            key = ""
            if score_total > 0:
                if score_delta_text.startswith("+"):
                    key = "human_correct"  # Player correct
                elif score_delta_text.startswith("-"):
                    key = "human_wrong"  # Player wrong
                else:
                    key = "human_neutral"  # Player wait-and-see
            else:
                if score_delta_text.startswith("-"):
                    key = "human_wrong"  # Player all-in but wrong
                else:
                    pass  # Player has already failed

            hue = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2HSV)[:, :, 0].mean()
            if my_hue is None or hue < my_hue:
                # Lowest hue is my rank, other ranks are opponent players
                if my_key:
                    # Previously recorded is opponent player rank
                    result[my_key] += 1
                # My rank should not be recorded in the result, so keep it
                my_hue = hue
                my_key = key
            else:
                # Surely opponent player rank
                if key:
                    result[key] += 1

        return result


class DatasetGenerator:
    VERSION_NAME = "Honeydew"

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
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        recognizer = GameRoundRecognizer(image)
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
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        recognizer = GameRankRecognizer(image)
        rank_data = recognizer.get_rank_data()
        if rank_data["human_correct"] + rank_data["human_wrong"] + rank_data["human_neutral"] == 0:
            raise ValueError("No human player rank data recognized")
        return rank_data

    def _mp_worker(self, task_queue: mp.Queue, result_queue: mp.Queue):
        generator = DatasetGenerator(self._image_dir, self._include_ranking)
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

        def filterer(filename: str):
            lower_name = filename.lower()
            return lower_name.endswith(".png") and (lower_name.startswith("l_") or lower_name.startswith("r_"))

        filelist = os.listdir(self._image_dir)
        filelist = list(filter(filterer, filelist))
        filelist = sorted(filelist, key=lambda x: "".join(reversed(x)))  # Pseudo random order

        task_queue = mp.Queue()
        result_queue = mp.Queue()
        process_list: list[mp.Process] = []

        for _ in range(self._num_processes):
            p = mp.Process(target=self._mp_worker, args=(task_queue, result_queue))
            p.start()
            process_list.append(p)

        for filename in filelist:
            task_queue.put(filename)

        for _ in range(self._num_processes):
            task_queue.put(None)

        for _ in filelist:
            while True:
                try:
                    filename, entry, err = result_queue.get(timeout=0.1)
                    if err is None:
                        print(f"Parsed {filename}: {entry}")
                        dataset["data"].append(entry)
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

                    return dataset

        for p in process_list:
            p.join()

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
