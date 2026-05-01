import os
import sys
import time

import cv2
import keyboard
import numpy as np
import pydirectinput as pdi
from PIL import Image, ImageGrab

from src.utils import imread, TemplateMatch

TEMPLATES_PATH = "assets/templates"

TEMPLATES = {
    "round_ready_1": imread(os.path.join(TEMPLATES_PATH, "token_round_ready_1.png")),
    "round_ready_2": imread(os.path.join(TEMPLATES_PATH, "token_round_ready_2.png")),
    "result_win": imread(os.path.join(TEMPLATES_PATH, "token_result_win.png")),
    "result_lose": imread(os.path.join(TEMPLATES_PATH, "token_result_lose.png")),
    "button_start": imread(os.path.join(TEMPLATES_PATH, "token_button_start.png")),
    "button_back": imread(os.path.join(TEMPLATES_PATH, "token_button_back.png")),
    "game_mode_2": imread(os.path.join(TEMPLATES_PATH, "token_game_mode_2.png")),
    "match_mode_3": imread(os.path.join(TEMPLATES_PATH, "token_match_mode_3.png")),
}


def screenshot(resize: tuple = (1920, 1080)) -> cv2.typing.MatLike:
    screen_pil = ImageGrab.grab()
    screen = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2BGR)
    screen = cv2.resize(screen, resize, interpolation=cv2.INTER_AREA)
    return screen


def click(x: int, y: int, *, jitter: int = 4, reset_x: int = -100, reset_y: int = -100):
    rand_x = x + np.random.randint(-jitter, jitter + 1)
    rand_y = y + np.random.randint(-jitter, jitter + 1)
    pdi.moveTo(rand_x, rand_y)
    pdi.leftClick(rand_x, rand_y)
    reset_rand_x = reset_x + np.random.randint(-jitter, jitter + 1)
    reset_rand_y = reset_y + np.random.randint(-jitter, jitter + 1)
    pdi.moveTo(reset_rand_x, reset_rand_y)


def save_jpeg(image: cv2.typing.MatLike, path: str, *, quality: int = 90, optimize: bool = True, subsampling: int = 0):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg
    img_pil.save(path, format="JPEG", quality=quality, optimize=optimize, subsampling=subsampling)


def loop(
    save_screenshot: bool,
    auto_start: bool,
    infer: bool,
    *,
    save_screen_shot_dir: str,
    infer_dataset_path: str,
    infer_model_path: str,
):
    interval = 0.5  # seconds
    tm_threshold = 0.65  # recommended: 0.6~0.9
    allin_threshold = 1.0  # for inference
    certain_threshold = 0.6  # for inference
    min_okay_interval = 120  # seconds
    min_round_interval = 20  # seconds

    last_okay_time = time.time()
    last_round_time = 0
    last_round_screen = None

    if infer:
        print("Loading infer model...")
        import torch

        from .dataset_generator import GameRoundRecognizer
        from .dataset import RawArkGuesserDataset
        from .model import ArkGuesserModelV0

        dataset = RawArkGuesserDataset(infer_dataset_path)
        print(f"  Dataset loaded from {infer_dataset_path}, {dataset.num_classes} classes")

        model = ArkGuesserModelV0(num_classes=dataset.num_classes)
        model.load_state_dict(torch.load(infer_model_path, map_location="cpu"))
        model.eval()
        print(f"  Model loaded from {infer_model_path}")

    while not keyboard.is_pressed("q"):
        time.sleep(interval)
        screen = screenshot()

        # Data collecting operations:

        tm_round = (
            TemplateMatch(screen, TEMPLATES["round_ready_1"])
            if save_screenshot
            else TemplateMatch(screen, TEMPLATES["round_ready_2"])
        )
        if tm_round.conf > tm_threshold:
            if time.time() - last_round_time > min_round_interval:
                print(f"Round ready")
                time.sleep(1)
                last_round_screen = screenshot()
                last_round_time = last_okay_time = time.time()
                options = [(350, 1000), (1600, 1000), (950, 700)]

                if infer:
                    recognizer = GameRoundRecognizer(last_round_screen)
                    group1, group2 = recognizer.get_member_data()
                    print("  Left:", group1)
                    print("  Right:", group2)
                    x = torch.zeros((1, 2, dataset.num_classes), dtype=torch.float32)
                    for k, v in group1.items():
                        if k in dataset.names:
                            x[0, 0, dataset.names[k]] = float(v)  # type: ignore
                    for k, v in group2.items():
                        if k in dataset.names:
                            x[0, 1, dataset.names[k]] = float(v)  # type: ignore
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=-1).numpy().flatten()
                    if probs[0] > certain_threshold:
                        print(f"\033[93m  Predicted: Left wins ({probs[0]:.0%})\033[0m")
                        if probs[0] > allin_threshold:
                            click(x=options[0][0] - 200, y=options[0][1])
                        else:
                            click(x=options[0][0], y=options[0][1])
                    elif probs[1] > certain_threshold:
                        print(f"\033[93m  Predicted: Right wins ({probs[1]:.0%})\033[0m")
                        if probs[1] > allin_threshold:
                            click(x=options[1][0] + 200, y=options[1][1])
                        else:
                            click(x=options[1][0], y=options[1][1])
                    else:
                        print(f"\033[93m  Predicted: Uncertain ({probs[0]:.0%} vs {probs[1]:.0%})\033[0m")
                        click(x=options[2][0], y=options[2][1])
                else:
                    choice = np.random.choice(len(options))
                    click(x=options[choice][0], y=options[choice][1])
                    print(f"  Voted: Random option {choice + 1}")
            time.sleep(5)
            continue

        tm_win = TemplateMatch(screen, TEMPLATES["result_win"])
        tm_lose = TemplateMatch(screen, TEMPLATES["result_lose"])
        if (tm_win.conf + tm_lose.conf) / 2 > tm_threshold:
            if last_round_screen is not None:
                print(f"Round finished")
                left_win = tm_win.x < tm_lose.x
                print("\033[96m  Left won!\033[0m" if left_win else "\033[96m  Right won!\033[0m")
                if save_screenshot:
                    file_id = f"{'L' if left_win else 'R'}_{time.strftime('%Y%m%d_%H%M%S')}"
                    file_path = os.path.join(save_screen_shot_dir, f"{file_id}.jpg")
                    print(f"  Saving screenshot to {file_path}")
                    save_jpeg(last_round_screen, file_path)
                    time.sleep(5)
                    rank_file_path = os.path.join(save_screen_shot_dir, f"Rank_{file_id}.jpg")
                    save_jpeg(screenshot(), rank_file_path)
                    print(f"  Saving ranking screenshot to {rank_file_path}")
                last_round_screen = None
                last_round_time = 0
                last_okay_time = time.time()
            time.sleep(5)
            continue

        # UI operations:

        if auto_start:
            if time.time() - last_okay_time > min_okay_interval:
                print("No action detected for a while")
                time.sleep(3)
                continue

            tm_start = TemplateMatch(screen, TEMPLATES["button_start"])
            if tm_start.conf > tm_threshold:
                print(f"Start button (conf={tm_start.conf:.2f})")
                click(x=tm_start.x_center, y=tm_start.y_center)
                time.sleep(3)
                continue

            tm_back = TemplateMatch(screen, TEMPLATES["button_back"])
            if tm_back.conf > tm_threshold:
                print(f"Back button (conf={tm_back.conf:.2f})")
                click(x=tm_back.x_center, y=tm_back.y_center)
                time.sleep(3)
                continue

            tm_gm_2 = TemplateMatch(screen, TEMPLATES["game_mode_2"])
            if tm_gm_2.conf > tm_threshold:
                print(f"Game mode 2 entrance (conf={tm_gm_2.conf:.2f})")
                click(x=tm_gm_2.x_center, y=tm_gm_2.y_center)
                time.sleep(3)
                continue

            tm_mm_3 = TemplateMatch(screen, TEMPLATES["match_mode_3"])
            if tm_mm_3.conf > tm_threshold:
                print(f"Match mode 3 entrance (conf={tm_mm_3.conf:.2f})")
                click(x=tm_mm_3.x_center, y=tm_mm_3.y_center)
                time.sleep(3)
                continue


def main(
    save_screenshot: bool,
    auto_start: bool,
    infer: bool,
    *,
    save_screen_shot_dir: str = "",
    infer_dataset_path: str = "",
    infer_model_path: str = "",
):
    if save_screenshot:
        if not save_screen_shot_dir:
            print("Please specify save_screen_shot_dir when save_screenshot is True.")
            sys.exit(1)
        if not os.path.isdir(save_screen_shot_dir):
            os.makedirs(save_screen_shot_dir, exist_ok=True)

    if infer:
        if not infer_dataset_path or not infer_model_path:
            print("Please specify infer_dataset_path and infer_model_path when infer is True.")
            sys.exit(1)
        if not os.path.isfile(infer_dataset_path):
            print("Please make sure infer_dataset_path is a valid file.")
            sys.exit(1)
        if not os.path.isfile(infer_model_path):
            print("Please make sure infer_model_path is a valid file.")
            sys.exit(1)

    print("Start!")
    print(
        "  Please make sure the game window is fullscreen in the primary monitor.\n"
        "  Please make sure the monitor resolution is 1920x1080.\n"
        "  Long press 'q' to quit."
    )
    loop(
        save_screenshot,
        auto_start,
        infer,
        save_screen_shot_dir=save_screen_shot_dir,
        infer_dataset_path=infer_dataset_path,
        infer_model_path=infer_model_path,
    )
    print("Done!")
