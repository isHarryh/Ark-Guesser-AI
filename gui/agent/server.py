import json
import random
import sys
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent  # <project>/gui/agent/
WORK_DIR = AGENT_DIR.parent  # <project>/gui/
ROOT_DIR = AGENT_DIR.parent.parent  # <project>/


def ensure_libraries():
    """Ensure the agent can import libraries from the linked site-packages."""
    linked_site = AGENT_DIR / "site-packages"
    if linked_site.exists() and str(linked_site) not in sys.path:
        sys.path.insert(0, str(linked_site))


ensure_libraries()

from maa.agent.agent_server import AgentServer
from maa.context import Context, Tasker
from maa.custom_action import CustomAction

import cv2
from PIL import Image

MAX_ROUND_DURATION = 120
MAX_RANKING_GENERATION_DURATION = 10

_LAST_ROUND_IMAGE: Image.Image | None = None
_LAST_ROUND_READY_TIME: float | None = None
_LAST_ROUND_END_TIME: float | None = None
_LAST_ROUND_FILE_ID: str | None = None


def save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg
    image.save(path, format="JPEG", quality=90, optimize=True, subsampling=0)


def _parse_screenshot_directory(param: object) -> Path | None:
    directory = None
    if isinstance(param, dict):
        directory = param.get("screenshot_directory")
    elif isinstance(param, str):
        try:
            parsed = json.loads(param)
            if isinstance(parsed, dict):
                directory = parsed.get("screenshot_directory")
        except json.JSONDecodeError:
            directory = None

    if not isinstance(directory, str) or not directory:
        return None

    rel_path = Path(directory)
    if rel_path.is_absolute():
        return None
    return ROOT_DIR / rel_path


@AgentServer.custom_action("OnRoundReady")
class OnRoundReady(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        global _LAST_ROUND_IMAGE, _LAST_ROUND_READY_TIME, _LAST_ROUND_END_TIME, _LAST_ROUND_FILE_ID
        raw_image: cv2.typing.MatLike = context.tasker.controller.cached_image
        pil_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        _LAST_ROUND_IMAGE = Image.fromarray(pil_image)
        _LAST_ROUND_READY_TIME = time.time()
        _LAST_ROUND_END_TIME = None
        _LAST_ROUND_FILE_ID = None
        print("[OnRoundReady] Received round image")

        click_positions = [(350, 1000), (1600, 1000), (950, 700)]
        x, y = random.choice(click_positions)
        x += random.randint(-4, -4)
        y += random.randint(-4, -4)
        context.tasker.controller.post_click(x, y).wait()
        print(f"[OnRoundReady] Posted click action at {x}, {y}")
        return True


@AgentServer.custom_action("OnRoundEnd")
class OnRoundEnd(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        global _LAST_ROUND_IMAGE, _LAST_ROUND_READY_TIME, _LAST_ROUND_END_TIME, _LAST_ROUND_FILE_ID
        if _LAST_ROUND_READY_TIME is None:
            print("[OnRoundEnd] Round ready time missing, skip")
            return True

        duration = time.time() - _LAST_ROUND_READY_TIME
        if duration > MAX_ROUND_DURATION:
            print(f"[OnRoundEnd] Round duration too long, skip")
            _LAST_ROUND_IMAGE = None
            _LAST_ROUND_READY_TIME = None
            return True

        param = argv.custom_action_param.lower()
        if "left_win" in param:
            left_win = True
        elif "right_win" in param:
            left_win = False
        else:
            print("[OnRoundEnd] Unknown round result:", argv.custom_action_param)
            return True

        if _LAST_ROUND_IMAGE is None:
            print("[OnRoundEnd] Round image missing, skip")
            return True

        file_id = f"{'L' if left_win else 'R'}_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"[OnRoundEnd] Round result recorded")
        _LAST_ROUND_END_TIME = time.time()
        _LAST_ROUND_FILE_ID = file_id
        return True


@AgentServer.custom_action("OnRankingGenerated")
class OnRankingGenerated(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        global _LAST_ROUND_IMAGE, _LAST_ROUND_READY_TIME, _LAST_ROUND_END_TIME, _LAST_ROUND_FILE_ID
        if _LAST_ROUND_END_TIME is None or _LAST_ROUND_FILE_ID is None:
            print("[OnRankingGenerated] Round end info missing, skip")
            return True

        duration = time.time() - _LAST_ROUND_END_TIME
        if duration > MAX_RANKING_GENERATION_DURATION:
            print(f"[OnRankingGenerated] Ranking generation too slow, skip")
            return True

        screenshot_dir = _parse_screenshot_directory(argv.custom_action_param)
        if screenshot_dir is None:
            print("[OnRankingGenerated] Screenshot directory missing, skip saving")
            return True

        if _LAST_ROUND_IMAGE is None:
            print("[OnRankingGenerated] Round image missing, skip saving")
            return True

        round_output = screenshot_dir / f"{_LAST_ROUND_FILE_ID}.jpg"
        save_jpeg(_LAST_ROUND_IMAGE, round_output)
        print(f"[OnRankingGenerated] Saved round image to {round_output}")

        time.sleep(1)  # Ensure the ranking UI is ready
        context.tasker.controller.post_screencap().wait()
        raw_image: cv2.typing.MatLike = context.tasker.controller.cached_image
        rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        rank_image = Image.fromarray(rgb_image)
        rank_output = screenshot_dir / f"Rank_{_LAST_ROUND_FILE_ID}.jpg"
        save_jpeg(rank_image, rank_output)
        print(f"[OnRankingGenerated] Saved ranking image to {rank_output}")

        _LAST_ROUND_IMAGE = None
        _LAST_ROUND_READY_TIME = None
        _LAST_ROUND_END_TIME = None
        _LAST_ROUND_FILE_ID = None
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python server.py <socket_id>")
        print("socket_id is provided by AgentIdentifier.")
        sys.exit(1)

    socket_id = sys.argv[-1]

    Tasker.set_log_dir(WORK_DIR / "debug")
    AgentServer.start_up(socket_id)
    AgentServer.join()
    AgentServer.shut_down()


if __name__ == "__main__":
    main()
