import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from collections.abc import Callable

MFW_REPO = "MaaXYZ/MaaFramework"
MFW_VERSION_REQUIRED = "v5.10.4"

MXU_REPO = "MistEO/MXU"
MXU_VERSION_REQUIRED = "v2.1.3"


def _get_os_keyword() -> str:
    try:
        return {"windows": "win", "linux": "linux", "darwin": "macos"}[platform.system().lower()]
    except KeyError as exc:
        raise RuntimeError(f"Unrecognized operating system: {platform.system().lower()}") from exc


def _get_arch_keyword() -> str:
    try:
        return {"amd64": "x86_64", "x86_64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}[
            platform.machine().lower()
        ]
    except KeyError as exc:
        raise RuntimeError(f"Unrecognized architecture: {platform.machine().lower()}") from exc


def _download_file(url: str, dest_path: Path) -> None:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Ark-Guesser-AI-Gui-Setup")
    with urllib.request.urlopen(req, timeout=30) as res:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(res, f)


def _get_release_asset_url(repo: str, tag_name: str, keywords: list[str]) -> tuple[str, str, str]:
    api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag_name}"
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    req = urllib.request.Request(api_url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "Ark-Guesser-AI-gui-setup")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")

    with urllib.request.urlopen(req, timeout=30) as res:
        release = res.read().decode("utf-8")

    data = json.loads(release)
    assert isinstance(data, dict), "Unexpected GitHub API response"

    if data.get("draft", False):
        raise ValueError(f"Release {tag_name} is a draft")

    assets = data.get("assets", [])
    assert isinstance(assets, list), "Unexpected GitHub API response"

    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = asset.get("name", "").lower()
        if all(k.lower() in name for k in keywords):
            release_tag_name = data.get("tag_name") or data.get("name") or tag_name
            return asset["browser_download_url"], asset["name"], str(release_tag_name)

    raise ValueError(f"No matching asset found in release {tag_name}")


def _read_version(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_version(path: Path, version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(version, encoding="utf-8")


def _is_artifact_usable(version_path: Path, artifact_path: Path, *, artifact_is_dir: bool = False) -> bool:
    if not _read_version(version_path):
        return False
    if artifact_is_dir:
        return artifact_path.is_dir() and any(artifact_path.iterdir())
    else:
        return artifact_path.is_file()


def _find_file(root: Path, names: list[str]) -> Path | None:
    lowered = [n.lower() for n in names]
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in lowered:
            return path
    return None


def _copy_tree(src: Path, dst: Path) -> None:
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for file_name in files:
            shutil.copy2(Path(root) / file_name, target_dir / file_name)
        for dir_name in dirs:
            (target_dir / dir_name).mkdir(parents=True, exist_ok=True)


def _create_directory_link(src: Path, dst: Path) -> bool:
    if dst.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    if _get_os_keyword() == "win":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise OSError(f"Failed to create junction (code {result.returncode}): {result.stderr}")
    else:
        dst.symlink_to(src, target_is_directory=True)
    return True


def _download_release_package(
    *,
    repo: str,
    tag: str,
    keywords: list[str],
    version_path: Path,
    display_name: str,
    currently_usable: bool,
    install: Callable[[Path, Path], None],
) -> None:
    print(f"Checking {display_name} release info...")
    url, filename, version = _get_release_asset_url(repo, tag, keywords)
    current_version = _read_version(version_path)
    if currently_usable and current_version == version:
        print(f"{display_name} is up to date ({tag}). Skip download.")
        return
    if current_version:
        print(f"{display_name} version change: {current_version} -> {version}")
    else:
        print(f"{display_name} version: {version}")
    print(f"Downloading {display_name}...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        download_path = tmp_path / filename
        _download_file(url, download_path)
        install(download_path, tmp_path)

    _write_version(version_path, version)
    print(f"{display_name} download complete.")


def download_mxu(gui_dir: Path, cache_dir: Path) -> None:
    mxu_name = "mxu.exe" if _get_os_keyword() == "win" else "mxu"
    version_path = cache_dir / "version_mxu.txt"
    mxu_dst = gui_dir / mxu_name

    def install(download_path: Path, tmp_path: Path) -> None:
        extract_root = tmp_path / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        print("Extracting MXU archive...")

        try:
            shutil.unpack_archive(download_path, extract_root)
        except shutil.ReadError as e:
            raise RuntimeError(f"MXU release asset cannot be unpacked") from e

        mxu_src = _find_file(extract_root, [mxu_name])
        if not mxu_src:
            raise RuntimeError(f"{mxu_name} not found in archive")
        shutil.copy2(mxu_src, mxu_dst)

    _download_release_package(
        repo=MXU_REPO,
        tag=MXU_VERSION_REQUIRED,
        keywords=["mxu", _get_os_keyword(), _get_arch_keyword()],
        version_path=version_path,
        display_name="MXU",
        currently_usable=_is_artifact_usable(version_path, mxu_dst, artifact_is_dir=False),
        install=install,
    )


def download_maafw(gui_dir: Path, cache_dir: Path) -> None:
    maafw_dir = gui_dir / "maafw"
    version_path = cache_dir / "version_maafw.txt"

    def install(download_path: Path, tmp_path: Path) -> None:
        extract_root = tmp_path / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        print("Extracting MaaFramework archive...")
        try:
            shutil.unpack_archive(download_path, extract_root)
        except shutil.ReadError as e:
            raise RuntimeError(f"MaaFramework release asset cannot be unpacked") from e

        sdk_root: Path | None = None
        for root, dir_names, _ in os.walk(extract_root):
            if "bin" in dir_names:
                sdk_root = Path(root)
                break
        if not sdk_root:
            raise RuntimeError("Cannot find MaaFramework bin directory in archive")

        bin_dir = sdk_root / "bin"
        if maafw_dir.exists():
            shutil.rmtree(maafw_dir)
        maafw_dir.mkdir(parents=True, exist_ok=True)
        _copy_tree(bin_dir, maafw_dir)

    _download_release_package(
        repo=MFW_REPO,
        tag=MFW_VERSION_REQUIRED,
        keywords=["maa", _get_os_keyword(), _get_arch_keyword()],
        version_path=version_path,
        display_name="MaaFramework",
        currently_usable=_is_artifact_usable(version_path, maafw_dir, artifact_is_dir=True),
        install=install,
    )


def prepare_deps() -> bool:
    gui_dir = Path.cwd() / "gui"
    interface_path = gui_dir / "interface.json"
    if not interface_path.exists():
        print("Cannot find MaaFramework interface.json file, please check your working directory.")
        return False

    try:
        cache_dir = gui_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            download_mxu(gui_dir, cache_dir)
        except Exception as ex:
            mxu_name = "mxu.exe" if _get_os_keyword() == "win" else "mxu"
            if _is_artifact_usable(cache_dir / "version_mxu.txt", gui_dir / mxu_name):
                print(f"MXU update failed, using cached version. Reason: {ex}")
            else:
                raise

        try:
            download_maafw(gui_dir, cache_dir)
        except Exception as ex:
            if _is_artifact_usable(cache_dir / "version_maafw.txt", gui_dir / "maafw"):
                print(f"MaaFramework update failed, using cached version. Reason: {ex}")
            else:
                raise
    except urllib.error.URLError as e:
        print(f"Network error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return False

    print("GUI dependencies are ready.")
    return True


def start_gui() -> subprocess.Popen:
    gui_dir = Path.cwd() / "gui"
    mxu_name = "mxu.exe" if _get_os_keyword() == "win" else "mxu"
    mxu_path = gui_dir / mxu_name
    if not mxu_path.exists():
        raise FileNotFoundError(f"{mxu_name} not found at {mxu_path}")

    python_dir = Path(sys.executable).resolve().parent
    python_env_root = python_dir.parent
    packages_dir = python_env_root / "Lib" / "site-packages"
    if not packages_dir.exists():
        raise FileNotFoundError(f"Python site-packages directory not found for Python environment: {packages_dir}")

    packages_link = gui_dir / "agent" / "site-packages"
    if _create_directory_link(packages_dir, packages_link):
        print(f"Linked Python packages: {packages_dir} -> {packages_link}")
    else:
        print(f"Skipped Python packages linking because link already exists: {packages_link}")

    print("Starting MXU GUI program...")
    return subprocess.Popen([str(mxu_path)], cwd=str(gui_dir))
