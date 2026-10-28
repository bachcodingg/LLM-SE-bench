"""
Record docs/assets/demo.gif: the demo in a terminal, then the dashboard.

Frames are rendered, not screen-captured, so the GIF is reproducible and
needs no recording tool:

1. ``python scripts/demo.py`` is run for real and its output drawn as a
   terminal, a line at a time.
2. The dashboard's refactoring page is screenshotted with headless Chrome.

    python scripts/dev/record_demo_gif.py [--chrome PATH] [--port 8061]

Needs Docker (for the demo), Pillow, Dash and a Chrome or Edge binary.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "docs" / "assets" / "demo.gif"
SIZE = (1100, 640)
BACKGROUND = (24, 26, 33)
FOREGROUND = (214, 218, 224)
COLOURS = {
    "PASS": (126, 211, 133), "PARTIAL": (240, 190, 92), "FAIL": (236, 112, 99),
    "==": (130, 180, 240), "why": (240, 190, 92), "$": (126, 211, 133),
}
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(line: str, width: int) -> list[str]:
    """Wrap at word boundaries, continuing under the value column."""
    if len(line) <= width:
        return [line]
    indent = " " * (len(line) - len(line.lstrip()) + 11)
    return textwrap.wrap(line, width, subsequent_indent=indent, drop_whitespace=False) or [line]


def _colour(line: str) -> tuple[int, int, int]:
    stripped = line.strip()
    for key in ("PASS", "PARTIAL", "FAIL"):
        if stripped.startswith("verdict") and key in stripped:
            return COLOURS[key]
    for key in ("==", "why", "$"):
        if stripped.startswith(key):
            return COLOURS[key]
    return FOREGROUND


def terminal_frames(lines: list[str]) -> list[tuple[Image.Image, int]]:
    """(frame, duration_ms) pairs revealing *lines* one at a time."""
    font = _font(17)
    line_height = 23
    columns = 96
    wrapped = [part for line in lines for part in _wrap(line, columns)]
    visible = (SIZE[1] - 60) // line_height

    frames: list[tuple[Image.Image, int]] = []
    for shown in range(1, len(wrapped) + 1):
        image = Image.new("RGB", SIZE, BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, SIZE[0], 30], fill=(40, 43, 54))
        for i, colour in enumerate([(236, 112, 99), (240, 190, 92), (126, 211, 133)]):
            draw.ellipse([14 + i * 22, 9, 26 + i * 22, 21], fill=colour)
        draw.text((90, 6), "llm-se-bench: evaluate_patch on three refactorings of GC_001",
                  font=_font(14), fill=(150, 156, 168))
        window = wrapped[max(0, shown - visible):shown]
        for row, text in enumerate(window):
            draw.text((20, 44 + row * line_height), text, font=font, fill=_colour(text))
        last = wrapped[shown - 1].strip()
        pause = 900 if last.startswith(("verdict", "why")) else 140 if last else 60
        frames.append((image, pause))
    image, _ = frames[-1]
    frames[-1] = (image, 3500)
    return frames


def _chrome(explicit: str | None) -> str:
    for candidate in [explicit, *CHROME_CANDIDATES]:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("No Chrome or Edge binary found; pass --chrome.")


def dashboard_frame(chrome: str, port: int) -> Image.Image:
    """Screenshot of the refactoring page, served by a short-lived dashboard."""
    server = subprocess.Popen(
        [sys.executable, "-m", "framework.dashboard.app", "--port", str(port)],
        cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/refactoring"
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        # A wider viewport than the GIF, same aspect ratio, scaled down: the
        # page lays out at desktop width and nothing is cropped or squashed.
        viewport = (int(SIZE[0] * 1.5), int(SIZE[1] * 1.5))
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "dashboard.png"
            subprocess.run(
                [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                 "--force-device-scale-factor=1",
                 f"--window-size={viewport[0]},{viewport[1]}",
                 "--virtual-time-budget=15000", f"--screenshot={shot}", url],
                check=True, capture_output=True, timeout=120,
            )
            image = Image.open(shot).convert("RGB")
            image.load()
    finally:
        server.terminate()
    return image.crop((0, 0, *viewport)).resize(SIZE, Image.LANCZOS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--chrome", default=None)
    parser.add_argument("--port", type=int, default=8061)
    args = parser.parse_args(argv)

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    demo = subprocess.run(
        [sys.executable, "scripts/demo.py"], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    lines = ["$ python scripts/demo.py", *[
        line for line in demo.stdout.splitlines() if " INFO " not in line
    ]]

    frames = terminal_frames(lines)
    dashboard = dashboard_frame(_chrome(args.chrome), args.port)
    frames.append((dashboard, 6000))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    images = [frame.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for frame, _ in frames]
    images[0].save(
        OUT, save_all=True, append_images=images[1:],
        duration=[duration for _, duration in frames], loop=0, optimize=True,
    )
    print(f"{OUT.relative_to(REPO_ROOT)}: {len(frames)} frames, "
          f"{OUT.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
