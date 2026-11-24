"""Run the whole-image decoder with the engines available on this host."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent
WORKFLOW = ROOT / "research/二维码双向延展/workflows"
spec = importlib.util.spec_from_file_location("historic_verify", WORKFLOW / "verify.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
VISION = ROOT / "work" / "read-vision"
USE_VISION = sys.platform == "darwin" and os.environ.get("ARTQR_OPENCV_ONLY") != "1"
ENGINES = [f"OpenCV {cv2.__version__} QRCodeDetector"]
if USE_VISION:
    ENGINES.insert(0, "macOS Vision VNDetectBarcodesRequest")


def make_reader():
    VISION.parent.mkdir(exist_ok=True)
    if USE_VISION and not VISION.is_file():
        subprocess.run(["swiftc", str(WORKFLOW / "read_vision.swift"), "-o", str(VISION)], check=True)


def decode_source(path):
    # normalize EXIF, including portrait phone photos, before invoking either decoder
    import numpy as np
    from PIL import ImageOps
    with Image.open(path) as image, TemporaryDirectory(prefix="source-qr-", dir=VISION.parent) as temp:
        raster = ImageOps.exif_transpose(image).convert("RGBA")
        white = Image.alpha_composite(Image.new("RGBA", raster.size, "white"), raster).convert("RGB")
        white.thumbnail((1800, 1800))
        sample = Path(temp) / "source.png"
        white.save(sample)
        values = []
        if USE_VISION:
            values = json.loads(subprocess.check_output([str(VISION), str(sample)]))[str(sample)]
        pixels = cv2.cvtColor(np.asarray(white), cv2.COLOR_RGB2BGR)
        try:
            value = cv2.QRCodeDetector().detectAndDecode(pixels)[0]
        except (cv2.error, UnicodeError):
            value = ""
        return set(filter(None, [*values, value]))


def scan_files(directory, names, payload):
    with TemporaryDirectory(prefix="verify-", dir=VISION.parent) as scratch:
        return {
            name: checker.decode_file(directory.resolve() / name, payload, VISION if USE_VISION else None, Path(scratch))
            for name in names
        }
