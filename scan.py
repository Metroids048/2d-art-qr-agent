"""Use the historical whole-image reader unchanged for comparable measurements."""
import importlib.util
import json
import subprocess
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
ENGINES = ["macOS Vision VNDetectBarcodesRequest", f"OpenCV {cv2.__version__} QRCodeDetector"]


def make_reader():
    VISION.parent.mkdir(exist_ok=True)
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
            name: checker.decode_file(directory.resolve() / name, payload, VISION, Path(scratch))
            for name in names
        }
