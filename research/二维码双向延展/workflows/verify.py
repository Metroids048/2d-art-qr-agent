import json
import subprocess
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def decode_file(relative_path, destination, vision, scratch):
    path = ROOT / relative_path
    native_path = path
    with Image.open(path) as original:
        if "A" in original.getbands() or "transparency" in original.info:
            rgba = original.convert("RGBA")
            image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
            native_path = scratch / f"{path.parent.name}-{path.stem}-white.png"
            image.save(native_path)
        else:
            image = original.convert("RGB")

    variants = [("原尺寸", native_path, image)]
    for side in (1200, 900, 720, 480):
        if side >= max(image.size):
            continue
        resized = ImageOps.contain(image, (side, side), Image.Resampling.LANCZOS)
        sample_path = scratch / f"{path.parent.name}-{path.stem}-{side}.png"
        resized.save(sample_path)
        variants.append((f"长边 {side} px", sample_path, resized))

    native = json.loads(
        subprocess.check_output([str(vision), *[str(sample) for _, sample, _ in variants]])
    )
    sizes = []
    for label, sample, raster in variants:
        vision_urls = native[str(sample)]
        result = {"size": list(raster.size), "variant": label}
        try:
            pixels = cv2.cvtColor(np.array(raster), cv2.COLOR_RGB2BGR)
            opencv_url = cv2.QRCodeDetector().detectAndDecode(pixels)[0]
        except cv2.error as error:
            opencv_url = ""
            result["opencv_error"] = str(error)
        result.update(
            vision=vision_urls,
            vision_match=destination in vision_urls,
            opencv=opencv_url,
            opencv_match=destination == opencv_url,
        )
        sizes.append(result)

    return {
        "file": str(relative_path),
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "expected": destination,
        "sizes": sizes,
    }
