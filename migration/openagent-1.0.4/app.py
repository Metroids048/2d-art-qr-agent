"""Local-only upload, design selection and artifact download."""
import argparse
import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import cv2
import numpy as np
from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from PIL import Image, ImageFont, ImageOps, UnidentifiedImageError
from qrcode.exceptions import DataOverflowError

from art import (DISPLAY_SIDES, DPI, FONT, LAYOUTS, QUIET, RESAMPLE, STYLES,
                 avoid_structures, composition_options, detail_cost,
                 halftone_art, input_pixels, isolate_subject, material,
                 matrix_options, mockup, module_art, motif, nearby_artwork, normal_qr,
                 refine_resized_detail, save_proof)
from illustration import ILLUSTRATIONS, ILLUSTRATION_LABELS, TONAL_RANGES, Illustration
from scan import ENGINES, decode_source, make_reader, scan_files
from source_fusion import MATERIALS, SourceFusion

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / "jobs"
JOBS.mkdir(exist_ok=True)
ENGINE = "artqr-color/4.7.6"
app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=45 * 1024 * 1024)

FILES = {
    "motif.png": "图案稿", "fallback.png": "普通备用码", "art-qr.png": "艺术码",
    "flat.png": "完整平稿", "proof.pdf": "实际尺寸打样 PDF", "cut-guide.png": "刀线示意",
    "mockup.png": "商务预览", "beads.csv": "拼豆配色格表", "scan.json": "逐图读码 JSON",
}
DISPLAY_FILES = {f"art-qr-{side}.png": f"{side} px 艺术码" for side in DISPLAY_SIDES}
ART_FILES = ("art-qr.png", *DISPLAY_FILES)
FILES.update(DISPLAY_FILES)
INPUT_NAMES = {"image": "主参考原图", "logo": "品牌字标", "context": "实物展示图",
               "background": "场景底图", "source_qr": "上传的原码"}
READABLE = ("fallback.png", "art-qr.png", "flat.png", "mockup.png")
CENTER_STYLES = {
    "solid-centers": {},
    "tinted-centers": {"dark": 49, "light": 220},
    "bead-colors": {},
    "bead-tinted": {"dark": 49, "light": 211},
}
FORM_LAYOUTS = {"auto": "按用途匹配", **LAYOUTS}
LAYOUT_CUES = {
    "beads": ("拼豆", "熔珠"),
    "bar": ("吧台", "台牌", "收银台"),
    "shelf": ("货架", "价签", "包装"),
    "cup": ("杯形", "杯柄"),
    "arch": ("门贴", "玻璃门", "橱窗"),
    "sticker": ("贴纸", "封口贴"),
    "label": ("展签", "展品", "馆藏", "展览"),
}
DEFAULTS = {
    "url": "", "title": "", "caption": "扫码了解详情", "brief": "", "layout": "cup",
    "width": 100, "height": 112, "qr_percent": 76, "focus_x": 50, "focus_y": 50,
    "zoom": 100, "frame": "full", "detail": "tone", "invert": False,
    "scene_x": 72, "scene_y": 54, "scene_width": 45, "scene_mount": "flush",
    "scene_corners": "", "graphic": "fusion", "art_px": 720,
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(file, value):
    file.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def number(form, name, minimum, maximum):
    try:
        value = float(form.get(name, DEFAULTS[name]))
    except ValueError:
        raise ValueError(f"{name} 必须为数字。") from None
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} 的范围是 {minimum}–{maximum}。")
    return round(value, 2)


def print_corners(text):
    try:
        points = [tuple(map(float, pair.split(","))) for pair in text.split(";")]
        if len(points) != 4 or any(len(point) != 2 for point in points):
            raise ValueError
        if any(not (0 <= value <= 100) for point in points for value in point):
            raise ValueError
    except ValueError:
        raise ValueError("请在底图上依次标出左上、右上、右下和左下四角。") from None
    for index, (x, y) in enumerate(points):
        nx, ny = points[(index + 1) % 4]
        tx, ty = points[(index + 2) % 4]
        if (nx - x) * (ty - ny) - (ny - y) * (tx - nx) <= 8:
            raise ValueError("四角顺序不正确，或落位过小。请沿面材顺时针重新标记。")
    return points


def settings(form):
    values = {key: form.get(key, DEFAULTS[key]).strip()[:limit]
              for key, limit in (("title", 60), ("caption", 85), ("brief", 450))}
    values.update(
        layout=form.get("layout", "cup"), detail=form.get("detail", "tone"),
        frame=form.get("frame", DEFAULTS["frame"]),
        fusion_style=form.get("fusion_style", "auto"),
        isolate=form.get("isolate") == "1",
        invert=form.get("invert") == "1",
        width=number(form, "width", 45, 260), height=number(form, "height", 45, 260),
        qr_percent=number(form, "qr_percent", 52, 80), focus_x=number(form, "focus_x", 0, 100),
        focus_y=number(form, "focus_y", 0, 100), zoom=number(form, "zoom", 12, 100),
        scene_x=number(form, "scene_x", 0, 100), scene_y=number(form, "scene_y", 0, 100),
        scene_width=number(form, "scene_width", 20, 90),
        scene_mount=form.get("scene_mount", DEFAULTS["scene_mount"]),
        graphic=form.get("graphic", DEFAULTS["graphic"]),
        scene_corners=form.get("scene_corners", "").strip(),
    )
    if (values["layout"] not in FORM_LAYOUTS or values["detail"] not in ("tone", "relief")
            or values["frame"] not in ("crop", "full")):
        raise ValueError("请选择可用的载体和图案提取方式。")
    if values["graphic"] not in ("fusion", "illustrated", "photo", "rounded", "mosaic"):
        raise ValueError("请选择可用的码面风格。")
    if values["fusion_style"] not in ("auto", *MATERIALS):
        raise ValueError("请选择版印、釉彩或织纹。")
    if values["graphic"] == "fusion":
        art_px = number({"art_px": form.get("art_px") or DEFAULTS["art_px"]}, "art_px", 120, 2400)
        if art_px != int(art_px):
            raise ValueError("艺术码像素尺寸必须为整数。")
        values["art_px"] = int(art_px)
    if values["scene_mount"] not in ("flush", "standing", "print-paper", "print-glass"):
        raise ValueError("请选择可用的预览摆放方式。")
    if values["scene_mount"].startswith("print-"):
        values["print_corners"] = print_corners(values["scene_corners"])
    if values["layout"] == "auto":
        matches = ((kind, cue) for kind, cues in LAYOUT_CUES.items()
                   for cue in cues if cue in values["brief"])
        layout, cue = next(matches, ("label", None))
        values["layout"] = layout
        values["layout_cue"] = cue or "未匹配关键词，使用展签"
    if values["layout"] == "beads" and values["graphic"] == "illustrated":
        values["graphic"] = "photo"
    brief = values["brief"]
    if re.search(r"字样|文字|字标|字母|标志|轮廓|笔画|细线|细长|间隙|线条|纹样|纹理|浮雕|格线|枝纹", brief):
        values["image_priority"] = "strokes"
    elif re.search(r"主色|色块|配色|颜色|色调|色彩", brief):
        values["image_priority"] = "color"
    else:
        values["image_priority"] = "balanced"
    if max(values["width"], values["height"]) > 2.4 * min(values["width"], values["height"]):
        raise ValueError("宽高比请控制在 1∶2.4 至 2.4∶1，给码区和标题留出空间。")
    if values["layout"] == "round" and (
            abs(values["width"] - values["height"]) > .01 or values["qr_percent"] > 66):
        raise ValueError("圆形贴适用等宽等高尺寸，码区占比需在 66% 以内。")
    if values["layout"] == "cup" and values["width"] > values["height"] * 1.25:
        raise ValueError("杯形门贴适合正方或竖版尺寸。横版可选吧台牌、货架卡或展签。")
    if form.get("omit_print_copy") == "1":
        values.update(title="", caption="")
    elif form.get("single_preview") == "1":
        # Conversational orders print only the user's copy.
        if "caption" not in form:
            values["caption"] = ""
    else:
        values["title"] = values["title"] or LAYOUTS[values["layout"]]
    return values


def take_image(upload, label, directory):
    upload.seek(0, os.SEEK_END)
    length = upload.tell()
    upload.seek(0)
    if length > 12 * 1024 * 1024:
        raise ValueError(f"{label}请缩小到 12 MB 以内。")
    try:
        with Image.open(upload) as image:
            formats = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}
            fmt = formats.get(image.format)
            if fmt is None or getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"{label}需要一张 PNG、JPG 或 WebP。")
            if min(image.size) < 48 or image.width * image.height > 24_000_000:
                raise ValueError(f"{label}短边需至少 48 px，且总像素不超过 2400 万。")
            sample = input_pixels(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError(f"无法打开{label}。请检查文件是否完整。") from None
    name = f"{label}.{fmt}"
    upload.seek(0)
    upload.save(directory / name)
    oriented_size = sample.size
    sample.thumbnail((2200, 2200), RESAMPLE, reducing_gap=3.0)
    if label == "motif-source" and sample.size != oriented_size and sample.getchannel("A").getextrema() == (255, 255):
        sample = refine_resized_detail(sample)
    return sample, {
        "stored_as": name, "original_name": Path(upload.filename).name[:180],
        "sha256": digest(directory / name), "oriented_size_px": oriented_size,
        "working_size_px": list(sample.size),
    }


def full_payload(url, original, original_path):
    if bool(url) == bool(original):
        raise ValueError("请在网页链接与原码文件之间任选一种。")
    if original:
        values = decode_source(original_path)
        if len(values) != 1:
            raise ValueError("原码须能读取一个唯一载荷。请换一张清晰的原码，或填写原链接。")
        payload = values.pop()
    else:
        try:
            parts = urlsplit(url)
            port = parts.port
            valid = (parts.scheme in ("http", "https") and bool(parts.hostname)
                     and parts.username is None and (port is None or port > 0)
                     and not re.search(r"[\x00-\x20\x7f\\<>]|%(?![0-9a-fA-F]{2})", url)
                     and not any(char.isspace() for char in url)
                     and not parts.netloc.endswith(":"))
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("请输入完整的 http 或 https 网页链接。")
        payload = url
    if any(char in payload for char in "\r\n\0") or len(payload.encode("utf-8")) > 512:
        raise ValueError("演示支持最多 512 字节的单行文本或网页载荷。")
    return payload


def bead_sheet(path, grid, image, *, fractional=False):
    cells = len(grid) + 2 * QUIET
    pitch = image.width / cells if fractional else image.width // cells
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("row", "column", "qr_dark", "center_rgb"))
        for row in range(len(grid) + 2 * QUIET):
            for col in range(len(grid) + 2 * QUIET):
                bit = 0
                if QUIET <= row < len(grid) + QUIET and QUIET <= col < len(grid) + QUIET:
                    bit = int(grid[row - QUIET, col - QUIET])
                color = image.getpixel((int((col + .5) * pitch), int((row + .5) * pitch)))
                writer.writerow((row + 1, col + 1, bit, "#{:02x}{:02x}{:02x}".format(*color)))


def fallback(directory, payload, art_mask):
    path = directory / "fallback.png"
    best = None
    for mask in (art_mask, *(n for n in range(8) if n != art_mask)):
        candidate = normal_qr(payload, mask)
        candidate.save(path, dpi=(DPI, DPI))
        decoded = scan_files(directory, (path.name,), payload)[path.name]
        rows = decoded["sizes"]
        score = (
            int(rows[0]["vision_match"]) + int(rows[0]["opencv_match"]),
            sum(row["vision_match"] + row["opencv_match"] for row in rows),
        )
        if best is None or score > best[0]:
            best = score, mask, candidate, decoded
        if score == (2, 2 * len(rows)):
            break
    _, mask, candidate, decoded = best
    path.unlink()
    return mask, candidate, decoded


def read_render(image):
    """Preflight a complete render before writing it. The final files are independently verified."""
    if image.mode == "RGBA":
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image)
    pixels = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    try:
        return cv2.QRCodeDetector().detectAndDecode(pixels)[0]
    except (cv2.error, UnicodeError):
        return ""


def render_candidate(payload, color, ink, key, settings, finish, compositions):
    """Compare compatible encodings without changing the submitted destination."""
    beads = settings["layout"] == "beads"
    glass = settings["scene_mount"] == "print-glass"
    s = settings
    original = [0, 0, color.width]
    if key == "rich" or beads:
        placements = [(color, original)]
    elif glass:
        placements = list(nearby_artwork(color))[1:]
    else:
        placements = [avoid_structures(color, payload), (color, original)]

    def options():
        prepared = (item for item in compositions if key == "clear" or item[1] == original)
        for artwork, box, features, matrix in list(prepared)[:12]:
            for profile in ("source-fusion", "source-fusion-strong"):
                yield artwork, box, matrix, profile, features
        if not beads:
            for artwork, box in placements:
                for matrix in matrix_options(payload, artwork):
                    for profile in ("fine", "soft", "balanced"):
                        yield artwork, box, matrix, profile, None
        if beads:
            profiles = ("bead-tinted", "bead-colors") if key == "rich" else ("bead-colors",)
        elif key == "clear" and not glass:
            profiles = ("tinted-centers", "solid-centers")
        else:
            profiles = ("solid-centers",)
        matrices = list(matrix_options(payload, color))
        for profile in profiles:
            for matrix in matrices:
                yield color, original, matrix, profile, None

    def render(artwork, matrix, profile, side, features):
        if profile in CENTER_STYLES:
            style = {**STYLES[key], **CENTER_STYLES[profile]}
            if glass:
                style = {**style, "curve": 6}
            return module_art(payload, artwork, ink, style, solid=beads,
                              graphic=s["graphic"], matrix=matrix)[0]
        return halftone_art(artwork, ink, *matrix[:2], side, s["graphic"], profile, features)

    best = None
    screened = []
    guided_checks = 0
    small, large = DISPLAY_SIDES
    for artwork, box, matrix, profile, features in options():
        if features is not None and guided_checks >= 4:
            continue
        qr = render(artwork, matrix, profile, small, features)
        small_image = qr.resize((small, small), RESAMPLE)
        small_match = read_render(small_image) == payload
        if features is not None:
            if small_match:
                guided_checks += 1
                candidate = (False, 0, artwork, box, matrix, profile, small_image, features)
                # Wide native and printed surfaces tolerate a larger ink footprint than small PNGs.
                native_profile = ({"source-fusion": "fine", "source-fusion-strong": "balanced"}[profile]
                                  if s["image_priority"] == "strokes" else profile)
                guided = evaluate_candidate(candidate, large, payload, render, finish, None, native_profile)
                if guided is not None:
                    return guided[1]
        elif glass and not beads:
            cost = detail_cost(artwork, small_image, len(matrix[0]), box)
            screened.append((not small_match, cost, artwork, box, matrix, profile, small_image, features))
        else:
            screened.append((not small_match, 0, artwork, box, matrix, profile, small_image, features))
            if small_match:
                # Most layouts decode on the first render. Try the full material path right away.
                candidate = screened.pop()
                best = evaluate_candidate(candidate, large, payload, render, finish, best, glass=glass)
                if best[0] == (True, 5):
                    return best[1]
    for candidate in sorted(screened, key=lambda item: item[:2]):
        best = evaluate_candidate(candidate, large, payload, render, finish, best, glass=glass)
        if best[0] == (True, 5):
            break
    return best[1]


def evaluate_candidate(candidate, large, payload, render, finish, best, native_profile=None, *, glass=False):
    small_miss, _, artwork, box, matrix, profile, small_image, features = candidate
    screen = render(artwork, matrix, profile, large, features)
    large_image = screen.resize((large, large), RESAMPLE)
    scans = int(not small_miss) + (read_render(large_image) == payload)
    displays_matched = scans == 2
    if best is not None and (displays_matched, scans + 3) <= best[0]:
        return best
    native_profile = native_profile or profile
    if features is not None and not displays_matched:
        return None
    native = screen if native_profile == profile else render(artwork, matrix, native_profile, large, None)
    native_matched = read_render(native) == payload
    if features is not None and not native_matched:
        return None
    flat, cut, placement, scene = finish(native, len(matrix[0]))
    flat_matched = read_render(flat) == payload
    scene_matched = read_render(scene) == payload
    if features is not None and not (flat_matched and scene_matched):
        return None
    required_matched = displays_matched and native_matched and (not glass or scene_matched)
    score = (required_matched, scans + native_matched + flat_matched + scene_matched)
    if best is None or score > best[0]:
        return score, (artwork, box, matrix, profile, native_profile, native, small_image, large_image,
                       flat, cut, placement, scene)
    return best


def render_illustration(illustration, color, payload, key, settings, finish):
    method = ILLUSTRATIONS[key]
    small, large = DISPLAY_SIDES
    best = None

    def render(*_):
        return sheet

    for tonal_range in TONAL_RANGES:
        for matrix in illustration.matrices:
            sheet = illustration.draw(*matrix[:2], method, tonal_range)
            small_image = sheet.resize((small, small), RESAMPLE)
            small_miss = read_render(small_image) != payload
            if small_miss and best is not None:
                continue
            candidate = (small_miss, 0, color, [0, 0, color.width], matrix, method, small_image, None)
            best = evaluate_candidate(candidate, large, payload, render, finish, best,
                                      glass=settings["scene_mount"] == "print-glass")
            if best[0] == (True, 5):
                return best[1]
    return best[1]


def rasterize_proof(pdf, raster=None):
    command = ["pdftoppm", "-singlefile", "-r", str(DPI)]
    if raster is not None:
        command.extend(("-png", str(pdf), str(raster.with_suffix(""))))
    else:
        command.append(str(pdf))
    return subprocess.run(command, check=True, capture_output=True, timeout=30).stdout


def render_source_fusion(fusion, payload, key, finish, settings, scratch, output_proof):
    best = None
    best_score = (-1, -1, -1)
    proof = scratch / "candidate.pdf"
    for matrix, method, qr in fusion.renders(key):
        native_ok = read_render(qr) == payload
        if best_score[0] == 1 and not native_ok:
            continue
        flat, cut, placement, scene = finish(qr, len(matrix[0]))
        flat_ok = read_render(flat) == payload
        scene_ok = read_render(scene) == payload
        raster_matches = int(flat_ok) + int(scene_ok)
        # Even a readable PDF cannot improve on the current best candidate.
        if best is not None and (native_ok, raster_matches + int(native_ok), native_ok) <= best_score:
            continue
        proof_ok = False
        if native_ok:
            save_proof(proof, flat, settings["width"], settings["height"])
            with Image.open(io.BytesIO(rasterize_proof(proof))) as page:
                proof_ok = read_render(page) == payload
        score = (native_ok, raster_matches + int(proof_ok), proof_ok)
        if score > best_score:
            if native_ok:
                # Keep the exact PDF that was screened for the selected raster.
                proof.replace(output_proof)
            artwork = fusion.source
            best = (artwork, [0, 0, *artwork.size], matrix, method, method, qr,
                    None, None, flat, cut, placement, scene)
            best_score = score
        if score == (True, 3, True):
            break
    return best


def required_art_files(settings):
    return ("art-qr.png",) if settings.get("graphic") == "fusion" else ART_FILES


def generate(directory, form, uploads):
    s = settings(form)
    single = form.get("single_preview") == "1"
    inputs = directory / "inputs"
    inputs.mkdir()
    images, sources = {}, {}
    labels = {"image": "motif-source", "logo": "brand-source", "context": "display-source",
              "background": "scene-source", "source_qr": "original-qr"}
    for key, label in labels.items():
        upload = uploads.get(key)
        if upload:
            images[key], sources[key] = take_image(upload, label, inputs)
    if "image" not in images:
        raise ValueError("请上传主参考图。")
    if s.get("print_corners") and "background" not in images:
        raise ValueError("材质印刷预览需要包含空白载体的场景底图。")
    original = sources.get("source_qr")
    original_path = inputs / original["stored_as"] if original else None
    payload = full_payload(form.get("url", "").strip(), original, original_path)
    if original and form.get("require_web_url") == "1":
        full_payload(payload, None, None)
    color, extension, crop, ink, accent, region = motif(images["image"], s)
    display = images.get("context", images["image"])
    shelf_photo = (s["layout"] == "shelf" and s["width"] >= s["height"] * 1.16
                   and s["graphic"] == "fusion" and not {"logo", "context"}.intersection(images))
    cutout = isolate_subject(display, retain_pixels=shelf_photo) if (
        shelf_photo or (s["isolate"] and s["layout"] in ("label", "bar", "shelf"))) else None
    s["preserve_shelf_subject"] = shelf_photo and cutout is not None
    fusion = None
    if s["graphic"] == "fusion":
        source = images["image"].crop(region)
        source = Image.alpha_composite(Image.new("RGBA", source.size, "white"), source).convert("RGB")
        if s["invert"]:
            source = ImageOps.invert(source)
        crop = source
        fusion = SourceFusion(source, payload, accent, s["art_px"],
                              s["image_priority"], s["fusion_style"])
    illustration = (Illustration(color, payload, accent, s["image_priority"])
                    if s["graphic"] == "illustrated" else None)
    compositions = (composition_options(color, payload, s["image_priority"])
                    if s["layout"] != "beads" and s["graphic"] in ("photo", "rounded") else [])
    s["scene_mode"] = ("uploaded" if "background" in images
                       else ("object_studio" if cutout is not None else "studio"))
    s["subject_isolated"] = cutout is not None
    assets = {}
    baseline = None
    for key in ("clear",) if single else STYLES:
        folder = directory / key
        folder.mkdir()
        def finish(qr, modules):
            flat, cut, placement = material(qr, cutout or display, extension, ink, accent,
                                            images.get("logo"), s, modules, editorial=key == "clear")
            scene = mockup(flat, s["layout"], images.get("background"), s, display, accent, cutout)
            return flat, cut, placement, scene

        proof = folder / "proof.pdf"
        if fusion:
            with TemporaryDirectory(prefix="proof-preview-", dir=directory) as scratch:
                rendered = render_source_fusion(fusion, payload, key, finish, s, Path(scratch), proof)
        elif illustration:
            rendered = render_illustration(illustration, color, payload, key, s, finish)
        else:
            rendered = render_candidate(payload, color, ink, key, s, finish, compositions)
        (artwork, artwork_box, (grid, _, mask), method, native_method, qr, small_image, large_image,
         flat, cut, placement, scene) = rendered
        label = ("拼豆原色" if s["layout"] == "beads" else
                 "细节半色调" if key == "rich" else "结构避让 · 半色调")
        if fusion:
            label = MATERIALS[method.split("/")[1]]
        elif illustration:
            label = ILLUSTRATION_LABELS[method]
        elif method == "solid-centers":
            label = "加强点阵" if key == "rich" else "对比优先"
        elif method == "tinted-centers":
            label = "原色强化点阵"
        elif method == "bead-tinted":
            label = "拼豆原色 · 柔和配色"
        elif method.startswith("source-fusion"):
            label = "原图融合" if key == "rich" else "轮廓避让 · 原图融合"
        if baseline is None:
            fallback_mask, baseline, fallback_record = fallback(directory, payload, mask)
        display_files = {} if fusion else dict(zip(DISPLAY_FILES, (small_image, large_image)))
        outputs = {
            "motif.png": artwork,
            "fallback.png": baseline, "art-qr.png": qr, "flat.png": flat, "cut-guide.png": cut,
            "mockup.png": scene, **display_files,
        }
        for name, picture in outputs.items():
            picture.save(folder / name, dpi=(DPI, DPI), compress_level=3)
        if not proof.is_file():
            save_proof(proof, flat, s["width"], s["height"])
        if s["layout"] == "beads":
            bead_sheet(folder / "beads.csv", grid, qr, fractional=bool(fusion))
        records = scan_files(folder, (*READABLE[1:], *display_files), payload)
        records["fallback.png"] = fallback_record.copy()
        with TemporaryDirectory(prefix="proof-read-", dir=directory) as temp:
            raster = Path(temp) / "proof.png"
            rasterize_proof(proof, raster)
            proof_record = scan_files(raster.parent, (raster.name,), payload)[raster.name]
            proof_record["raster_sha256"] = proof_record["sha256"]
            proof_record["sha256"] = digest(proof)
            proof_record["file"] = str(proof)
            records["proof.pdf"] = proof_record
        for record in records.values():
            record["file"] = Path(record["file"]).name
        status = {}
        for name, record in records.items():
            rows = record["sizes"]
            status[name] = {
                "vision": sum(row["vision_match"] for row in rows),
                "opencv": sum(row["opencv_match"] for row in rows), "count": len(rows),
                "native": rows[0]["vision_match"] or rows[0]["opencv_match"],
                "foreign": any((set(row["vision"]) - {payload}) or
                               (row["opencv"] and row["opencv"] != payload) for row in rows),
            }
        report = {
            "candidate": key, "style": label, "engines": ENGINES, "expected": payload,
            "input_hashes": {k: info["sha256"] for k, info in sources.items()},
            "graphic": s["graphic"], "mask": mask, "fallback_mask": fallback_mask, "modules": len(grid), "padding": "standard", "placement": placement,
            "matrix_sha256": sha256(grid.tobytes()).hexdigest(),
            "color_cells_sha256": sha256(artwork.resize(grid.shape, Image.Resampling.BOX).tobytes()).hexdigest(),
            "outer_color_sha256": sha256(extension.tobytes()).hexdigest(),
            "scope": "逐图整幅原尺寸与缩放读取。透明层按白底合成。模拟图未经过物理镜头、印刷或反光测试。",
            "status": status, "files": records,
            "sha256": {p.name: digest(p) for p in sorted(folder.iterdir())},
        }
        needs_scene_opencv = s["scene_mount"] == "print-glass"
        scene_opencv = records["mockup.png"]["sizes"][0]["opencv_match"]
        art_opencv_misses = [name for name in required_art_files(s) if records[name]["sizes"][0]["opencv"] != payload]
        report.update(method=method, native_method=native_method, image_priority=s["image_priority"], display_sizes_px=[] if fusion else list(DISPLAY_SIDES),
                      artwork_box_px=artwork_box, artwork_canvas_px=artwork.width,
                      native_target_px=qr.width if fusion else DISPLAY_SIDES[-1],
                      art_opencv_misses=art_opencv_misses,
                      scene_opencv_required=needs_scene_opencv,
                      scene_opencv_matched=scene_opencv)
        write_json(folder / "scan.json", report)
        assets[key] = {
            "style": label, "status": status, "placement": placement,
            "visual_review_required": fusion is not None or illustration is not None,
            "selectable": (not art_opencv_misses and (not needs_scene_opencv or scene_opencv) and
                           all(item["native"] and not item["foreign"] for item in status.values())),
            "files": [name for name in FILES if (folder / name).is_file()],
        }
    crop.save(directory / "crop.png")
    write_json(directory / "job.json", {
        "id": directory.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": s, "inputs": sources, "payload": payload,
        "payload_source": "original-qr" if original else "url", "focal_box_px": region,
        "candidates": assets, "engine": ENGINE,
    })


def get_job(job_id):
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        abort(404)
    directory = JOBS / job_id
    if not (directory / "job.json").is_file():
        abort(404)
    job = json.loads((directory / "job.json").read_text())
    selection = directory / "selection.json"
    job["selection"] = json.loads(selection.read_text()) if selection.exists() else None
    job["selected"] = job["selection"]["candidate"] if job["selection"] else None
    return directory, job


@app.before_request
def local_requests():
    if request.host.split(":")[0] not in ("localhost", "127.0.0.1"):
        abort(403)
    if request.method == "POST" and request.headers.get("Sec-Fetch-Site") == "cross-site":
        abort(403)


@app.after_request
def local_documents(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob: data:; object-src 'none'; frame-ancestors 'none'; form-action 'self'"
    return response


@app.get("/")
def home():
    return render_template("index.html", values=DEFAULTS, layouts=FORM_LAYOUTS)


@app.errorhandler(413)
def oversized(_error):
    message = "上传总大小超过 45 MB。请缩小文件后再试。"
    if request.accept_mimetypes.best == "application/json":
        return {"error": message}, 413
    return render_template("index.html", values=DEFAULTS, layouts=FORM_LAYOUTS,
                           error=message), 413


@app.post("/jobs")
def create():
    values = {**DEFAULTS, **request.form}
    with TemporaryDirectory(prefix=".generating-", dir=JOBS) as staging:
        native = app.config.get("ARTQR_CREATE")
        identity = native["job"] if native else uuid4().hex[:12]
        target = JOBS / identity
        if native and target.is_dir():
            binding = target / "native.json"
            if not binding.is_file() or json.loads(binding.read_text()) != native:
                abort(409, description="预览与这次聊天的上传记录不一致。")
            return redirect(url_for("result", job_id=identity), 303)
        directory = Path(staging) / identity
        directory.mkdir()
        try:
            if any(len(request.form.getlist(key)) > 1 for key in request.form):
                raise ValueError("同一项只能提交一个值，请确认目标网址和设计选项。")
            if any(len(request.files.getlist(key)) > 1 for key in request.files):
                raise ValueError("每类素材只接收一张，请确认要使用的图片。")
            generate(directory, request.form, request.files)
        except (ValueError, DataOverflowError) as error:
            message = str(error) or "载荷太长，请换较短的原链接。"
            if request.accept_mimetypes.best == "application/json":
                return {"error": message}, 422
            return render_template("index.html", values=values, layouts=FORM_LAYOUTS,
                                   error=message), 422
        if native:
            write_json(directory / "native.json", native)
        os.rename(directory, target)
    return redirect(url_for("result", job_id=identity), 303)


@app.get("/jobs/<job_id>")
def result(job_id):
    directory, job = get_job(job_id)
    reports = {k: json.loads((directory / k / "scan.json").read_text()) for k in STYLES}
    return render_template("result.html", job=job, reports=reports, names=FILES,
                           input_names=INPUT_NAMES, layouts=LAYOUTS)


@app.get("/jobs/<job_id>/inputs/<field>")
def original_input(job_id, field):
    directory, job = get_job(job_id)
    source = job.get("inputs", {}).get(field)
    if field not in INPUT_NAMES or source is None:
        abort(404)
    path = directory / "inputs" / source["stored_as"]
    if not path.is_file():
        abort(404)
    if digest(path) != source["sha256"]:
        abort(409, description="原图与本次提交记录不一致。")
    return send_file(path)


@app.post("/jobs/<job_id>/select/<key>")
def select(job_id, key):
    directory, job = get_job(job_id)
    if key not in job["candidates"]:
        abort(404)
    if not job["candidates"][key]["selectable"]:
        abort(409, description="该候选的整图读码未通过，请在读码明细中查看失败文件。")
    selection = {"candidate": key, "selected_at": datetime.now(timezone.utc).isoformat()}
    binding_file = directory / "native.json"
    if binding_file.is_file():
        binding = json.loads(binding_file.read_text())
        approval = app.config.get("ARTQR_APPROVAL", {})
        if (any(approval.get(k) != binding[k] for k in ("origin", "session", "job"))
                or approval.get("candidate") != key or not approval.get("event")):
            abort(409, description="请在本次聊天里确认这版预览。")
        selection["native_confirmation"] = approval
    if requires_visual_review(job, key):
        if request.form.get("visual_review") != "1":
            abort(409, description="请先对照上传原图检查成稿，再确认选稿。")
        selection["visual_review"] = verified_art(directory, job, key)
    with TemporaryDirectory(prefix=".select-", dir=directory) as temp:
        path = Path(temp) / "selection.json"
        write_json(path, selection)
        os.replace(path, directory / "selection.json")
    return redirect(url_for("result", job_id=job_id) + "#delivery", 303)


def requires_visual_review(job, key):
    return job["settings"].get("graphic") == "fusion" or job["candidates"][key].get("visual_review_required")


def verified_art(directory, job, key):
    folder = directory / key
    report = json.loads((folder / "scan.json").read_text())
    source = job["inputs"]["image"]
    original = directory / "inputs" / source["stored_as"]
    if (not original.is_file() or digest(original) != source["sha256"]
            or report["input_hashes"].get("image") != source["sha256"]):
        abort(409, description="原图与本次选稿不一致，请重新上传。")
    hashes = {"source": source["sha256"]}
    for name in required_art_files(job["settings"]):
        path = folder / name
        record = report["files"].get(name, {})
        rows = record.get("sizes", [])
        if (not path.is_file() or not rows or rows[0]["opencv"] != job["payload"]
                or digest(path) != record.get("sha256")):
            abort(409, description="艺术码与本次 OpenCV 原尺寸读码记录不一致，请重新生成。")
        hashes[name] = record["sha256"]
    return hashes


def require_downloadable(directory, job, key):
    if not key or job["selected"] != key or not job["candidates"][key]["selectable"]:
        abort(409, description="请先选定通过读码的一稿。")
    binding_file = directory / "native.json"
    if binding_file.is_file():
        binding = json.loads(binding_file.read_text())
        approval = job["selection"].get("native_confirmation", {})
        if (any(approval.get(k) != binding[k] for k in ("origin", "session", "job"))
                or approval.get("candidate") != key or not approval.get("event")):
            abort(409, description="请选择并确认本次聊天的候选。")
    if (requires_visual_review(job, key)
            and job["selection"].get("visual_review") != verified_art(directory, job, key)):
        abort(409, description="请先核对并确认本次选稿对应的原图和成稿。")


@app.get("/jobs/<job_id>/crop.png")
def crop(job_id):
    directory, _job = get_job(job_id)
    return send_file(directory / "crop.png")


@app.get("/jobs/<job_id>/asset/<key>/<name>")
def asset(job_id, key, name):
    directory, job = get_job(job_id)
    if key not in STYLES or name not in job["candidates"][key]["files"]:
        abort(404)
    attachment = request.args.get("download") == "1"
    protected = attachment or name in ("proof.pdf", "beads.csv")
    if protected:
        require_downloadable(directory, job, key)
    path = directory / key / name
    if protected and name != "scan.json":
        report = json.loads((directory / key / "scan.json").read_text())
        if digest(path) != report["sha256"][name]:
            abort(409, description="文件与本次读码记录不一致。请重新生成。")
    return send_file(path, as_attachment=attachment)


@app.get("/jobs/<job_id>/bundle.zip")
def bundle(job_id):
    directory, job = get_job(job_id)
    key = job["selected"]
    require_downloadable(directory, job, key)
    report = json.loads((directory / key / "scan.json").read_text())
    stream = io.BytesIO()
    with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
        for name in job["candidates"][key]["files"]:
            path = directory / key / name
            if name != "scan.json" and digest(path) != report["sha256"][name]:
                abort(409, description="文件与本次读码记录不一致。请重新生成。")
            compression = ZIP_STORED if path.suffix in (".png", ".pdf") else ZIP_DEFLATED
            archive.write(path, name, compress_type=compression)
        for source in job["inputs"].values():
            name = source["stored_as"]
            path = directory / "inputs" / name
            if digest(path) != source["sha256"]:
                abort(409, description="源文件已变更。请重新上传。")
            archive.write(path, f"inputs/{name}", compress_type=ZIP_STORED)
        for name in ("job.json", "crop.png"):
            compression = ZIP_STORED if name == "crop.png" else ZIP_DEFLATED
            archive.write(directory / name, name, compress_type=compression)
        archive.writestr("selection.json", json.dumps(job["selection"], ensure_ascii=False))
    stream.seek(0)
    return send_file(stream, mimetype="application/zip", as_attachment=True,
                     download_name=f"art-qr-{job_id}-{key}.zip")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--port", type=int, default=8785)
    args = cli.parse_args()
    if not shutil.which("pdftoppm"):
        cli.error("打样 PDF 读码需要 Poppler 的 pdftoppm。")
    try:
        ImageFont.truetype(FONT, 16)
    except OSError:
        cli.error(f"无法加载物料排版字体 {FONT}。请设置 ARTQR_FONT 指向已安装的中文字体。")
    make_reader()
    app.run(host="127.0.0.1", port=args.port, threaded=True)
