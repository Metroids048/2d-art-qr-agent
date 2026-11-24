"""Input-driven module art and material layouts for the local studio."""
from io import BytesIO
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import qrcode
from PIL import Image, ImageCms, ImageDraw, ImageFilter, ImageFont, ImageOps
from qrcode import util
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

DPI = 300
QUIET = 4
PAPER = (255, 253, 248)
RESAMPLE = Image.Resampling.LANCZOS
DISPLAY_SIDES = (240, 320)
SYSTEM_FONT = ("/System/Library/Fonts/Hiragino Sans GB.ttc" if sys.platform == "darwin"
               else "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
FONT = os.environ.get("ARTQR_FONT", SYSTEM_FONT)
STYLES = {
    "rich": {"label": "原色构图", "core": 16, "edge": 2, "dark": 36, "light": 231},
    "clear": {"label": "强化点阵", "core": 17, "edge": 2, "dark": 34, "light": 235},
}
LAYOUTS = {"cup": "杯形门贴", "bar": "横向吧台牌", "arch": "拱形门贴",
           "label": "展签", "shelf": "货架卡", "beads": "拼豆图样",
           "sticker": "圆角薄贴 / 小牌", "round": "圆形玻璃贴"}
LUMA = np.array([.299, .587, .114], dtype=np.float32)


def input_pixels(image):
    """Normalize the working copy to oriented sRGB pixels, keeping transparency."""
    image = ImageOps.exif_transpose(image)
    rgba = image.convert("RGBA")
    if profile := image.info.get("icc_profile"):
        color = image if image.mode in ("RGB", "CMYK", "L", "LAB") else (
            image.getchannel("L") if image.mode == "LA" else rgba.convert("RGB"))
        try:
            corrected = ImageCms.profileToProfile(
                color, BytesIO(profile), ImageCms.createProfile("sRGB"), outputMode="RGB")
        except (ImageCms.PyCMSError, OSError, ValueError):
            return rgba
        corrected.putalpha(rgba.getchannel("A"))
        return corrected
    return rgba


def refine_resized_detail(image):
    """Restore a little midtone detail lost in a large downsample."""
    pixels = np.asarray(image, dtype=np.float32)
    colors = pixels[:, :, :3]
    luma = colors @ LUMA
    detail = luma - cv2.GaussianBlur(luma, (0, 0), .85)
    strength = np.abs(detail)
    weight = np.clip((strength - 2) / 4, 0, 1) * np.clip((30 - strength) / 12, 0, 1)
    change = np.clip(detail * .22 * weight, np.maximum(-3, -colors.min(axis=2)),
                     np.minimum(3, 255 - colors.max(axis=2)))
    colors += change[:, :, None]
    return Image.fromarray(np.uint8(np.rint(pixels)))


def qr_code(payload, mask=None, version=None):
    code = qrcode.QRCode(version=version, error_correction=qrcode.constants.ERROR_CORRECT_H,
                         border=0, mask_pattern=mask)
    code.add_data(payload)
    code.make(fit=version is None)
    return code


def normal_qr(payload, mask, box_size=24):
    code = qr_code(payload, mask)
    code.border = QUIET
    code.box_size = box_size
    return code.make_image(fill_color="black", back_color="white").convert("RGB")


def functional_modules(version):
    # Use the pinned encoder's own placement routines, before any data bit is laid down.
    code = qrcode.QRCode(version=version)
    n = code.modules_count = 17 + 4 * version
    code.modules = [[None] * n for _ in range(n)]
    for row, col in ((0, 0), (n - 7, 0), (0, n - 7)):
        code.setup_position_probe_pattern(row, col)
    code.setup_position_adjust_pattern()
    code.setup_timing_pattern()
    code.setup_type_info(False, 0)
    if version >= 7:
        code.setup_type_number(False)
    return np.not_equal(code.modules, None)


def motif(source, settings):
    """Sample a square focal area and keep the adjacent artwork for layout continuation."""
    source = source.convert("RGBA")
    full = settings.get("frame") == "full"
    original_size = source.size
    if full:
        side = max(source.size)
        fill = "white"
        if settings.get("graphic") == "illustrated":
            preview = source.resize((64, 64), RESAMPLE)
            border = np.asarray(Image.alpha_composite(Image.new("RGBA", preview.size, "white"), preview))
            perimeter = np.concatenate((border[0], border[-1], border[:, 0], border[:, -1]))
            fill = tuple(np.uint8(np.median(perimeter, axis=0)).tolist())
        source = ImageOps.pad(source, (side, side), color=fill)
    side = round(min(source.size) * (1 if full else settings["zoom"] / 100))
    cx = round(source.width * settings["focus_x"] / 100)
    cy = round(source.height * settings["focus_y"] / 100)
    x = min(max(cx - side // 2, 0), source.width - side)
    y = min(max(cy - side // 2, 0), source.height - side)
    rgb = Image.alpha_composite(Image.new("RGBA", source.size, "white"), source).convert("RGB")
    margin = round(side * .32)
    padded = np.pad(np.asarray(rgb), ((margin, margin), (margin, margin), (0, 0)), mode="reflect")
    view = Image.fromarray(padded[y:y + side + 2 * margin, x:x + side + 2 * margin])
    view = view.resize((512, 512), RESAMPLE)
    edge = round(margin * 512 / (side + 2 * margin))
    box = (edge, edge, 512 - edge, 512 - edge)
    center = view.crop(box)
    pixels = np.asarray(view)
    # A light edge-aware pass removes camera noise without erasing colored outlines.
    smooth = cv2.bilateralFilter(pixels, 7, 21, 4)
    if settings["detail"] == "relief":
        lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
        lightness = lab[:, :, 0].astype(np.float32)
        surround = cv2.GaussianBlur(lightness, (0, 0), 13)
        shadow = np.clip((surround - lightness) * 3.1, 0, 60)
        flank = np.clip((lightness - surround) * 1.5, 0, 29)
        lab[:, :, 0] = np.uint8(np.clip(lightness - shadow + flank, 0, 255))
        lab[:, :, 1] = np.uint8(np.clip(128 + (lab[:, :, 1].astype(float) - 128) * 1.4 - 4, 0, 255))
        smooth = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if settings.get("graphic") in ("rounded", "mosaic"):
        # Flatten camera noise into poster-like color while keeping the uploaded hue and silhouette.
        hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.24, 0, 255)
        light = hsv[..., 2] / 255
        hsv[..., 2] = np.clip(.40 * hsv[..., 2] + .60 * (np.round(light * 6) * 42.5), 0, 255)
        smooth = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if settings["invert"]:
        smooth = 255 - smooth
    extension = Image.fromarray(smooth)
    art = extension.crop(box)
    colors = np.asarray(art.resize((64, 64), RESAMPLE), dtype=float).reshape(-1, 3)
    chroma = np.ptp(colors, axis=1)
    # Favor saturated pixels but keep accent hues from actual pixels, not an RGB average.
    colored = colors[(chroma >= max(18, np.quantile(chroma, .7))) & (colors @ LUMA < 224)]
    if len(colored):
        _, bins = np.unique(np.uint8(colored) // 30, axis=0, return_inverse=True)
        accent = np.median(colored[bins == np.bincount(bins).argmax()], axis=0)
    else:
        accent = np.median(colors, axis=0)
    ink_color = accent
    if settings.get("graphic") == "rounded" and accent @ LUMA > 150:
        dark_chromatic = colors[(chroma >= 60) & (colors @ LUMA < 110)]
        if len(dark_chromatic) >= len(colors) * .08:
            ink_color = np.median(dark_chromatic, axis=0)
    ink = ink_color * min(1, 46 / max(1, ink_color @ LUMA))
    region = (0, 0, *original_size) if full else (x, y, x + side, y + side)
    return art, extension, center, ink, accent, region


def matrix_options(payload, source):
    standard = qr_code(payload)
    n = len(standard.modules)
    fixed = functional_modules(standard.version)
    levels = np.asarray(source.convert("L").resize((n, n), Image.Resampling.BOX), dtype=float) / 255
    patterns = [qr_code(payload, m, standard.version).get_matrix() for m in range(8)]
    penalties = [util.lost_point(matrix) for matrix in patterns]
    scores = [np.sum((2 * levels[~fixed] - 1) * (1 - 2 * np.array(m, dtype=int)[~fixed]))
              for m in patterns]
    floor = min(penalties)
    # Prefer source agreement while favoring QR masks with fewer visual collisions.
    ordered = sorted(range(8), key=lambda m: (penalties[m] > floor * 1.20,
                                              .07 * (penalties[m] - floor) - scores[m]))
    return [(np.asarray(patterns[m], dtype=bool), fixed, m) for m in ordered]


def design_matrix(payload, source):
    return matrix_options(payload, source)[0]


def module_art(payload, source, ink, style, solid=False, graphic="photo", matrix=None):
    data, fixed, mask = matrix if matrix is not None else design_matrix(payload, source)
    n = len(data)
    size = 24
    quiet = QUIET * size
    raster = Image.new("RGB", ((n + 2 * QUIET) * size,) * 2, PAPER)
    # Keep a continuous source image. QR bits pull the local RGB toward ink or paper
    # only as much as required near the module center. Finder/timing cells stay solid.
    pixels = np.asarray(source.resize((n * size, n * size), RESAMPLE), dtype=np.float32).copy()
    coordinates = np.arange(size) + .5 - size / 2
    radius = np.maximum(np.abs(coordinates[:, None]), np.abs(coordinates[None, :]))
    if graphic == "rounded" and not solid:
        # A softly rounded square leaves a broad, contrast-stable sample in the middle.
        curve = style.get("curve", 3)
        radius = (np.abs(coordinates[:, None])**curve + np.abs(coordinates[None, :])**curve)**(1/curve)
    alpha = np.clip((style["core"] / 2 + style["edge"] - radius) / style["edge"], 0, 1)[..., None]
    if graphic == "mosaic" and not solid:
        colors = np.asarray(source.resize((n, n), Image.Resampling.BOX), dtype=np.float32)
    for row, col in np.ndindex(data.shape):
        bit = data[row, col]
        x, y = col * size, row * size
        block = pixels[y:y + size, x:x + size]
        if fixed[row, col]:
            block[:] = ink if bit else PAPER
            continue
        if solid:
            block[:] = np.mean(block, axis=(0, 1))
        elif graphic == "mosaic":
            block[:] = block * .53 + colors[row, col] * .47
        lightness = (block @ LUMA)[..., None]
        if bit:
            corrected = block * np.minimum(1, style["dark"] / np.maximum(lightness, 1))
        else:
            corrected = 255 - (255 - block) * np.minimum(1, (255 - style["light"]) /
                                                          np.maximum(255 - lightness, 1))
        block[:] = corrected if solid else block + (corrected - block) * alpha
    raster.paste(Image.fromarray(np.uint8(np.clip(pixels, 0, 255))), (quiet, quiet))
    return raster, data, mask


def avoid_structures(source, payload):
    """Look for a nearby uncropped placement with fewer hard QR edges across the artwork."""
    code = qr_code(payload)
    n = code.modules_count
    stride = max(1, min(4, 132 // n))
    side = n * stride
    fixed = functional_modules(code.version).repeat(stride, axis=0).repeat(stride, axis=1)
    levels = np.where(np.array(code.modules), 38, 248).repeat(stride, axis=0).repeat(stride, axis=1)[fixed]
    source_gray = np.asarray(source.resize((side, side), RESAMPLE), dtype=np.float32) @ LUMA
    best = None
    for width in range(side, round(side * .82) - 1, -2):
        shrunk = cv2.resize(source_gray, (width, width), interpolation=cv2.INTER_AREA)
        for y in range(0, side - width + 1, 2):
            for x in range(0, side - width + 1, 2):
                canvas = np.full((side, side), 255, dtype=np.float32)
                canvas[y:y + width, x:x + width] = shrunk
                dx = cv2.Sobel(canvas, cv2.CV_32F, 1, 0)
                dy = cv2.Sobel(canvas, cv2.CV_32F, 0, 1)
                energy = np.minimum(1, np.hypot(dx[fixed], dy[fixed]) / 400)
                energy += .2 * (canvas[fixed] < 248)
                conflict = np.sum(energy * ((canvas[fixed] - levels) / 255) ** 2)
                center = (side - width) / 2
                cost = conflict * (side / width) ** 2 + 8 * (abs(x - center) + abs(y - center)) / side
                if best is None or cost < best[0]:
                    best = (cost, width, x, y)
    _, width, x, y = best
    scaled, dx, dy = (round(value * source.width / side) for value in (width, x, y))
    output = Image.new("RGB", source.size, "white")
    output.paste(source.resize((scaled, scaled), RESAMPLE), (dx, dy))
    return output, [dx, dy, scaled]


HALFTONES = {
    "fine": (142, 50, .08, .44),
    "soft": (156, 60, .11, .425),
    "balanced": (175, 70, .11, .425),
}


def artwork_features(source, modules, priority):
    """Find color boundaries and narrow strokes at the smallest actual display size."""
    side = round(DISPLAY_SIDES[0] * modules / (modules + 2 * QUIET))
    rgb = np.asarray(source.resize((side, side), RESAMPLE))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    smoothed = cv2.GaussianBlur(lab, (0, 0), .85)
    dx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0) / 8
    dy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1) / 8
    boundary = np.linalg.norm(np.hypot(dx, dy), axis=2)
    light = smoothed[..., 0]
    # Black-hat and top-hat protect both dark strokes and bright gaps between them.
    window = (max(5, round(2.1 * side / modules)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window, window))
    strokes = (cv2.morphologyEx(light, cv2.MORPH_BLACKHAT, kernel) +
               cv2.morphologyEx(light, cv2.MORPH_TOPHAT, kernel))
    color_weight, stroke_weight = {"balanced": (1., .8), "strokes": (1., 1.25),
                                   "color": (1.3, .65)}[priority]
    weight = np.clip(color_weight * boundary / 30 + stroke_weight * strokes / 58, 0, 1)
    return cv2.dilate(weight, np.ones((3, 3), np.uint8))


def composition_options(source, payload, priority):
    """Align source features with function cells and data bits before styling."""
    code = qr_code(payload)
    fixed = functional_modules(code.version)
    grids = [np.asarray(qr_code(payload, mask, code.version).modules, dtype=bool)
             for mask in range(8)]
    targets = np.where(grids, 60., 221.)
    cell_weight = np.where(fixed, 3., 1.)
    modules = len(fixed)
    original = source.width
    options = []
    stroke_sensitive = priority == "strokes"
    scales = (1., .97, .94, .91, .88) if stroke_sensitive else (1., .95, .89)
    steps = 4 if stroke_sensitive else 2
    if stroke_sensitive:
        stroke_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for proportion in scales:
        side = round(original * proportion)
        gap = original - side
        offsets = (0,) if not gap else tuple(round(gap * i / steps) for i in range(steps + 1))
        motif = source.resize((side, side), RESAMPLE)
        for y in offsets:
            for x in offsets:
                image = Image.new("RGB", source.size, "white")
                image.paste(motif, (x, y))
                features = artwork_features(image, modules, priority)
                importance = cv2.resize(features, (modules, modules), interpolation=cv2.INTER_AREA)
                if stroke_sensitive:
                    letter_plane = np.asarray(image.convert("L").resize((3 * modules,) * 2, RESAMPLE))
                    dark_strokes = cv2.morphologyEx(letter_plane, cv2.MORPH_BLACKHAT, stroke_kernel)
                    importance += cv2.resize(dark_strokes.astype(np.float32) / 102, (modules,) * 2,
                                             interpolation=cv2.INTER_AREA)
                light = np.asarray(image.resize((modules, modules), Image.Resampling.BOX)) @ LUMA
                mismatch = ((light - targets) / 255)**2
                overlap = (mismatch * (.12 + importance) * cell_weight).sum(axis=(1, 2))
                displacement = (abs(x - gap / 2) + abs(y - gap / 2)) / original
                for mask, cost in enumerate(overlap):
                    total = cost + (1 - proportion) * 10 + displacement * 2
                    options.append((total, image, [x, y, side], features, (grids[mask], fixed, mask)))
    return [option[1:] for option in sorted(options, key=lambda option: option[0])]


def halftone_art(source, ink, data, fixed, target_px, graphic, profile="balanced", features=None):
    """Adapt module coverage to the output grid and avoid erasing local edges."""
    n = len(data)
    pitch = min(24, max(8, 1152 // n))
    pixels = np.asarray(source.resize((n * pitch, n * pitch), RESAMPLE), dtype=np.float32)
    blocks = pixels.reshape(n, pitch, n, pitch, 3).transpose(0, 2, 1, 3, 4)
    lightness = blocks @ LUMA
    edge = np.clip(lightness.std(axis=(-2, -1)) / 50, 0, 1)
    coverage = (blocks.min(axis=-1) < 218).mean(axis=(-2, -1))
    blank = np.clip((.65 - coverage) * 2, 0, 1)
    px_per_module = target_px / (n + 2 * QUIET)
    size_adjustment = .022 * (target_px - DISPLAY_SIDES[0]) / (DISPLAY_SIDES[1] - DISPLAY_SIDES[0])
    guided = profile in ("source-fusion", "source-fusion-strong")
    light, dark, edge_fade, paper_radius = HALFTONES.get(profile, HALFTONES["fine"])
    if profile == "source-fusion-strong":
        light, dark, edge_fade, paper_radius = HALFTONES["balanced"]
    radius = np.clip(.44 - edge_fade * edge - size_adjustment, min(.46, 1.4 / px_per_module), .46)
    radius = radius * (1 - blank) + paper_radius * blank
    axis = np.abs((np.arange(pitch) + .5) / pitch - .5)
    distance = (axis[:, None]**4 + axis[None, :]**4)**.25
    if graphic == "mosaic":
        distance = np.maximum(axis[:, None], axis[None, :])
    alpha = np.clip((radius[..., None, None] + .058 - distance) / .0875, 0, 1)
    if guided:
        protected = cv2.resize(features, (n * pitch, n * pitch), interpolation=cv2.INTER_LINEAR)
        protected = protected.reshape(n, pitch, n, pitch).transpose(0, 2, 1, 3)
        # Keep the original sampling contrast at the center, and restore source detail around it.
        outer = np.clip((distance - .23) / .20, 0, 1)
        alpha *= 1 - .78 * protected * outer
    low = (dark * (1 - blank) + 28 * blank)[..., None, None]
    high = (light * (1 - blank) + 240 * blank)[..., None, None]
    target = np.where(data[..., None, None], np.minimum(lightness, low),
                      np.maximum(lightness, high))
    shadows = blocks * np.minimum(1, target / np.maximum(lightness, 1))[..., None]
    # Lift brightness in the original hue before mixing toward white.
    gain = np.minimum(255 / np.maximum(blocks.max(axis=-1), 1),
                      target / np.maximum(lightness, 1))
    saturated = blocks * gain[..., None]
    highlights = 255 - (255 - saturated) * np.minimum(
        1, (255 - target) / np.maximum(255 - saturated @ LUMA, 1))[..., None]
    tint = np.where((target < lightness)[..., None], shadows, highlights)

    # The timing strip keeps every whole module and its bit. Its ink follows the adjacent hue.
    timing = np.zeros_like(fixed)
    timing[6, 8:n - 8] = True
    timing[8:n - 8, 6] = True
    alpha[fixed] = 1
    hard = fixed & ~timing
    tint[hard] = np.where(data[hard, None], ink, PAPER)[:, None, None, :]
    layer = blocks + (tint - blocks) * alpha[..., None]
    rgb = layer.transpose(0, 2, 1, 3, 4).reshape(pixels.shape)
    raster = Image.new("RGB", ((n + 2 * QUIET) * pitch,) * 2, PAPER)
    raster.paste(Image.fromarray(np.uint8(np.clip(rgb, 0, 255))), (QUIET * pitch,) * 2)
    return raster


def nearby_artwork(source):
    """Keep every pixel of the selected source while moving its edge off fixed modules."""
    yield source, [0, 0, source.width]
    for scale in (.94, .89):
        side = round(source.width * scale)
        guide = (source.width - side) // 2
        motif = source.resize((side, side), RESAMPLE)
        for x in (2 * guide - 1, 1):
            image = Image.new("RGB", source.size, "white")
            image.paste(motif, (x, guide))
            yield image, [x, guide, side]


def detail_cost(source, preview, modules, placement):
    """Weight contrast loss on the narrow dark and bright features of the real input."""
    margin = round(preview.width * QUIET / (modules + 2 * QUIET))
    inside = preview.crop((margin, margin, preview.width - margin, preview.height - margin))
    reference = np.asarray(source.resize(inside.size, RESAMPLE), dtype=np.float32)
    source_light = reference @ LUMA
    changes = np.asarray(inside, dtype=np.float32) @ LUMA - source_light
    feature_width = (round(3.8 * preview.width / (modules + 2 * QUIET)) | 1)
    neighborhood = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feature_width,) * 2)
    valleys = cv2.morphologyEx(source_light, cv2.MORPH_BLACKHAT, neighborhood) / 255
    ridges = cv2.morphologyEx(source_light, cv2.MORPH_TOPHAT, neighborhood) / 255
    # Exclude near-white padding and emphasize holes in strokes and highlights.
    present = (source_light < 247) | (ridges > .2)
    loss = np.abs(changes) * .26 + np.maximum(changes, 0) * valleys * 2
    loss += np.maximum(-changes, 0) * ridges * .7
    geometry_cost = (1 - placement[2] / source.width) * 7
    return float((loss * present).sum() / max(1, present.sum())) + geometry_cost


def fitted_text(draw, position, content, max_width, px, color, anchor="lt"):
    px = max(8, int(px))
    font = ImageFont.truetype(FONT, px)
    width = font.getlength(content)
    if width > max_width:
        font = ImageFont.truetype(FONT, max(7, int(px * max_width / width)))
    draw.text(position, content, font=font, fill=color, anchor=anchor)


def shelf_surface(source, accent):
    """Sample the product's main color without repeating its outline or lettering."""
    sample = source.copy()
    sample.thumbnail((256, 256), RESAMPLE)
    matching = (np.linalg.norm(np.asarray(sample, dtype=np.float32) - accent, axis=2) < 36).astype(np.uint8)
    enclosed = np.pad(matching, 1)
    distances = cv2.distanceTransform(enclosed, cv2.DIST_L2, 5)[1:-1, 1:-1]
    _, distance, _, (x, y) = cv2.minMaxLoc(distances)
    radius = int(distance * .54)
    if radius < 6:
        return Image.new("RGB", (128, 128), tuple(np.uint8(accent)))
    sx, sy = source.width / sample.width, source.height / sample.height
    return source.crop((round((x - radius) * sx), round((y - radius) * sy),
                        round((x + radius) * sx), round((y + radius) * sy)))


def shelf_card(art, subject, ink, s, size, x, y, side, gutter, flipped):
    """Place an intact product and its printed code on one continuous card."""
    w, h = size
    pixels = np.asarray(art.convert("RGB"))
    edges = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    paper = tuple(np.uint8(np.median(edges, axis=0)))
    page = Image.new("RGB", size, paper)
    left, right = (x + side + gutter, w - gutter) if flipped else (gutter, x - gutter)
    span = right - left
    image_top = round(h * (.17 if s["title"] else .065))
    hero = ImageOps.contain(subject.convert("RGBA"), (span, h - image_top - gutter), RESAMPLE)
    page.paste(hero, (left + (span - hero.width) // 2,
                      (image_top + h - gutter - hero.height) // 2), hero)
    draw = ImageDraw.Draw(page)
    fitted_text(draw, ((left + right) // 2, round(h * .075)), s["title"], span, h * .057, ink, "mt")
    fitted_text(draw, (x + side // 2, y + side + gutter // 2), s["caption"], side, h * .037, ink, "mt")
    return page


def material(art, source, extended, ink, accent, logo, s, modules, editorial=False):
    w, h = (round(s[k] / 25.4 * DPI) for k in ("width", "height"))
    unit = min(w, h)
    layout = s["layout"]
    side = round(unit * s["qr_percent"] / 100)
    ink = tuple(np.uint8(ink))
    accent = np.asarray(accent)
    deep = tuple(np.uint8(accent * min(.86, 58 / max(1, accent @ LUMA))))
    pale = tuple(np.uint8(accent * .15 + np.array(PAPER) * .85))
    outer = Image.new("L", (w, h))
    mask = ImageDraw.Draw(outer)
    inset = max(2, round(unit * .01))
    body_w = w
    if layout == "cup":
        body_w = round(w * .87)
        handle = (round(w * .71), round(h * .36), w - inset, round(h * .65))
        mask.ellipse(handle, outline=255, width=round(unit * .073))
        mask.rounded_rectangle((inset, inset, body_w - 1, h - inset - 1),
                               radius=round(unit * .17), fill=255)
    elif layout == "round":
        mask.ellipse((inset, inset, w-inset-1, h-inset-1), fill=255)
    elif layout == "sticker":
        mask.rounded_rectangle((inset, inset, w-inset-1, h-inset-1), radius=round(unit * .075), fill=255)
    elif layout == "arch":
        mask.rounded_rectangle((inset, inset, w-inset-1, h-inset-1), radius=round(unit * .15), fill=255)
    else:
        mask.rectangle((0, 0, w-1, h-1), fill=255)

    landscape = layout in ("bar", "label", "shelf", "sticker") and w >= h * 1.16
    flipped = landscape and editorial
    gutter = round(unit * .065)
    x = (gutter if flipped else w - side - gutter) if landscape else (body_w - side) // 2
    y = (h - side) // 2 if landscape else max(gutter, (h - side) // 2 - round(unit * .025))
    if outer.crop((x, y, x + side, y + side)).getextrema() != (255, 255):
        raise ValueError("当前码区触到刀线。请调小码区占比，或增加物料尺寸。")

    compact = layout in ("sticker", "round")
    keep_subject = s.get("preserve_shelf_subject")
    if keep_subject:
        page = shelf_card(art, source, ink, s, (w, h), x, y, side, gutter, flipped)
    else:
        # Continue the focal window outside the quiet zone at the same scale.
        page = Image.new("RGB", (w, h), PAPER if compact else pale if editorial else deep)
        sub = side * modules / (modules + 2 * QUIET)
        ext_side = round(sub * 1.64)
        extent = (x + (side - ext_side) // 2, y + (side - ext_side) // 2)
        continuity = extended.resize((ext_side, ext_side), RESAMPLE)
        if compact:
            axis = np.linspace(-1, 1, ext_side, dtype=np.float32)
            radius = np.hypot(axis[:, None], axis[None, :])
            strength = 62 if layout == "sticker" else 42
            fade = Image.fromarray(np.uint8(strength * np.clip((1 - radius) * 2.2, 0, 1)))
        else:
            fade = Image.new("L", (ext_side, ext_side))
            ImageDraw.Draw(fade).rounded_rectangle((0, 0, ext_side-1, ext_side-1),
                                                  radius=round(ext_side * .16), fill=255)
        page.paste(continuity, extent, fade)

    draw = ImageDraw.Draw(page)
    if compact:
        if landscape:
            left = x + side + gutter if flipped else gutter
            right = w - gutter if flipped else x - gutter
            span = right - left
            if logo is not None and span > unit * .22:
                mark = ImageOps.contain(logo, (span, round(h * .23)), RESAMPLE)
                rgba = np.asarray(mark.convert("RGBA"))
                painted = rgba[..., :3][rgba[..., 3] > 192]
                if painted.size and np.median(painted @ LUMA) > 226:
                    # Use a one-ink version of light wordmarks on cream paper.
                    recolored = Image.new("RGBA", mark.size, ink)
                    recolored.putalpha(mark.getchannel("A"))
                    mark = recolored
                page.paste(mark, (left, round(h * .15)), mark)
            else:
                draw.rounded_rectangle((left, round(h * .21), left + span * .30, round(h * .235)),
                                       radius=max(2, unit // 100), fill=deep)
            fitted_text(draw, (left, round(h * .43)), s["title"], span, unit*.115, ink)
            fitted_text(draw, (left, round(h * .65)), s["caption"], span, unit*.062, deep)
        else:
            if layout == "round":
                ring = round(unit * .042)
                draw.ellipse((ring, ring, w-ring-1, h-ring-1), outline=ink, width=max(3, unit//125))
                rim = ring + max(4, unit // 72)
                draw.arc((rim, rim, w-rim-1, h-rim-1), 25, 155,
                         fill=tuple(np.uint8(accent)), width=max(2, unit // 145))
            title_top = min(round(h * .105), y - round(unit * .068))
            fitted_text(draw, (w//2, title_top), s["title"], round(w*.51), unit * .062, ink, "mt")
            fitted_text(draw, (w//2, round(h*.827)), s["caption"], round(w*.57), unit * .046, ink, "mt")
    elif landscape and not keep_subject:
        left = x + side + gutter if flipped else gutter
        right = w - gutter if flipped else x - gutter
        span = max(8, right - left)
        panel_edge = x + side + gutter // 2 if flipped else x - gutter // 2
        panel_box = (panel_edge, 0, w, h) if flipped else (0, 0, panel_edge, h)
        panel_color = PAPER if editorial else deep
        panel_text = ink if editorial else PAPER
        draw.rectangle(panel_box, fill=panel_color)
        if span > unit * .29:
            art_height = round(h * (.49 if logo is not None else .56))
            area = (span, art_height)
            subject = source.convert("RGBA")
            hero = ImageOps.fit(subject, area, RESAMPLE) if layout == "bar" else ImageOps.contain(subject, area, RESAMPLE)
            hx = left + ((span - hero.width) // 2 if layout == "label" else 0)
            hy = round(h * .25) + (art_height - hero.height) // 2
            page.paste(hero, (hx, hy), hero)
            draw.line((left, round(h * .824), right, round(h * .824)), fill=panel_text, width=max(1, unit//450))
        fitted_text(draw, (left, round(h * .105)), s["title"], span, unit * .081, panel_text)
        fitted_text(draw, (left, round(h * .875)), s["caption"], span, unit * .036, panel_text)
        if logo is not None and span > unit * .34:
            stamp = ImageOps.contain(logo, (round(span*.28), round(h*.13)), RESAMPLE)
            row = round(h * .73)
            page.paste(stamp, (right-stamp.width, row-stamp.height//2), stamp)
    elif not keep_subject:
        footer = y + side + max(3, (h - y - side) // 5)
        panel_text = ink if editorial else PAPER
        text_width = round(body_w * .82)
        title_size = min(unit * .081, (h - footer) * .32)
        if layout == "cup":
            # A colored handle joins the cup body, using the same overall silhouette.
            draw.ellipse(handle, outline=deep, width=round(unit * .073))
        footer_y = footer - max(2, unit // 88)
        draw.rectangle((inset, footer_y, body_w - 1, h - 1), fill=pale if editorial else deep)
        fitted_text(draw, (body_w // 2, footer), s["title"], text_width, title_size, panel_text, "mt")
        fitted_text(draw, (body_w // 2, footer + int(title_size * 1.30)), s["caption"],
                    text_width, title_size * .60, panel_text, "mt")
        if logo is not None and layout == "cup":
            stamp = ImageOps.contain(logo, (unit//9, max(10, unit//20)), RESAMPLE)
            if stamp.width < (body_w - text_width) // 2:
                page.paste(stamp, (round(body_w*.12), footer), stamp)
    page.paste(art.resize((side, side), RESAMPLE), (x, y))

    design = page.convert("RGBA")
    design.putalpha(outer)
    eroded = outer.filter(ImageFilter.MinFilter(5))
    trace = Image.fromarray(np.asarray(outer) - np.asarray(eroded))
    if layout in ("bar", "label", "shelf", "beads"):
        ImageDraw.Draw(trace).rectangle((1, 1, w-2, h-2), outline=255, width=2)
    guide = design.copy()
    guide.paste("#c71f81", (0, 0, w, h), trace)
    return design, guide, {
        "dimensions_mm": [s["width"], s["height"]],
        "canvas_px": [w, h],
        "qr_with_quiet_mm": [round(v / DPI * 25.4, 2) for v in (x, y, side, side)],
        "quiet_zone_modules": QUIET,
        "module_pitch_mm": round(side / (modules + 2 * QUIET) / DPI * 25.4, 3),
        "trim_bbox_px": list(outer.getbbox()),
        "trim_reference": "cut-guide.png",
    }


def isolate_subject(source, *, retain_pixels=False):
    """Separate a centrally framed object from a simple background for display."""
    original = source.convert("RGBA")
    im = original.copy()
    if not retain_pixels or original.getchannel("A").getextrema()[0] != 0:
        im.thumbnail((680, 680), RESAMPLE)
    if im.getchannel("A").getextrema()[0] == 0:
        box = im.getbbox()
        return im.crop(box) if box else None
    rgb = np.asarray(im.convert("RGB"))
    h, w = rgb.shape[:2]
    border = max(2, min(h,w) // 32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    edge_chroma = np.concatenate((lab[:border, :, 1:].reshape(-1, 2),
                                  lab[-border:, :, 1:].reshape(-1, 2),
                                  lab[:, :border, 1:].reshape(-1, 2),
                                  lab[:, -border:, 1:].reshape(-1, 2)))
    distance = np.linalg.norm(lab[:, :, 1:] - np.median(edge_chroma, axis=0), axis=-1)
    edge_spread = np.quantile(np.linalg.norm(edge_chroma - np.median(edge_chroma, axis=0), axis=-1), .95)
    threshold = max(8, edge_spread + 5)
    chromatic = distance >= threshold
    labels = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
    labels[distance < threshold * .60] = cv2.GC_PR_BGD
    likelihood = cv2.erode(chromatic.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    labels[likelihood] = cv2.GC_FGD
    labels[:border, :] = labels[-border:, :] = cv2.GC_BGD
    labels[:, :border] = labels[:, -border:] = cv2.GC_BGD
    use_color = .055 < likelihood.mean() < .65 and edge_spread < 17
    mode = cv2.GC_INIT_WITH_MASK if use_color else cv2.GC_INIT_WITH_RECT
    cv2.grabCut(rgb, labels, (border, border, w - border * 2, h - border * 2),
                np.zeros((1, 65)), np.zeros((1, 65)), 3, mode)
    mask = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    components, component, stats, _ = cv2.connectedComponentsWithStats(mask)
    if components > 1:
        mask = (component == stats[1:, cv2.CC_STAT_AREA].argmax()+1).astype(np.uint8)
    if not .035 < mask.mean() < .78:
        return None
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    alpha = Image.fromarray(mask * 255).filter(ImageFilter.GaussianBlur(.65))
    if retain_pixels:
        im = original
        alpha = alpha.resize(im.size, RESAMPLE)
    im.putalpha(alpha)
    box = im.getchannel("A").getbbox()
    return im.crop(box) if box else None


def surface_print(flat, background, corners, glass):
    """Project ink density into the photograph and keep the captured surface light."""
    w, h = background.size
    bw, bh = flat.size
    quad = np.float32(corners) * np.float32((w / 100, h / 100))
    x, y = np.maximum(0, np.floor(quad.min(axis=0)).astype(int) - 5)
    right, bottom = np.minimum((w, h), np.ceil(quad.max(axis=0)).astype(int) + 6)
    width, height = right - x, bottom - y
    # Integrate coverage at the projected scale, especially on the foreshortened side.
    # A half-pixel offset aligns the averaged samples with the original photo pixels.
    local_quad = 2 * (quad - np.float32((x, y))) + .5
    transform = cv2.getPerspectiveTransform(
        np.float32([(0, 0), (bw - 1, 0), (bw - 1, bh - 1), (0, bh - 1)]), local_quad)
    rgb = np.asarray(flat.convert("RGB"), dtype=np.float32)
    coverage = np.asarray(flat.getchannel("A"), dtype=np.float32) / 255
    reflected_light = .095 if glass else .038
    optical_density = (1 - np.minimum(rgb / np.float32(PAPER), 1)) * (1 - reflected_light)
    pigment = optical_density * coverage[..., None]
    transfer = cv2.warpPerspective(pigment, transform, (width * 2, height * 2), flags=cv2.INTER_LINEAR)
    transfer = cv2.resize(transfer, (width, height), interpolation=cv2.INTER_AREA)
    transfer = cv2.GaussianBlur(transfer, (5, 5), .35 if glass else .48)
    scene = background.convert("RGB")
    lit_surface = np.asarray(scene.crop((x, y, right, bottom)), dtype=np.float32)
    scene.paste(Image.fromarray(np.uint8(np.clip(lit_surface * (1 - transfer), 0, 255))), (x, y))
    return scene


def mockup(flat, layout, background, settings, source, accent, cutout):
    if settings.get("print_corners"):
        return surface_print(flat, background, settings["print_corners"],
                             settings["scene_mount"] == "print-glass")
    w, h = 1320, 980
    door = layout in ("arch", "cup")
    if background is not None:
        scene = background.convert("RGB")
        w, h = scene.size
        scale = settings["scene_width"] / 100
        cx, cy = (settings[k] / 100 for k in ("scene_x", "scene_y"))
    else:
        color = np.asarray(accent, dtype=float)
        k = 1.45 if layout == "label" else 1.15
        wall = np.clip(np.array((185, 183, 176)) * .72 + color * .28, 48, 240)
        floor = np.clip(wall * .61 + np.array((20, 15, 9)), 0, 255)
        y, x = np.ogrid[:h, :w]
        spotlight = np.maximum(0, 1 - np.sqrt(((x - w *.23) / w)**2 + ((y - h *.12) / h)**2))
        noise = np.random.default_rng(42).normal(0, 1.1, (h, w))
        texture = np.where(y < h*.69, 0, np.sin(y*.65 + np.sin(x*.007)*2) * 1.6)
        base = np.where((y < h*.69)[..., None], wall, floor)
        scene = Image.fromarray(np.uint8(np.clip(base + (spotlight * 50 * k - 14 + noise + texture)[..., None],
                                                 0, 255)))
        d = ImageDraw.Draw(scene)
        if door:
            d.rounded_rectangle((60, 38, w-62, h+50), radius=18, fill="#292d2b")
            d.rounded_rectangle((83, 58, w-84, h+55), radius=10, fill=tuple(np.uint8(wall*.61)))
            d.line((w//2, 60, w//2, h), fill="#282d2a", width=15)
            reflect = Image.new("RGBA", (w,h))
            rd = ImageDraw.Draw(reflect)
            rd.polygon(((87, 674),(w-89, 161),(w-89, 213),(87, 740)), fill=(255,255,245,26))
            scene = Image.alpha_composite(scene.convert("RGBA"), reflect).convert("RGB")
        elif source is not None and layout in ("label", "bar", "shelf"):
            if cutout is not None:
                subject = ImageOps.contain(cutout, (round(w*.53), round(h*.64)), RESAMPLE)
                xx = round(w*.05) + (round(w*.45) - subject.width) // 2
                yy = round(h*.72) - subject.height
                stage = (round(w*.035), round(h*.71), round(w*.54), round(h*.85))
                d.rounded_rectangle(stage, radius=13, fill=tuple(np.uint8(wall*.64)))
                d.ellipse((xx-8, yy+subject.height-15, xx+subject.width+21, yy+subject.height+17),
                          fill=tuple(np.uint8(wall*.41)))
                scene.paste(subject, (xx,yy), subject)
            elif layout == "bar":
                picture = ImageOps.fit(source.convert("RGB"), (w, round(h*.69)), RESAMPLE, centering=(.69, .49))
                scene.paste(Image.blend(picture, Image.new("RGB", picture.size, "#38291e"), .20), (0,0))
                d.line((0, round(h*.69), w, round(h*.69)), fill="#513420", width=12)
            else:
                subject = ImageOps.fit(source.convert("RGB"), (round(w*.46), round(h*.59)), RESAMPLE)
                print_x, print_y = round(w*.035), round(h*.06)
                scene.paste(subject, (print_x, print_y))
        if layout == "beads":
            scale, cx, cy = .65, .50, .50
        elif door:
            scale, cx, cy = .52, .72, .53
        else:
            scale, cx, cy = .65, .665, .65

    hero = ImageOps.contain(flat, (round(w * scale), round(h * .88)), RESAMPLE)
    x = np.clip(round(w * cx) - hero.width // 2, 0, w - hero.width)
    y = np.clip(round(h * cy) - hero.height // 2, 0, h - hero.height)
    a, b = hero.size
    pw, ph = a - 1, b - 1
    skew = .024 if background is not None else (.055 if layout in ("bar", "shelf", "label") else .035)
    quad = np.float32([(x, y+ph*skew), (x+pw, y), (x+pw*(1-skew), y+ph), (x+pw*skew, y+ph*(1-skew))])
    transform = cv2.getPerspectiveTransform(np.float32([(0,0),(pw,0),(pw,ph),(0,ph)]), quad)
    # Warp premultiplied channels to avoid black fringing at the trim edge.
    paper = hero.convert("RGBa")
    warped = cv2.warpPerspective(np.asarray(paper), transform, (w, h), flags=cv2.INTER_CUBIC)
    mounted = Image.fromarray(warped, "RGBa").convert("RGBA")
    shadow = Image.new("L", (w,h))
    if settings.get("scene_mount") == "standing":
        top_right, front_right, front_left = map(tuple, quad[1:])
        rear = (min(w - 1, front_right[0] + a * .15), front_right[1] - b * .11)
        contact = (front_left[0] - a * .04, front_left[1] + b * .025)
        ImageDraw.Draw(shadow).polygon((contact, (rear[0] + a * .06, rear[1]),
                                        (rear[0], front_right[1] + b * .07), front_right), fill=106)
        shadow = shadow.filter(ImageFilter.GaussianBlur(max(2, a * .025)))
        scene.paste("#21170e", (0, 0, w, h), shadow)
        draw = ImageDraw.Draw(scene)
        draw.polygon((top_right, rear, front_right), fill="#735336")
        draw.line((top_right, front_right), fill="#c8a77b", width=max(1, round(a * .005)))
    else:
        shadow.paste(mounted.getchannel("A"), (10, 14))
        shadow = shadow.filter(ImageFilter.GaussianBlur(13)).point(lambda a: round(a * .53))
        scene.paste("#27251f", (0,0,w,h), shadow)
    scene.paste(mounted, (0,0), mounted)
    return scene


def save_proof(path: Path, flat, width, height):
    page_width, page_height = width * mm, height * mm
    page = Canvas(str(path), pagesize=(page_width, page_height), pageCompression=1)
    page.setTitle("Art QR - actual-size digital proof")
    # Keep the page's exact physical size and align the artwork to the 300 dpi
    # pixel grid. Stretching to the page resamples scan-critical QR details.
    image_width, image_height = (pixels * 72 / DPI for pixels in flat.size)
    page.drawImage(ImageReader(flat), 0, page_height - image_height,
                   image_width, image_height, mask="auto")
    page.save()
