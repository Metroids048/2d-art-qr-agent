"""Render QR modules and three finder patterns with the uploaded image’s own colors."""
import cv2
import numpy as np
import qrcode
from PIL import Image, ImageOps
from qrcode import constants, util

from art import LUMA, QUIET, functional_modules
from research.二维码双向延展.workflows.fit_art import optimize

PITCH = 16
BORDER = QUIET
MATERIALS = {"print": "原图融合 · 彩色版印",
             "glass": "原图融合 · 釉彩嵌片",
             "weave": "原图融合 · 彩色织纹"}


def source_pigments(pixels):
    """Keep separate source hues, even when they would cancel out in an RGB median."""
    pixels = pixels[::max(1, len(pixels) // 4096)].astype(np.float32, copy=False)
    chroma = np.ptp(pixels, axis=1)
    colored = pixels[(chroma >= 14) & (pixels @ LUMA < 238)]
    if len(colored) < max(8, len(pixels) * .045):
        return []
    hue = cv2.cvtColor(colored.reshape(-1, 1, 3) / 255, cv2.COLOR_RGB2HSV)[:, 0, 0]
    distance = abs((hue[:, None] - np.arange(0, 360, 15) + 180) % 360 - 180)
    neighborhoods = distance <= 23
    weights = neighborhoods.T @ np.ptp(colored, axis=1)
    pigments = []
    for _ in range(3):
        peak = weights.argmax()
        nearby = neighborhoods[:, peak]
        if weights[peak] == 0 or nearby.sum() < max(8, .055 * len(colored)):
            break
        pigments.append(np.median(colored[nearby], axis=0))
        offsets = (np.arange(len(weights)) - peak + 12) % 24 - 12
        weights[abs(offsets) <= 3] = 0
    return pigments


def as_ink(pigment):
    luma = pigment @ LUMA
    ink = pigment * min(1, 105 / max(1, luma))
    return np.maximum(0, ink + (ink - ink @ LUMA) * .30)


def ramp(value, lo, hi):
    position = np.clip((value - lo) / (hi - lo), 0, 1)
    return position**2 * (3 - 2 * position)


def lay_source(original, length, paper, framed):
    """Keep the entire source and its proportions on a square support."""
    side = length - (2 * BORDER * PITCH if framed else 0)
    placed = ImageOps.contain(original, (side, side), Image.Resampling.LANCZOS)
    left = (length - placed.width) // 2
    top = (length - placed.height) // 2
    px = np.asarray(placed, dtype=np.float32)
    right = length - placed.width - left
    bottom = length - placed.height - top
    extended = cv2.copyMakeBorder(px, top, bottom, left, right, cv2.BORDER_REPLICATE)
    y, x = np.indices((length, length))
    distance = np.maximum.reduce([top - y, y - (top + placed.height - 1),
                                  left - x, x - (left + placed.width - 1)])
    blur = cv2.GaussianBlur(extended, (0, 0), length * .024)
    soften = ramp(distance, -1, length * .07)[:, :, None]
    substrate = extended + (blur - extended) * soften
    fade = ramp(distance, 0, max(left, top, length * .05))[:, :, None]
    if framed:
        distance_fade = ramp(distance, -length * .022, max(length * .04, max(left, top) * .8))[:, :, None]
        return substrate + (paper - substrate) * distance_fade
    return substrate + (paper - substrate) * fade * .82


def lift_studio_matte(reference, paper):
    """Relight a smooth neutral studio backdrop with a continuous paper tone."""
    width = max(2, reference.shape[1] // 18)
    left = np.median(reference[:, :width], axis=1)
    right = np.median(reference[:, -width:], axis=1)
    edges = np.concatenate((left, right))
    rows = (left + right) * .5
    luma = rows @ LUMA
    if (np.median(luma[:max(2, len(rows) // 8)]) > 93
            or np.quantile(np.ptp(edges, axis=1), .80) > 16
            or np.quantile(abs((left - right) @ LUMA), .80) > 16):
        return reference
    smooth = cv2.GaussianBlur(rows[:, None, :], (1, 0), sigmaX=1, sigmaY=len(rows) * .045)
    if np.quantile(abs((rows[:, None, :] - smooth) @ LUMA), .80) > 11:
        return reference
    residual = reference - smooth
    neutral = 1 - ramp(np.ptp(reference, axis=2), 6, 28)
    matching = 1 - ramp(abs(residual @ LUMA), 12, 38)
    cover = matching * neutral
    opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    candidate = cv2.morphologyEx((cover > .68).astype(np.uint8), cv2.MORPH_OPEN, opening)
    _, regions = cv2.connectedComponents(candidate, connectivity=8)
    labels = np.unique(np.concatenate((regions[:, 0], regions[:, -1])))
    connected = np.isin(regions, labels[labels != 0])
    if np.mean(connected) < .20:
        return reference
    cover *= cv2.GaussianBlur(connected.astype(np.float32), (0, 0), 2.2)
    backdrop = paper * .92 + smooth * .08 + residual * .49
    return reference + (backdrop - reference) * cover[:, :, None]


def keep_hues(rgb, tone, strength):
    existing = rgb @ LUMA
    darken = np.minimum(tone / np.maximum(existing, 1), 1)
    compressed = rgb * darken[:, :, None]
    lift = np.maximum(tone - existing, 0) / np.maximum(255 - existing, 1)
    baseline = compressed + (255 - compressed) * lift[:, :, None]
    chroma = rgb - existing[:, :, None]
    gray = tone[:, :, None]
    limit = np.where(chroma >= 0, (255 - gray) / np.maximum(chroma, .001),
                     gray / np.maximum(-chroma, .001))
    amount = np.minimum(1.18, limit.min(axis=2))[:, :, None]
    colored = gray + chroma * amount
    return baseline + (colored - baseline) * strength


def source_matrix(payload, reference, plain):
    version = plain.version
    fixed = functional_modules(version)
    side = len(fixed)
    border = BORDER * PITCH
    block = reference[border:-border, border:-border]
    gray = block @ LUMA
    average = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)
    local = cv2.GaussianBlur(gray, (0, 0), PITCH * 1.10)
    local = cv2.resize(local, (side, side), interpolation=cv2.INTER_AREA)
    cutoff = np.clip(.57 * np.median(average) + .43 * local, 94, 184)
    target = np.asarray(average < cutoff, dtype=np.uint8)
    weight = .6 + np.minimum(abs(average - cutoff) / 80, 1.5)
    weight[fixed] = 0
    options = []
    for mask in range(8):
        plain.makeImpl(False, mask)
        grid = np.asarray(plain.modules, dtype=np.uint8)
        options.append((float(np.sum(weight * (grid == target))), grid, fixed, mask))
    _, grid, fixed, mask = max(options, key=lambda item: item[0])
    standard = (grid, fixed, mask)
    # The proven fitter uses spare post-terminator bits. Larger payloads use the
    # best matching standard mask to bound its cost and retain the exact payload.
    if version <= 7:
        try:
            grid, _, recipe = optimize(payload, version, target, weight)
        except ValueError:
            return standard, standard
        return (grid, fixed, recipe["mask"]), standard
    return standard, standard


class SourceFusion:
    def __init__(self, source, payload, accent, output_px, priority="balanced", style="auto"):
        self.priority = priority
        self.style = style
        self.source = source
        self.output_px = output_px
        palette = np.asarray(source.resize((64, 64)), dtype=np.float32)
        perimeter = np.concatenate((palette[0], palette[-1], palette[:, 0], palette[:, -1]))
        edge = np.median(perimeter, axis=0)
        pigments = source_pigments(palette.reshape(-1, 3)) or [np.asarray(accent)]
        self.ink = as_ink(pigments[0])
        color = pigments[0] - pigments[0] @ LUMA
        self.paper = np.asarray([247, 243, 235], dtype=np.float32) * .94 + edge * .06 + color * .11
        edge_luma = edge @ LUMA
        lift = np.clip((210 - edge_luma) / max(1, 255 - edge_luma), 0, 1)
        diffuse = edge + (255 - edge) * lift
        paper_mix = ramp(np.median(np.ptp(perimeter, axis=1)), 65, 180) * .64
        self.paper += (diffuse - self.paper) * paper_mix
        # Product shots often already include their own empty margin. Keep
        # their original scale when that margin can carry the QR quiet band.
        band = np.ones((64, 64), dtype=bool)
        band[6:-6, 6:-6] = False
        self.has_margin = np.quantile(palette[band] @ LUMA, .08) >= 219
        detail = np.mean(abs((palette - cv2.GaussianBlur(palette, (0, 0), 1.4)) @ LUMA) > 12)
        self.clear_style = "weave" if detail > .38 else "glass"
        code = qrcode.QRCode(version=5, error_correction=constants.ERROR_CORRECT_M, border=0)
        code.add_data(util.QRData(payload, mode=util.MODE_8BIT_BYTE))
        code.best_fit(start=5)
        length = (17 + 4 * code.version + 2 * BORDER) * PITCH
        self.length = length
        self.code = code
        self.payload = payload
        self.compositions = {}

    def matte_reference(self, framed):
        placed = ImageOps.contain(self.source, (self.length, self.length), Image.Resampling.LANCZOS)
        pixels = np.asarray(placed, dtype=np.float32)
        lifted = lift_studio_matte(pixels, self.paper)
        matte = self.source if lifted is pixels else Image.fromarray(lifted.clip(0, 255).astype(np.uint8))
        return lay_source(matte, self.length, self.paper, framed)

    def renders(self, key):
        style = ("print" if key == "rich" else self.clear_style) if self.style == "auto" else self.style
        first = not self.has_margin
        attempts = ((first, True, 0, 0), (first, True, 1, 0), (not first, True, 1, 0),
                    (True, True, 2, 0), (True, False, 2, 1), (False, False, 2, 1))
        if key == "clear":
            attempts = ((True, True, 2, 1), (first, True, 2, 1),
                        (True, False, 2, 0), (False, False, 2, 0))
        for framed, balanced, level, option in dict.fromkeys(attempts):
            if framed not in self.compositions:
                reference = self.matte_reference(framed)
                self.compositions[framed] = reference, source_matrix(self.payload, reference, self.code)
            reference, matrices = self.compositions[framed]
            matrix = matrices[option]
            qr = paint(reference, matrix[0], self.paper, self.ink, style, level,
                       self.output_px, balanced, self.priority)
            yield matrix, f"source-fusion/{style}/{int(framed)}-{int(balanced)}-{level}-{option}", qr
        # Keep the original colors and edges when the expressive marks cannot
        # carry the payload at the requested size. Try the image-fitted mask
        # before the standard one, with the entire source framed first.
        for framed, option in ((True, 0), (True, 1), (False, 0), (False, 1)):
            reference, matrices = self.compositions[framed]
            if option == 1 and np.array_equal(matrices[0][0], matrices[1][0]):
                continue
            matrix = matrices[option]
            qr = contrast_render(reference, matrix[0], matrix[1], self.ink, style, self.output_px)
            yield matrix, f"source-fusion/{style}/contrast-{int(framed)}-{option}", qr

        # Strong highlights and deep shadows can overlap at the default
        # contrast. Keep light and dark data centers separated on downsizing.
        retention = .42 if key == "rich" else .34
        for framed in (True, False):
            reference, matrices = self.compositions[framed]
            matrix = matrices[1]
            qr = contrast_render(reference, matrix[0], matrix[1], self.ink, style,
                                 self.output_px, retention=retention)
            yield matrix, f"source-fusion/{style}/separated-contrast-{int(framed)}", qr


def contrast_render(reference, modules, fixed, ink, style, output_px, *, retention=.59):
    border = BORDER * PITCH
    body = reference[border:-border, border:-border]
    bits = modules.astype(bool).repeat(PITCH, axis=0).repeat(PITCH, axis=1)
    fixed = fixed.repeat(PITCH, axis=0).repeat(PITCH, axis=1)
    lightness = body @ LUMA
    coords = (np.arange(len(body)) + .5) / PITCH
    offset = abs(coords % 1 - .5)
    distance = np.maximum(offset[:, None], offset[None, :])
    # Only the centers need a large lightness shift. Preserve the hue and use
    # the whole source lightness range inside both dark and light data marks.
    depth = np.where(fixed, 1, 1 - ramp(distance, .37, .555))
    target = retention * lightness + np.where(bits, 22, 253 - retention * 255)
    target = np.where(fixed, .23 * lightness + np.where(bits, 17, 191), target)
    grain = grain_field(coords[None, :], coords[:, None], style)
    target += grain * bits * .22
    tone = lightness + (target - lightness) * depth
    # Dye the solid function marks with source ink while keeping their
    # geometry and contrast. The quiet band keeps its source tint.
    pigment = ink - ink @ LUMA
    body = body + (fixed & bits)[:, :, None] * pigment * .44
    sheet = keep_hues(reference, np.maximum(reference @ LUMA, 225), .82)
    sheet[border:-border, border:-border] = keep_hues(body, tone.clip(1, 254), .92)
    return Image.fromarray(sheet.clip(0, 255).astype(np.uint8)).resize(
        (output_px, output_px), Image.Resampling.LANCZOS
    )


def paint(reference, modules, paper, ink, style, level, output_px, balanced, priority):
    length = len(reference)
    side = len(modules)
    border = BORDER * PITCH
    base_luma = reference @ LUMA
    lightness = base_luma.copy()
    sheet = reference.copy()
    # A faint tint in smooth highlights gives a studio source and its marks the
    # same stock. Leave colored areas and fine image detail at their own hue.
    paper_dye = ink - ink @ LUMA
    neutral_highlight = ramp(base_luma, 197, 246) * (1 - ramp(np.ptp(reference, axis=2), 10, 35))
    sheet += paper_dye * neutral_highlight[:, :, None] * .13

    samples = slice(border, border + side * PITCH)
    field = lightness[samples, samples]
    bright_goal = [200, 211, 223][level]
    dark_goal = [120, 106, 89][level]
    def expand(cells):
        return cells.repeat(PITCH, axis=0).repeat(PITCH, axis=1)

    bits = expand(modules.astype(bool))
    fixed = functional_modules((side - 17) // 4)
    square = expand(fixed)

    coords = (np.arange(len(field)) + .5) / PITCH
    x, y = np.meshgrid(coords % 1 - .5, coords % 1 - .5)
    ax, ay = abs(x), abs(y)
    # A brush can join cells on the same patch of the source. At a logo stroke
    # or packaging edge it breaks, so the QR rhythm follows the image boundary.
    tile_color = cv2.resize(reference[samples, samples], (side, side), interpolation=cv2.INTER_AREA)
    horizontal = (modules[:, :-1] == modules[:, 1:]) & ~fixed[:, :-1] & ~fixed[:, 1:]
    vertical = (modules[:-1] == modules[1:]) & ~fixed[:-1] & ~fixed[1:]
    horizontal = horizontal * (1 - ramp(np.linalg.norm(np.diff(tile_color, axis=1), axis=2), 19, 77))
    vertical = vertical * (1 - ramp(np.linalg.norm(np.diff(tile_color, axis=0), axis=2), 19, 77))
    east, west = np.pad(horizontal, ((0, 0), (0, 1))), np.pad(horizontal, ((0, 0), (1, 0)))
    south, north = np.pad(vertical, ((0, 1), (0, 0))), np.pad(vertical, ((1, 0), (0, 0)))
    same_row = np.where(x > 0, expand(east), expand(west))
    same_col = np.where(y > 0, expand(south), expand(north))
    connected = expand(np.maximum.reduce([east, west, south, north]))
    height = [.25, .30, .38][level]
    if style == 'print':
        # Use the same tangent estimate as the illustration renderer, at QR-cell scale.
        softened = cv2.GaussianBlur(base_luma[samples, samples].astype(np.float32), (0, 0), 1.7)
        dx = cv2.Sobel(softened, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(softened, cv2.CV_32F, 0, 1)
        products, balance, edges = (
            cv2.resize(values, (side, side), interpolation=cv2.INTER_AREA)
            for values in (dx * dy, dy**2 - dx**2, dx**2 + dy**2)
        )
        angles = np.clip(.5 * np.arctan2(-2 * products, balance), -.58, .58) * ramp(edges, 50, 2800)
        tilt = expand(angles)
        cross = y * np.cos(tilt) - x * np.sin(tilt)
        line = 1 - ramp(abs(cross), height * .74, .39)
        depth = line * (1 - ramp(ax, .29, .51))
        bridge = same_row * (1 - ramp(ay, height * .74, .41))
        depth = np.maximum(depth, bridge * (.95 - abs(tilt) * .60))
        grain = (np.sin(cross * 25 + x * 8) + np.sin(cross * 46 + x * 17)) * 3.5
        bridge = same_col * (1 - ramp(ax, .12, .31))
        depth = np.maximum(depth, bridge * abs(tilt) * .9)
    elif style == 'weave':
        parity = (coords.astype(int)[:, None] + coords.astype(int)[None, :]) % 2 == 0
        u, v = np.where(parity, ax, ay), np.where(parity, ay, ax)
        depth = 1 - ramp((u**4 + (v * 1.32)**4)**.25, height, .48)
        bridge = np.maximum(same_row * (1 - ramp(ay, .18, .32)), same_col * (1 - ramp(ax, .18, .32)))
        depth = np.maximum(depth, bridge * .75)
        grain = np.sin(u * 61) * 4.2 + np.sin(v * 108) * 2.8
    else:
        depth = 1 - ramp((ax**4 + ay**4)**.25, height, .53)
        bridge = np.maximum(same_row * (1 - ramp(ay, .25, .41)), same_col * (1 - ramp(ax, .25, .41)))
        depth = np.maximum(depth, bridge * .83)
        highlight = (1 - ramp(abs(y + .20), .012, .10)) * (1 - ramp(ax, .23, .43))
        bevel = (1 - ramp(abs(y - .26), .03, .13)) * (1 - ramp(ax, .30, .46))
        grain = highlight * 13 - bevel * 7
    if balanced:
        local = cv2.GaussianBlur(field, (0, 0), PITCH * .63)
        dark = np.minimum(field, 77 + .245 * local)
        color_strength = ramp(np.ptp(reference[samples, samples], axis=2), 24, 106)
        preserve = (1 - ramp(local, 71, 160)) * color_strength * [28, 18, 8][level]
        light = np.maximum(field, [145, 155, 171][level] + .185 * local - preserve)
        target = np.where(bits, dark, light)
        depth = np.maximum(depth, .16 + .17 * bits)
    else:
        target = np.where(bits, np.minimum(field, dark_goal), np.maximum(field, bright_goal))
    if level < 2:
        # Keep the center readable and give the longer marks more presence than
        # single dots, especially where the original already has a hard edge.
        tile_luma = expand(tile_color @ LUMA)
        detail = abs(field - tile_luma)
        amount = [.38, .21][level] * (1 - .65 * connected) + .19 * ramp(detail, 15, 65)
        amount += .28 * (1 - ramp(tile_luma, 58, 144))
        if priority == "strokes":
            amount += .24 * ramp(detail, 8, 45)
        amount = np.minimum(amount, .86)
        depth *= 1 - amount * ramp(np.maximum(ax, ay), .13, .44)
    depth = np.maximum(depth, square * .95)
    field += (target - field + grain * bits) * depth
    # Blend the same paper tint through pale cells to keep the printed or stained hues.
    delta = field - base_luma[samples, samples]
    paper_mix = np.maximum(delta, 0) / 255
    sheet[samples, samples] += (paper - 255) * paper_mix[:, :, None] * .44
    pigment_field = cv2.GaussianBlur(reference, (0, 0), PITCH * 1.6)
    local_dye = pigment_field - (pigment_field @ LUMA)[:, :, None]
    local_strength = ramp(np.ptp(local_dye, axis=2), 12, 102)
    chroma = local_dye * .63 + paper_dye * (1.04 - .78 * local_strength)[:, :, None]
    # Ink over an already dark label should not replace the label's own hue.
    dye_amount = .18 + .82 * ramp(base_luma[samples, samples], 70, 185)
    data_dye = chroma[samples, samples] * dye_amount[:, :, None]
    sheet[samples, samples] += data_dye * np.maximum(-delta, 0)[:, :, None] * .0038
    if balanced:
        sheet[samples, samples] += data_dye * (.20 + .38 * bits * depth)[:, :, None]

    coords = (np.arange(length) + .5) / PITCH - BORDER
    gx, gy = np.meshgrid(coords, coords)
    distance = np.maximum.reduce([-gx, -gy, gx - side, gy - side])
    outside = distance > 0
    # Quiet band retains the source's color while gently lifting its darkest marks.
    lightness += np.maximum(222 - lightness, 0) * outside

    eyes = [(3.5, 3.5), (side - 3.5, 3.5), (3.5, side - 3.5)]
    smooth_luma = pigment_field @ LUMA
    local = base_luma - smooth_luma
    exponent = {"print": 9, "glass": 10, "weave": 7}[style]
    striation = grain_field(gx, gy, style)
    for cx, cy in eyes:
        dx, dy = abs(gx - cx), abs(gy - cy)
        area = np.maximum(dx, dy)
        radius = (dx**exponent + dy**exponent)**(1 / exponent)
        outer = 1 - ramp(radius, 3.39, 3.54)
        solid = np.maximum(1 - ramp(radius, 1.43, 1.56), outer * ramp(radius, 2.44, 2.57))
        envelope = 1 - ramp(area, 4.28, 4.49)
        grain = striation
        if style == "glass":
            rim = (1 - ramp(np.minimum(abs(radius - 2.67), abs(radius - 1.29)), .022, .18))
            grain = grain + rim * 15
        # Use the same pigment field and underlying texture inside the finder
        # bars and pale channels. Only their scan-critical contrast is fixed.
        dark = [102, 89, 69][level] + (smooth_luma - 145) * .11
        dark = np.minimum(base_luma, dark) + local.clip(-35, 35) * .26 + grain
        pale = np.maximum(lightness, [197, 208, 220][level] + local.clip(-35, 35) * .18)
        target = dark * solid + pale * (1 - solid)
        lightness += (target - lightness) * envelope
        pigment = reference + chroma * (.21 + .84 * solid)[:, :, None]
        sheet += (pigment - sheet) * envelope[:, :, None]

    rgb = keep_hues(sheet.clip(0, 255), lightness.clip(1, 254), .94 if priority == "color" else .8)
    return Image.fromarray(rgb.clip(0, 255).astype('uint8')).resize((output_px, output_px), Image.Resampling.LANCZOS)


def grain_field(x, y, style):
    if style == 'print':
        return 8 * np.sin(y * 22 + x * 4)
    if style == 'weave':
        return 5 * np.sin(x * 39) + 5 * np.sin(y * 57)
    return 3 * np.sin(y * 11 + x * 3)
