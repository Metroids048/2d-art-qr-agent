"""Source-driven ink strokes and inset tiles on the workbench's standard QR grid."""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from art import LUMA, PAPER, QUIET, RESAMPLE, matrix_options


ILLUSTRATIONS = {"rich": "ink-strokes", "clear": "inset-tiles"}
ILLUSTRATION_LABELS = {"ink-strokes": "丝网墨线", "inset-tiles": "立体嵌片"}
# From the widest source tonal range to the strongest separation for OpenCV.
TONAL_RANGES = ((5, 158, 75, 180), (12, 137, 90, 165),
                (18, 100, 123, 130), (38, 68, 155, 95))


class Illustration:
    def __init__(self, source, payload, accent, priority):
        self.matrices = matrix_options(payload, source)
        self.n = len(self.matrices[0][0])
        self.pitch = 20
        n, p = self.n, self.pitch
        self.extent = n * p
        # Simplify photographic texture into continuous shape and colour fields.
        field = np.asarray(source.resize((240, 240), RESAMPLE))
        lab = cv2.cvtColor(field, cv2.COLOR_RGB2LAB)
        smooth = cv2.bilateralFilter(lab, 7, 19 if priority == "strokes" else 26, 5)
        poster = smooth.astype(np.float32)
        steps = np.array([12, 14, 14] if priority == "color" else [9, 12, 12])
        poster = .25 * poster + .75 * np.round(poster / steps) * steps
        rgb = Image.fromarray(cv2.cvtColor(np.uint8(np.clip(poster, 0, 255)), cv2.COLOR_LAB2RGB))
        self.source = np.asarray(rgb.resize((self.extent, self.extent), RESAMPLE), dtype=np.float32)
        self.tone = self.source @ LUMA
        self.chroma = self.source - self.tone[..., None]
        self.accent = np.asarray(accent)
        self.level = (self.tone / 255)**.76
        if np.quantile(np.ptp(self.source, axis=2), .8) < 20 and np.median(self.tone) < 115:
            # Two inks separate the highlights in a low-key monochrome image without
            # requiring a larger brightness jump between neighbouring QR modules.
            shadow_ink = np.array([-35, 7, 82])
            highlight_ink = np.array([20, -2, -33])
            highlight = np.clip(self.level * 1.8, 0, 1)[..., None]
            self.chroma += (1 - highlight) * shadow_ink + highlight * highlight_ink
            self.accent = self.accent + shadow_ink
        gray = smooth[..., 0].astype(np.float32)
        dx, dy = (cv2.Sobel(gray, cv2.CV_32F, i, 1-i) for i in (1, 0))
        level = dx * dx + dy * dy
        products = cv2.resize(dx * dy, (n, n), interpolation=cv2.INTER_AREA)
        edges = cv2.resize(level, (n, n), interpolation=cv2.INTER_AREA)
        # Follow the tangent of the local outline, keeping each sampling centre solid.
        balance = cv2.resize(dy * dy - dx * dx, (n, n), interpolation=cv2.INTER_AREA)
        normal = .5 * np.arctan2(-2 * products, balance)
        self.angle = normal * np.clip(edges / 520, 0, 1)
        axis = (np.arange(p) + .5) / p - .5
        self.x, self.y = np.meshgrid(axis, axis)

    @staticmethod
    def rgb(tone, chroma, saturation):
        return np.clip(tone[..., None] + saturation * chroma, 0, 255)

    def draw(self, data, fixed, style, tonal_range):
        n, p = self.n, self.pitch
        x, y = self.x, self.y
        if style == "ink-strokes":
            angle = self.angle[..., None, None]
            cross = y * np.cos(angle) - x * np.sin(angle)
            along = x * np.cos(angle) + y * np.sin(angle)
            shape = (np.abs(cross) < .30) & (np.abs(along) < .49)
            finish = np.where(cross > .21, -12, 0) * shape
        else:
            distance = (x**4 + y**4)**.25
            shape = np.broadcast_to(distance < .454, (n, n, p, p))
            finish = np.where(y < -.27, 15, np.where(y > .27, -14, 0)) * shape

        core = x * x + y * y < .265**2
        mask = np.where(fixed[..., None, None], True, shape | core)
        mask = (mask & data[..., None, None]).transpose(0, 2, 1, 3).reshape(self.tone.shape)
        trim = (finish * ~fixed[..., None, None]).transpose(0, 2, 1, 3).reshape(self.tone.shape)
        low, width, pale, pale_width = tonal_range
        paper = self.rgb(pale + pale_width * self.level, self.chroma, .77)
        ink = self.rgb(low + width * self.level + trim, self.chroma, 1.08)
        view = np.where(mask[..., None], ink, paper)
        # Restore all scanning structures except finder eyes, which receive the same finish
        # as the data marks. All structures maintain their original sampling centres.
        accent_chroma = self.accent - self.accent @ LUMA
        function_ink = self.rgb(np.array(51), accent_chroma, .8)
        function_paper = self.rgb(np.array(243), accent_chroma, .13)
        # Timing and alignment marks carry scene colours too, with extra contrast reserve.
        for row, col in np.argwhere(fixed):
            cell = np.s_[row*p:(row+1)*p, col*p:(col+1)*p]
            if data[row, col]:
                view[cell] = .60 * function_ink + .40 * ink[cell]
            else:
                view[cell] = .78 * function_paper + .22 * paper[cell]
        margin = p * QUIET
        sheet = Image.new("RGB", ((n + 2*QUIET)*p,) * 2, PAPER)
        sheet.paste(Image.fromarray(np.uint8(view)), (margin, margin))
        pen = ImageDraw.Draw(sheet)
        for row, col in ((0, 0), (0, n-7), (n-7, 0)):
            y0, x0 = margin + row*p, margin + col*p
            color = .70 * function_ink + .30 * ink[row*p:(row+7)*p, col*p:(col+7)*p].mean(axis=(0, 1))
            self.finder(pen, x0, y0, style, color, function_paper)
        return sheet

    def finder(self, pen, x, y, style, ink, paper):
        p = self.pitch
        color = tuple(np.uint8(ink).tolist())
        pale = tuple(np.uint8(paper).tolist())
        edge = tuple(np.uint8(np.clip(ink + 41, 0, 255)).tolist())
        shade = tuple(np.uint8(np.clip(ink - 14, 0, 255)).tolist())
        pen.rectangle((x, y, x+7*p-1, y+7*p-1), fill=pale)
        for inset, width, fill in ((0, 7, color), (1, 5, pale), (2, 3, color)):
            left, top = x + inset*p, y + inset*p
            span = width*p
            pen.rounded_rectangle((left, top, left+span-1, top+span-1), radius=p//2, fill=fill)
        if style == "inset-tiles":
            for inset, width in ((0, 7), (2, 3)):
                left, top = x + inset*p, y + inset*p
                right, bottom = left + width*p, top + width*p
                pen.line((left+p//2, top+5, right-p//2, top+5), fill=edge, width=4)
                pen.line((left+p//2, bottom-5, right-p//2, bottom-5), fill=shade, width=4)
        else:
            for inset, width in ((0, 7), (2, 3)):
                left, top = x + inset*p, y + inset*p
                for stripe in (6, 13):
                    for band in (top, top+(width-1)*p):
                        pen.line((left+p//2, band+stripe, left+width*p-p//2, band+stripe),
                                 fill=edge, width=2)
