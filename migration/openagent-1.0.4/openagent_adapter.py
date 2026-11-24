"""Bridge OpenAgent input and output to the local workbench."""
import argparse
from contextlib import contextmanager
from fcntl import LOCK_EX, flock
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from threading import Thread
from zipfile import BadZipFile, ZipFile

from PIL import Image, ImageDraw, ImageFont
from werkzeug.serving import make_server

from app import FONT, INPUT_NAMES, ROOT, app, make_reader
from art import input_pixels
from migration.smoke import post_select, read_png, request, upload


CANDIDATES = {"rich": "浓郁版", "clear": "清晰版"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@contextmanager
def workbench():
    if not shutil.which("pdftoppm"):
        raise RuntimeError("找不到打样 PDF 所需的 pdftoppm")
    ImageFont.truetype(FONT, 16)
    make_reader()
    with make_server("127.0.0.1", 0, app) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": .05})
        thread.start()
        try:
            yield server.server_port
        finally:
            server.shutdown()
            thread.join()


def get(port, route):
    status, _, result = request(port, route)
    if status != 200:
        raise RuntimeError(f"工作台 GET {route} 返回 HTTP {status}")
    return result


def comparison(source, png, key, out, *, material=False, single=False):
    font = ImageFont.truetype(FONT, 25)
    with Image.open(io.BytesIO(png)) as image:
        art = image.convert("RGBA")
    art.thumbnail((720, 720), Image.Resampling.LANCZOS)
    sheet = Image.new("RGB", (1280, 796), "#f7f4ee")
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 18), "本次上传原图", fill="#26231f", font=font)
    label = "物料平稿" if material else "候选预览"
    draw.text((535, 18), label if single else f"{CANDIDATES[key]} · {label}",
              fill="#26231f", font=font)
    sheet.paste(source, (24 + (485 - source.width) // 2, 55 + (720 - source.height) // 2), source)
    sheet.paste(art, (535 + (720 - art.width) // 2, 55 + (720 - art.height) // 2), art)
    sheet.save(out, compress_level=3)


def preview(args, port):
    files = {field: getattr(args, field) for field in INPUT_NAMES if getattr(args, field)}
    fields = {"title": args.title, "brief": args.brief, "layout": args.layout, "require_web_url": "1"}
    if getattr(args, "single_preview", False):
        fields["single_preview"] = "1"
    if getattr(args, "omit_print_copy", False):
        fields["omit_print_copy"] = "1"
    for field in ("url", "caption", "width", "height", "qr_percent", "art_px", "fusion_style"):
        value = getattr(args, field, None)
        if value is not None:
            fields[field] = value
    with TemporaryDirectory(prefix="openagent-artqr-") as folder:
        wire = {}
        for field, path in files.items():
            suffix = path.suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                raise ValueError(f"{INPUT_NAMES[field]}请使用 PNG、JPG 或 WebP。")
            name = path.name if not re.search(r'["\\\r\n]', path.name) else f"{field}{suffix}"
            wire[field] = Path(folder) / field / name
            wire[field].parent.mkdir()
            shutil.copyfile(path, wire[field])
        source_hash = sha256(wire["image"].read_bytes()).hexdigest()
        route = upload(port, fields, wire)
    job_path = ROOT / route.lstrip("/")
    job = json.loads((job_path / "job.json").read_text())
    original = get(port, f"{route}/inputs/image")
    if sha256(original).hexdigest() != source_hash or job["inputs"]["image"]["sha256"] != source_hash:
        raise RuntimeError(f"任务 {job['id']} 的原图 SHA-256 不一致")
    delivery = job_path / "openagent"
    delivery.mkdir(exist_ok=True)
    result = {"job": job["id"], "renderer": job["engine"], "expected": job["payload"],
              "payload_source": job["payload_source"], "input_sha256": source_hash,
              "selection": "pending", "candidates": {}}
    single = len(job["candidates"]) == 1
    if single:
        result["single_preview"] = True
    receipt_path = delivery / "preview-receipt.json"
    if receipt_path.is_file():
        previous = json.loads(receipt_path.read_text())
        if (previous["job"] != job["id"] or previous["input_sha256"] != source_hash
                or previous["expected"] != job["payload"]):
            raise RuntimeError("保存的预览与这次输入不一致")
        if all(Path(row["preview"]).is_file() and
               sha256(Path(row["preview"]).read_bytes()).hexdigest() == row.get("comparison_sha256")
               for row in previous["candidates"].values()):
            return previous
    use_cue = job["settings"].get("layout_cue")
    material = (args.layout != "auto" or bool(args.title)
                or use_cue not in (None, "未匹配关键词，使用展签")
                or any(value is not None for value in (args.width, args.height, args.caption)))
    with Image.open(io.BytesIO(original)) as image:
        reference = input_pixels(image)
    reference.thumbnail((485, 690), Image.Resampling.LANCZOS, reducing_gap=3.0)
    for key in job["candidates"]:
        asset = f"{route}/asset/{key}"
        png = get(port, f"{asset}/art-qr.png")
        decoded, size = read_png(png)
        scan = json.loads(get(port, f"{asset}/scan.json"))
        art_hash = sha256(png).hexdigest()
        if scan["files"]["art-qr.png"]["sha256"] != art_hash:
            raise RuntimeError(f"任务 {job['id']} 的成码哈希与工作台回执不符")
        sheet = delivery / f"{job['id']}-{key}-原图与候选预览.png"
        preview_name = "flat.png" if material else "art-qr.png"
        preview_image = get(port, f"{asset}/{preview_name}") if material else png
        preview_hash = sha256(preview_image).hexdigest()
        if preview_hash != scan["files"][preview_name]["sha256"]:
            raise RuntimeError("预览与工作台回执不一致")
        with TemporaryDirectory(prefix=".preview-", dir=delivery) as temp:
            staged = Path(temp) / sheet.name
            comparison(reference, preview_image, key, staged, material=material, single=single)
            os.replace(staged, sheet)
        result["candidates"][key] = {
            "preview": str(sheet), "comparison_sha256": sha256(sheet.read_bytes()).hexdigest(), "art_sha256": art_hash, "opencv": decoded, "size_px": size,
            "preview_asset": preview_name, "preview_asset_sha256": preview_hash,
            "exact_match": decoded == job["payload"], "engines": scan["engines"],
            "selectable": job["candidates"][key]["selectable"],
            "png_download_before_selection": request(port, f"{asset}/art-qr.png?download=1")[0],
        }
    result["zip_before_selection"] = request(port, f"{route}/bundle.zip")[0]
    with TemporaryDirectory(prefix=".receipt-", dir=delivery) as temp:
        staged = Path(temp) / receipt_path.name
        staged.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        os.replace(staged, receipt_path)
    return result


def delivery(args, port):
    job_path = ROOT / "jobs" / args.job
    with (job_path / "openagent" / ".delivery.lock").open("a") as lock:
        flock(lock, LOCK_EX)
        return deliver_selected(args, port, job_path)


def deliver_selected(args, port, job_path):
    route = f"/jobs/{args.job}"
    output = job_path / "openagent"
    receipt = json.loads((output / "preview-receipt.json").read_text())
    if receipt["job"] != args.job:
        raise RuntimeError("选稿回执与本次任务不一致")
    if (job_path / "native.json").is_file():
        binding = json.loads((job_path / "native.json").read_text())
        approval = app.config.get("ARTQR_APPROVAL", {})
        if (any(approval.get(k) != binding[k] for k in ("origin", "session", "job"))
                or approval.get("candidate") != args.candidate or not approval.get("event")):
            raise ValueError("请确认这次聊天里的预览。")
    entry = receipt["candidates"][args.candidate]
    if not entry["exact_match"] or not entry["selectable"]:
        raise RuntimeError("该候选未通过实际选稿门槛")
    folder = f"{route}/asset/{args.candidate}"
    asset = f"{folder}/art-qr.png"
    art = get(port, asset)
    decoded, _ = read_png(art)
    if sha256(art).hexdigest() != entry["art_sha256"] or decoded != receipt["expected"]:
        raise RuntimeError("当前完整艺术 PNG 与原候选的 OpenCV 核验不一致")
    if entry.get("preview_asset_sha256"):
        preview = art if entry["preview_asset"] == "art-qr.png" else get(port, f"{folder}/{entry['preview_asset']}")
        if sha256(preview).hexdigest() != entry["preview_asset_sha256"]:
            raise RuntimeError("当前作品与选稿时的预览不一致")
    selection_path = job_path / "selection.json"
    selected = json.loads(selection_path.read_text()) if selection_path.is_file() else {}
    if selected and (job_path / "native.json").is_file() and selected["candidate"] != args.candidate:
        raise ValueError("这次已经选定一版。如需调整，请先重新预览。")
    if selected.get("candidate") != args.candidate and post_select(port, route, args.candidate, review=True) != 303:
        raise RuntimeError("工作台拒绝本次确认选稿")
    png = get(port, f"{asset}?download=1")
    bundle = get(port, f"{route}/bundle.zip")
    filename = f"art-qr-{args.job}-{args.candidate}"
    png_path, zip_path = output / f"{filename}.png", output / f"{filename}.zip"
    with ZipFile(io.BytesIO(bundle)) as archive:
        if png != art or archive.read("art-qr.png") != png:
            raise RuntimeError("预览、独立下载与 ZIP 内的完整艺术 PNG 字节不一致")
        if zip_path.is_file():
            with ZipFile(zip_path) as previous:
                names = archive.namelist()
                previous_selection = json.loads(previous.read("selection.json"))
                previous_selection.pop("selected_at", None)
                current_selection = json.loads(archive.read("selection.json"))
                current_selection.pop("selected_at", None)
                if (previous.namelist() != names or previous_selection != current_selection
                        or any(previous.read(name) != archive.read(name)
                               for name in names if name != "selection.json")):
                    raise RuntimeError("已落盘的 ZIP 与本次可下载文件不一致")
            bundle = zip_path.read_bytes()
    if png_path.is_file() and png_path.read_bytes() != png:
        raise RuntimeError("已落盘的 PNG 与本次可下载文件不一致")
    with TemporaryDirectory(prefix=".export-", dir=output) as temp:
        for path, content in ((png_path, png), (zip_path, bundle)):
            if not path.is_file():
                staged = Path(temp) / path.name
                staged.write_bytes(content)
                os.replace(staged, path)
    return {"job": args.job, "candidate": args.candidate, "png": str(png_path), "zip": str(zip_path),
            "png_sha256": sha256(png).hexdigest(), "zip_sha256": sha256(bundle).hexdigest(),
            "opencv": decoded}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    subs = cli.add_subparsers(dest="action", required=True)
    make = subs.add_parser("preview")
    for field in INPUT_NAMES:
        if field == "source_qr":
            continue
        make.add_argument("--" + field.replace("_", "-"), type=Path, required=field == "image")
    destination = make.add_mutually_exclusive_group(required=True)
    destination.add_argument("--url")
    destination.add_argument("--source-qr", type=Path)
    make.add_argument("--title", default="")
    make.add_argument("--brief", default="")
    make.add_argument("--caption")
    make.add_argument("--layout", default="auto")
    make.add_argument("--width", type=float)
    make.add_argument("--height", type=float)
    make.add_argument("--qr-percent", type=float)
    make.add_argument("--art-px", type=int)
    make.add_argument("--fusion-style", choices=("auto", "print", "glass", "weave"), default="auto")
    make.add_argument("--single-preview", action="store_true")
    done = subs.add_parser("deliver")
    done.add_argument("--job", required=True)
    done.add_argument("--candidate", choices=CANDIDATES, required=True)
    done.add_argument("--visual-confirmed", action="store_true", required=True)
    args = cli.parse_args()
    if args.action == "deliver" and not re.fullmatch(r"[a-f0-9]{12}", args.job):
        cli.error("任务编号格式无效")
    try:
        with workbench() as port:
            result = preview(args, port) if args.action == "preview" else delivery(args, port)
    except (AssertionError, BadZipFile, OSError, RuntimeError, ValueError) as exc:
        cli.exit(1, f"艺术二维码工作台失败：{exc}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
