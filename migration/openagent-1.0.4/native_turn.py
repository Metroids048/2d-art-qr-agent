"""Resolve artwork inputs and approvals from the current OpenClaw turn."""
import fcntl
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

from PIL import Image

from app import ROOT, app, full_payload, make_reader
from openagent_adapter import CANDIDATES, delivery, preview, workbench
from scan import decode_source

HOME = Path.home() / ".openclaw"
SESSIONS = HOME / "agents/main/sessions"
INBOUND = HOME / "media/inbound"
FORMAL_INPUT = Path("/oasn/in/invocation.json")
FORMAL_RUNS = Path("/oasn/.oasn-shared-runs")
FORMAL_SESSION = "agent:main:main"
DASHBOARD = re.compile(r"agent:main:dashboard:[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
FORMAL_RUN = re.compile(r"run_[0-9a-f]{32}")
DEVELOPER_RUN = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
FORMAL_REFERENCE = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{24}(?![0-9A-Fa-f])")
SELECTION_CODE = re.compile(r"\s+(\d{6})(?=[\s。！!」”]*$)")
MATERIALS = "把你想放进码里的原图和扫码要打开的完整网址发给我。"
INTRO = ("你好，我是艺术二维码。我可以把照片、插画或店标融进二维码里。你想把它用在哪儿？"
         "把原图和扫码要打开的完整网址发我。要是做贴纸、展签或台卡，可以顺便说说大概多大、"
         "图里哪部分一定要留。我先做一张给你看。")
RENEW = "刚才的预览暂时打不开。请把原图和网址一起重发，我再做一张。"
RETRY_REPLY = "这次没能发出完整结果。你可以回复「重试」，我会先找这单已有的预览。"
IMAGE_TYPES = {"PNG": (".png", "image/png"), "JPEG": (".jpg", "image/jpeg"),
               "WEBP": (".webp", "image/webp")}
MAX_ATTACHMENT_BYTES = 12 * 1024 * 1024
SUFFIXES = {".jpg", ".jpeg", ".webp", ".png"}
URL = re.compile(r"""https?://[^\s<>"'“”‘’「」《》【】，。；！？、]+""", re.IGNORECASE)
URL_HINT = re.compile(r"https?[:：/]|\bwww\.|(?:网址|链接)\s*[：:]\s*\S", re.IGNORECASE)
USE_OLD = re.compile(r"(?:沿用|使用|采用|用|读取|提取|扫|从)(?:这次|本次|我发的|上传的|附上的|附件中的|附件里的|这个|这张|的|原来那张|一下|一下这张|我附的|我上传的|旧的|原有的|\s){0,5}(?:旧(?:二维码|码)|原(?:二维码|码)|这张二维码)")
REJECT = re.compile(r"不要|不选|不确认|不满意|暂不|先别|先看|等我|只看|还要|再改|不想|别发|不合适|不通过|不是|不行|还是像|但|不过|否则|如果|吗|[?？]")
DECLINE_SELECTION = re.compile(r"不要|不选|不确认|不满意|先别|暂不|先不|不合适|不通过|不行")
RECOVER = re.compile(r"重新发|重新预览|再发(?:给我|我)?(?:一次|一遍|一下|一份|预览)|补发|重新下载|下载不了|再给我.*(?:文件|成品)|找不到.*(?:文件|成品)")
RETRY = re.compile(r"重试|再试一次|继续制作|接着做")
OMIT_PRINT_COPY = re.compile(
    r"(?:^|[，,。；;！!？?\n])\s*(?:(?:这张|本张|成品)\s*)?(?:请\s*)?"
    r"(?:(?:不(?:要|用)?|别)(?:额外)?(?:加|添加|印|放)(?:上)?(?:任何|额外的?)?(?:文字|文案|字样|字)"
    r"|无(?:文字|文案)|只要图[，,]?\s*不要(?:文字|文案))"
    r"\s*(?=$|[，,。；;！!？?\n])"
)
CANDIDATE_WORDS = {word: key for key, word in CANDIDATES.items()}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def atomic_write(path, data):
    with NamedTemporaryFile(dir=path.parent, prefix=".artqr-", delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(data)
            file.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def save_state(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2).encode())
    kept = value.get("intake", {}).get("file")
    for cached in path.parent.glob(f"{path.stem}.intake-*"):
        if cached.name != kept:
            cached.unlink(missing_ok=True)


@contextmanager
def session_state(root, session_id):
    directory = root / "openagent-sessions"
    directory.mkdir(mode=0o700, exist_ok=True)
    name = sha256(session_id.encode()).hexdigest()
    with (directory / f"{name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield directory / f"{name}.json"


def current_transcript(context):
    # sessionKey/sessionId arrive from the gateway tool factory, never from model arguments.
    key, session = context.get("sessionKey"), context.get("sessionId")
    run_id = context.get("runId")
    has_bound_run = isinstance(run_id, str) and FORMAL_RUN.fullmatch(run_id)
    if not isinstance(key, str) or not key or not (
            key == FORMAL_SESSION or DASHBOARD.fullmatch(key) or has_bound_run):
        raise ValueError("请在当前网页聊天里继续提交图片或确认预览。")
    index = read_json(SESSIONS / "sessions.json", {})
    entry = index.get(key, {})
    if not isinstance(session, str) or entry.get("sessionId") != session:
        raise ValueError("没能对应到这条聊天，请在当前会话重发这条消息。")
    location = entry.get("sessionFile")
    path = Path(location) if location else SESSIONS / f"{session}.jsonl"
    if path.parent != SESSIONS or path.name != f"{session}.jsonl" or path.resolve() != path:
        raise ValueError("当前聊天的记录还没准备好，请稍等再试。")

    transcript = {}
    tip = None
    with path.open() as file:
        for line in file:
            if not line.endswith("\n"):
                raise ValueError("这条消息还在接收，请稍等再试。")
            event = json.loads(line)
            if event.get("id") and "parentId" in event:
                transcript[event["id"]] = event
                tip = event["id"]
    events = []
    visited = set()
    while tip and tip not in visited and tip in transcript:
        visited.add(tip)
        event = transcript[tip]
        events.append(event)
        tip = event["parentId"]
    events.reverse()
    for event in reversed(events):
        message = event.get("message", {})
        if message.get("role") == "user":
            return session, events, event
        if message.get("oasnWebchatFinalReply"):
            break
    raise ValueError("这条消息还没同步到当前聊天，请重发这一条。")


def message_text(message):
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if part.get("type") == "text")


def paths_for(message, inbound=None):
    # Only native attachment metadata. Ignore file names and paths in chat text.
    paths = message.get("MediaPaths") or []
    kinds = message.get("MediaTypes") or []
    if len(paths) != len(kinds):
        raise ValueError("这张图没有传完整，请把图片重新附一次。")
    images = []
    for name, kind in zip(paths, kinds):
        path = Path(name)
        if (kind not in {"image/png", "image/jpeg", "image/webp"}
                or path.parent != (inbound or INBOUND) or path.suffix.lower() not in SUFFIXES
                or path.resolve() != path or not path.is_file()
                or path.stat().st_size > MAX_ATTACHMENT_BYTES):
            raise ValueError("请重新附一张可打开的 PNG、JPG 或 WebP 主图。")
        _, mime = verified_image_type(path)
        if mime != kind:
            raise ValueError("请重新附一张可打开的 PNG、JPG 或 WebP 主图。")
        images.append(path)
    return images


def verified_image_type(path):
    try:
        with Image.open(path) as image:
            image_type = IMAGE_TYPES[image.format]
            image.verify()
    except (OSError, SyntaxError, KeyError, Image.DecompressionBombError):
        raise ValueError("请重新附一张可打开的 PNG、JPG 或 WebP 主图。") from None
    return image_type


def url_from(text, use_old=False):
    urls = []
    for match in URL.finditer(text):
        value = match.group()
        if match.start() > 0 and text[match.start() - 1] == "(" and value.endswith(")") and value.count(")") > value.count("("):
            value = value[:-1]
        urls.append(value)
    urls = list(dict.fromkeys(urls))
    if len(urls) > 1:
        raise ValueError("你发了多个链接。扫码要打开其中哪一个？")
    if urls:
        try:
            return full_payload(urls[0], None, None)
        except ValueError:
            raise ValueError("这个网址不完整。请发以 https:// 或 http:// 开头的完整链接。") from None
    if URL_HINT.search(text) and not use_old:
        raise ValueError("请发以 https:// 或 http:// 开头的完整链接。")
    return None


@dataclass
class Intake:
    image: Path | None = None
    url: str | None = None
    source_qr: Path | None = None
    image_event: str | None = None
    target_event: str | None = None

    def reply(self, *, sized=False):
        return missing_material_reply(self.image, self.url or self.source_qr, sized=sized)


def missing_material_reply(image_received, address_received, *, sized=False):
    if image_received and not address_received:
        return "主图已收到。扫码要打开哪个网页？把链接发我就行。"
    if address_received and not image_received:
        return "扫码地址有了。把要做成二维码的主图发我就行。"
    if not image_received:
        if sized:
            return "尺寸记下了。把原图和扫码要打开的完整网址发我，我就按这个大小先做一张。"
        return MATERIALS
    return None


def save_incomplete_intake(path, inputs, body):
    intake = {"url": inputs.url, "image_event": inputs.image_event,
              "target_event": inputs.target_event, "text": body[:450]}
    attachment = inputs.image or inputs.source_qr
    if attachment:
        data = attachment.read_bytes()
        fingerprint = sha256(data).hexdigest()
        cached = path.with_name(f"{path.stem}.intake-{fingerprint}{attachment.suffix.lower()}")
        atomic_write(cached, data)
        intake.update(file=cached.name, sha256=fingerprint,
                      role="image" if inputs.image else "source_qr")
    return intake


def supplement_intake(path, previous, inputs):
    if not (inputs.image or inputs.url or inputs.source_qr) or (
            inputs.image and (inputs.url or inputs.source_qr)):
        return False
    same_side = (inputs.image and previous["image_event"]) or (
        (inputs.url or inputs.source_qr) and previous["target_event"])
    if same_side:
        return False
    cached_name = previous.get("file")
    if cached_name:
        cached = path.parent / cached_name
        if (cached.parent != path.parent or not cached.name.startswith(f"{path.stem}.intake-")
                or not cached.is_file() or sha256(cached.read_bytes()).hexdigest() != previous["sha256"]):
            return False
        setattr(inputs, previous["role"], cached)
    if previous["image_event"]:
        inputs.image_event = previous["image_event"]
    else:
        inputs.url = previous["url"]
        inputs.target_event = previous["target_event"]
    return True


def collect_inputs(turns, inbound=None):
    choice = Intake()
    pending_address_error = None
    pending_image_error = None
    for event in turns:
        message = event["message"]
        body = message_text(message)
        old = bool(USE_OLD.search(body) and not re.search(r"(?:不要|不用|别)(?:再)?(?:沿用|使用|用)?[^。！]{0,10}(?:旧码|原码|旧二维码)", body))
        try:
            url = url_from(body, old)
        except ValueError as error:
            pending_address_error = error
            choice.url = choice.source_qr = None
            url = None
        if url:
            pending_address_error = None
            choice.url, choice.source_qr, choice.target_event = url, None, event["id"]
        try:
            paths = paths_for(message, inbound)
        except ValueError as error:
            pending_image_error = error
            choice.image = None
            paths = []
        if old:
            if not paths:
                pending_address_error = ValueError("请把要沿用的旧二维码附在这条消息里。")
                choice.url = choice.source_qr = None
                continue
            if url:
                pending_address_error = ValueError("你同时给了网址和旧码。扫码该用哪一个？")
                choice.url = choice.source_qr = None
                continue
            make_reader()
            try:
                decoded = [(path, decode_source(path)) for path in paths]
                old_codes = [(path, values) for path, values in decoded if values]
                if len(old_codes) != 1 or len(paths) > 2 or len(old_codes[0][1]) != 1:
                    raise ValueError
                source_qr, values = old_codes[0]
                full_payload(next(iter(values)), None, None)
            except (OSError, ValueError):
                pending_address_error = ValueError("请发一张能扫出网页的旧码，并把主图另附。")
                choice.url = choice.source_qr = None
                continue
            choice.url, choice.source_qr, choice.target_event = None, source_qr, event["id"]
            pending_address_error = None
            paths = [path for path in paths if path != choice.source_qr]
        if len(paths) > 1:
            pending_image_error = ValueError("收到多张图片。请单独发要做成二维码的那张主图。")
            choice.image = None
        if len(paths) == 1:
            choice.image, choice.image_event = paths[0], event["id"]
            pending_image_error = None
    if pending_address_error:
        raise pending_address_error
    if pending_image_error:
        raise pending_image_error
    return choice


def approval(body):
    handover = r"(?:成品(?:包|图)?|(?:印刷)?文件|二维码|最终稿|定稿)"
    current_preview = r"(?:这(?:一)?张|这(?:一)?版|这(?:一)?稿|这个)"
    acceptance = (
        rf"(?:好(?:的|了)?|可以了?|满意|没问题|确认|就(?:用|选)?{current_preview}"
        rf"|(?:用|选){current_preview}|{current_preview}(?:我)?(?:可以|满意|没问题))"
    )
    if re.fullmatch(
        rf"\s*(?:{acceptance}[，,。！!\s]+)?"
        r"(?:(?:可以|能不能|能|麻烦|请|直接)\s*)?"
        rf"(?:(?:发我|发给我|给我|帮我)(?:下载(?:链接)?|{handover}|导出)"
        rf"|发(?:送)?{handover}|把(?:{current_preview}的?)?{handover}(?:发(?:给)?我|给我))"
        r"(?:吗|吧|一下)?[。！!？?\s]*",
        body,
    ):
        return "single"
    if REJECT.search(body):
        return None
    clean = body.strip()
    if re.fullmatch(
        rf"{acceptance}"
        rf"(?:[，,。！! ]*(?:就用{current_preview}|(?:就)?按{current_preview}(?:出|发){handover}"
        rf"|(?:麻烦|请)?(?:发我|给我|发一下|发|出)(?:{handover}|下载|链接)?(?:吧|一下)?))?"
        r"(?:吧|了)?[。！! ]*"
        rf"|(?:挺好[，,。！! ]*)?(?:就)?按{current_preview}来(?:吧)?[。！! ]*"
        r"|(?:麻烦|请)?(?:发我|给我|发一下|发)(?:这张(?:的)?|成品|下载)(?:成品|文件|链接)?(?:吧|一下)?[。！! ]*",
        clean,
    ):
        return "single"
    if len([word for word in CANDIDATE_WORDS if word in body]) != 1:
        return None
    match = re.fullmatch(
        r"(?:好(?:的)?[，, ]*)?(?:(?:麻烦|请)\s*)?"
        r"(?:我(?:选|确认|要|用)|确认|就选|选定|选用|选|就用|就要|用|要"
        r"|(?:给我)?发(?:我|一下)?|给我)?"
        r"[ 「“]*(浓郁版|清晰版)[ 」”]*(?:[，,。 ]*(?:请)?(?:发我|给我|下载|发一下|出)"
        r"(?:这版|成品图|成品|打包文件|二维码|和|、|文件|一下| |[，,])*)?"
        r"(?:就好|就行|吧)?[。！! ]*",
        clean,
    )
    return CANDIDATE_WORDS[match[1]] if match else None


def preview_shown(events, origin, current, job, candidate):
    def is_turn(event, run):
        message = event.get("message", {})
        return message.get("role") == "user" and (
            event.get("id") == run or message.get("idempotencyKey") == f"{run}:user")

    end = next((i for i, event in enumerate(events) if is_turn(event, current)), None)
    if end is None:
        return False
    start = next((i for i, event in enumerate(events[:end]) if is_turn(event, origin)), None)
    if start is None:
        return False
    name = f"{job}-{candidate}-原图与候选预览.png"
    for event in events[start + 1:end]:
        message = event.get("message", {})
        if message.get("role") == "assistant" and message.get("oasnWebchatFinalReply") is True:
            for part in message.get("content", []):
                if isinstance(part, dict) and part.get("type") == "image" and part.get("alt") == name:
                    return True
    return False


NEW_REQUEST = re.compile(r"另做|再做一(?:个|张|份)|新做|新任务|换个(?:网址|链接|二维码)|重新做")
KEEP_ART = re.compile(r"(?:沿用|保留|继续用)(?:同一张|上一张|原来的)?(?:主图|原图)|原图不变|只(?:改|换)(?:网址|链接)")


def result(message, status, media=()):
    return {"message": message, "status": status, "media": list(media)}


def preview_reply(receipt):
    entries = receipt["candidates"]
    if receipt.get("single_preview"):
        row = next(iter(entries.values()))
        if row["exact_match"] and row["selectable"]:
            message = ("左边是原图，右边是做好的预览。扫码会打开你给的网页。"
                       "想把哪个细节再做得明显一点，可以直接告诉我。这张合适就回「就用这张」。")
        else:
            message = ("先发预览给你看。这张还没通过读码检查，暂时不能出成品。"
                       "可以告诉我能不能把码做大一点，或换一张轮廓更清楚的原图。")
        return result(message, "preview", [row["preview"]])
    good = [CANDIDATES[key] for key, row in entries.items()
            if row["exact_match"] and row["selectable"]]
    choices = [f"「{name}」" for name in good]
    if len(good) == 2:
        message = ("两版预览都做好了，读码都得到了你给的完整网址。浓郁版更突出颜色和纹理，清晰版的码点对比更强。"
                   f"看好后直接回{'或'.join(choices)}，我就发这版的成品。")
    elif good:
        message = (f"预览做好了。{good[0]}已读回完整网址，可以出成品。另一版可以对照看。"
                   f"喜欢这版就直接回{choices[0]}。")
    else:
        message = "先给你看两版预览。目前两版都没通过读码确认。可以把原图和网址发来试一组更大的码，或者换一张轮廓更清楚的图。"
    return result(message, "preview", [row["preview"] for row in entries.values()])


def source_still_matches(receipt):
    for row in receipt["candidates"].values():
        path = Path(row["preview"])
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != row["comparison_sha256"]:
            return False
    return True


def revision_inputs(session, job_id):
    jobs = ROOT / "jobs"
    error = "这单的原素材暂时找不到。请把原图和扫码地址一起重发，我就能继续改。"
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{12}", job_id):
        raise ValueError(error)
    directory = jobs / job_id
    try:
        binding = read_json(directory / "native.json", {})
        job = read_json(directory / "job.json", {})
        if binding.get("session") != session or binding.get("job") != job_id:
            raise ValueError
        paths = {}
        for name, expected in binding["inputs"].items():
            source = job["inputs"][name]
            path = directory / "inputs" / source["stored_as"]
            if (path.resolve().parent != directory / "inputs" or path.is_symlink()
                    or source["sha256"] != expected or sha256(path.read_bytes()).hexdigest() != expected):
                raise ValueError
            paths[name] = path
        url = binding["url"]
        if "source_qr" in paths:
            if url is not None or job["payload_source"] != "original-qr":
                raise ValueError
        elif url != job["payload"] or job["payload_source"] != "url":
            raise ValueError
        return Intake(paths["image"], url, paths.get("source_qr"), binding["image_event"], binding["target_event"])
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        raise ValueError(error) from None


def done_reply(state, paths):
    intro = ("成品可以下载了。" if state["receipt"].get("single_preview")
             else f"你选的{CANDIDATES[state['candidate']]}可以下载了。")
    return result(intro + "下面可以直接保存二维码图片，也可以下载整套文件。", "delivered",
                  [paths["png"], paths["zip"]])


def options_from(raw):
    # These are design options only. No path, URL, approval, or job ID can come from the model.
    names = ("title", "brief", "caption", "layout", "width", "height", "qr_percent", "art_px", "fusion_style")
    values = {key: raw.get(key) for key in names}
    values["layout"] = values["layout"] or "auto"
    values["fusion_style"] = values["fusion_style"] or "auto"
    values["title"] = values["title"] or ""
    values["brief"] = values["brief"] or ""
    if raw.get("omit_print_copy") is True:
        values["omit_print_copy"] = True
    return values


def supplied_options(raw):
    return {key: value for key, value in options_from(raw).items()
            if value not in (None, "", "auto")}


def question_reply(body):
    question = body.strip().strip("。！？!?.，, ")
    greeting = r"你好[呀啊]?|您好|哈[喽啰]|嗨|hi|hello"
    introduction = r"怎么开始|你能做什么|你是做什么的|你是谁|(?:这个)?怎么用|你会做什么|(?:可以|能不能)?(?:请)?介绍一下(?:你自己|自己|你的功能)?"
    if re.fullmatch(rf"(?:{greeting})|(?:(?:{greeting})[，,！! ]+)?(?:{introduction})(?:吗|呀)?", question, re.I):
        return INTRO
    if re.fullmatch(rf"(?:(?:{greeting})[，,！! ]+)?我想做(?:一?个|一?张)?艺术二维码[。！!，, ]+(?:你能帮我做什么|要先发你什么|怎么开始)", question, re.I):
        return INTRO
    if ("两版" in question or all(name in question for name in CANDIDATE_WORDS)) and re.search(r"区别|不同|怎么选|哪个好", question):
        return "现在默认按你的用途做一张预览。你可以直接告诉我图里哪部分要保留、哪里要改。如果是在取以前的稿子，可以回那张预览下的版名。"
    asks_about_print_scan = (
        re.search(r"印刷|打印|试扫|都能扫|保证.*扫|扫描", question)
        and re.search(r"保证|肯定|一定|都能|都扫|手机|反光|打样", question)
    )
    print_scan_reply = "出件前会用软件读回你给的完整网址。准备印刷时，可以按实际尺寸和材质试印一张，用手机在实际距离和光线下试扫。"
    if re.search(r"CMYK|TIFF|矢量|SVG|出血|(?:动态|批量|活码)(?:二维码|码)?|访问统计|点击统计", question, re.I):
        reply = ("我做的是通往固定网页的艺术二维码，可交付 RGB PNG、ZIP 和实际尺寸打样 PDF。"
                 "CMYK、可编辑矢量源稿、印厂出血、动态改链、访问统计和批量出码需要另用相应工具。")
        return reply + print_scan_reply if asks_about_print_scan else reply
    if (re.search(r"弧面|曲面|瓶贴|玻璃瓶", question)
            and re.search(r"印|打样|试扫|尺寸|多大|摆放|贴在哪", question)):
        return ("把二维码放在瓶身正面中央，四周留白，别让码的两边绕到瓶侧。"
                "尺寸要看瓶身直径、贴纸宽高和通常的扫码距离。把这些告诉我，我可以帮你定一个打样尺寸。"
                "先用正式材质按实际大小试印，贴到瓶上，用手机从正面和略偏两侧试扫，再换个光线检查反光。")
    if asks_about_print_scan:
        return print_scan_reply
    if re.search(r"图片|原图|主图|照片|文件", question) and re.search(r"能传|怎么传|多大|限制|上限|支持.*(?:格式|图片)|清晰度|压缩|画质", question):
        return "发一张清楚的 PNG、JPG 或 WebP 原图，12 MB 以内，短边至少 48 像素，总像素不超过 2400 万。我会保留原图文件，成品用无损 PNG，二维码画布会按用途和读码要求排版。"
    if re.search(r"格式|PNG|PDF|ZIP|(?:什么|哪些|哪几种|几种)文件", question, re.I) and re.search(r"下载|导出|成品|交付|支持|可以|有", question):
        return "选好后可以下载二维码 PNG 和 ZIP 成品包。做展签、台卡或货架卡时，包里还会有按成品尺寸制作的打样 PDF。"
    if re.search(r"印刷|打印|印(?:成)?(?:展签|贴纸|台卡|货架卡)", question) and re.search(r"能|可以|适合|怎么|多大", question):
        return ("可以。成品准备放在哪儿，大概多宽、多高？比如「贴在包装盒上，宽 6 厘米、高 7 厘米」。"
                "还没量好也可以先说用途，把原图和网址发来，我先做一张给你看。")
    return None


def on_turn(session, events, current, design, *, formal_inbound=None, reply_only=False):
    message = current["message"]
    body = message_text(message)
    received_files = bool(message.get("MediaPaths") or message.get("MediaTypes"))
    has_address = bool(URL_HINT.search(body))
    recovery_text = re.sub(r"吗?[?？]\s*$", "", body)
    recover = (not received_files and not has_address and not REJECT.search(recovery_text)
               and RECOVER.search(recovery_text))
    if not received_files and not has_address:
        answer = question_reply(body)
        if answer:
            return result(answer, "answered")
    if reply_only and (received_files or has_address or not (approval(body) or recover)):
        return None
    current_id = current["id"]
    formal = formal_inbound is not None
    managed_key = current.get("managed_key")
    with session_state(ROOT, session) as state_path:
        state = read_json(state_path, {})
        if state.get("managed_key") not in (None, managed_key):
            return result("这段聊天还没有可选的预览。请把主图和网址一起发来。", "missing_preview")
        reference = state.get("reference")
        code = state.get("selection_code")
        # A verified invocation can continue the preview stored for this chat
        # even when the channel does not expose OpenClaw's transcript markers.
        same_managed_chat = not events and managed_key and managed_key == state.get("managed_key")
        design_change = supplied_options({
            **design, "omit_print_copy": bool(OMIT_PRINT_COPY.search(body))
        })
        if state.get("event") == current_id and state.get("receipt"):
            if source_still_matches(state["receipt"]):
                return preview_reply(state["receipt"])
        retry_request = (not received_files and not design_change and not has_address
                         and not REJECT.search(body) and RETRY.search(body))
        retry_pending = not formal and state.get("pending") and retry_request
        same_chat_recovery = not formal or bool(managed_key and managed_key == state.get("managed_key"))
        if (state.get("pending") and not state.get("selection_paused") and not received_files
                and not NEW_REQUEST.search(body) and (state.get("event") == current_id
                or (same_chat_recovery and (recover or retry_request)))):
            prepared = ROOT / "jobs" / state["job"] / "openagent" / "preview-receipt.json"
            if prepared.is_file():
                receipt = read_json(prepared)
                if receipt["job"] == state["job"] and source_still_matches(receipt):
                    state.update(receipt=receipt, pending=False)
                    state.pop("intake", None)
                    state.pop("revision_source", None)
                    save_state(state_path, state)
                    return preview_reply(receipt)
            if formal and state.get("event") != current_id:
                return result("这单还没有完整预览。请把主图和网址一起重发，我再继续制作。", "needs_input")
        candidate = approval(body) if not received_files and not has_address else None
        code_ok = False
        if formal and not received_files and not has_address:
            supplied_refs = FORMAL_REFERENCE.findall(body)
            supplied_code = SELECTION_CODE.search(body)
            stripped = SELECTION_CODE.sub("", FORMAL_REFERENCE.sub("", body)).strip()
            candidate = approval(stripped)
            if supplied_refs:
                code_ok = (len(supplied_refs) == 1 and not supplied_code
                           and supplied_refs[0].lower() == reference)
            elif supplied_code:
                code_ok = supplied_code[1] == code
            if (candidate or recover) and (supplied_refs or supplied_code) and not code_ok:
                return result("这条选稿回复对应的是另一组预览。请看当前聊天里最新一组，再回复图下方的选项。", "missing_preview")
        elif formal:
            recover = False
        if state.get("receipt") and not NEW_REQUEST.search(body) and (candidate or recover):
            entries = state["receipt"]["candidates"]
            if state.get("selection_paused"):
                return result(RENEW, "missing_preview")
            if recover and not candidate and "预览" in body:
                if source_still_matches(state["receipt"]):
                    return preview_reply(state["receipt"])
                return result(RENEW, "missing_preview")
            if candidate == "single":
                if len(entries) != 1:
                    return preview_reply(state["receipt"])
                candidate = next(iter(entries))
            if candidate and candidate not in entries:
                return preview_reply(state["receipt"])
            if state.get("candidate") and (candidate is None or candidate == state["candidate"]):
                candidate = state["candidate"]
                event_id = state["approved_event"]
            else:
                event_id = current_id
            if not candidate:
                if not source_still_matches(state["receipt"]):
                    return result(RENEW, "missing_preview")
                return preview_reply(state["receipt"])
            row = entries[candidate]
            if not (row["selectable"] and row["exact_match"]):
                return preview_reply(state["receipt"])
            if not source_still_matches(state["receipt"]):
                return result(RENEW, "missing_preview")
            if not (code_ok or same_managed_chat
                    or preview_shown(events, state["origin"], current_id, state["job"], candidate)):
                if events:
                    return preview_reply(state["receipt"])
                return result(RENEW, "missing_preview")
            if state.get("candidate") and state["candidate"] != candidate:
                return result("这组已经出了成品。想换一版，请把主图和网址一起发来，注明要换哪版。", "already_selected")
            app.config["ARTQR_APPROVAL"] = {
                "session": session, "origin": state["origin"], "job": state["job"],
                "candidate": candidate, "event": event_id,
            }
            with workbench() as port:
                paths = delivery(SimpleNamespace(job=state["job"], candidate=candidate), port)
            state.update(candidate=candidate, approved_event=event_id, delivered=paths)
            save_state(state_path, state)
            return done_reply(state, paths)
        if candidate:
            return result("这段聊天里还没有预览。先发原图和完整网址，我做好后发给你看。", "missing_preview")
        if (not received_files and not has_address and not NEW_REQUEST.search(body) and state.get("receipt")
                and DECLINE_SELECTION.search(body) and not design_change):
            return result("可以。你想改图片里的哪个细节，还是要改文字、尺寸或排版？我可以用这单的原图和网址继续调整。", "needs_input")
        if reply_only:
            return None
        text_only = not received_files and not has_address and not USE_OLD.search(body)
        revision_source = None
        if formal and same_chat_recovery and text_only and not NEW_REQUEST.search(body):
            if state.get("pending") and state.get("event") == current_id:
                revision_source = state.get("revision_source")
            elif state.get("receipt") and not state.get("selection_paused") and design_change:
                if ({"width", "height", "layout", "fusion_style", "title", "caption", "omit_print_copy"}
                        & design_change.keys() or re.search(r"改|调整|加大|缩小|放大|移到|保留|突出", body)):
                    revision_source = state.get("job")
        if formal and text_only and not revision_source:
            if NEW_REQUEST.search(body):
                state = {}
                save_state(state_path, state)
            receipt = state.get("receipt")
            if receipt and source_still_matches(receipt):
                if any(name in body for name in CANDIDATE_WORDS):
                    return preview_reply(receipt)
            if managed_key and not receipt and not state.get("pending"):
                if design_change:
                    state.update(managed_key=managed_key, start=state.get("start", current_id),
                                 supplied_design={**state.get("supplied_design", {}), **design_change},
                                 request_text="\n".join(filter(None, (state.get("request_text"), body)))[-450:])
                    save_state(state_path, state)
                intake = state.get("intake", {})
                reply = missing_material_reply(intake.get("image_event"), intake.get("target_event"),
                                               sized={"width", "height"} <= design_change.keys())
                return result(reply, "needs_input")
            return result(MATERIALS, "needs_input")

        # After a preview, a changed picture or an explicit new task starts a
        # fresh pair. A design-only revision can reuse the same input pair.
        start = state.get("start")
        fresh_pair = ((formal and not revision_source) or NEW_REQUEST.search(body)
                      or (state.get("receipt") and received_files and "沿用" not in body and "同一个" not in body)
                      or (state.get("receipt") and has_address and not received_files and not KEEP_ART.search(body)))
        if fresh_pair:
            start = current_id
        elif (state.get("receipt") and not received_files and not URL_HINT.search(body)
              and "重新预览" not in body
              and not design_change):
            if source_still_matches(state["receipt"]):
                return preview_reply(state["receipt"])
            return result("原预览文件没找到。请回复「重新预览」。", "missing_preview")
        relevant = [current] if formal else []
        if not formal:
            in_scope = start is None
            for event in events:
                in_scope = in_scope or event["id"] == start
                if in_scope and event.get("message", {}).get("role") == "user":
                    relevant.append(event)
                    if retry_pending and event["id"] == state["event"]:
                        break
        if not relevant:
            relevant = [current]
            start = current_id
        # Error feedback comes from the actual files and exact address. Save a
        # fresh pair boundary even when one item has not yet arrived.
        previous = state.get("intake") if formal and same_chat_recovery and not NEW_REQUEST.search(body) else None
        try:
            inputs = (revision_inputs(session, revision_source) if revision_source
                      else collect_inputs(relevant, formal_inbound))
        except ValueError as error:
            if not previous and not revision_source:
                old_design = {} if fresh_pair else state.get("supplied_design", {})
                save_state(state_path, {"start": start or relevant[0]["id"],
                                        "supplied_design": {**old_design, **design_change}})
            return result(str(error), "needs_input")
        supplemented = bool(previous and supplement_intake(state_path, previous, inputs))
        if supplemented:
            start = state["start"]
        carry_request = same_chat_recovery and not NEW_REQUEST.search(body) and (supplemented or not previous)
        request_text = state.get("request_text") if carry_request else None
        old_design = state.get("supplied_design", {}) if supplemented or not fresh_pair or request_text else {}
        supplied = {**old_design, **design_change}
        if "omit_print_copy" not in design_change and {"title", "caption"} & design_change.keys():
            supplied.pop("omit_print_copy", None)
        if pending := inputs.reply(sized={"width", "height"} <= supplied.keys()):
            state = {"start": start or relevant[0]["id"], "supplied_design": supplied}
            if formal and (inputs.image or inputs.url or inputs.source_qr):
                state.update(managed_key=managed_key, intake=save_incomplete_intake(state_path, inputs, body))
            if request_text:
                state["request_text"] = request_text
            save_state(state_path, state)
            return result(pending, "needs_input")

        origin = relevant[-1]["id"] if retry_pending else current_id
        resuming = state.get("pending") and state.get("event") == origin
        design = state["design"] if resuming else {**options_from(supplied), "single_preview": True}
        user_texts = [message_text(event["message"]) for event in relevant]
        brief = "\n".join(user_texts[-4:])
        if supplemented:
            user_texts.insert(0, previous["text"])
            brief = f"{previous['text']}\n{brief}"
        if request_text:
            user_texts.insert(0, request_text)
            brief = f"{request_text}\n{brief}"
        if not resuming:
            for key, blank in (("title", ""), ("caption", None)):
                if revision_source and key not in design_change:
                    design[key] = state["design"][key]
                    continue
                if design[key] and not any(design[key] in text for text in user_texts):
                    design[key] = blank
            for text in reversed(user_texts):
                if OMIT_PRINT_COPY.search(text):
                    design["omit_print_copy"] = True
                    break
                if any(design[key] and design[key] in text for key in ("title", "caption")):
                    design.pop("omit_print_copy", None)
                    break
        # Use the actual user wording as the brief when no summary is supplied.
        if revision_source:
            if not resuming:
                revision = (design_change.get("brief") or body)[-450:]
                old_brief = state["design"]["brief"][:max(0, 449 - len(revision))]
                design["brief"] = "\n".join(filter(None, (old_brief, revision)))
        else:
            design["brief"] = brief[:450] if request_text else design["brief"] or brief[:450]
        request = SimpleNamespace(image=inputs.image, url=inputs.url, source_qr=inputs.source_qr,
                                  logo=None, context=None, background=None, **design)
        sha = {key: sha256(path.read_bytes()).hexdigest() for key, path in
               (("image", inputs.image), ("source_qr", inputs.source_qr)) if path is not None}
        identity = {"session": session, "origin": origin, "image_event": inputs.image_event,
                    "target_event": inputs.target_event, "url": inputs.url, "inputs": sha, "design": design}
        job = sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
        state = {"start": start or relevant[0]["id"], "origin": origin, "event": origin,
                 "job": job, "pending": True, "design": design, "supplied_design": supplied}
        if managed_key:
            state["managed_key"] = managed_key
        if supplemented:
            state["intake"] = previous
        if revision_source:
            state["revision_source"] = revision_source
        save_state(state_path, state)
        app.config["ARTQR_CREATE"] = {**identity, "job": job}
        with workbench() as port:
            print("ARTQR_PREVIEW_STARTED", file=sys.stderr, flush=True)
            receipt = preview(request, port)
        state.update(receipt=receipt, pending=False)
        state.pop("intake", None)
        state.pop("revision_source", None)
        save_state(state_path, state)
        return preview_reply(receipt)


def formal_options(body, proposed):
    # The model extracts options, but the current prompt must contain the value.
    design = {key: value for key, value in supplied_options(proposed).items()
              if key in {"title", "caption", "brief"} and isinstance(value, str) and value in body}
    number = r"(\d+(?:\.\d+)?)"
    units = r"(毫米|mm|厘米|公分|cm)"
    dimensions = re.search(rf"{number}\s*[×xX*]\s*{number}\s*{units}", body, re.I)
    if dimensions:
        scale = 1 if dimensions[3].lower() in {"毫米", "mm"} else 10
        design.update(width=float(dimensions[1]) * scale, height=float(dimensions[2]) * scale)
    else:
        square = re.search(rf"{number}\s*{units}\s*见方", body, re.I)
        named_dimensions = re.search(rf"宽\s*{number}\s*{units}[^。；！？]{{0,10}}高\s*{number}\s*{units}", body, re.I)
        if square:
            side = float(square[1]) * (1 if square[2].lower() in {"毫米", "mm"} else 10)
            design.update(width=side, height=side)
        elif named_dimensions:
            width, width_unit, height, height_unit = named_dimensions.groups()
            design.update(width=float(width) * (1 if width_unit.lower() in {"毫米", "mm"} else 10),
                          height=float(height) * (1 if height_unit.lower() in {"毫米", "mm"} else 10))
    layouts = {"cup": "杯形", "round": "圆形", "sticker": "贴纸", "bar": "吧台",
               "arch": "拱形", "label": "展签", "shelf": "货架卡", "beads": "拼豆"}
    layout = proposed.get("layout")
    if layout in layouts and layouts[layout] in body:
        design["layout"] = layout
    styles = {"print": "印刷肌理", "glass": "玻璃感", "weave": "编织感"}
    style = proposed.get("fusion_style")
    if style in styles and styles[style] in body:
        design["fusion_style"] = style
    return design


def stage_formal_attachments(attachments, incoming, files):
    if not isinstance(attachments, list) or not all(isinstance(name, str) for name in attachments):
        raise ValueError("附件没有传完整，请重新发送本次的图片和网址。")
    if len(attachments) > 2:
        raise ValueError("请只附一张主图。要沿用旧码时，再另外附上那张旧码。")
    paths, types = [], []
    for index, name in enumerate(attachments):
        path = Path(name)
        if (path.parent != files or path.resolve() != path or not path.is_file()
                or path.stat().st_size > MAX_ATTACHMENT_BYTES):
            raise ValueError("请重新附一张可打开的 PNG、JPG 或 WebP 主图。")
        suffix, mime = verified_image_type(path)
        staged = incoming / f"image-{index}{suffix}"
        copyfile(path, staged)
        paths.append(str(staged))
        types.append(mime)
    return paths, types


def input_retry(reason, session=None, key=None):
    print(f"artqr_input_retry: {reason}", file=sys.stderr)
    if session and key:
        with session_state(ROOT, f"formal:{session}") as state_path:
            state = read_json(state_path, {})
            if state.get("managed_key") == key:
                if state.get("intake"):
                    save_state(state_path, {})
                elif state.get("receipt") or state.get("pending"):
                    state["selection_paused"] = True
                    save_state(state_path, state)
    return result("我这次没收到完整的内容。请在这里重发刚才的消息；有图片的话，把图片也一起附上。", "retry")


def formal_turn(context, proposed, *, reply_only=False):
    def reject(reason, session=None, key=None):
        return None if reply_only else input_retry(reason, session, key)

    session, key, tool_run = context.get("sessionId"), context.get("sessionKey"), context.get("runId")
    if not isinstance(session, str) or not session or not isinstance(key, str) or not key:
        return reject("session_unbound")
    if not isinstance(tool_run, str) or not tool_run:
        return reject("gateway_run_unbound", session, key)

    missing = object()
    envelope = missing
    if FORMAL_RUN.fullmatch(tool_run):
        invocation = FORMAL_RUNS / tool_run / "in" / "invocation.json"
        envelope = read_json(invocation, missing)
    if envelope is missing:
        invocation = FORMAL_INPUT
        envelope = read_json(invocation, missing)
    developer_run = key == FORMAL_SESSION and DEVELOPER_RUN.fullmatch(tool_run)
    stale_fixed_envelope = (developer_run and invocation == FORMAL_INPUT
                            and isinstance(envelope, dict)
                            and envelope.get("schema_version") == "oasn.invocation.v1"
                            and envelope.get("run_id") != tool_run)
    if envelope is missing or stale_fixed_envelope:
        # A developer chat can coexist with an older formal invocation.
        # Bind the developer transcript to this run without reading that older input.
        if key != FORMAL_SESSION or not (developer_run or FORMAL_RUN.fullmatch(tool_run)):
            return reject("invocation_missing", session, key)
        try:
            _, events, observed = current_transcript(context)
        except (OSError, ValueError):
            return reject("developer_message_unbound", session, key)
        message = observed["message"]
        if message.get("idempotencyKey") != f"{tool_run}:user":
            return reject("developer_message_unbound", session, key)
        current = {"id": tool_run, "message": message, "managed_key": key}
        return on_turn(f"formal:{session}", events, current,
                       formal_options(message_text(message), proposed), formal_inbound=INBOUND, reply_only=reply_only)

    # The platform owns this read-only file, and the plugin supplies the run ID
    # from the gateway's tool-call hook. A match binds the current prompt and
    # attachments without depending on how the channel wrote its chat history.
    if not isinstance(envelope, dict) or envelope.get("schema_version") != "oasn.invocation.v1":
        return reject("invocation_schema", session, key)
    run_id = envelope.get("run_id")
    if run_id != tool_run:
        return reject("invocation_run_mismatch", session, key)
    body = envelope.get("input")
    if not isinstance(body, dict) or not isinstance(body.get("user_prompt"), str):
        return reject("invocation_input_invalid", session, key)
    prompt = body["user_prompt"]
    if reply_only and body.get("attachments"):
        return None
    events = ()
    try:
        _, observed_events, observed = current_transcript(context)
        if observed["message"].get("idempotencyKey") == f"{run_id}:user":
            events = observed_events
    except (OSError, ValueError):
        pass  # The envelope is authoritative. A missing transcript only affects selection.
    with TemporaryDirectory(prefix="artqr-current-") as directory:
        incoming = Path(directory).resolve()
        try:
            paths, types = stage_formal_attachments(body.get("attachments"), incoming,
                                                    invocation.parent / "files")
        except ValueError:
            with session_state(ROOT, f"formal:{session}") as state_path:
                state = read_json(state_path, {})
                if (state.get("event") != run_id and state.get("managed_key") in (None, key)
                        and (not state.get("intake") or NEW_REQUEST.search(prompt))):
                    save_state(state_path, {})
            raise
        message = {"role": "user", "content": prompt, "MediaPaths": paths, "MediaTypes": types}
        current = {"id": run_id, "message": message, "managed_key": key}
        return on_turn(f"formal:{session}", events, current, formal_options(prompt, proposed),
                       formal_inbound=incoming, reply_only=reply_only)


def main():
    reply_only = False
    try:
        data = json.load(sys.stdin)
        context = data["context"]
        proposed = data.get("design") or {}
        reply_only = data.get("replyOnly") is True
        key = context.get("sessionKey") or ""
        if reply_only and not context.get("sessionId") and key:
            index = read_json(SESSIONS / "sessions.json", {})
            context["sessionId"] = index.get(key, {}).get("sessionId")
        if DASHBOARD.fullmatch(key):
            session_id, events, current = current_transcript(context)
            if reply_only and current["message"].get("idempotencyKey") != f"{context.get('runId')}:user":
                outcome = None
            else:
                outcome = on_turn(session_id, events, current, proposed, reply_only=reply_only)
        else:
            outcome = formal_turn(context, proposed, reply_only=reply_only)
    except json.JSONDecodeError as error:
        print(str(error), file=sys.stderr)
        outcome = result("这次没有读到完整消息。请在本聊天重试。", "retry")
    except ValueError as error:
        outcome = None if reply_only else result(str(error), "needs_input")
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        outcome = result(RETRY_REPLY, "retry")
    print(json.dumps(outcome, ensure_ascii=False))


if __name__ == "__main__":
    main()
