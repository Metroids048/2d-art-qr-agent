"""HTTP and QR decoding helpers shared by the conversation adapter."""

from http.client import HTTPConnection
from urllib.parse import urlencode
from uuid import uuid4

import cv2
import numpy as np


def request(port, path, *, method="GET", body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=180)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        result = connection.getresponse()
        return result.status, result.getheader("Location"), result.read()
    finally:
        connection.close()

def upload(port, fields, files):
    boundary = uuid4().hex
    sections = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    ]
    for field, path in files.items():
        sections.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'
            'Content-Type: application/octet-stream\r\n\r\n'.encode()
            + path.read_bytes() + b"\r\n"
        )
    sections.append(f"--{boundary}--\r\n".encode())
    status, location, body = request(
        port, "/jobs", method="POST", body=b"".join(sections),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert status == 303 and location and location.startswith("/jobs/"), (status, body[:200])
    return location

def read_png(png):
    raster = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert raster is not None
    value = cv2.QRCodeDetector().detectAndDecode(raster)[0]
    return value, list(raster.shape[:2][::-1])

def post_select(port, route, candidate, review=False):
    return request(
        port, f"{route}/select/{candidate}", method="POST",
        body=urlencode({"visual_review": "1"} if review else {}),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )[0]
