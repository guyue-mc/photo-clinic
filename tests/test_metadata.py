"""metadata.py：图片解码校验、EXIF/PNG 启发式、C2PA 降级。"""
from __future__ import annotations

import base64
import io
import sys

import pytest
from PIL import Image

from photo_clinic.metadata import (
    ImageTooLargeError,
    InvalidImageError,
    DecodedImage,
    decode_image,
    detect_pale_skin,
    inspect_metadata,
)


def _decoded_from_rgb(width: int, height: int, color: tuple[int, int, int]) -> DecodedImage:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG")
    return DecodedImage(data=buf.getvalue(), media_type="image/jpeg", format="JPEG",
                        width=width, height=height)


def test_detect_pale_skin_flags_white_skin():
    # 青白/灰白皮肤：低饱和高亮
    assert detect_pale_skin(_decoded_from_rgb(400, 400, (230, 228, 224))) is True


def test_detect_pale_skin_ok_for_warm_skin():
    # 正常暖调肤色：高饱和，落在肤色区间
    assert detect_pale_skin(_decoded_from_rgb(400, 400, (205, 155, 115))) is False


def test_detect_pale_skin_ok_for_dark_background():
    # 深色背景（无人脸信息）不应误判
    assert detect_pale_skin(_decoded_from_rgb(400, 400, (30, 60, 90))) is False

MAX = 10 * 1024 * 1024


def test_decode_valid_jpeg(jpeg_image):
    img = decode_image(jpeg_image, MAX)
    assert img.format == "JPEG"
    assert img.media_type == "image/jpeg"
    assert (img.width, img.height) == (64, 64)


def test_decode_text_file_raises():
    payload = base64.b64encode(b"this is not an image at all").decode()
    with pytest.raises(InvalidImageError):
        decode_image(payload, MAX)


def test_decode_bad_base64_raises():
    with pytest.raises(InvalidImageError):
        decode_image("not-valid-base64!!!", MAX)


def test_decode_too_large_raises(jpeg_image):
    with pytest.raises(ImageTooLargeError):
        decode_image(jpeg_image, max_bytes=10)


def test_decode_auto_compress_resizes_and_shrinks(large_noise_png):
    img = decode_image(large_noise_png, max_bytes=4 * 1024 * 1024, auto_compress=True)
    assert img.format == "JPEG"
    assert (img.width, img.height) == (2000, 1333)
    assert len(img.data) <= 4 * 1024 * 1024


def test_decode_auto_compress_still_enforces_hard_limit(large_noise_png):
    # 硬上限 = max_bytes × 5（此处 5MB < 17MB 图），超过仍拒绝
    with pytest.raises(ImageTooLargeError):
        decode_image(large_noise_png, max_bytes=1024 * 1024, auto_compress=True)


def test_compress_rgba_png_flattens_to_white_jpeg():
    buf = io.BytesIO()
    noise = Image.effect_noise((3000, 2000), 120).convert("RGB")
    noise.putalpha(Image.new("L", noise.size, 128))
    noise.save(buf, format="PNG")
    img = decode_image(base64.b64encode(buf.getvalue()).decode(), 3 * 1024 * 1024, auto_compress=True)
    assert img.format == "JPEG"
    assert img.width <= 2000 and img.height <= 2000
    assert len(img.data) <= 3 * 1024 * 1024


def test_decode_base64_oversized_payload_raises():
    """超长 base64 串在解码前即被拒绝，不做整体解码（内存 DoS 防护）。"""
    payload = base64.b64encode(b"A" * 100).decode()
    with pytest.raises(ImageTooLargeError):
        decode_image(payload, max_bytes=10)


def test_decode_unsupported_format_raises():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="BMP")
    payload = base64.b64encode(buf.getvalue()).decode()
    with pytest.raises(InvalidImageError):
        decode_image(payload, MAX)


def test_decode_base64_with_whitespace(jpeg_image):
    chunked = "\n".join(jpeg_image[i : i + 32] for i in range(0, len(jpeg_image), 32))
    img = decode_image(" " + chunked + "\n", MAX)
    assert img.format == "JPEG"


def test_decode_gif():
    buf = io.BytesIO()
    Image.new("P", (8, 8)).save(buf, format="GIF")
    img = decode_image(base64.b64encode(buf.getvalue()).decode(), MAX)
    assert img.format == "GIF"
    assert img.media_type == "image/gif"


def test_decode_webp():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="WEBP")
    img = decode_image(base64.b64encode(buf.getvalue()).decode(), MAX)
    assert img.format == "WEBP"
    assert img.media_type == "image/webp"


def test_decode_empty_base64_raises():
    with pytest.raises(InvalidImageError):
        decode_image("", MAX)


def test_inspect_png_parameters_chunk(png_with_parameters):
    flags = inspect_metadata(decode_image(png_with_parameters, MAX))
    assert flags.has_parameters_chunk is True
    assert flags.evidence_strength == "strong"


def test_inspect_exif_ai_software(jpeg_with_ai_software):
    flags = inspect_metadata(decode_image(jpeg_with_ai_software, MAX))
    assert flags.suspicious_software == "Midjourney"
    assert flags.evidence_strength == "strong"


def test_inspect_clean_jpeg(jpeg_image):
    flags = inspect_metadata(decode_image(jpeg_image, MAX))
    assert flags.evidence_strength == "none"
    assert flags.suspicious_software is None
    assert flags.has_parameters_chunk is False
    # c2pa 未安装 → c2pa_available=False 且 c2pa_ai_manifest=None
    if not flags.c2pa_available:
        assert flags.c2pa_ai_manifest is None


def test_c2pa_missing_degrades_gracefully(jpeg_image, monkeypatch):
    import photo_clinic.metadata as metadata_mod

    monkeypatch.setattr(metadata_mod, "_C2PA_LOADABLE", None)  # 重置缓存
    monkeypatch.setitem(sys.modules, "c2pa", None)  # 模拟未安装
    flags = inspect_metadata(decode_image(jpeg_image, MAX))
    assert flags.c2pa_available is False
    assert flags.c2pa_ai_manifest is None
    assert flags.evidence_strength == "none"


def test_c2pa_native_load_failure_degrades_gracefully(jpeg_image, monkeypatch):
    """原生库加载失败（如服务器 glibc 过旧抛 RuntimeError）同样视为不可用。"""
    import builtins

    import photo_clinic.metadata as metadata_mod

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "c2pa":
            raise RuntimeError("GLIBC_2.33 not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(metadata_mod, "_C2PA_LOADABLE", None)
    assert metadata_mod.c2pa_available() is False
    # 评审链路不受影响
    flags = inspect_metadata(decode_image(jpeg_image, MAX))
    assert flags.c2pa_available is False
    assert flags.c2pa_ai_manifest is None
    assert flags.evidence_strength == "none"


def test_inspect_ai_software_case_insensitive():
    buf = io.BytesIO()
    img = Image.new("RGB", (8, 8))
    exif = Image.Exif()
    exif[0x0131] = "MiDjOuRnEy 6.1"
    img.save(buf, format="JPEG", exif=exif)
    flags = inspect_metadata(decode_image(base64.b64encode(buf.getvalue()).decode(), MAX))
    assert flags.suspicious_software == "MiDjOuRnEy 6.1"
    assert flags.evidence_strength == "strong"


def test_inspect_clean_software_ignored():
    buf = io.BytesIO()
    img = Image.new("RGB", (8, 8))
    exif = Image.Exif()
    exif[0x0131] = "Adobe Photoshop 2024"
    img.save(buf, format="JPEG", exif=exif)
    flags = inspect_metadata(decode_image(base64.b64encode(buf.getvalue()).decode(), MAX))
    assert flags.suspicious_software is None
    assert flags.evidence_strength == "none"


def test_inspect_c2pa_ai_manifest_strong(jpeg_image, monkeypatch):
    monkeypatch.setattr("photo_clinic.metadata._c2pa_ai_manifest", lambda data, media_type: True)
    flags = inspect_metadata(decode_image(jpeg_image, MAX))
    assert flags.c2pa_available is True
    assert flags.c2pa_ai_manifest is True
    assert flags.evidence_strength == "strong"


def test_inspect_c2pa_no_manifest(jpeg_image, monkeypatch):
    monkeypatch.setattr("photo_clinic.metadata._c2pa_ai_manifest", lambda data, media_type: False)
    flags = inspect_metadata(decode_image(jpeg_image, MAX))
    assert flags.c2pa_available is True
    assert flags.c2pa_ai_manifest is False
    assert flags.evidence_strength == "none"
