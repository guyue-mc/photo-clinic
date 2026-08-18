"""图片解码校验与 AI 生成元数据检测（C2PA / EXIF / PNG tEXt）。"""
from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass
from functools import cached_property

from PIL import Image

from photo_clinic.schemas import MetadataFlags

SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "GIF"})

FORMAT_TO_MEDIA_TYPE = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}

# EXIF Software 命中即高置信判 AI（不区分大小写包含匹配）
AI_SOFTWARE_HINTS = (
    "midjourney",
    "novelai",
    "fooocus",
    "stablediffusion",
    "stable diffusion",
    "dall-e",
    "dall·e",
    "firefly",
    "comfyui",
    "flux",
    "sdxl",
    "kandinsky",
    "dreamstudio",
    "ideogram",
    "leonardo",
)

# C2PA 标准值：内容由采样训练的模型算法生成
_TRAINED_ALGORITHMIC_MEDIA = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
)


class InvalidImageError(Exception):
    """base64 解码失败或不是支持的图片格式。"""


class InvalidMediaTypeError(Exception):
    """请求声明的 media_type 与图片实际格式不符。"""


class ImageTooLargeError(Exception):
    """解码后字节数超过上限。"""


def _strip_whitespace(s: str) -> str:
    """去除 base64 串中的空白字符；常规路径（无空白）只做一遍扫描，零复制。"""
    if not any(ch.isspace() for ch in s):
        return s
    return "".join(s.split())


@dataclass(frozen=True)
class DecodedImage:
    data: bytes
    media_type: str
    format: str
    width: int
    height: int

    @cached_property
    def b64(self) -> str:
        """base64 编码（惰性缓存）：一次请求多次 LLM 调用复用，不再重复编码整份图片。"""
        return base64.b64encode(self.data).decode()


# 自动压缩目标：长边上限（像素）
AUTO_COMPRESS_MAX_EDGE = 2000
# LLM 评审用图上限：长边 ≤3000px 且 ≤2MB（保留更多皮肤纹理细节供模型判断；上传自动压缩护栏不受影响）
LLM_IMAGE_MAX_EDGE = 3000
LLM_IMAGE_MAX_BYTES = 2 * 1024 * 1024
# 硬上限 = max_bytes 的 5 倍（防内存 DoS；自动压缩只在硬上限以内生效）
_HARD_LIMIT_MULTIPLIER = 5


def decode_image(
    image_base64: str, max_bytes: int, *, auto_compress: bool = False
) -> DecodedImage:
    """解码并校验图片；超过 max_bytes 时若 auto_compress 则自动压缩（长边 ≤2000px 且 ≤max_bytes）。"""
    compact = _strip_whitespace(image_base64)
    hard_limit = max_bytes * _HARD_LIMIT_MULTIPLIER
    # 解码前预检长度上限（base64 膨胀约 4/3），避免超大 payload 先整体解码再判大
    if len(compact) > (hard_limit * 4 + 2) // 3:
        raise ImageTooLargeError(f"图片超过 {hard_limit // 1024 // 1024}MB 硬上限")
    try:
        data = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError("base64 解码失败") from exc
    if len(data) > hard_limit:
        raise ImageTooLargeError(f"图片超过 {hard_limit // 1024 // 1024}MB 硬上限")
    if len(data) > max_bytes:
        if not auto_compress:
            raise ImageTooLargeError(f"图片超过 {max_bytes // 1024 // 1024}MB 上限")
        data = compress_image(data, max_bytes=max_bytes)
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        width, height = img.size
        img.verify()
    except Exception as exc:
        raise InvalidImageError(f"无法解析为图片: {exc}") from exc
    if fmt not in SUPPORTED_FORMATS:
        raise InvalidImageError(f"不支持的图片格式: {fmt}")
    media_type = FORMAT_TO_MEDIA_TYPE[fmt]
    return DecodedImage(data=data, media_type=media_type, format=fmt, width=width, height=height)


def compress_image(
    data: bytes, *, max_bytes: int, max_edge: int = AUTO_COMPRESS_MAX_EDGE
) -> bytes:
    """压缩到长边 ≤ max_edge 且字节 ≤ max_bytes（JPEG 逐级降质；透明图白底合成）。"""
    out, _, _ = _resize_reencode(data, max_edge=max_edge, max_bytes=max_bytes)
    return out


def downscale_image(
    img: DecodedImage,
    *,
    max_edge: int = LLM_IMAGE_MAX_EDGE,
    max_bytes: int = LLM_IMAGE_MAX_BYTES,
) -> DecodedImage:
    """LLM 评审用小图：长边与字节均已达标则原样返回，否则重编码为 JPEG。

    必须在元数据检测之后调用：重编码会丢弃 EXIF / PNG 参数块等 AI 证据。
    """
    if img.width <= max_edge and img.height <= max_edge and len(img.data) <= max_bytes:
        return img
    data, width, height = _resize_reencode(img.data, max_edge=max_edge, max_bytes=max_bytes)
    return DecodedImage(data=data, media_type="image/jpeg", format="JPEG", width=width, height=height)


def _resize_reencode(
    data: bytes, *, max_edge: int, max_bytes: int
) -> tuple[bytes, int, int]:
    """重编码为 JPEG：长边缩至 ≤ max_edge，逐级降质压到 ≤ max_bytes。"""
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        img.thumbnail((max_edge, max_edge))
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            img = background
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        width, height = img.size
        smallest = None
        for quality in (85, 70, 60, 50, 40):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            out = buf.getvalue()
            smallest = out
            if len(out) <= max_bytes:
                return out, width, height
        return smallest, width, height  # 最低质量仍超限（极端情况），返回最小体积版本


# c2pa 可用性缓存：导入可能因未安装（ImportError）或原生库加载失败（如系统
# glibc 过旧，RuntimeError）而失败，失败后不再每次重试（导入失败不会被 Python 缓存）
_C2PA_LOADABLE: bool | None = None


def c2pa_available() -> bool:
    global _C2PA_LOADABLE
    if _C2PA_LOADABLE is None:
        try:
            import c2pa  # noqa: F401

            _C2PA_LOADABLE = True
        except Exception:  # noqa: BLE001 — 未安装或原生库无法加载都视为不可用
            _C2PA_LOADABLE = False
    return _C2PA_LOADABLE


def _c2pa_ai_manifest(data: bytes, media_type: str) -> bool | None:
    """True=AI manifest；False=库可用但未命中；None=库不可用（未安装/无法加载）。"""
    if not c2pa_available():
        return None
    try:
        from c2pa import Reader
    except ImportError:
        return None
    try:
        with Reader(media_type, io.BytesIO(data)) as reader:
            manifest = reader.get_active_manifest()
        if not manifest:
            return False
        return manifest.get("digital_source_type") == _TRAINED_ALGORITHMIC_MEDIA
    except Exception:  # noqa: BLE001 — c2pa 库对无签名媒体可能抛各种异常，一律视为未命中
        return False


def inspect_metadata(img: DecodedImage) -> MetadataFlags:
    """检测 AI 生成元数据证据；strong 命中由 pipeline 短路判 AI。"""
    c2pa = _c2pa_ai_manifest(img.data, img.media_type)
    suspicious_software = None
    has_parameters = False
    with Image.open(io.BytesIO(img.data)) as im:
        try:
            software = im.getexif().get(0x0131)  # IFD0 Software 标签
        except Exception:  # noqa: BLE001 — 部分格式（如 GIF/WEBP）读 EXIF 会抛异常，视为无 EXIF
            software = None
        if software:
            value = str(software)
            if any(hint in value.lower() for hint in AI_SOFTWARE_HINTS):
                suspicious_software = value
        # PNG 的 tEXt/zTXt 键值在 open 时已进入 img.info
        has_parameters = "parameters" in (im.info or {})

    strength = "strong" if (c2pa is True or has_parameters or suspicious_software) else "none"
    return MetadataFlags(
        c2pa_available=c2pa is not None,
        c2pa_ai_manifest=c2pa,
        suspicious_software=suspicious_software,
        has_parameters_chunk=has_parameters,
        evidence_strength=strength,
    )
