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

# ===== 皮肤发白像素检测（确定性判定，不依赖 LLM 感知）=====
# 采样区域：画面中央偏上（人脸最可能出现的位置），长宽各取 40%
_FACE_CROP = (0.30, 0.15, 0.70, 0.55)
# 正常肤色像素：红-黄色相、中等饱和、非暗部
_SKIN_HUE_MAX, _SKIN_HUE_MIN = 30, 235  # HSV 0-255 刻度下的红-黄区间
_SKIN_SAT_MIN, _SKIN_SAT_MAX = 38, 166  # S 0.15~0.65
_SKIN_VAL_MIN = 102  # V > 0.4
# 发白像素：低饱和 + 高亮（白到不真实）
_WHITE_SAT_MAX = 77  # S < 0.3
_WHITE_VAL_MIN = 191  # V > 0.75
# 判定阈值：发白像素占比高 且 正常肤色占比极低 → 皮肤发白
# 皮肤占比 <5%：真发白的脸几乎落不进正常肤色色域（1.jpg=1%）；
# 正常脸+白衣服/天空场景皮肤占比仍高（8/11/12.jpg=9-57%），不会被误判
PALE_SKIN_WHITE_RATIO = 0.15
PALE_SKIN_MAX_SKIN_RATIO = 0.05


_BLUE_HUE_LO, _BLUE_HUE_HI = 90, 160  # 蓝色调色相区间（HSV 0-255 刻度）
_BLUE_SAT_MIN = 30
_BLUE_VAL_MIN = 120
_BLUE_MAX_RATIO = 0.65  # 蓝色占比超过该值视为水体/天空场景，跳过判定


def detect_pale_skin(img: DecodedImage) -> bool:
    """像素级肤色发白检测：发白像素占比 > 15% 且 正常肤色像素占比 < 30% 即判定。

    发白皮肤（青白/灰白/惨白）低饱和高亮，几乎不落在正常肤色区间；
    正常皮肤（含暖调、健康肤色）占据采样区大部。阈值经 1/2/5/6/7.jpg 实测校准。
    蓝色主导（>65%）的采样区视为水体/天空场景，跳过判定（防水族馆等场景误报）。
    """
    w, h = img.width, img.height
    crop = (
        Image.open(io.BytesIO(img.data))
        .convert("RGB")
        .crop((int(w * _FACE_CROP[0]), int(h * _FACE_CROP[1]), int(w * _FACE_CROP[2]), int(h * _FACE_CROP[3])))
        .resize((200, 150))
        .convert("HSV")
    )
    px = list(crop.getdata())
    if not px:
        return False
    skin = 0
    white = 0
    blue = 0
    for hh, ss, vv in px:
        if vv > _SKIN_VAL_MIN and _SKIN_SAT_MIN < ss < _SKIN_SAT_MAX and (hh <= _SKIN_HUE_MAX or hh >= _SKIN_HUE_MIN):
            skin += 1
        if ss < _WHITE_SAT_MAX and vv > _WHITE_VAL_MIN:
            white += 1
        if _BLUE_HUE_LO < hh < _BLUE_HUE_HI and ss > _BLUE_SAT_MIN and vv > _BLUE_VAL_MIN:
            blue += 1
    n = len(px)
    if blue / n > _BLUE_MAX_RATIO:
        return False
    return (white / n) > PALE_SKIN_WHITE_RATIO and (skin / n) < PALE_SKIN_MAX_SKIN_RATIO


_EXIF_FOCAL_LENGTH = 0x920A  # FocalLength
_EXIF_FOCAL_35MM = 0xA405  # FocalLengthIn35mmFilm（优先，可直接判断焦段）


def extract_focal_length(data: bytes) -> float | None:
    """从 EXIF 读取实际焦距（mm）；优先 35mm 等效，回退原始焦距。无 EXIF/无焦距返回 None。"""
    try:
        exif = Image.open(io.BytesIO(data)).getexif()
        if not exif:
            return None
        if _EXIF_FOCAL_35MM in exif:
            value = exif[_EXIF_FOCAL_35MM]
            if isinstance(value, (int, float)):
                return float(value)
        if _EXIF_FOCAL_LENGTH in exif:
            value = exif[_EXIF_FOCAL_LENGTH]
            if isinstance(value, (int, float)):
                return float(value)
    except Exception:
        return None
    return None


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
