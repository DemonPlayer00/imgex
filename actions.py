import hashlib
import os

import cv2
import numpy as np

# 默认密码（未通过 -p 提供时使用；其 SHA-256 摘要作为写入扩展区的校验码）
DEFAULT_PASSWORD = "477a3d43f692aeaf1c7f40c0c91bffde3e2e638d8e90c668422373ee82a18521"
# payload 格式版本（位于 SHA-256 校验码之后，1 字节）
PAYLOAD_VERSION = 1
# 每像素参与写入的通道数（BGR）
_CHANNELS = 3


def expand_image(image: np.ndarray, border_size: int) -> np.ndarray:
    if border_size == 0:
        return image.copy()  # 无扩展时返回原图副本

    # 使用 BORDER_REFLECT_101 模式（以边缘像素为轴镜像）
    expanded = cv2.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_REFLECT_101
    )
    return expanded


def calc_expand_pixels(width: int, height: int, need: int) -> int:
    """解 4k^2 + 2k(width+height) >= need，返回最小扩展圈数 k。"""
    if need <= 0:
        return 0
    k = 0
    while True:
        extra = 4 * k * k + 2 * k * (width + height)
        if extra >= need:
            return k
        k += 1


# ---------- 位拆分 / 合并 ----------

def split_byte(value: int):
    """将 0~255 按位奇偶拆为两个 0~15 的值。

    奇数位（bit 1,3,5,7）组成第一个值，偶数位（bit 2,4,6,8）组成第二个值。
    """
    odd = (((value >> 7) & 1) << 3) | (((value >> 5) & 1) << 2) | (((value >> 3) & 1) << 1) | ((value >> 1) & 1)
    even = (((value >> 6) & 1) << 3) | (((value >> 4) & 1) << 2) | (((value >> 2) & 1) << 1) | (value & 1)
    return odd, even


def merge_nibbles(odd: int, even: int) -> int:
    """split_byte 的逆运算：两个 0~15 的值合并回 0~255。"""
    value = 0
    for i in range(4):
        value |= ((odd >> i) & 1) << (2 * i + 1)
        value |= ((even >> i) & 1) << (2 * i)
    return value


def _payload_to_nibbles(payload: bytes) -> np.ndarray:
    """向量化位拆分：payload 每字节按位奇偶拆为两个 0~15 值（先奇位后偶位）。

    与逐字节 split_byte 结果完全一致。
    """
    arr = np.frombuffer(payload, dtype=np.uint8)
    odd = np.zeros(len(arr), dtype=np.uint8)
    even = np.zeros(len(arr), dtype=np.uint8)
    for i in range(4):
        odd |= ((arr >> (2 * i + 1)) & 1) << i   # bit 1,3,5,7 -> 值A 的 bit 0..3
        even |= ((arr >> (2 * i)) & 1) << i      # bit 2,4,6,8 -> 值B 的 bit 0..3
    nibbles = np.empty(2 * len(arr), dtype=np.uint8)
    nibbles[0::2] = odd
    nibbles[1::2] = even
    return nibbles


def _nibbles_to_bytes(nibbles: np.ndarray) -> bytes:
    """向量化位合并：nibble 流按 (奇位值, 偶位值) 成对合并回字节。

    与逐字节 merge_nibbles 结果完全一致；尾部奇数个 nibble 丢弃。
    """
    n = len(nibbles) // 2 * 2
    odd = nibbles[0:n:2]
    even = nibbles[1:n:2]
    merged = np.zeros(len(odd), dtype=np.uint8)
    for i in range(4):
        merged |= ((odd >> i) & 1) << (2 * i + 1)
        merged |= ((even >> i) & 1) << (2 * i)
    return merged.tobytes()


# ---------- 扩展区坐标 ----------

def expand_coords(height: int, width: int, k: int):
    """行主序返回 k 圈扩展区像素坐标（numpy 数组 (yy, xx)），从左上角 (0,0) 开始。

    数量 = 4k^2 + 2k(height + width)；跳过内部区域。
    """
    eh, ew = height + 2 * k, width + 2 * k
    yy, xx = np.mgrid[0:eh, 0:ew]
    ext = (yy < k) | (yy >= height + k) | (xx < k) | (xx >= width + k)
    return yy[ext], xx[ext]


# ---------- RLE 区间 ----------

def scan_runs(mask: np.ndarray, direction: int):
    """扫描连续不一致区间。

    direction=0：逐行扫描；direction=1：逐列扫描。
    返回 [(index, [(start, end), ...]), ...]，仅包含非空行/列。
    """
    m = mask if direction == 0 else mask.T
    entries = []
    for i in range(m.shape[0]):
        runs = []
        j = 0
        n = m.shape[1]
        row = m[i]
        while j < n:
            if row[j]:
                s = j
                while j < n and row[j]:
                    j += 1
                runs.append((s, j - 1))
            else:
                j += 1
        if runs:
            entries.append((i, runs))
    return entries


def serialize_entries(entries, direction: int) -> bytes:
    """区间数据序列化：方向(1B) + 非空行/列数(4B) + 每项[索引(4B), 区间数(4B), 区间*(起点(4B), 终点(4B))]。"""
    out = bytearray([direction])
    out += len(entries).to_bytes(4, "big")
    for idx, runs in entries:
        out += idx.to_bytes(4, "big")
        out += len(runs).to_bytes(4, "big")
        for s, e in runs:
            out += s.to_bytes(4, "big")
            out += e.to_bytes(4, "big")
    return bytes(out)


def parse_entries(data: bytes, pos: int):
    """serialize_entries 的逆运算。返回 (direction, entries, 结束位置)。"""
    direction = data[pos]
    pos += 1
    count = int.from_bytes(data[pos:pos + 4], "big")
    pos += 4
    entries = []
    for _ in range(count):
        idx = int.from_bytes(data[pos:pos + 4], "big")
        pos += 4
        n = int.from_bytes(data[pos:pos + 4], "big")
        pos += 4
        runs = []
        for _ in range(n):
            s = int.from_bytes(data[pos:pos + 4], "big")
            pos += 4
            e = int.from_bytes(data[pos:pos + 4], "big")
            pos += 4
            runs.append((s, e))
        entries.append((idx, runs))
    return direction, entries, pos


def _entries_to_coords(entries, direction: int):
    """把区间列表展开为 (ys, xs) 索引数组（保持区间扫描顺序）。"""
    ys_list = []
    xs_list = []
    for idx, runs in entries:
        for s, e in runs:
            if direction == 0:
                xs = np.arange(s, e + 1, dtype=np.intp)
                ys = np.full(len(xs), idx, dtype=np.intp)
            else:
                ys = np.arange(s, e + 1, dtype=np.intp)
                xs = np.full(len(ys), idx, dtype=np.intp)
            ys_list.append(ys)
            xs_list.append(xs)
    if not ys_list:
        return None, None
    return np.concatenate(ys_list), np.concatenate(xs_list)


def serialize_values(image: np.ndarray, entries, direction: int) -> bytes:
    """按区间扫描顺序逐像素取原图的 BGR 值（每像素 3 字节，向量化）。"""
    ys, xs = _entries_to_coords(entries, direction)
    if ys is None:
        return b""
    return image[ys, xs].tobytes()


def apply_values(image: np.ndarray, entries, direction: int, values: bytes):
    """把原值流按区间顺序写回 image（还原原图，向量化）。返回消费的字节数。"""
    ys, xs = _entries_to_coords(entries, direction)
    if ys is None:
        return 0
    vals = np.frombuffer(values, dtype=np.uint8)
    n = len(ys)
    image[ys, xs] = vals[:n * _CHANNELS].reshape(n, _CHANNELS)
    return n * _CHANNELS


def build_payload(password: str, diff: np.ndarray, original: np.ndarray):
    """组装 payload：SHA-256(密码) + 版本 + 压缩后的差异记录 + 原像素值。

    横向/纵向两种区间编码选体积较小者。返回 (payload, entries, direction)。
    """
    h_entries = scan_runs(diff, 0)
    v_entries = scan_runs(diff, 1)
    h_blob = serialize_entries(h_entries, 0)
    v_blob = serialize_entries(v_entries, 1)
    if len(h_blob) <= len(v_blob):
        blob, direction, entries = h_blob, 0, h_entries
    else:
        blob, direction, entries = v_blob, 1, v_entries
    values = serialize_values(original, entries, direction)
    sha = hashlib.sha256(password.encode("utf-8")).hexdigest().encode("ascii")
    return sha + bytes([PAYLOAD_VERSION]) + blob + values, entries, direction


def encode(original_path, coded_path, password=None, output_path=None):
    """编码模式：将原图与处理后图的差异写入处理后图的镜像扩展区。

    payload 每字节按位奇偶拆为两个 0~15 的值，按 BGR 顺序写入扩展区像素：
    默认对理论镜像值做加法偏移，若加法溢出（>255）则改做减法（差值恒 >=226，
    不会双向溢出），解码端取 |实际 - 理论| 绝对值即可还原。
    """
    password = password or DEFAULT_PASSWORD
    original = cv2.imread(original_path)
    coded = cv2.imread(coded_path)
    if original is None or coded is None:
        raise ValueError("无法读取图片（路径不存在或格式不支持）")
    if original.shape != coded.shape:
        raise ValueError("原图与处理后图的尺寸/通道数不一致")

    height, width = coded.shape[:2]
    diff = np.any(original != coded, axis=2)
    payload, entries, direction = build_payload(password, diff, original)

    # 每字节 2 个 nibble，每像素 3 个通道
    need_pixels = (2 * len(payload) + _CHANNELS - 1) // _CHANNELS
    k = calc_expand_pixels(width, height, need_pixels)
    max_k = (min(height, width) - 1) // 2
    if k > max_k:
        raise ValueError(f"图片过小，扩展区无法容纳差异数据（需要 {k} 圈，上限 {max_k} 圈）")

    expanded = expand_image(coded, k)

    # 向量化写入：扩展区理论值 + 偏移（加法溢出转减法，保证不双向溢出）
    yy, xx = expand_coords(height, width, k)
    theory_vals = expanded[yy, xx].astype(np.int16)     # (扩展像素, 3)
    nibbles = _payload_to_nibbles(payload)
    d_img = np.zeros((len(yy), _CHANNELS), dtype=np.int16)
    d_img.ravel()[:len(nibbles)] = nibbles
    added = theory_vals + d_img
    new_vals = np.where(added > 255, theory_vals - d_img, added).astype(np.uint8)
    expanded[yy, xx] = new_vals

    out = output_path or os.path.splitext(coded_path)[0] + "_encoded.png"
    if not cv2.imwrite(out, expanded):
        raise ValueError(f"保存失败：{out}")

    print("【编码模式】")
    print(f"  原图: {original_path}")
    print(f"  处理后图: {coded_path}")
    print(f"  差异像素: {int(diff.sum())}（{100 * diff.mean():.2f}%）")
    print(f"  RLE 方向: {'横向' if direction == 0 else '纵向'}（{len(entries)} 行/列）")
    print(f"  payload: {len(payload)} 字节，扩展 {k} 圈")
    print(f"  输出: {out}")


def decode(coded_path, password=None, output_path=None):
    """解码模式：从外圈向内逐圈测试，SHA-256 校验通过后确认扩展圈数并还原原图。

    每圈假设下：内部图像按 BORDER_REFLECT_101 重建理论扩展区，
    偏移量 = |实际像素 - 理论像素|（绝对值化，与编码端加减方向无关）。
    """
    password = password or DEFAULT_PASSWORD
    img = cv2.imread(coded_path)
    if img is None:
        raise ValueError("无法读取图片（路径不存在或格式不支持）")

    eh, ew = img.shape[:2]
    expect = hashlib.sha256(password.encode("utf-8")).hexdigest().encode("ascii")
    max_k = (min(eh, ew) - 1) // 2

    payload = None
    chosen_k = 0
    for k in range(1, max_k + 1):
        inner = img[k:eh - k, k:ew - k]
        theory = expand_image(inner, k)  # 与 img 同尺寸的理论镜像扩展图
        height, width = inner.shape[:2]
        yy, xx = expand_coords(height, width, k)

        # 向量化收集：全图一次差分 + 扩展区花式索引（行主序，通道连续）
        nibbles = cv2.absdiff(img, theory)[yy, xx].ravel()
        if not nibbles.any():
            break  # 该圈及更外没有任何偏移，数据不可能在更外层

        byte_len = len(nibbles) // 2
        if byte_len < 64:
            continue
        data = _nibbles_to_bytes(nibbles)
        if data[:64] == expect:
            payload = data
            chosen_k = k
            break

    if payload is None:
        raise ValueError("未找到匹配的 SHA-256 校验码（密码错误或图片不是编码产物）")

    version = payload[64]
    if version != PAYLOAD_VERSION:
        raise ValueError(f"不支持的 payload 版本：{version}")
    direction, entries, pos = parse_entries(payload, 65)

    restored = img[chosen_k:eh - chosen_k, chosen_k:ew - chosen_k].copy()
    apply_values(restored, entries, direction, payload[pos:])

    out = output_path or os.path.splitext(coded_path)[0] + "_decoded.png"
    if not cv2.imwrite(out, restored):
        raise ValueError(f"保存失败：{out}")

    print("【解码模式】")
    print(f"  输入: {coded_path}（确认扩展 {chosen_k} 圈）")
    print(f"  RLE 方向: {'横向' if direction == 0 else '纵向'}（{len(entries)} 行/列）")
    print(f"  输出: {out}")
