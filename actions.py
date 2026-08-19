import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


# 默认密码（未通过 -p 提供时使用；其 SHA-256 摘要作为写入扩展区的校验码）
DEFAULT_PASSWORD = "477a3d43f692aeaf1c7f40c0c91bffde3e2e638d8e90c668422373ee82a18521"
# payload 格式版本（每通道独立编解码）
PAYLOAD_VERSION = 2


def _imread(path):
    """读取图片，兼容含中文/非 ASCII 的路径。

    cv2.imread 在 Windows 上走 ANSI API（fopen），非 ASCII 路径会失败返回 None；
    改为 np.fromfile（宽字符路径）+ imdecode 解码。
    使用 IMREAD_UNCHANGED 保留原始通道数（灰度 1 / BGR 3 / BGRA 4），
    便于逐通道独立编解码与输出同通道数。
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is not None and img.dtype != np.uint8:
        raise ValueError("仅支持 8 位图像（当前位深：%d）" % (img.dtype.itemsize * 8))
    return img


def _imwrite(path, img):
    """写入图片，兼容含中文/非 ASCII 的路径。

    cv2.imwrite 在 Windows 上同样受 ANSI API 限制；改为 imencode + tofile。
    """
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def expand_image(image: np.ndarray, border_size: int) -> np.ndarray:
    if border_size == 0:
        return image.copy()  # 无扩展时返回原图副本

    # OpenCV 把 (h, w, 1) 单通道数组当作 2D 处理（copyMakeBorder 会丢维度），
    # 单通道时用 2D 扩展后再补回第 3 维，保证通道数不变
    single = image.ndim == 3 and image.shape[2] == 1
    src = image[:, :, 0] if single else image

    # 使用 BORDER_REFLECT_101 模式（以边缘像素为轴镜像）
    expanded = cv2.copyMakeBorder(
        src,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_REFLECT_101
    )
    return expanded[:, :, None] if single else expanded


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

def _payload_to_nibbles(payload: bytes) -> np.ndarray:
    """向量化位拆分：payload 每字节按位奇偶拆为两个 0~15 值（先奇位后偶位）。

    奇位（bit 1,3,5,7）在前，偶位（bit 2,4,6,8）在后。
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

    尾部奇数个 nibble 丢弃。
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

def parse_entries(data: bytes, pos: int):
    """段解析（_rle_encode_channel 的分离布局）：
    方向(1B) + 行数(4B) + 行头区[idx,n_runs]×rows(8B×rows) + run区[s,e]×Σn(8B×Σn)。
    返回 (direction, entries, 结束位置)。全 numpy 读头区/run区，逐行组装 entries。
    """
    direction = data[pos]
    pos += 1
    count = int.from_bytes(data[pos:pos + 4], "big")
    pos += 4
    hdr = np.frombuffer(data, dtype=">u4", count=2 * count, offset=pos)
    pos += 8 * count
    idxs = hdr[0::2]
    ns = hdr[1::2]
    total = int(ns.sum())
    runs = np.frombuffer(data, dtype=">u4", count=2 * total, offset=pos)
    pos += 8 * total
    rs = runs[0::2]
    es = runs[1::2]
    entries = []
    off = 0
    for i in range(count):
        n = int(ns[i])
        entries.append((int(idxs[i]), [(int(rs[off + j]), int(es[off + j])) for j in range(n)]))
        off += n
    return direction, entries, pos


def _coords_from_runs(rows_arr, starts_arr, ends_arr, direction):
    """把 run 描述（行号/起点列/终点列，均按 run 对齐）展开为 (ys, xs) 索引数组。

    全 numpy 向量化（释放 GIL，供多线程并行）：
    lens = 各 run 长度；xs 在 run 内连续，用 cumsum 前缀和定位每个像素的列偏移。
    direction=1 时 rows_arr 是列号、区间沿行方向，交换 ys/xs。
    """
    lens = ends_arr - starts_arr + 1
    total = int(lens.sum())
    if total == 0:
        return None, None
    inner = np.arange(total, dtype=np.intp) - np.repeat(np.cumsum(lens) - lens, lens)
    xs = inner + np.repeat(starts_arr, lens)
    ys = np.repeat(rows_arr, lens)
    if direction == 1:
        ys, xs = xs, ys
    return ys, xs


def _entries_to_coords(entries, direction: int):
    """把区间列表展开为 (ys, xs) 索引数组（保持区间扫描顺序）。向量化。"""
    if not entries:
        return None, None
    rows = []
    starts = []
    ends = []
    for idx, runs in entries:
        n = len(runs)
        if n == 0:
            continue
        rows.extend([idx] * n)
        starts.extend(s for s, _ in runs)
        ends.extend(e for _, e in runs)
    if not rows:
        return None, None
    return _coords_from_runs(np.asarray(rows, dtype=np.intp), np.asarray(starts, dtype=np.intp),
                             np.asarray(ends, dtype=np.intp), direction)


def _rle_encode_channel(mask: np.ndarray, plane: np.ndarray):
    """单通道 RLE 编码：独立检测差异区间 + 序列化 + 原值提取。

    全程 numpy 向量化（np.diff 找 run 边界、大端 tobytes 组装、前缀和展开坐标），
    不持有 Python GIL，可在线程池中按通道并行。
    返回 (direction, blob, values_bytes)：direction 取 h/v 体积较小者。
    """
    best = None
    for d in (0, 1):
        m = mask if d == 0 else mask.T
        dm = np.diff(m.astype(np.int8), axis=1, prepend=0, append=0)
        starts = np.argwhere(dm == 1)   # (K, 2)：每 run 起点 (行, 列)
        ends = np.argwhere(dm == -1)    # (K, 2)：每 run 终点+1
        k = len(starts)
        if k == 0:
            blob = bytes([d]) + b"\x00\x00\x00\x00"
            values = b""
        else:
            rows = starts[:, 0]
            # ⛔ 行分组必须用布尔比较：np.diff(rows, prepend=-1) 会把相邻自然数行号
            #（10→11 的 diff=1）误判为新组，导致行头与 run 区错位
            first = np.flatnonzero(np.concatenate([[True], rows[1:] != rows[:-1]]))
            row_ids = rows[first]
            counts = np.diff(np.append(first, k))
            # 行头区 [idx(4B) + n_runs(4B)] × 行数，run 区 [起点(4B) + 终点(4B)] × K，均大端
            row_hdr = np.empty((len(row_ids), 2), dtype=">u4")
            row_hdr[:, 0] = row_ids
            row_hdr[:, 1] = counts
            run_arr = np.empty((k, 2), dtype=">u4")
            run_arr[:, 0] = starts[:, 1]
            run_arr[:, 1] = ends[:, 1] - 1
            blob = bytes([d]) + len(row_ids).to_bytes(4, "big") + row_hdr.tobytes() + run_arr.tobytes()
            # 原值：按 run 顺序展开坐标后取平面值
            ys, xs = _coords_from_runs(np.repeat(row_ids, counts), starts[:, 1], ends[:, 1] - 1, d)
            values = plane[ys, xs].tobytes()
        total = len(blob) + len(values)
        if best is None or total < best[0]:
            best = (d, blob, values)
    return best


def _encode_channels_parallel(masks, planes):
    """各通道独立 RLE 编码，每通道一个线程并行（numpy 操作释放 GIL）。"""
    nc = len(masks)
    if nc <= 1:
        return [_rle_encode_channel(masks[0], planes[0])]
    with ThreadPoolExecutor(max_workers=nc) as ex:
        return list(ex.map(lambda c: _rle_encode_channel(masks[c], planes[c]), range(nc)))


def _apply_channels_parallel(restored, sections, directions):
    """各通道独立恢复原值，每通道一个线程并行（不同通道写不同列，无竞争）。"""
    nc = len(sections)
    if nc <= 1:
        entries, values = sections[0]
        apply_values_channel(restored, entries, directions[0], values, 0)
        return
    with ThreadPoolExecutor(max_workers=nc) as ex:
        list(ex.map(lambda c: apply_values_channel(restored, sections[c][0], directions[c], sections[c][1], c),
                    range(nc)))


def build_payload(password: str, masks, original: np.ndarray):
    """payload：每通道独立 RLE 编码 + 独立原值，头部记录通道数与各段长度。

    各通道编码在线程池中并行（numpy 操作释放 GIL，互不阻塞）。
    返回 (header, segments, directions, n_pixels)：
      header  = SHA-256(64) + version(1) + n_channels(1) + direction_i(1B × n)
                + n_pixels_i(4B × n) + entries_bytes_i(4B × n)
      segments[c] = 通道 c 的 entries + 原值（写入时独立成流）
    每通道独立选择 h/v 方向中体积较小者。
    """
    nc = len(masks)
    results = _encode_channels_parallel(masks, [original[:, :, c] for c in range(nc)])
    directions = [r[0] for r in results]
    blobs = [r[1] for r in results]
    vals = [r[2] for r in results]
    n_pixels = [len(v) for v in vals]

    header = bytearray(hashlib.sha256(password.encode("utf-8")).hexdigest().encode("ascii"))
    header += bytes([PAYLOAD_VERSION, nc]) + bytes(directions)
    for c in range(nc):
        header += n_pixels[c].to_bytes(4, "big")
    for c in range(nc):
        header += len(blobs[c]).to_bytes(4, "big")
    segments = [blobs[c] + vals[c] for c in range(nc)]
    return bytes(header), segments, directions, n_pixels


def parse_payload_streams(data0: bytes, diff_ext: np.ndarray):
    """从解码端恢复的每通道独立流解析。

    data0    = 通道 0 流还原的字节（header + 通道 0 段）
    diff_ext = 扩展区 |实际-理论| 数组 (N, nc)，用于取各通道流
    返回 (n_channels, directions, [(entries, values), ...])。
    """
    nc = data0[65]
    directions = list(data0[66:66 + nc])
    Lh = 66 + 9 * nc
    if len(data0) < Lh:
        raise ValueError("通道 0 流过短，无法读取头部")
    n_pixels = [int.from_bytes(data0[66 + nc + 4 * c:66 + nc + 4 * c + 4], "big") for c in range(nc)]
    e_bytes = [int.from_bytes(data0[66 + 5 * nc + 4 * c:66 + 5 * nc + 4 * c + 4], "big") for c in range(nc)]
    seg_lens = [e + n for e, n in zip(e_bytes, n_pixels)]

    sections = []
    for c in range(nc):
        if c == 0:
            if len(data0) < Lh + seg_lens[0]:
                raise ValueError("通道 0 段不完整")
            seg = data0[Lh:Lh + seg_lens[0]]
        else:
            bc = _nibbles_to_bytes(diff_ext[:, c])
            if len(bc) < seg_lens[c]:
                raise ValueError(f"通道 {c} 流不完整")
            seg = bc[:seg_lens[c]]
        _, entries, end = parse_entries(seg, 0)
        if end != e_bytes[c]:
            raise ValueError(f"通道 {c} 的 entries 长度不一致（头部 {e_bytes[c]}，实际 {end}）")
        sections.append((entries, seg[e_bytes[c]:]))
    return nc, directions, sections


def apply_values_channel(image: np.ndarray, entries, direction: int, values: bytes, channel: int):
    """把单通道原值流按区间顺序写回 image[:, :, channel]（向量化）。"""
    ys, xs = _entries_to_coords(entries, direction)
    if ys is None:
        return
    vals = np.frombuffer(values, dtype=np.uint8)
    image[ys, xs, channel] = vals[:len(ys)]


def encode(original_path, coded_path, password=None, output_path=None):
    """编码模式：将原图与处理后图的差异写入处理后图的镜像扩展区。

    每个通道独立检测差异、独立 RLE、独立记录原值；支持 1/3/4 通道
    （灰度/BGR/BGRA），输出保持与输入相同的通道数。
    payload 每字节按位奇偶拆为两个 0~15 的值，按通道线性写入扩展区像素：
    默认对理论镜像值做加法偏移，若加法溢出（>255）则改做减法（差值恒 >=226，
    不会双向溢出），解码端取 |实际 - 理论| 绝对值即可还原。
    """
    password = password or DEFAULT_PASSWORD
    original = _imread(original_path)
    coded = _imread(coded_path)
    if original is None or coded is None:
        raise ValueError("无法读取图片（路径不存在或格式不支持）")
    orig_ndim = original.ndim
    if original.ndim == 2:
        original = original[:, :, None]
    if coded.ndim == 2:
        coded = coded[:, :, None]
    if original.shape != coded.shape:
        raise ValueError("原图与处理后图的尺寸/通道数不一致")

    height, width = coded.shape[:2]
    nc = coded.shape[2]
    masks = [original[:, :, c] != coded[:, :, c] for c in range(nc)]
    header, segments, directions, n_pixels = build_payload(password, masks, original)

    # 每通道独立流：通道 0 流 = header + 段 0，其余通道流 = 各自段；
    # 扩展区大小取所有通道中需求最大者（每像素提供 1 个槽位/通道）
    parts = [header + segments[0]] + segments[1:]
    nibble_lens = [2 * len(p) for p in parts]
    need_pixels = max(nibble_lens)
    k = calc_expand_pixels(width, height, need_pixels)
    max_k = (min(height, width) - 1) // 2
    if k > max_k:
        raise ValueError(f"图片过小，扩展区无法容纳差异数据（需要 {k} 圈，上限 {max_k} 圈）")

    expanded = expand_image(coded, k)

    # 向量化写入：扩展区理论值 + 偏移（加法溢出转减法，保证不双向溢出）
    yy, xx = expand_coords(height, width, k)
    theory_vals = expanded[yy, xx].astype(np.int16)     # (扩展像素, nc)
    d_img = np.zeros((len(yy), nc), dtype=np.int16)
    for c in range(nc):
        d_img[:nibble_lens[c], c] = _payload_to_nibbles(parts[c])
    added = theory_vals + d_img
    new_vals = np.where(added > 255, theory_vals - d_img, added).astype(np.uint8)
    expanded[yy, xx] = new_vals

    if orig_ndim == 2:
        expanded = expanded[:, :, 0]  # 灰度输入 → 灰度输出

    out = output_path or os.path.splitext(coded_path)[0] + "_encoded.png"
    if not _imwrite(out, expanded):
        raise ValueError(f"保存失败：{out}")

    print("【编码模式】")
    print(f"  原图: {original_path}")
    print(f"  处理后图: {coded_path}")
    for c in range(nc):
        print(f"  通道 {c} 差异像素: {n_pixels[c]}")
    dirs_str = "/".join("横" if d == 0 else "纵" for d in directions)
    print(f"  RLE 方向: {dirs_str}")
    total = sum(len(p) for p in parts)
    print(f"  payload: {total} 字节（通道流最大 {max(nibble_lens) // 2} 字节），扩展 {k} 圈")
    print(f"  输出: {out}")


def decode(coded_path, password=None, output_path=None):
    """解码模式：从外圈向内逐圈测试，SHA-256 校验通过后确认扩展圈数并还原原图。

    每圈假设下：内部图像按 BORDER_REFLECT_101 重建理论扩展区，
    偏移量 = |实际像素 - 理论像素|（绝对值化，与编码端加减方向无关）。
    每通道独立原值；输出保持输入通道数。
    """
    password = password or DEFAULT_PASSWORD
    img = _imread(coded_path)
    if img is None:
        raise ValueError("无法读取图片（路径不存在或格式不支持）")
    ndim = img.ndim
    if ndim == 2:
        img = img[:, :, None]

    eh, ew = img.shape[:2]
    expect = hashlib.sha256(password.encode("utf-8")).hexdigest().encode("ascii")
    max_k = (min(eh, ew) - 1) // 2

    streams = None
    chosen_k = 0
    for k in range(1, max_k + 1):
        inner = img[k:eh - k, k:ew - k]
        theory = expand_image(inner, k)  # 与 img 同尺寸的理论镜像扩展图
        height, width = inner.shape[:2]
        yy, xx = expand_coords(height, width, k)

        # 向量化收集：全图一次差分 + 扩展区花式索引（行主序，通道连续）
        diff_ext = cv2.absdiff(img, theory)[yy, xx]
        if diff_ext.ndim == 1:
            diff_ext = diff_ext[:, None]  # OpenCV 对单通道 (h,w,1) 返回 2D，补回通道维
        if not diff_ext.any():
            break  # 该圈及更外没有任何偏移，数据不可能在更外层

        # 通道 0 独立流校验（header 位于通道 0 流开头）
        data0 = _nibbles_to_bytes(diff_ext[:, 0])
        if len(data0) >= 64 and data0[:64] == expect:
            streams = (data0, diff_ext)
            chosen_k = k
            break

    if streams is None:
        raise ValueError("未找到匹配的 SHA-256 校验码（密码错误或图片不是编码产物）")

    restored = img[chosen_k:eh - chosen_k, chosen_k:ew - chosen_k].copy()
    data0, diff_ext = streams
    version = data0[64]
    if version != PAYLOAD_VERSION:
        raise ValueError(f"不支持的 payload 版本：{version}")
    nc, directions, sections = parse_payload_streams(data0, diff_ext)
    _apply_channels_parallel(restored, sections, directions)
    dirs_str = "/".join("横" if d == 0 else "纵" for d in directions)

    if ndim == 2:
        restored = restored[:, :, 0]  # 灰度输入 → 灰度输出

    out = output_path or os.path.splitext(coded_path)[0] + "_decoded.png"
    if not _imwrite(out, restored):
        raise ValueError(f"保存失败：{out}")

    print("【解码模式】")
    print(f"  输入: {coded_path}（确认扩展 {chosen_k} 圈）")
    print(f"  payload 版本: v{version}")
    print(f"  RLE 方向: {dirs_str}")
    print(f"  输出: {out}")
