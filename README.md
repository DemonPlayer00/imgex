# imgex

**图片编码解码工具**：把原图与处理后图的逐像素差异编码进单张 PNG，解码时无损还原原图。

## 工作原理

- **编码**：对比「原图」与「处理后图」，逐通道检测差异像素，经 RLE 压缩后写入处理后图的**镜像扩展区**（BORDER_REFLECT_101），输出一张 PNG。
- **解码**：从外圈向内逐圈扫描，SHA-256 校验通过后确认扩展圈数，还原出与原图逐像素一致的结果。

## 特性

- **无损还原**：原图差异完整保留，解码结果与原图逐像素一致
- **逐通道独立编解码**：支持灰度（1 通道）/ BGR（3 通道）/ BGRA（4 通道），输出保持输入通道数
- **可选密码**：未提供时使用内置默认密码；密码的 SHA-256 摘要作为扩展区校验码
- **中文路径兼容**：Windows/Linux 下非 ASCII 路径均可正常读写
- **双模式**：命令行 + 图形界面（支持拖放，Windows/Linux 单文件可执行）

## 安装

需要 Python 3 及依赖：

```bash
pip install -r requirements.txt
```

> GUI 模式需要 tkinter；Arch 用户：`sudo pacman -S python-tk`，Debian/Ubuntu 用户：`sudo apt install python3-tk`。
> 拖放支持需要 `tkinterdnd2`（已在 requirements.txt 中）。

打包为单文件可执行程序：

```bash
pyinstaller imgex.spec
# 产物：dist/imgex（Linux）/ dist/imgex.exe（Windows）
```

## 使用教程

### 模式判定

| 参数组合 | 模式 |
|---------|------|
| `-o` + `-c` | 编码（原图 → 处理后图） |
| 仅 `-c` | 解码（编码图 → 还原图） |
| 仅 `-o` 或无参数 | 无效组合 / 启动图形界面 |

### 命令行模式

**编码**——将原图与处理后图的差异写入处理后图：

```bash
imgex -o 原图.png -c 处理后图.png
# 输出：处理后图_encoded.png（与处理后图同目录）

imgex -o 原图.png -c 处理后图.png -p 我的密码 -O 输出.png
# 指定密码与输出路径
```

**解码**——从编码图还原原图：

```bash
imgex -c 处理后图_encoded.png
# 输出：处理后图_encoded_decoded.png

imgex -c 编码图.png -p 我的密码 -O 还原.png
# 编码时用了密码，解码必须使用相同密码
```

### 图形界面模式

直接运行（无参数）启动 GUI：

```bash
imgex
```

- 三个展示框：**原图** / **处理后图** / **输出结果**
- 点击或**拖放**图片到前两个框
- 编码 = 原图 + 处理后图；解码 = 仅放编码图到第二个框
- 密码留空使用默认值
- 点击「开始」后输出结果显示在第三个框

### 注意事项

- 原图与处理后图必须**尺寸、通道数完全一致**
- 仅支持 **8 位图像**（PNG / JPG / BMP / WEBP）
- 解码必须使用与编码相同的密码，否则校验失败
- 编码图请以 PNG 等无损格式保存，有损压缩会破坏扩展区数据

## 关联项目

本软件是 **OpenArt 许可证/计划**的基础设施之一。如果您感兴趣，请查看 https://github.com/DemonPlayer00/OpenArt-Licenses-Library
