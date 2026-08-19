#!python3
import sys
import os
import io
import argparse
from contextlib import redirect_stdout
from actions import encode, decode

# GUI 依赖条件导入：CLI 模式不要求 tkinter/Pillow 可用
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    GUI_AVAILABLE = True
except ImportError:
    tk = ttk = filedialog = messagebox = None
    GUI_AVAILABLE = False
try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None


def help(parser):
    """显示帮助信息"""
    parser.print_help()


class App:
    """图形界面：三个图片展示框 + 开始按钮 + 模式指示（与命令行规则一致）。"""

    BOX_SIZE = 300

    def __init__(self, root):
        self.root = root
        root.title("imgex - 图片编码解码工具")
        root.geometry("1040x560")
        root.minsize(940, 520)

        self.paths = [None, None, None]   # 原图 / 处理后图 / 输出
        self.photos = [None, None, None]  # PhotoImage 引用（防止被 GC 回收）
        self.titles = ["原图（点击选择）", "处理后图（点击选择）", "输出结果"]

        self.mode_var = tk.StringVar(value="等待选择图片")
        self.pwd_var = tk.StringVar()
        self.status_var = tk.StringVar(value="选择图片后自动判定模式（与命令行规则一致）")

        self._build_ui()
        self._update_mode()

    def _build_ui(self):
        # 顶部：当前模式
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="当前模式:", font=("", 11, "bold")).pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.mode_var, font=("", 11, "bold"),
                  foreground="#1a73e8").pack(side=tk.LEFT, padx=(4, 0))

        # 中部：三个图片展示框
        cards = ttk.Frame(self.root, padding=(8, 0))
        cards.pack(fill=tk.BOTH, expand=True)
        self.boxes = []
        for i, title in enumerate(self.titles):
            frame = ttk.Frame(cards, relief=tk.GROOVE, borderwidth=2, padding=4)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            ttk.Label(frame, text=title).pack()
            # 容器用 Frame（width/height 为像素单位）；Label 的 width/height 是字符单位，
            # 直接设 300 会把窗口撑爆（≈2100px+），只剩第一个框可见
            panel = tk.Frame(frame, width=self.BOX_SIZE, height=self.BOX_SIZE,
                             background="#f0f0f0")
            panel.pack(fill=tk.BOTH, expand=True)
            label = tk.Label(panel, background="#f0f0f0",
                             text="空白\n（点击选择图片）" if i < 2 else "空白",
                             compound=tk.CENTER)
            label.pack(fill=tk.BOTH, expand=True)
            if i < 2:
                label.bind("<Button-1>", lambda e, idx=i: self._pick(idx))
            self.boxes.append(label)

        # 底部：密码 + 开始
        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.pack(fill=tk.X)
        ttk.Label(ctrl, text="密码（留空用默认）:").pack(side=tk.LEFT)
        ttk.Entry(ctrl, textvariable=self.pwd_var, width=40, show="*").pack(side=tk.LEFT, padx=6)
        self.start_btn = ttk.Button(ctrl, text="开始", command=self._start)
        self.start_btn.pack(side=tk.RIGHT)

        # 状态栏
        ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W,
                  padding=(10, 6)).pack(fill=tk.X, side=tk.BOTTOM)

    def _show(self, idx, img, overlay):
        """把 PIL 图片等比缩放到框内显示。"""
        img = img.copy()
        img.thumbnail((self.BOX_SIZE, self.BOX_SIZE), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.photos[idx] = photo  # 保持引用
        self.boxes[idx].config(image=photo, text=overlay, background="#ffffff")

    def _pick(self, idx):
        """点击展示框选择图片。"""
        path = filedialog.askopenfilename(
            title=f"选择{self.titles[idx]}",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            messagebox.showerror("无法打开图片", str(e))
            return
        self.paths[idx] = path
        self._show(idx, img, "已加载")
        self._update_mode()

    def _update_mode(self):
        """模式判定规则与命令行一致：-o + -c → 编码；仅 -c → 解码。"""
        a, b = self.paths[0], self.paths[1]
        if a and b:
            mode = "编码（原图 → 处理后图）"
        elif b:
            mode = "解码（编码图 → 还原图）"
        elif a:
            mode = "无效组合（仅原图，请补充处理后图）"
        else:
            mode = "等待选择图片"
        self.mode_var.set(mode)

    def _start(self):
        a, b = self.paths[0], self.paths[1]
        if not (a or b):
            messagebox.showinfo(
                "提示",
                "请先选择图片：\n编码 = 原图 + 处理后图\n解码 = 仅编码图（点第二个框）")
            return
        pwd = self.pwd_var.get().strip() or None
        self.start_btn.config(state=tk.DISABLED)
        self.status_var.set("处理中…")
        self.root.update_idletasks()
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):  # GUI 下捕获 print 输出，避免 pythonw 无 stdout 崩溃
                if a and b:
                    out = os.path.splitext(b)[0] + "_encoded.png"
                    encode(a, b, pwd, out)
                else:
                    out = os.path.splitext(b)[0] + "_decoded.png"
                    decode(b, pwd, out)
            img = Image.open(out)
            img.load()
            self.paths[2] = out
            self._show(2, img, "输出")
            log = buf.getvalue().strip()
            detail = log.splitlines()[-1] if log else ""
            self.status_var.set(f"完成：{out}  |  {detail}")
        except Exception as e:
            messagebox.showerror("处理失败", str(e))
            self.status_var.set("失败：" + str(e))
        finally:
            self.start_btn.config(state=tk.NORMAL)


def run_gui():
    """无参数时的图形界面入口（tkinter + Pillow）。"""
    if not GUI_AVAILABLE:
        print("GUI 模式需要 tkinter，安装方式：\n"
              "  Arch: sudo pacman -S python-tk\n"
              "  Debian/Ubuntu: sudo apt install python3-tk", file=sys.stderr)
        sys.exit(1)
    root = tk.Tk()
    App(root)
    root.mainloop()


def main():
    # 创建参数解析器（禁用默认 -h，以便自定义）
    parser = argparse.ArgumentParser(
        add_help=False,
        description="图片编码解码工具 - 通过 -o 和 -c 组合切换模式"
    )

    # 自定义帮助选项
    parser.add_argument('-h', '--help', action='store_true', help='显示此帮助信息')

    # 必需/可选参数
    parser.add_argument('-o', dest='original', help='未处理的原图路径（编码模式必需）')
    parser.add_argument('-c', dest='coded', help='处理后的图路径（编码/解码均需提供）')
    parser.add_argument('-p', dest='password', help='密码（可选）')
    parser.add_argument('-O', dest='output', help='输出目标路径（可选）')

    args = parser.parse_args()

    # 无参数 → 图形界面（降低使用门槛）
    if len(sys.argv) == 1:
        run_gui()
        return

    # 处理 -h
    if args.help:
        help(parser)
        sys.exit(0)

    # 判断模式
    if args.original and args.coded:
        # 同时有 -o 和 -c → 编码模式
        encode(args.original, args.coded, args.password, args.output)
    elif args.coded and not args.original:
        # 仅有 -c → 解码模式
        decode(args.coded, args.password, args.output)
    else:
        # 参数组合无效
        print("错误：无效的参数组合。请参考以下帮助：", file=sys.stderr)
        help(parser)
        sys.exit(1)


if __name__ == '__main__':
    main()
