#!python3
import sys
import argparse
from actions import encode, decode

def help(parser):
    """显示帮助信息"""
    parser.print_help()


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