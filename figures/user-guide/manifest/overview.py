"""Draw overview.svg: the TileOPs components as a layered component diagram on a fixed grid.

Graphviz reorders the components within a layer once edges cross layers, so this diagram is
laid out by hand: three layers of equal width, three columns, and orthogonal edges that run in
the gaps between columns. Run it as ``overview.py <zh|en> <output.svg>``; render.sh does both.
"""

import sys
from pathlib import Path

FONT = "'Noto Sans SC','PingFang SC','Microsoft YaHei',sans-serif"
INK = "#2A2540"
SUB = "#5B5670"
LINE = "#6B5FA0"
BAND_FILL = "#FBFAFE"
BAND_LINE = "#CFC6E6"
BAND_INK = "#4A2C8F"
# Fill of each category, and the border and title colour its boxes take.
DEV = "#EFE9FA"
SYS = "#E2F6F8"
PUB = "#E8F6EA"
EDGE = {DEV: ("#8E6CCF", "#4A2C8F"), SYS: ("#3AA9B8", "#0B5E6A"), PUB: ("#6DBA7A", "#2E6B3A")}

WIDTH = 700
COL_W = 184
GAP = 28
BAND_X = 20
BAND_W = WIDTH - 2 * BAND_X
COL_X = [BAND_X + 24 + i * (COL_W + GAP) for i in range(3)]


def text_width(s: str, size: float) -> float:
    return sum(size if ord(c) > 0x2E7F else size * 0.56 for c in s)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.boxes: dict[str, tuple[float, float, float, float]] = {}

    def text(self, x, y, s, size=13, weight="normal", fill=INK, anchor="start"):
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>'
        )

    def band(self, y, h, label):
        tab_w = text_width(label, 13) * (1.0 if LANG == "zh" else 1.12) + 22
        self.parts.append(
            f'<path d="M{BAND_X},{y} h{tab_w:.1f} l6,20 H{BAND_X + BAND_W} V{y + h} H{BAND_X} Z" '
            f'fill="{BAND_FILL}" stroke="{BAND_LINE}" stroke-width="1.2"/>'
        )
        self.parts.append(
            f'<line x1="{BAND_X}" y1="{y + 20}" x2="{BAND_X + tab_w + 6:.1f}" y2="{y + 20}" '
            f'stroke="{BAND_LINE}" stroke-width="1.2"/>'
        )
        self.text(BAND_X + 10, y + 15, label, weight="bold", fill=BAND_INK)

    def box(self, key, col, y, title, subs=(), fill=SYS, h=None):
        x = COL_X[col]
        h = h or 30 + 16 * len(subs)
        stroke, ink = EDGE[fill]
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{COL_W}" height="{h}" rx="5" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.2"/>'
        )
        self.text(x + 10, y + 20, title, weight="bold", fill=ink)
        for i, s in enumerate(subs):
            self.text(x + 10, y + 38 + 16 * i, s, size=11, fill=SUB)
        self.boxes[key] = (x, y, COL_W, h)

    def edge(self, points, label=None, at=0, dashed=False, dx=6, dy=-4, anchor="start"):
        """A polyline through *points*, arrowhead at the end, label beside segment *at*."""
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="1.3"{dash} '
            f'marker-end="url(#arrow)"/>'
        )
        if label:
            (x1, y1), (x2, y2) = points[at], points[at + 1]
            mx, my = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
            w = text_width(label, 12)
            bx = mx if anchor == "start" else mx - w / 2 if anchor == "middle" else mx - w
            self.parts.append(
                f'<rect x="{bx - 2:.1f}" y="{my - 11:.1f}" width="{w + 4:.1f}" height="15" '
                f'fill="#FFFFFF" fill-opacity="0.85"/>'
            )
            self.text(mx, my, label, size=12, fill=INK, anchor=anchor)

    # Anchor points of a box.
    def top(self, k, f=0.5):
        x, y, w, _ = self.boxes[k]
        return (x + w * f, y)

    def bottom(self, k, f=0.5):
        x, y, w, h = self.boxes[k]
        return (x + w * f, y + h)

    def left(self, k):
        x, y, _, h = self.boxes[k]
        return (x, y + h / 2)

    def right(self, k):
        x, y, w, h = self.boxes[k]
        return (x + w, y + h / 2)

    def render(self, height) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" '
            f'width="{WIDTH}" height="{height}" font-family="{FONT}">'
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
            f'markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{LINE}"/></marker></defs>'
            f'<rect width="{WIDTH}" height="{height}" fill="#FFFFFF"/>'
        )
        return head + "".join(self.parts) + "</svg>\n"


def L(zh: str, en: str) -> str:
    """The label in the language being drawn."""
    return zh if LANG == "zh" else en


def draw() -> str:
    s = Svg()
    s.text(WIDTH / 2, 28, L("TileOPs 的组件与调用关系", "TileOPs components and how they call each other"), size=15, weight="bold", fill=BAND_INK, anchor="middle")

    # 顶行:图例在左侧,spec 居中。
    s.box("spec", 1, 48, "spec", ["src/tileops/manifest/spec/"], fill=DEV)
    lx, ly = COL_X[0], 44
    s.parts.append(
        f'<rect x="{lx}" y="{ly}" width="{COL_W + 20}" height="76" rx="4" fill="#FFFFFF" '
        f'stroke="{BAND_LINE}" stroke-width="1"/>'
    )
    for i, (color, name) in enumerate([(DEV, L("开发者编写", "Written by the developer")), (SYS, L("系统提供，开发者不修改", "Provided by the system")), (PUB, L("发布", "Publication"))]):
        yy = ly + 12 + 21 * i
        s.parts.append(
            f'<rect x="{lx + 10}" y="{yy}" width="18" height="13" fill="{color}" stroke="{EDGE[color][0]}" stroke-width="1"/>'
        )
        s.text(lx + 36, yy + 11, name, size=12)

    # 实现层。
    y1 = 140
    s.band(y1, 228, L("实现", "Implementation"))
    r1 = y1 + 48
    s.box("gen", 0, r1, L("代码生成", "Code generation"), [L("调用检查、形状推导", "call checks, shape inference"), L("dtype 检查、eval_roofline", "dtype checks, eval_roofline")])
    s.box("op", 1, r1, L("Op 类", "Op class"), [L("__init__、forward、docstring", "__init__, forward, docstring"), L("kernel_types、interfaces", "kernel_types, interfaces")], fill=DEV)
    s.box("base", 2, r1, L("Op 基类", "Op base"), [L("kernel_for：选择实现", "kernel_for: selection"), L("entry 缓存、target 选择", "entry cache, target selection")])
    s.box("kernel", 2, r1 + 104, L("Kernel 实现", "Kernel implementation"), [L("继承 kernel 接口", "inherits a kernel interface"), L("refusal、entry_for 按需声明", "refusal, entry_for as needed")], fill=DEV)

    # 验证与测量层。
    y2 = y1 + 228 + 40
    s.band(y2, 294, L("验证与测量", "Validation and measurement"))
    q1 = y2 + 36
    q2 = q1 + 96
    q3 = q2 + 82
    s.box("ref", 0, q1, L("参考实现", "Reference"), [L("workloads/ 的 ref_program", "ref_program in workloads/")], fill=DEV)
    s.box("val", 1, q1, "validator", ["validate_manifest.py"])
    s.box("inst", 2, q1, L("workload 实例化", "Workload instantiation"), ["instantiate"])
    s.box("test", 0, q2, L("正确性测试", "Correctness tests"), ["tests/ops/"], fill=DEV)
    s.box("bench", 1, q2, L("benchmark 函数", "Benchmark function"), ["benchmarks/ops/"], fill=DEV)
    s.box("mtest", 2, q2, L("manifest 测试", "Manifest tests"), [L("meta 调用、target conformance", "meta calls, target conformance")])
    s.box("mb", 1, q3, "bench.Runner", [L("校验、计时、FLOPs 与字节数", "checks, timing, FLOPs, bytes")])

    # 发布层。
    y3 = y2 + 294 + 40
    s.band(y3, 112, L("发布", "Publication"))
    p1 = y3 + 36
    s.box("nightly", 0, p1, "nightly", [L("运行全部 benchmark", "runs every benchmark"), L("按 case id 记录历史", "keeps history by case id")], fill=PUB)
    s.box("m5", 1, p1, L("roofline 工具", "Roofline tool"), [L("SOL 效率", "SOL efficiency"), L("与瓶颈判定", "and bound")], fill=PUB)
    s.box("site", 2, p1, L("文档站", "Docs site"), [L("读取 spec、docstring", "reads specs, docstrings"), L("与 benchmark 结果", "and benchmark results")], fill=PUB)
    height = y3 + 112 + 20

    # spec 到两层的输入。
    sb = s.bottom("spec", 0.3)
    turn = (sb[1] + y1) / 2 + 8
    gx0 = s.top("gen")[0]
    s.edge([sb, (sb[0], turn), (gx0, turn), s.top("gen")], L("签名、roofline", "signature, roofline"), at=0, dx=6, dy=4)
    sx, sy = s.right("spec")
    rail = BAND_X + BAND_W + 10
    drop = COL_X[2] + COL_W / 2
    s.edge(
        [(sx, sy), (rail, sy), (rail, y2 - 18), (drop, y2 - 18), (drop, y2 + 20)],
        L("workload 行、全部字段", "workload rows, all fields"),
        at=0,
        dx=0,
        dy=-5,
        anchor="middle",
    )

    # 实现层内部。
    ay = r1 + 20
    s.edge([(s.right("gen")[0], ay), (s.left("op")[0], ay)])
    s.edge([(s.right("op")[0], ay), (s.left("base")[0], ay)])
    s.text(s.right("gen")[0] + GAP / 2, r1 - 7, L("安装生成的方法", "installs methods"), size=12, anchor="middle")
    s.text(s.right("op")[0] + GAP / 2, r1 - 7, "kernel_for", size=12, anchor="middle")
    s.edge([s.bottom("base"), s.top("kernel")], L("构造并缓存", "builds and caches"))

    # 验证与测量层对实现层的调用与核对。
    # Clear of the band's tab, whose width follows the language of its label.
    tab_end = BAND_X + text_width(L("验证与测量", "Validation and measurement"), 13) * (1.0 if LANG == "zh" else 1.12) + 22
    tx = max(COL_X[0] + COL_W / 2, tab_end + 24)
    s.edge(
        [(tx, y2 + 20), (tx, y1 + 228)],
        L("测试与 benchmark 执行调用", "tests and benchmarks call the op"),
        dashed=True,
        dx=6 if LANG == "zh" else -6,
        anchor="start" if LANG == "zh" else "end",
    )
    vx = s.top("val")[0]
    s.edge([(vx, q1), (vx, s.bottom("op")[1])], L("核对接口", "checks the interface"), at=0, dy=30, dashed=True)

    # 验证与测量层内部。
    s.edge([s.bottom("ref", 0.3), s.top("test", 0.3)], L("数值参考", "reference"))
    gap_y1 = (q1 + 46 + q2) / 2
    s.edge(
        [s.bottom("ref", 0.8), (s.bottom("ref", 0.8)[0], gap_y1), (s.top("bench", 0.2)[0], gap_y1), s.top("bench", 0.2)],
        L("torch 基线", "torch baseline"),
        at=1,
        dx=0,
        dy=-4,
        anchor="middle",
    )
    s.edge([s.bottom("inst", 0.7), s.top("mtest", 0.7)], L("调用", "calls"))
    gx = COL_X[2] - GAP / 2
    ix = s.bottom("inst", 0.2)[0]
    mby = s.right("mb")[1]
    s.edge(
        [s.bottom("inst", 0.2), (ix, gap_y1), (gx, gap_y1), (gx, mby), s.right("mb")],
        L("调用", "calls"),
        at=3,
        dx=6,
        dy=-4,
    )
    s.edge([s.bottom("bench"), s.top("mb")], L("对比实现", "implementations"))

    # 验证与测量层到发布层。
    mbx, mbb = s.bottom("mb")
    nx = s.top("nightly")[0]
    mid = (mbb + p1) / 2 + 6
    s.edge([(mbx, mbb), (mbx, mid), (nx, mid), (nx, p1)], "device time", at=1, dx=0, dy=-4, anchor="middle")
    s.edge([s.right("nightly"), s.left("m5")])
    s.edge([s.right("m5"), s.left("site")])

    return s.render(height)


LANG = "zh"

if __name__ == "__main__":
    LANG = sys.argv[1]
    Path(sys.argv[2]).write_text(draw(), encoding="utf-8")
