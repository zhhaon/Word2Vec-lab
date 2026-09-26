"""第 6 步：模型推理。

把训练好的词向量当成一个「语义计算器」：
    nn   <词> [k]              最近邻：谁和它意思最像
    sim  <词1> <词2>           余弦相似度
    ana  <a> <b> <c>           类比 a:b :: c:?   即 vec(b)-vec(a)+vec(c)
    arith <表达式>             自由向量算术，如 king - man + woman
    like <词1> - <词2> [k]     找「像词1 但不像词2」的词

用法：
    交互：  python -m src.infer --vectors runs/full/vectors.npz
    单次：  python -m src.infer --vectors runs/full/vectors.npz -c "ana king man woman"
"""
from __future__ import annotations

import argparse
import re
from typing import List, Optional

from .utils import human_int, resolve_path
from .vectors import EmbeddingMatrix, load_vectors

HELP = """可用命令：
  nn <词> [k]                最近邻（默认 k=10）
  sim <词1> <词2>            余弦相似度
  ana <a> <b> <c>            类比 a:b :: c:?  （求 d）
                             例：ana man king woman  ->  queen
                                 含义是 king - man + woman
  arith <表达式>             直接写向量算术，如 king - man + woman
  like <词1> - <词2> [k]     像词1 但不像词2
  help / ?                   显示帮助
  q / quit / exit            退出

提示：输入单个词等价于 nn <词>"""


def _fmt(results: List, limit: int = 10) -> str:
    return "\n".join(f"    {i + 1:2d}. {w:<20s} {s:+.4f}"
                     for i, (w, s) in enumerate(results[:limit]))


class SemanticCalculator:
    def __init__(self, emb: EmbeddingMatrix) -> None:
        self.emb = emb

    # ---------- 单个命令 ----------
    def run(self, line: str) -> Optional[str]:
        line = line.strip()
        if not line:
            return None
        low = line.lower()
        if low in ("q", "quit", "exit", ":q"):
            return "__QUIT__"
        if low in ("help", "?", "h"):
            return HELP

        tokens = re.split(r"\s+", line)
        cmd = tokens[0].lower()

        try:
            if cmd == "nn":
                return self.nn(tokens[1], int(tokens[2]) if len(tokens) > 2 else 10)
            if cmd == "sim":
                return self.sim(tokens[1], tokens[2])
            if cmd == "ana" or cmd == "analogy":
                return self.ana(tokens[1], tokens[2], tokens[3])
            if cmd == "arith":
                return self.arith(line[len(tokens[0]):].strip())
            if cmd == "like":
                return self.like(line[len(tokens[0]):].strip())
            # 未知命令：若第一个词在词表里，按 nn 处理
            if cmd in self.emb:
                return self.nn(cmd, 10)
            return f"未知命令：{tokens[0]}\n{HELP}"
        except KeyError as exc:
            word = str(exc.args[0]) if exc.args else ""
            hint = self.emb.suggest(word)
            extra = f"\n    你是不是想输入：{', '.join(hint)}" if hint else ""
            return f"  词表里没有「{word}」（词表大小 {human_int(len(self.emb))}）{extra}"
        except IndexError:
            return "  参数不足，输入 help 查看用法"

    # ---------- 具体功能 ----------
    def nn(self, word: str, k: int = 10) -> str:
        res = self.emb.most_similar(word, topk=k)
        return f"  与「{word}」最相近的 {len(res)} 个词：\n{_fmt(res)}"

    def sim(self, w1: str, w2: str) -> str:
        s = self.emb.similarity(w1, w2)
        bar = "#" * int(round((s + 1) / 2 * 30))
        return f"  cos({w1}, {w2}) = {s:+.4f}   [{bar:<30s}]"

    def ana(self, a: str, b: str, c: str) -> str:
        res = self.emb.analogy(a, b, c, topk=5)
        head = f"  {a} : {b}  ::  {c} : ?"
        return f"{head}\n{_fmt(res)}\n    （最优答案：{res[0][0] if res else '无'}）"

    def arith(self, expr: str) -> str:
        pos, neg = self._parse_expr(expr)
        res = self.emb.arithmetic(pos, neg, topk=10)
        pretty = " ".join([f"+{w}" for w in pos] + [f"-{w}" for w in neg]).strip()
        return f"  {pretty}\n{_fmt(res)}"

    def like(self, expr: str) -> str:
        # like word1 - word2 [k]
        parts = expr.split()
        k = 10
        if parts and parts[-1].isdigit():
            k = int(parts[-1])
            parts = parts[:-1]
        pos, neg = self._parse_expr(" ".join(parts))
        res = self.emb.arithmetic(pos, neg, topk=k)
        pretty = " ".join([f"+{w}" for w in pos] + [f"-{w}" for w in neg]).strip()
        return f"  像 {pretty} 的词：\n{_fmt(res)}"

    @staticmethod
    def _parse_expr(expr: str):
        expr = expr.replace("+", " +").replace("-", " -")
        pos, neg, sign = [], [], 1
        for tok in expr.split():
            if tok == "+":
                sign = 1
            elif tok == "-":
                sign = -1
            else:
                (pos if sign == 1 else neg).append(tok.lower())
        if not pos and not neg:
            raise IndexError("表达式为空")
        return pos, neg


def repl(calc: SemanticCalculator, emb: EmbeddingMatrix) -> None:
    print(f"词向量已加载：{human_int(len(emb))} 个词，{emb.dim} 维")
    print(HELP)
    while True:
        try:
            line = input("word2vec> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        out = calc.run(line)
        if out == "__QUIT__":
            break
        if out:
            print(out)


def main() -> None:
    from .utils import force_utf8_stdout

    force_utf8_stdout()
    p = argparse.ArgumentParser(description="词向量推理 / 交互探索")
    p.add_argument("--vectors", default="runs/tiny/vectors.npz",
                   help="vectors.npz 或 vectors.txt")
    p.add_argument("-c", "--command", default=None, help="执行单条命令后退出")
    a = p.parse_args()

    emb = load_vectors(resolve_path(a.vectors))
    calc = SemanticCalculator(emb)
    if a.command:
        out = calc.run(a.command)
        print(out if out != "__QUIT__" else "")
        return
    repl(calc, emb)


if __name__ == "__main__":
    main()
