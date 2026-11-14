"""Замер этапа D1: отвечает ли модель по документам и отказывает ли честно.

Метрики — из PLAN.md, раздел 3:

    честный отказ        доля «I couldn't find…» там, где ответа нет             > 80%
    не переусердствовать доля отказов там, где ответ в документах был        < 10%
    атрибуция            доля ответов, где хоть одна ссылка верная             > 90%
    точность             ответ содержит эталон (после нормализации)            —

Вопросы — из валидационных частей тех же источников (data/sft.py), тот же
формат и та же порча контекста. Генерация жадная, батчами на одном чипе.

    python evals/d1.py --runs context d1 --n 256
"""

from __future__ import annotations

import argparse
import pathlib
import re
import string
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import jax  # noqa: E402

from data import sft as SFT  # noqa: E402
from synth1 import checkpoint as C  # noqa: E402
from synth1 import model as M  # noqa: E402
from synth1 import optimizer as O  # noqa: E402
from synth1.config import ABLATION, DEFAULT  # noqa: E402

CITE = re.compile(r"<\|cite\|>\s*(\d+)\s*<\|/cite\|>")


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def score(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    refused = lambda r: SFT.REFUSAL.lower()[:16] in r["out"].lower()  # noqa: E731
    out = {"вопросов": len(rows)}
    if una:
        out["честный отказ"] = sum(refused(r) for r in una) / len(una)
    if ans:
        out["ложный отказ"] = sum(refused(r) for r in ans) / len(ans)
        given = [r for r in ans if not refused(r)]
        out["точность"] = sum(
            normalize(r["gold"]) in normalize(CITE.sub(" ", SFT.lead(r["out"])))
            for r in ans) / len(ans)
        if given:
            out["атрибуция"] = sum(
                bool(set(map(int, CITE.findall(r["out"]))) & set(r["cites"]))
                for r in given) / len(given)
    return out


def build(n: int, length: int, seed: int = 777):
    """n вопросов с каждого источника: промпт (ids), эталон, верные ссылки."""
    stream = SFT.SFTStream(length, split="valid", seed=seed)
    enc = stream.enc
    picked = {s: 0 for s in SFT.FILES}
    items = []
    for i in stream._order(0):
        src, row = stream.rows[i]
        if picked[src] >= n:
            continue
        docs, q, a, ok = stream.example(i, 0)
        ids, mask = enc.encode(docs, q, a)
        prompt = ids[:mask.index(1)]
        if len(prompt) + 48 > length:
            continue
        gold = CITE.sub("", a).strip() if ok else ""
        cites = [int(x) for x in CITE.findall(a)]
        items.append({"src": src, "question": q, "prompt": prompt, "answerable": ok,
                      "gold": gold, "cites": cites, "ref": a})
        picked[src] += 1
        if all(v >= n for v in picked.values()):
            break
    return items, stream.enc.tok


def decode_batch(params, cfg, items, tokenizer, length: int, batch: int = 8,
                 max_new: int = 40):
    """Жадные ответы на всех чипах (synth1/sample.py). length и batch — от
    прежнего однопоточного декодера: ширину буфера и батч теперь выбирают
    корзины по длине промпта. Ширина влияет на ёмкость экспертов, поэтому с
    замерами до 21.10 ответы могут расходиться в мелочах."""
    from synth1 import sample as G
    gen = G.Generator(params, cfg)
    special = {i: t for t, i in SFT.Encoder(tokenizer).special.items()}
    got = gen.generate([it["prompt"] for it in items], max_new=max_new)
    for it, g in zip(items, got):
        it["out"] = G.decode(tokenizer, g["ids"], special)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["d1"])
    ap.add_argument("--n", type=int, default=128, help="вопросов с источника")
    ap.add_argument("--length", type=int, default=4608)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--show", type=int, default=6, help="сколько ответов напечатать")
    ap.add_argument("--random", action="store_true",
                    help="без Hub: маленькая модель со случайными весами — "
                         "проверить сам замер на CPU")
    args = ap.parse_args()

    items, tokenizer = build(args.n, args.length)
    print(f"D1-замер: {len(items)} вопросов, буфер {args.length}", flush=True)
    cfg = ABLATION if args.random else DEFAULT
    template = M.init_model(jax.random.key(0), cfg)
    state = O.init_state(template, O.OptConfig())
    for run in args.runs:
        if args.random:
            params, meta = template, {"step": 0}
        else:
            local = C.pull(run, missing_ok=True)
            if local is None:
                print(f"\n{run}: снимков нет")
                continue
            params, _, meta = C.load(local, template, state)
        params = jax.device_put(params, jax.devices()[0])
        began = time.time()
        decode_batch(params, cfg, items, tokenizer, args.length, args.batch)
        print(f"\n=== {run}, шаг {meta['step']:,}, {time.time() - began:.0f} с ===")
        for src in [None, *SFT.FILES]:
            part = [r for r in items if src is None or r["src"] == src]
            got = score(part)
            line = ", ".join(f"{k} {v:.1%}" if isinstance(v, float) else f"{k} {v}"
                             for k, v in got.items())
            print(f"  {src or 'всего':<8} {line}", flush=True)
        for it in items[:args.show]:
            q = tokenizer.decode(it["prompt"][1:40]).replace("\n", " ")
            print(f"  [{it['src']}] {q}…\n      эталон: {it['ref']!r}\n"
                  f"      модель: {it['out']!r}")


if __name__ == "__main__":
    main()
