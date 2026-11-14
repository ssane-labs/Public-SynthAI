"""Замер поиска со средой: растёт ли точность от поиска и не растут ли выдумки.

Критерий этапа D2 из PLAN.md. Среда — абзацы валидационной части HotpotQA и
MuSiQue (data/wiki_search.py, data/search_env.py: BM25, пять результатов,
окно страницы). Модель ведёт диалог со средой по ходам: пишет до закрывающего
тега, среда дописывает выдачу или страницу, и так до ответа, но не больше
--turns вызовов и 8192 токенов. Вопросы валидации модель не видела: поиск
учился на train.

    HotpotQA   два шага по двум статьям
    MuSiQue    2–4 шага, трудные

Для каждой группы — то же без поиска (инструменты запрещены): ответ по памяти.

    верно         ответ совпал с эталоном (search_sft.correct)
    отказ         «I couldn't find…» (с найденным близким и предложением искать иначе)
    выдумка       ответил, но неверно
    без ответа    не дописал ответ за лимит
    ходов         среднее число search + open + find

    python evals/d2_env.py --runs gv3
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import config  # noqa: E402
from data import calib as CB  # noqa: E402
from data import search_sft as SS  # noqa: E402
from data import sft as SFT  # noqa: E402
from synth1 import sample as G  # noqa: E402

from synth1.agent import CITE, run_agent  # noqa: E402


def score(items, states) -> dict:
    n = max(len(items), 1)
    right = refused = wrong = none = cited = 0
    for it, s in zip(items, states):
        a = s["answer"]
        if a is None:
            none += 1
            continue
        if CB.refused(a):
            refused += 1
            continue
        if SS.correct(CITE.sub(" ", SFT.lead(a)), it["gold"]):
            right += 1
            nums = {int(x) for x in CITE.findall(a)}
            cited += bool(nums) and nums <= set(s["src"])
        else:
            wrong += 1
    return {"верно": right / n, "отказ": refused / n, "выдумка": wrong / n,
            "без ответа": none / n, "ссылки на найденное": cited / max(right, 1),
            "ходов": float(np.mean([s["calls"] for s in states]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["d2"])
    ap.add_argument("--hotpot", type=int, default=128, help="вопросов HotpotQA")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--musique", type=int, default=128, help="вопросов MuSiQue")
    ap.add_argument("--only-hotpot", action="store_true",
                    help="только HotpotQA — быстрый замер (~10 мин)")
    ap.add_argument("--show", type=int, default=2)
    args = ap.parse_args()
    if args.only_hotpot:
        args.musique = 0

    began = time.time()
    from data import wiki_search as WS
    from tokenizers import Tokenizer
    enc = SS.Encoder(Tokenizer.from_file(str(config.TOKENIZER_OUT / "synth1.json")))
    groups = []
    for name, source, n in (("HotpotQA", "hotpot", args.hotpot),
                            ("MuSiQue", "musique", args.musique)):
        if not n:
            continue
        # Корпус — абзацы всей валидации источника (тысячи страниц), вопросы —
        # n случайных из неё. Модель этих вопросов не видела: поиск учился на
        # train (data/wiki_search.py).
        rows = WS.rows_for(source, "valid")
        env = WS.make_env(source, rows)
        random.Random(0).shuffle(rows)
        items = []
        for r in rows[:n]:
            gold = r["answer"]
            items.append({"question": r["question"], "gold": gold,
                          "prompt": enc.encode_part("user", r["question"])})
        groups.append((name, items, env))
        print(f"среда {name}: {env.n:,} абзацев, {len(items)} вопросов", flush=True)
    print(f"среды собраны за {time.time() - began:.0f} с", flush=True)

    for run in args.runs:
        try:
            params, cfg, meta = CB.load_run(run)
        except SystemExit as exc:
            print(f"\n{run}: снимка нет ({exc})")
            continue
        gen = G.Generator(params, cfg)
        print(f"\n=== {run}, шаг {meta['step']:,} ===", flush=True)
        for name, items, env_ in groups:
            t0 = time.time()
            with_tools = run_agent(gen, enc, env_, items, args.turns)
            closed = run_agent(gen, enc, env_, items, 0, tools=False)
            a, b = score(items, with_tools), score(items, closed)
            fmt = lambda d: ", ".join(  # noqa: E731
                f"{k} {v:.1%}" if k != "ходов" else f"{k} {v:.1f}" for k, v in d.items())
            print(f"  {name} ({len(items)}):\n    с поиском:  {fmt(a)}\n"
                  f"    без поиска: {fmt(b)}  ({time.time() - t0:.0f} с)", flush=True)
            for it, s in list(zip(items, with_tools))[:args.show]:
                print(f"    [вопрос] {it['question'][:90]!r}\n      эталон: {it['gold'][:60]!r}\n"
                      f"      ход: {s['trace'][:500]!r}\n      ответ: {str(s['answer'])[:120]!r}")


if __name__ == "__main__":
    main()
