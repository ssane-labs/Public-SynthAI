"""Замер этапа D2 без среды: начинает ли модель с поиска и как пишет запрос.

Модели дают только вопрос — с валидации HotpotQA, MuSiQue и SQuAD (поиск
учился на train, data/wiki_search.py) — и смотрят первое действие. Полный
замер со средой — evals/d2_env.py.

    доля «ищет»       первое действие — <|search|> (после необязательного
                      <|think|>), а не ответ из памяти
    пересечение       доля слов эталонного первого запроса в запросе модели
    формат            запрос закрыт <|/search|>

    python evals/d2.py --runs d2v3 --n 64
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import jax  # noqa: E402

import config  # noqa: E402
from data import search_sft as SS  # noqa: E402
from evals import d1 as D1  # noqa: E402
from synth1 import checkpoint as C  # noqa: E402
from synth1 import model as M  # noqa: E402
from synth1 import optimizer as O  # noqa: E402
from synth1.config import DEFAULT  # noqa: E402

THINK = re.compile(r"<\|think\|>.*?(?:<\|/think\|>|$)", re.S)


def build(n: int):
    """Вопросы валидации HotpotQA, MuSiQue, SQuAD (поиск учился на train) и
    эталонный первый запрос — из траектории data/wiki_search.py."""
    from tokenizers import Tokenizer

    from data import wiki_search as WS
    tok = Tokenizer.from_file(str(config.TOKENIZER_OUT / "synth1.json"))
    enc = SS.Encoder(tok)
    items = []
    for source in ("hotpot", "musique", "squad"):
        got, _, _ = WS.trajectories(source, "valid", max(n, 1) * 2, seed=777)
        for segs in got[:max(1, n // 3)]:
            ref = next((t for k, t in segs if k == "search"), "")
            items.append({"prompt": enc.encode_part("user", segs[0][1]), "ref": ref})
    return items, tok


def score(items) -> dict:
    searched = closed = 0
    overlap = []
    for it in items:
        # Первое действие после всех рассуждений: модель часто думает в
        # два-три блока подряд и только потом ищет.
        body = THINK.sub("", it["out"]).lstrip()
        if body.startswith("<|search|>"):
            searched += 1
            q = body[len("<|search|>"):]
            if "<|/search|>" in q:
                closed += 1
                q = q.split("<|/search|>", 1)[0]
            ref = set(D1.normalize(it["ref"]).split())
            got = set(D1.normalize(q).split())
            if ref:
                overlap.append(len(ref & got) / len(ref))
    n = max(len(items), 1)
    return {"ищет": searched / n, "запрос закрыт": closed / max(searched, 1),
            "пересечение с учителем": sum(overlap) / max(len(overlap), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["d2"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--show", type=int, default=4)
    # 60 токенов не хватало: рассуждение до первого запроса бывает длиннее,
    # и такой ответ считался «не ищет» (замер 21.10).
    ap.add_argument("--max-new", type=int, default=160)
    args = ap.parse_args()

    items, tok = build(args.n)
    items = [it for it in items if len(it["prompt"]) + 64 < args.length]
    print(f"D2-замер: {len(items)} отложенных вопросов", flush=True)
    cfg = DEFAULT
    template = M.init_model(jax.random.key(0), cfg)
    state = O.init_state(template, O.OptConfig())
    for run in args.runs:
        local = C.pull(run, missing_ok=True)
        if local is None:
            print(f"\n{run}: снимков нет")
            continue
        params, _, meta = C.load(local, template, state)
        params = jax.device_put(params, jax.devices()[0])
        D1.decode_batch(params, cfg, items, tok, args.length, max_new=args.max_new)
        got = score(items)
        print(f"\n=== {run}, шаг {meta['step']:,} ===")
        print("  " + ", ".join(f"{k} {v:.1%}" for k, v in got.items()), flush=True)
        for it in items[:args.show]:
            q = tok.decode(it["prompt"][1:60]).replace("\n", " ")
            print(f"  [вопрос] {q}…\n      эталон ищет: {it['ref']!r}\n"
                  f"      модель: {it['out'][:300]!r}")


if __name__ == "__main__":
    main()
