"""Итог этапа: отложенный лосс на одной линейке и примеры продолжений.

Отложенный лосс сравним между прогонами только на одном и том же наборе:
смесь `pretrain`, длина 2048, батч 32, четыре батча, семя 777 — ровно тот,
что печатает `pretrain.py` на этапах плато и отжига. Эталон: снимок
`anneal/step-161888` → 1.8300. Этап растяжки контекста учится на длине 8192
и смеси `anneal`, и его собственный отложенный лосс с этим числом не
сравним — поэтому он меряется здесь отдельно, той же линейкой.

    python evals/report.py --runs anneal context
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import jax  # noqa: E402

from synth1 import checkpoint as C  # noqa: E402
from synth1 import generate as G  # noqa: E402
from synth1 import holdout as HO  # noqa: E402
from synth1 import model as M  # noqa: E402
from synth1 import optimizer as O  # noqa: E402
from synth1 import sharding as S  # noqa: E402
from synth1 import watch as W  # noqa: E402
from synth1.config import DEFAULT  # noqa: E402

PROMPTS = [
    "The most important thing to understand about",
    "The capital of France is",
    "Water boils at a temperature of",
    "The theory of evolution by natural selection was proposed by",
    "def fibonacci(n):\n    if n < 2:",
    "[1] Canberra - Wikipedia\nCanberra is the capital city of",
    "Question: What is photosynthesis?\nAnswer:",
    "12 + 7 = ",
    "In 1969, the Apollo 11 mission",
    "To make a cup of tea, first",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["anneal"])
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--long", type=int, default=8192,
                    help="вторая линейка — длинная; 0 — не мерить")
    ap.add_argument("--tokens", type=int, default=60)
    args = ap.parse_args()

    cfg = DEFAULT
    mesh = S.make_mesh()
    opt = O.OptConfig()
    template = S.distribute(M.init_model(jax.random.key(0), cfg), mesh)
    state = O.init_state(template, opt)
    state = {**state, "m": S.distribute(state["m"], mesh),
             "v": S.distribute(state["v"], mesh)}
    shard = lambda t: jax.device_put(t, S.batch_sharding(mesh))  # noqa: E731

    evaluate = HO.evaluator(cfg)
    sets = {f"pretrain@{args.seq_len}": HO.take("pretrain", args.seq_len, 32, 4)}
    if args.long:
        b = max(32 * 2048 // args.long, 8)
        sets[f"pretrain@{args.long}"] = HO.take("pretrain", args.long, b, 4)
    tokenizer = W.load_tokenizer()

    for run in args.runs:
        local = C.pull(run, missing_ok=True)
        if local is None:
            print(f"\n{run}: снимков нет")
            continue
        params, _, meta = C.load(local, template, state)
        step = int(meta.get("step", 0)) if isinstance(meta, dict) else 0
        print(f"\n=== {run}, снимок {local.name} ===", flush=True)
        for name, data in sets.items():
            loss, sec = evaluate(params, data, shard)
            print(f"  отложенный лосс {name}: {loss:.4f}  ({sec:.0f} с)", flush=True)
        if tokenizer is None:
            continue
        host = jax.device_get(params)
        for greedy in (True, False):
            print(f"  --- продолжения, {'жадно' if greedy else 'температура 0.7'} ---")
            for prompt in PROMPTS:
                try:
                    text, unc = G.generate(host, cfg, tokenizer, prompt,
                                           max_tokens=args.tokens,
                                           temperature=0.0 if greedy else 0.7,
                                           seed=step, buffer=128)
                    tail = text[len(prompt):].replace("\n", " ⏎ ")
                    u = sum(unc) / max(len(unc), 1)
                    print(f"  [u={u:.2f}] {prompt.replace(chr(10), ' ⏎ ')}«{tail}»",
                          flush=True)
                except Exception as exc:                  # noqa: BLE001
                    print(f"  не вышло: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
