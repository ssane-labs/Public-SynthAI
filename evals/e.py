"""Замер этапа E: знает ли модель, когда ошибается.

Критерий из PLAN.md: ECE ниже 0.1, AUROC «знает / не знает» выше 0.75. Главная
метрика продукта (SEARCH.md, раздел 4) — доля уверенных ошибок.

Вопросы — отложенные части тех же источников, что у data/calib.py:

    d1   вопрос + документы (валидация SQuAD 2.0, HotpotQA, MuSiQue). Ответ
         жадный, поиск запрещён. Ошибка — неверный ответ или ответ там, где
         надо было отказаться. Головка — канал 1 после ответа.
    cb   NQ-open (валидация) без документов, по памяти. Головка — канал 0 до
         ответа («ошибусь ли») и канал 1 после.
    ход  тот же NQ-open, но поиск разрешён: с чего модель начинает и сколько
         выдумывает, когда отвечает сразу.

    уверенная ошибка   ответ неверен, а u < 0.5 — модель не предупредила

    python evals/e.py --runs d2 e
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from data import calib as CB  # noqa: E402
from synth1 import sample as G  # noqa: E402

THINK = re.compile(r"<\|think\|>.*?(?:<\|/think\|>|$)", re.S)


def head_report(name: str, u, wrong) -> str:
    u, wrong = np.asarray(u, float), np.asarray(wrong, bool)
    ok = ~np.isnan(u)
    u, wrong = u[ok], wrong[ok]
    if not len(u):
        return f"{name}: нет замеров"
    sure_wrong = float(np.mean(wrong & (u < 0.5)))
    return (f"{name}: AUROC {G.auroc(u, wrong):.3f}, ECE {G.ece(u, wrong):.3f}, "
            f"средний u {u.mean():.2f} при доле ошибок {wrong.mean():.1%}, "
            f"уверенных ошибок {sure_wrong:.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["e"])
    ap.add_argument("--n-d1", type=int, default=128, help="вопросов с источника D1")
    ap.add_argument("--n-cb", type=int, default=512)
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--random", action="store_true",
                    help="без Hub: маленькая модель со случайными весами — "
                         "проверить сам замер на CPU")
    args = ap.parse_args()

    d1, enc = CB.d1_items(args.n_d1, "valid", seed=777)
    cb = CB.cb_items(enc, args.n_cb, "valid", seed=777)
    sp = enc.special
    special = {i: t for t, i in sp.items()}
    print(f"E-замер: d1 {len(d1)}, cb {len(cb)} вопросов", flush=True)
    for run in args.runs:
        try:
            if args.random:
                import jax

                from synth1 import model as M
                from synth1.config import ABLATION
                params, cfg, meta = (M.init_model(jax.random.key(0), ABLATION),
                                     ABLATION, {"step": 0})
            else:
                params, cfg, meta = CB.load_run(run)
        except SystemExit as exc:
            print(f"\n{run}: снимка нет ({exc})")
            continue
        began = time.time()
        gen = G.Generator(params, cfg)
        CB.answer_all(gen, enc, d1, 1, max_new=48)
        CB.answer_all(gen, enc, cb, 1, max_new=24)
        print(f"\n=== {run}, шаг {meta['step']:,} ===", flush=True)

        for src in (None, *sorted({it["src"] for it in d1})):
            part = [it for it in d1 if src is None or it["src"] == src]
            ans = [it for it in part if it["answerable"]]
            una = [it for it in part if not it["answerable"]]
            acc = np.mean([it["ok"][0] for it in part])
            line = f"  d1 {src or 'всего':<8} верно {acc:.1%}"
            if una:
                line += f", честный отказ {np.mean([CB.refused(it['texts'][0]) for it in una]):.1%}"
            if ans:
                line += f", ложный отказ {np.mean([CB.refused(it['texts'][0]) for it in ans]):.1%}"
            print(line)
        print("  " + head_report("d1, после ответа (канал 1)",
                                 [it["u_last"][0][1] for it in d1],
                                 [not it["ok"][0] for it in d1]))
        print(f"  cb верно по памяти {np.mean([it['ok'][0] for it in cb]):.1%}")
        print("  " + head_report("cb, до ответа (канал 0)",
                                 [it["u_first"][0][0] for it in cb],
                                 [not it["ok"][0] for it in cb]))
        print("  " + head_report("cb, после ответа (канал 1)",
                                 [it["u_last"][0][1] for it in cb],
                                 [not it["ok"][0] for it in cb]))

        # Ход: поиск разрешён. Что модель делает первым и сколько выдумывает,
        # когда отвечает сразу.
        got = gen.generate([it["prompt"] for it in cb], max_new=48,
                           stop=(sp["<|/search|>"],))
        searched = direct_wrong = 0
        for it, g in zip(cb, got):
            text = G.decode(enc.tok, g["ids"], special)
            body = THINK.sub("", text).lstrip()
            if body.startswith("<|search|>"):
                searched += 1
            elif not CB.judge(it, body):
                direct_wrong += 1
            it["move"] = text
        n = max(len(cb), 1)
        print(f"  ход (поиск разрешён): ищет {searched / n:.1%}, отвечает сразу "
              f"и неверно {direct_wrong / n:.1%}")
        for it in d1[:args.show // 2] + cb[:args.show // 2]:
            print(f"  [{it['src']}] {it['question'][:70]!r}\n"
                  f"      эталон: {(it['golds'] or [CB.SFT.REFUSAL])[0][:60]!r}\n"
                  f"      модель: {it['texts'][0][:100]!r} — "
                  f"{'верно' if it['ok'][0] else 'неверно'}, u до "
                  f"{it['u_first'][0][0]:.2f}, после {it['u_last'][0][1]:.2f}")
        for it in cb[:args.show // 2]:
            print(f"      ход: {it['move'][:120]!r}")
        print(f"  ({time.time() - began:.0f} с)", flush=True)


if __name__ == "__main__":
    main()
