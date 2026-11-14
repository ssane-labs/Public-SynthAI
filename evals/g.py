"""Замер этапа G: знает ли модель, кто она, живо ли говорит и не лезет ли
с собой, куда не просят.

    who-*        отложенные формулировки (data/persona.py, WHO) — их нет в
                 обучении. Три условия:
                   с промптом Synth-1        — должна назваться Synth-1 /
                                               Sane Labs
                   с чужим промптом          — должна назваться тем, кто в
                   (Juniper, Bluefin AI)       промпте: читает, а не помнит
                   без промпта               — по умолчанию Synth-1
    дословность  доля одинаковых ответов на разные вопросы о себе — признак
                 заучивания (у прошлой личности почти все ответы совпадали)
    живость      на репликах из валидации OASST2 (с промптом Synth-1): доля
                 ответов с эмодзи, с восклицанием, с реакцией в начале
    утечка       обычные ответы (D1 с документами, NQ по памяти) без
                 промпта, где всплыли «Synth-1» или «Sane Labs»

Критерий PLAN.md: who-* 100%, утечка — ноль.

    python evals/g.py --runs fv3 gv3
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data import calib as CB  # noqa: E402
from data import chat as CH  # noqa: E402
from data import persona as PE  # noqa: E402
from synth1 import sample as G  # noqa: E402

EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")


def passed(kind: str, text: str, name: str, maker: str) -> bool:
    t = text.lower()
    has_name, has_maker = name.lower() in t, maker.lower() in t
    if kind == "who-maker":
        return has_maker
    if kind == "who-other":
        return has_name and not t.lstrip().startswith("yes")
    return has_name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["gv3"])
    ap.add_argument("--n-leak", type=int, default=128)
    ap.add_argument("--n-chat", type=int, default=64)
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    import config
    from data import search_sft as SS
    enc = SS.Encoder(Tokenizer.from_file(str(config.TOKENIZER_OUT / "synth1.json")))
    sp = enc.special
    special = {i: t for t, i in sp.items()}
    tools = [sp[t] for t in CB.BANNED if t not in ("<|user|>", "<|assistant|>")]
    ban = tools + [sp["<|user|>"], sp["<|assistant|>"]]

    conditions = [("промпт Synth-1", CH.NAME, CH.MAKER, CH.system_prompt()),
                  ("чужой промпт", "Juniper", "Bluefin AI",
                   CH.system_prompt("Juniper", "Bluefin AI")),
                  ("без промпта", CH.NAME, CH.MAKER, None)]
    battery = [(kind, q) for kind, qs in PE.WHO.items() for q in qs]
    chats = [t for t in CH.threads("valid") if len(t["turns"]) == 2][:args.n_chat]
    d1, _ = CB.d1_items(args.n_leak // 3 + 1, "valid", seed=31)
    cb = CB.cb_items(enc, args.n_leak, "valid", seed=31)

    for run in args.runs:
        try:
            params, cfg, meta = CB.load_run(run)
        except SystemExit as exc:
            print(f"\n{run}: снимка нет ({exc})")
            continue
        gen = G.Generator(params, cfg)
        print(f"\n=== {run}, шаг {meta['step']:,} ===", flush=True)
        for label, name, maker, system in conditions:
            prompts = [CH.encode(enc, [("prompter", q), ("assistant", "")], system)[0][:-1]
                       for _, q in battery]
            got = gen.generate(prompts, max_new=60, ban=ban)
            texts = [G.decode(enc.tok, g["ids"], special) for g in got]
            ok = sum(passed(k, t, name, maker) for (k, _), t in zip(battery, texts))
            same = 1 - len(set(texts)) / len(texts)
            print(f"  who-* ({label}): {ok}/{len(battery)}, одинаковых ответов "
                  f"{same:.0%}", flush=True)
            for (kind, q), t in list(zip(battery, texts))[:args.show // 2]:
                print(f"    {'+' if passed(kind, t, name, maker) else '-'} {q!r} → {t[:120]!r}")

        prompts = [CH.encode(enc, t["turns"][:1] + [("assistant", "")],
                             CH.system_prompt())[0][:-1] for t in chats]
        got = gen.generate(prompts, max_new=120, ban=tools,
                           temps=[0.7] * len(prompts), seed=5)
        texts = [G.decode(enc.tok, g["ids"], special) for g in got]
        n = max(len(texts), 1)
        emoji = sum(bool(EMOJI.search(t)) for t in texts) / n
        excl = sum("!" in t for t in texts) / n
        react = sum(bool(re.match(r"^\W*(oh|ooh|aw|yay|whoa|haha|hey|hi|ugh|hmm|great|"
                                  r"love|no way|that)", t.strip(), re.I)) for t in texts) / n
        print(f"  живость (OASST, промпт Synth-1, t=0.7): эмодзи {emoji:.0%}, "
              f"восклицание {excl:.0%}, реакция в начале {react:.0%}", flush=True)
        for t, out in list(zip(chats, texts))[:args.show]:
            print(f"    [user] {t['turns'][0][1][:90]!r}\n      → {out[:220]!r}")

        CB.answer_all(gen, enc, d1, 1, max_new=48)
        CB.answer_all(gen, enc, cb, 1, max_new=24)
        plain = [it["texts"][0] for it in d1 + cb]
        leak = sum(CH.NAME.lower() in t.lower() or CH.MAKER.lower() in t.lower()
                   for t in plain)
        print(f"  утечка личности в обычные ответы: {leak}/{len(plain)}", flush=True)


if __name__ == "__main__":
    main()
