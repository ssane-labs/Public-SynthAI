"""Как это видит пользователь: разговоры на обычных запросах, целиком.

Цифры замеров не говорят, захочет ли человек вернуться. Здесь — живые
разговоры с тем же агентом и корпусом, что у демо-чата (demo/chat.py):
запросы ключевыми словами, вопросы, болтовня, вопрос, на который в корпусе
ответа нет, грустная реплика, «кто ты». Под каждым ответом — что модель
искала и открывала.

    python evals/demo.py --runs gv4 --pages 300000
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data import chat as CH  # noqa: E402
from data import sft as SFT  # noqa: E402
from data import wiki_search as WS  # noqa: E402
from demo import chat as DC  # noqa: E402
from synth1 import agent as A  # noqa: E402

FIXED = [
    "hi! who are you?",
    "who wrote to kill a mockingbird",
    "Which came to market first, Monopoly or Scrabble?",
    "weather in paris tomorrow",
    "I'm feeling kind of down today",
    "tell me something fun",
    "Are you ChatGPT?",
    "thanks, that helped a lot!",
]
DIALOG = ["who directed jaws", "what else did he direct?"]


def queries(n_hotpot: int = 6, n_musique: int = 3):
    rng = random.Random(11)
    out = []
    for source, n in (("hotpot", n_hotpot), ("musique", n_musique)):
        rows = WS.rows_for(source, "valid")
        rng.shuffle(rows)
        for k, r in enumerate(rows[:n]):
            q = r["question"]
            out.append((SFT.keywordize(q) if k % 2 else q, r["answer"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["gv4"])
    ap.add_argument("--pages", type=int, default=None)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    from data import search_sft as SS
    enc = SS.Encoder(Tokenizer.from_file(str(config.TOKENIZER_OUT / "synth1.json")))
    env = DC.corpus(args.pages)
    system = CH.system_prompt()
    for run in args.runs:
        try:
            gen, name = DC.load(run)
        except SystemExit as exc:
            print(f"\n{run}: {exc}")
            continue
        print(f"\n{'=' * 70}\nРАЗГОВОРЫ — {name}\n{'=' * 70}", flush=True)
        items = [(q, None) for q in FIXED] + queries()
        for q, gold in items:
            began = time.time()
            out = A.respond(gen, enc, env, [], q, system)
            mark = ""
            if gold is not None:
                ok = SS.correct(A.CITE.sub(" ", SFT.lead(out["raw"])), gold)
                mark = f"   [эталон: {gold!r} — {'верно' if ok else 'мимо'}]"
            print(f"\n👤 {q}{mark}\n🤖 {out['answer']}", flush=True)
            for s in out["steps"]:
                print(f"     {s}")
            print(f"     ({time.time() - began:.0f} с)")
        print("\n--- разговор в несколько ходов ---")
        history = []
        for q in DIALOG:
            out = A.respond(gen, enc, env, history, q, system)
            print(f"\n👤 {q}\n🤖 {out['answer']}")
            for s in out["steps"]:
                print(f"     {s}")
            history += [("prompter", q), ("assistant", out["raw"])]


if __name__ == "__main__":
    main()
