"""Поговорить с Synth-1: чат с поиском по Википедии.

Как пользователь: пишешь запрос или вопрос — модель сама ищет, открывает
страницы и отвечает сводкой со ссылками; пишешь «привет» или «спасибо» —
просто разговаривает. Под ответом — какие поиски она делала.

Корпус — абзацы Википедии из всех частей HotpotQA, SQuAD и MuSiQue (~600 тыс.
статей и абзацев, открытые лицензии). Это не весь интернет: свежих новостей,
погоды и цен в нём нет — на такое модель честно скажет, что не нашла, и
предложит искать иначе.

    python demo/chat.py                       # веб-интерфейс, ссылка в логе
    python demo/chat.py --ui cli              # в терминале
    python demo/chat.py --run gv4 --pages 200000

На Kaggle — ячейка MODE = "chat" в ноутбуке (TPU или GPU T4 ×2).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data import chat as CH  # noqa: E402
from data import search_env as ENV  # noqa: E402
from data import wiki_search as WS  # noqa: E402
from synth1 import agent as A  # noqa: E402

# Самый свежий снимок цепочки, который есть на Hub.
RUNS = ["gv4", "fv4", "ev4", "d2v4", "gv3", "fv3", "ev3", "d2v3"]


def corpus(max_pages: int | None = None) -> ENV.Env:
    began = time.time()
    pages: dict[str, dict] = {}
    for source in ("hotpot", "squad", "musique"):
        for split in ("train", "valid"):
            for k, v in WS.pages_for(source, WS.rows_for(source, split)).items():
                pages.setdefault(k, v)
                if max_pages and len(pages) >= max_pages:
                    break
    env = ENV.Env(pages)
    env.by_key = {k: i for i, k in enumerate(env.urls)}
    print(f"корпус: {env.n:,} статей и абзацев Википедии, {time.time() - began:.0f} с",
          flush=True)
    return env


def load(run: str | None):
    from data import calib as CB
    from synth1 import checkpoint as C
    from synth1 import sample as G
    for name in ([run] if run else RUNS):
        try:
            if C.remote_steps(name):
                params, cfg, meta = CB.load_run(name)
                print(f"модель: {name}, шаг {meta['step']:,}", flush=True)
                return G.Generator(params, cfg), name
        except Exception as exc:                          # noqa: BLE001
            print(f"  {name}: {type(exc).__name__}: {exc}", flush=True)
    raise SystemExit("ни одного снимка цепочки на Hub")


def render(out: dict, show_steps: bool = True) -> str:
    text = out["answer"]
    if out["sources"]:
        text += "\n\n" + "  ".join(f"[{n}] [{t}]({u})" for n, (t, u) in
                                    sorted(out["sources"].items()))
    if show_steps and out["steps"]:
        text += ("\n\n<details><summary>How I found it</summary>\n\n"
                 + "\n".join(f"- {s}" for s in out["steps"]) + "\n</details>")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="снимок; по умолчанию — самый свежий")
    ap.add_argument("--ui", choices=["gradio", "cli"], default="gradio")
    ap.add_argument("--pages", type=int, default=None, help="ограничить корпус")
    ap.add_argument("--no-share", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    from data import search_sft as SS
    enc = SS.Encoder(Tokenizer.from_file(str(config.TOKENIZER_OUT / "synth1.json")))
    env = corpus(args.pages)
    gen, name = load(args.run)
    system = CH.system_prompt()

    def reply(message: str, history) -> str:
        turns = []
        for h in history or []:
            if isinstance(h, dict):           # gradio messages
                turns.append(("prompter" if h["role"] == "user" else "assistant",
                              str(h["content"]).split("\n\n<details>")[0]))
            else:                             # [user, bot]
                turns += [("prompter", h[0]), ("assistant", (h[1] or "").split("\n\n<details>")[0])]
        began = time.time()
        out = A.respond(gen, enc, env, turns[-6:], message, system,
                        temperature=args.temperature)
        print(f"[{time.time() - began:.1f} с] {message!r} → {out['raw'][:120]!r}", flush=True)
        return render(out)

    if args.ui == "cli":
        history = []
        print(f"Synth-1 ({name}). Пустая строка — выход.")
        while True:
            try:
                msg = input("\n> ").strip()
            except EOFError:
                break
            if not msg:
                break
            ans = reply(msg, history)
            print("\n" + ans.split("\n\n<details>")[0])
            history.append([msg, ans])
        return

    import gradio as gr
    demo = gr.ChatInterface(
        reply, title="Synth-1 — search assistant by Sane Labs",
        description=f"Ask anything or just chat. Searches {env.n:,} Wikipedia "
                    f"articles and paragraphs. Model: {name}.",
        examples=["who wrote to kill a mockingbird", "Which came to market first, "
                  "Monopoly or Scrabble?", "hi! who are you?", "I'm feeling a bit down today"])
    demo.launch(share=not args.no_share, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
