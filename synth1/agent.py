"""Агент: модель ведёт разговор с поисковиком по ходам.

Модель пишет до закрывающего тега действия — среда отвечает выдачей или
страницей; до <|eot|> — это ответ; упёрлась в max_new посреди мысли —
продолжаем. После `turns` вызовов инструменты запрещены — остаётся ответить.

Одна и та же петля у замера (evals/d2_env.py), у демо-чата (demo/chat.py) и
у показа примеров (evals/demo.py): что меряем, то и показываем.
"""

from __future__ import annotations

import re

from synth1 import sample as G

THINK = re.compile(r"<\|think\|>.*?(?:<\|/think\|>|$)", re.S)
CITE = re.compile(r"<\|cite\|>\s*(\d+)\s*<\|/cite\|>")
ACTIONS = ("search", "open", "find")


def run_agent(gen, enc, env, items, turns: int, budget: int = 8192,
              max_new: int = 192, tools: bool = True, temperature: float = 0.0):
    """items — [{"prompt": ids, "question": текст}]. Возвращает состояния:
    answer (или None), trace, calls, src (номер → документ), events."""
    sp = enc.special
    special = {i: t for t, i in sp.items()}
    stops = tuple(sp[f"<|/{a}|>"] for a in ACTIONS)
    tool_ids = tuple(sp[f"<|{a}|>"] for a in ACTIONS) + stops
    states = [{"ids": list(it["prompt"]), "src": {}, "cur": None, "last_q": "",
               "calls": 0, "done": False, "answer": None, "trace": "", "last": "",
               "events": []}
              for it in items]
    for _ in range(2 * turns + 4):
        active = [k for k, s in enumerate(states) if not s["done"]]
        if not active:
            break
        for banned in (False, True):
            group = [k for k in active
                     if (not tools or states[k]["calls"] >= turns) == banned]
            if not group:
                continue
            got = gen.generate([states[k]["ids"] for k in group], max_new=max_new,
                               stop=stops, ban=tool_ids if banned else (),
                               temps=[temperature] * len(group))
            for k, g in zip(group, got):
                s, it = states[k], items[k]
                text = G.decode(enc.tok, g["ids"], special)
                s["ids"] += g["ids"]
                s["trace"] += text
                s["last"] += text
                if g["ids"] and g["ids"][-1] in stops:
                    obs = observe(env, enc, s, it, text)
                    s["ids"] += obs
                    s["trace"] += G.decode(enc.tok, obs, special)[:200] + "…"
                    s["calls"] += 1
                    s["last"] = ""
                elif len(g["ids"]) < max_new:
                    s["done"] = True
                    s["answer"] = final_answer(s["last"])
                if not s["done"] and len(s["ids"]) + max_new + 16 > budget:
                    s["done"] = True
    return states


def final_answer(text: str) -> str | None:
    """Ответ — то, что после последней мысли. Пусто или недописанное
    действие — ответа нет."""
    text = text.rsplit("<|/think|>", 1)[-1]
    text = THINK.sub("", text).strip()
    if not text or any(f"<|{a}|>" in text for a in ACTIONS):
        return None
    return text


def observe(env, enc, s, it, text: str) -> list[int]:
    sp = enc.special
    a = next(x for x in ACTIONS if text.endswith(f"<|/{x}|>"))
    arg = text.rsplit(f"<|{a}|>", 1)[-1][:-len(f"<|/{a}|>")].strip()
    if a == "search":
        s["last_q"] = arg
        hits = env.search(arg)
        lines = []
        for i in hits:
            num = len(s["src"]) + 1
            s["src"][num] = i
            lines.append(env.result_line(num, i, arg))
        s["events"].append(("search", arg, [env.pages[i]["title"] for i in hits]))
        body = "\n".join(lines) or "No results."
        return [sp["<|result|>"]] + enc._text(body) + [sp["<|/result|>"]]
    if a == "open":
        i = None
        if arg.isdigit() and int(arg) in s["src"]:
            i = s["src"][int(arg)]
        else:
            i = env.by_url.get(arg)
        if i is None:
            body = "[page not found]"
            s["events"].append(("open", arg, None))
        else:
            s["cur"] = i
            body = env.view(i, it["question"] + " " + s["last_q"]) or "[empty page]"
            s["events"].append(("open", arg, env.pages[i]["title"]))
        return [sp["<|page|>"]] + enc._text(body) + [sp["<|/page|>"]]
    body = (env.find(s["cur"], arg) if s["cur"] is not None else "") or "[no matches]"
    s["events"].append(("find", arg, None))
    return [sp["<|page|>"]] + enc._text(body) + [sp["<|/page|>"]]


# --- Разговор: для демо и показа примеров -----------------------------------

FALLBACK = ("Sorry, I lost my train of thought there 😅 Could you ask that "
            "again, maybe in other words?")


def prompt_ids(enc, history, message: str, system: str | None):
    """Промпт: системный, прошлые ходы (только ответы, без поисков) и вопрос."""
    from data import chat as CH
    turns = [(role, CITE.sub(lambda m: f"[{m.group(1)}]", text)) for role, text in history]
    return CH.encode(enc, turns + [("prompter", message), ("assistant", "")], system)[0][:-1]


def respond(gen, enc, env, history, message: str, system: str | None,
            turns: int = 6, temperature: float = 0.0) -> dict:
    """Один ход разговора: ответ (markdown со ссылками), шаги и источники."""
    ids = prompt_ids(enc, history, message, system)
    s = run_agent(gen, enc, env, [{"prompt": ids, "question": message}], turns,
                  temperature=temperature)[0]
    raw = s["answer"] or FALLBACK
    sources = {}

    def link(m):
        n = int(m.group(1))
        i = s["src"].get(n)
        if i is None:
            return f"[{n}]"
        sources[n] = (env.pages[i]["title"], env.pages[i].get("url", ""))
        return f"[[{n}]]({sources[n][1]})" if sources[n][1] else f"[{n}]"

    text = CITE.sub(link, raw)
    text = re.sub(r"\]\)\s*\[\[", "]) [[", text)
    steps = []
    for kind, arg, extra in s["events"]:
        if kind == "search":
            steps.append(f"🔎 searched “{arg}” → " + (", ".join(extra[:3]) or "nothing"))
        elif kind == "open":
            steps.append(f"📄 opened [{arg}] {extra or '(not found)'}")
        else:
            steps.append(f"🔦 looked for “{arg}” on the page")
    return {"answer": text, "raw": raw, "steps": steps, "sources": sources}
