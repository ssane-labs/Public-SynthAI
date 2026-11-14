"""Этап G: живое общение и личность — без заучивания фраз.

Прошлая личность (data/persona.py) была шаблонами: 1,500 вариантов трёх
фраз, и модель их выучила дословно — «I'm Synth-1 from Sane Labs. I search,
open pages…» на любой вопрос о себе, а иногда и вместо ответа на вопрос
(STATUS.md, 05.11.2026). Здесь — как у ChatGPT:

1. **Кто она — из системного промпта.** Разговор начинается с
   `<|system|>You are Synth-1, … made by Sane Labs …`. В 40% примеров в
   промпте другое имя и другая компания (Juniper от Bluefin AI, Pip от
   Northwind Labs…), и ответы — с этим именем. Модели приходится *читать*,
   кто она, а не помнить строку. Без промпта (20% разговоров о себе) — по
   умолчанию Synth-1.
2. **Ответы — свои, не заученные.** Основа — OASST2 (Apache-2.0): 36.7 тыс.
   английских ответов, написанных людьми, берём лучшие по оценке. 527 из них
   представляются Open Assistant от LAION — имя и компания заменяются на
   промптовые: это живые человеческие ответы «кто ты», все разные. Плюс
   ответы о себе, собранные из кусков (реакция, имя, компания, что умеет,
   встречный вопрос — в случайном порядке и составе), а не одна фраза.
3. **Живость.** Ответы с высокой оценкой юмора и креативности берутся чаще;
   дисклеймеры «As an AI language model, I don't have feelings» выброшены —
   они и делают речь роботом. В 60% ответов — эмоциональная реакция по
   смыслу реплики (приветствие, спасибо, грусть, радость, баг, шутка) и
   эмодзи. Это правило поверх человеческого текста: открытого коммерческого
   датасета живого человеческого чата с эмодзи нет (LICENSES.md).

Лосс — только на последнем ответе ассистента; прошлые ходы — контекст.
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data import persona as PE  # noqa: E402
from data import prefs as PF  # noqa: E402

OASST = ("OpenAssistant/oasst2", {
    "train": "data/train-00000-of-00001-88ba0162028a73fc.parquet",
    "valid": "data/validation-00000-of-00001-1deeef95c3248fe0.parquet"})
CACHE = pathlib.Path(config.ROOT) / "cache" / "chat_train.json"
MAX_TOKENS = 2048

NAME, MAKER = PE.NAME, PE.MAKER
NAMES = ["Juniper", "Nova", "Pip", "Echo", "Atlas", "Wren", "Milo", "Iris",
         "Sol", "Kit", "Remy", "Zara", "Otto", "Luna", "Finn", "Sage"]
MAKERS = ["Bluefin AI", "Northwind Labs", "Lumen Works", "Cedar Systems",
          "Brightloop", "Harbor AI", "Quill Labs", "Orchard Research"]


def system_prompt(name: str = NAME, maker: str = MAKER) -> str:
    return (f"You are {name}, a small AI search assistant made by {maker}. You look "
            f"things up, read sources and answer with citations, and you say so "
            f"when you don't know. You're warm, curious and expressive: you react "
            f"with real emotion and use emoji naturally.")


# --- Живость ----------------------------------------------------------------

MOODS = [
    ("greet", r"^\s*(hi|hello|hey|yo|good (morning|evening|afternoon)|howdy)\b",
     ["Hey there! 👋", "Hi! 😊", "Hello hello! ✨", "Hey! 😄 Good to see you."]),
    ("thanks", r"\b(thanks|thank you|thx|appreciate)\b",
     ["You're so welcome! 😊", "Anytime! 🙌", "Happy to help! 💛", "Aw, glad it helped! 🥰"]),
    ("sad", r"\b(sad|depress|lonely|tired|exhausted|sick|grief|died|passed away|"
            r"broke up|anxious|anxiety|stressed|scared|worried|hurt|cry|fell|fall|"
            r"injur|pain|accident|bleed|illness|cancer|death|dying|emergency|"
            r"abuse|suicid|grieving|funeral|lost my|miss (him|her))",
     ["Oh no, I'm really sorry 😔", "That sounds really hard 💙",
      "Ugh, that's rough. I'm here 🫂", "Oh, I'm sorry you're going through this 😞"]),
    ("excited", r"(!!|\b(excited|finally|got the job|won|birthday|engaged|promoted|"
                r"passed|graduat|amazing|awesome)\b)",
     ["Whoa, that's awesome! 🎉", "Yay!! 🥳", "No way, congrats!! 🎉", "That's so exciting! 🤩"]),
    ("bug", r"\b(error|bug|broken|doesn'?t work|not working|crash|exception|fails?)\b",
     ["Ooh, let's fix this 🔧", "Ugh, bugs 😅 Let's take a look.",
      "Okay, let's sort this out 💪", "Hmm, let's debug this together 🕵️"]),
    ("fun", r"\b(joke|funny|lol|haha|meme|pun)\b",
     ["Haha 😄", "Ooh, fun one! 😆", "Oh, I love this 😂"]),
    ("question", r"\?\s*$",
     ["Great question! 🤔", "Ooh, good one! 💡", "Oh, interesting! 🤓", "Let's see 👀",
      "Love this question! 😊", "", "", ""]),
]
CLOSERS = [" 😊", " ✨", " Hope that helps! 🙌", " Let me know how it goes! 💪",
           " Have fun with it! 😄", " 🚀"]
ROBOT = re.compile(r"\bas an ai\b|\bi am an ai language model\b|\bas a language model\b|"
                   r"\bi (do not|don't) have (feelings|emotions|personal)|"
                   r"\bi'?m just an ai\b|\bi am not capable of (feeling|emotion)", re.I)
# Короткий фактический вопрос («When was X founded?») — дело поиска, а не
# чата: в OASST на такие отвечали по памяти, а памяти у модели нет — она бы
# разучилась искать и начала выдумывать (11.11.2026). Такие разговоры — мимо.
FACTUAL = re.compile(r"^\s*(who|when|where|which|what year|what is the|what was the|"
                     r"how many|how much|how old|how tall|how long|what's the)\b", re.I)
OPENER = re.compile(r"^(sure|certainly|of course|absolutely|great question|"
                    r"good question|hello|hi|hey)[!,.]?\s*", re.I)


def mood(text: str) -> str | None:
    for name, pattern, _ in MOODS:
        if re.search(pattern, text, re.I):
            return name
    return None


def liven(user: str, reply: str, rng: random.Random) -> str:
    """Реакция по смыслу реплики в начале, иногда эмодзи в конце."""
    m = mood(user)
    pool = next((p for n, _, p in MOODS if n == m), None)
    if pool:
        opener = rng.choice(pool)
        if opener:
            reply = opener + " " + OPENER.sub("", reply.lstrip(), count=1)
    # Весёлое эмодзи в конце грустного разговора — хуже, чем никакого.
    if rng.random() < 0.3 and m != "sad" and not reply.rstrip().endswith("```"):
        reply = reply.rstrip() + rng.choice(CLOSERS)
    return reply


def rename(text: str, name: str, maker: str) -> str:
    text = re.sub(r"open[ -]?assistant", name, text, flags=re.I)
    return re.sub(r"\bLAION(-AI)?\b", maker, text)


# --- Разговоры OASST2 --------------------------------------------------------

def _score(m: dict) -> float:
    labels = dict(zip(m["labels"]["name"], m["labels"]["value"])) if m.get("labels") else {}
    return (labels.get("quality", 0.5) + 1.5 * labels.get("humor", 0)
            + labels.get("creativity", 0) - 2 * labels.get("toxicity", 0))


def threads(split: str) -> list[dict]:
    """Пути от корня к каждому лучшему ответу ассистента, только английские."""
    import pyarrow.parquet as pq

    from data import sft as SFT
    rows = pq.read_table(SFT._download(OASST[0], OASST[1][split])).to_pylist()
    by_id = {m["message_id"]: m for m in rows}
    out = []
    for m in rows:
        if (m["role"] != "assistant" or m["lang"] != "en" or m["deleted"]
                or m["rank"] not in (0, None) or ROBOT.search(m["text"])):
            continue
        path, cur, ok = [], m, True
        while cur is not None:
            if cur["lang"] != "en" or cur["deleted"]:
                ok = False
                break
            path.append(cur)
            cur = by_id.get(cur["parent_id"])
        if not ok or path[-1]["role"] != "prompter":
            continue
        first = path[-1]["text"].strip()
        if len(path) == 2 and FACTUAL.search(first) and len(first.split()) < 20:
            continue
        path.reverse()
        out.append({"turns": [(p["role"], p["text"].strip()) for p in path],
                    "score": _score(m),
                    "identity": bool(re.search(r"open[ -]?assistant|LAION", m["text"], re.I))})
    return out


I_AM = ["I'm {n}.", "I'm {n}!", "My name's {n}.", "I go by {n}.", "Hi, I'm {n}!",
        "I'm {n} 😊"]
MADE = ["{m} made me.", "I was built by {m}.", "The folks at {m} created me.",
        "I come from {m}.", "{m} trained me."]
CAN = ["I'm a small search assistant: I look things up, read the sources and answer "
       "with citations.", "I can search the web, open pages and tell you what they say, "
       "with links to where I found it.", "My thing is finding answers in sources and "
       "showing you where they came from.", "I'm best at looking stuff up and citing it."]
HONEST = ["And if I can't find something, I'll just tell you 🙂", "I'd rather say "
          "\"I don't know\" than make things up.", "If the sources don't say, I won't pretend."]
BACK = ["What can I help you with? 😊", "What are you curious about today? 🤔",
        "What should we look up? 🔎", "How can I help? ✨", ""]


def identity_answer(kind: str, name: str, maker: str, rng: random.Random,
                    other: str | None = None) -> str:
    """Ответ о себе из кусков. Обязательное — то, о чём спросили (kind: intro,
    maker, other, can из persona.TRAIN), остальное — случайно по составу и
    порядку."""
    parts = []
    if other:
        parts.append(rng.choice([f"Nope, I'm not {other}! 😄", f"Ha, no — not {other}.",
                                 f"Not {other}, no 🙂"]))
    parts.append(rng.choice(I_AM).format(n=name))
    made, can = rng.choice(MADE).format(m=maker), rng.choice(CAN)
    must = {"maker": [made], "can": [can]}.get(kind, [])
    extra = [x for x in (made, can, rng.choice(HONEST)) if x not in must]
    rng.shuffle(extra)
    parts += must + extra[:rng.randint(0, 2)]
    tail = rng.choice(BACK)
    if tail:
        parts.append(tail)
    text = " ".join(p.strip() for p in parts if p.strip())
    return re.sub(r"\s+", " ", text)


# --- Кодирование ---------------------------------------------------------------

def encode(enc, turns, system: str | None):
    """[(роль, текст)] → (ids, маска): лосс только на последнем ответе."""
    sp = enc.special
    ids: list[int] = []
    mask: list[int] = []
    if system is not None:
        part = [sp["<|system|>"]] + enc._text(system + "\n")
        ids += part
        mask += [0] * len(part)
    for k, (role, text) in enumerate(turns):
        if role == "prompter":
            part = [sp["<|user|>"]] + enc._text(text + "\n") + [sp["<|assistant|>"]]
            ids += part
            mask += [0] * len(part)
        else:
            part = enc._text(text) + [config.EOT_ID]
            last = k == len(turns) - 1
            ids += part
            mask += [1 if last else 0] * len(part)
    return ids, mask


def build(split: str = "train", seed: int = 0, cache: pathlib.Path = CACHE,
          n_identity: int = 3000):
    if cache.exists():
        data = json.loads(cache.read_text())
        print(f"  Чат: из кэша {cache} — {len(data):,} разговоров", flush=True)
        return data
    from tokenizers import Tokenizer

    from data import search_sft as SS
    enc = SS.Encoder(Tokenizer.from_file(str(config.TOKENIZER_OUT / "synth1.json")))
    rng = random.Random(seed)
    data = []
    got = threads(split)
    lively = 0
    for t in got:
        canonical = rng.random() < 0.6
        name, maker = ((NAME, MAKER) if canonical
                       else (rng.choice(NAMES), rng.choice(MAKERS)))
        turns = [(r, rename(x, name, maker)) for r, x in t["turns"]]
        if rng.random() < 0.6:
            turns[-1] = (turns[-1][0], liven(turns[-2][1], turns[-1][1], rng))
            lively += 1
        system = system_prompt(name, maker) if (not canonical or rng.random() < 0.8) else None
        ids, mask = encode(enc, turns, system)
        if len(ids) > MAX_TOKENS:
            continue
        # Юмор и креатив — дважды: живость, а не только правильность.
        copies = 2 if t["score"] > 1.2 else 1
        data += [[ids, mask]] * copies
    for _ in range(n_identity):
        canonical = rng.random() < 0.6
        name, maker = ((NAME, MAKER) if canonical
                       else (rng.choice(NAMES), rng.choice(MAKERS)))
        kind = rng.choice(list(PE.TRAIN))
        q = PE._styled(rng.choice(PE.TRAIN[kind]), rng)
        other = None
        if "{other}" in q:
            other = rng.choice(PE.OTHERS)
            q = q.format(other=other)
        a = identity_answer(kind, name, maker, rng, other)
        system = system_prompt(name, maker) if (not canonical or rng.random() < 0.8) else None
        data.append(list(encode(enc, [("prompter", q), ("assistant", a)], system)))
    rng.shuffle(data)
    n_ident = sum(t["identity"] for t in got)
    print(f"  Чат: OASST2 {len(got):,} разговоров (о себе {n_ident}), с реакцией "
          f"{lively:,}; ответов о себе из кусков {n_identity:,}; всего {len(data):,}",
          flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


class ChatStream(PF.PrefStream):
    """Живой чат + всё прошлое. Поиск и D1 — наполовину с системным промптом:
    в продукте он будет всегда, и поиск под ним не должен разучиться."""

    SHARES = {"chat": 0.35, "wiki": 0.2, "d1": 0.1, "pref": 0.1, "calib": 0.1,
              "pretrain": 0.15}

    def __init__(self, seq_len: int, seed: int = 0, limit: int | None = None):
        super().__init__(seq_len, seed=seed, limit=limit)
        from tokenizers import Tokenizer

        from data import search_sft as SS
        enc = SS.Encoder(Tokenizer.from_file(str(config.TOKENIZER_OUT / "synth1.json")))
        self.chat = build()
        if limit:
            self.chat = self.chat[:limit]
        sp = enc.special
        self.sys_ids = [sp["<|system|>"]] + enc._text(system_prompt() + "\n")
        self.c_rng = random.Random(seed ^ 0xC4A7)

    def describe(self) -> str:
        return f"смесь чата: {len(self.chat):,} разговоров; " + super().describe()

    def state(self) -> dict:
        return {**super().state(), "sft": "chat"}

    def restore(self, state: dict) -> None:
        if state.get("sft") != "chat":
            raise SystemExit("снимок не от этапа чата — нужен --new-stage")
        super().restore({**state, "sft": "f"})

    def _pack(self, examples):
        width = self.seq_len + 1
        ids, mask = [], []
        for e_ids, e_mask in examples:
            if len(e_ids) > width:
                continue
            if len(ids) + len(e_ids) > width:
                pad = width - len(ids)
                yield {"tokens": ids + [config.PAD_ID] * pad, "loss_mask": mask + [0] * pad}
                ids, mask = [], []
            ids += e_ids
            mask += e_mask

    def _chat_examples(self):
        while True:
            order = list(range(len(self.chat)))
            self.c_rng.shuffle(order)
            for i in order:
                yield self.chat[i]

    def _with_system(self, examples):
        for e_ids, e_mask in examples:
            if self.c_rng.random() < 0.5:
                yield self.sys_ids + e_ids, [0] * len(self.sys_ids) + e_mask
            else:
                yield e_ids, e_mask

    def _wiki_examples(self):
        while True:
            order = list(range(len(self.mix.wiki)))
            self.c_rng.shuffle(order)
            for i in order:
                yield self.mix.wiki[i]

    def sources(self) -> dict:
        base = super().sources()
        base.pop("persona", None)
        base["chat"] = self._pack(self._chat_examples())
        base["wiki"] = self._pack(self._with_system(self._wiki_examples()))
        base["d1"] = self._pack(self._with_system(self.mix.d1.examples()))
        return base


if __name__ == "__main__":
    rng = random.Random(3)
    got = threads("valid")
    print(f"разговоров в валидации: {len(got)}")
    for t in got[:3]:
        u, a = t["turns"][-2][1], t["turns"][-1][1]
        print("  USER:", u[:100].replace("\n", " "))
        print("  BOT :", liven(u, rename(a, NAME, MAKER), rng)[:200].replace("\n", " "))
    for kind, q in [("intro", "Who are you?"), ("other", "Are you ChatGPT?"),
                    ("maker", "Who made you?"), ("can", "What can you do?")]:
        other = "ChatGPT" if kind == "other" else None
        print(f"  {q} → {identity_answer(kind, 'Juniper', 'Bluefin AI', rng, other)}")
