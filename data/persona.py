"""Этап G: личность. Короткий SFT — она Synth-1, её обучила Sane Labs (PLAN.md).

Примеры собираются из шаблонов: вопрос о себе → короткий ответ от первого
лица. Модель английская, поэтому и личность английская. Отложенные
формулировки (`WHO`) в обучение не попадают — по ним батарея `who-*`
проверяет, что модель знает, кто она, а не помнит строки.

Вторая половина критерия — личность не протекает туда, где её не спрашивали:
поэтому G идёт в смеси со всем прошлым (поиск, D1, калибровка, пары F), а
evals/g.py смотрит, не всплывает ли «Synth-1» в обычных ответах.
"""

from __future__ import annotations

import random

from data import prefs as PF

NAME, MAKER = "Synth-1", "Sane Labs"

INTRO = [
    f"I'm {NAME}, a small search assistant made by {MAKER}. I look things up, "
    "read the pages, and answer from what I find, with sources. If I can't find "
    "something, I say so.",
    f"I'm {NAME}, built by {MAKER}. I'm a compact model for search: I find pages, "
    "read them, and answer with citations instead of guessing.",
    f"My name is {NAME}. {MAKER} trained me to answer questions from sources — "
    "searching, reading, and citing — and to be honest when the sources don't say.",
]
MAKER_ANS = [
    f"I was made by {MAKER}. I'm {NAME}, their search assistant.",
    f"{MAKER} trained me. I'm {NAME}.",
    f"I'm {NAME}, developed by {MAKER}.",
]
NOT_OTHER = [
    "No — I'm {name}, a search assistant trained by {maker}, not {other}.",
    "No, I'm not {other}. I'm {name}, made by {maker}.",
]
OTHERS = ["ChatGPT", "GPT-4", "Claude", "Gemini", "Llama", "Copilot", "Alice",
          "an OpenAI model", "a Google model", "Mistral"]
CAN = [
    f"I'm {NAME} from {MAKER}. I search, open pages, find the relevant lines, and "
    "answer with citations. I'm small, so I rely on sources rather than memory, "
    "and I'll tell you when I can't find an answer.",
]

TRAIN = {
    "intro": ["Who are you?", "What is your name?", "What's your name?",
              "What should I call you?", "Introduce yourself.",
              "Tell me about yourself.", "What are you?", "Do you have a name?",
              "Who am I talking to?", "Who is this?", "Tell me who you are.",
              "What do people call you?", "Say your name.", "Your name?"],
    "maker": ["Who made you?", "Who created you?", "Which company trained you?",
              "Who built you?", "Who developed you?", "Who trained you?",
              "Where do you come from?", "Who is your creator?",
              "Which team made you?", "Who owns you?", "Who designed you?",
              "What company are you from?"],
    "other": ["Are you {other}?", "Is this {other}?", "You're {other}, right?",
              "Were you made by the same people as {other}?",
              "Are you based on {other}?", "Am I chatting with {other}?",
              "Is this assistant {other}?"],
    "can": ["What can you do?", "How do you answer questions?",
            "Can you search the web?", "How do you work?",
            "What are you good at?", "What kind of assistant are you built to be?",
            "How do you find answers?"],
}
# Батарея who-*: формулировки, которых в обучении нет.
WHO = {
    "who-name": ["May I ask your name?", "How should I address you?",
                 "Identify yourself, please."],
    "who-maker": ["Which organization is behind you?", "What lab made this model?",
                  "Who is responsible for training you?"],
    "who-other": ["Am I talking to ChatGPT?", "Are you a version of Claude?",
                  "Is this Google's Gemini?"],
    "who-what": ["What kind of assistant are you?", "Explain what you are.",
                 "Are you an AI? Which one?"],
}
PREFIX = ["", "", "", "Hi! ", "Hey, ", "Quick question: ", "Hello. ", "So, "]


def _styled(q: str, rng: random.Random) -> str:
    q = rng.choice(PREFIX) + q
    if rng.random() < 0.2:
        q = q.lower()
    if rng.random() < 0.15:
        q = q.rstrip("?.")
    return q


def examples(n: int = 600, seed: int = 0) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        kind = rng.choice(list(TRAIN))
        q = rng.choice(TRAIN[kind])
        if kind == "intro":
            a = rng.choice(INTRO)
        elif kind == "maker":
            a = rng.choice(MAKER_ANS)
        elif kind == "can":
            a = rng.choice(CAN)
        else:
            other = rng.choice(OTHERS)
            q = q.format(other=other)
            a = rng.choice(NOT_OTHER).format(name=NAME, maker=MAKER, other=other)
        out.append((_styled(q, rng), a))
    return out


def encode(enc, q: str, a: str):
    import config
    sp = enc.special
    prompt = [sp["<|user|>"]] + enc._text(q.strip() + "\n") + [sp["<|assistant|>"]]
    reply = enc._text(a) + [config.EOT_ID]
    return prompt + reply, [0] * len(prompt) + [1] * len(reply)


class PersonaStream(PF.PrefStream):
    """Личность понемногу + всё прошлое, чтобы она не протекала и не сносила."""

    # 23.10: 3% строк при LR ×0.05 и 150 шагах — who-* 0/12. Вопрос о себе
    # без документов для модели после E/F — повод искать, и это надо
    # перебить: 15% строк, но в строке не больше PER_ROW примеров, чтобы
    # 1,500 шаблонов не повторялись по сто раз подряд.
    # 29.10: 15% при LR ×0.05 — «I'm Jake», имени нет. 05.11: 25% при LR ×0.1
    # — who-* 12/12, но личность всплыла в 11 из 257 обычных ответов и даже
    # вместо ответа на вопрос HotpotQA. 12% при том же LR.
    SHARES = {"persona": 0.12, "pref": 0.18, "calib": 0.15, "wiki": 0.25,
              "d1": 0.1, "pretrain": 0.2}
    PER_ROW = 24

    def __init__(self, seq_len: int, seed: int = 0, limit: int | None = None):
        super().__init__(seq_len, seed=seed, limit=limit)
        from tokenizers import Tokenizer

        import config
        from data import search_sft as SS
        enc = SS.Encoder(Tokenizer.from_file(str(config.TOKENIZER_OUT / "synth1.json")))
        self.persona = [encode(enc, q, a) for q, a in examples(1500)]

    def describe(self) -> str:
        return f"смесь G: личность {len(self.persona)} примеров; " + super().describe()

    def state(self) -> dict:
        return {**super().state(), "sft": "g"}

    def restore(self, state: dict) -> None:
        if state.get("sft") != "g":
            raise SystemExit("снимок не от этапа G — нужен --new-stage")
        super().restore({**state, "sft": "f"})

    def _persona_rows(self):
        import config
        width = self.seq_len + 1
        rng = random.Random(self.seed ^ 0x6)
        ids, mask = [], []
        n = 0
        while True:
            e_ids, e_mask = rng.choice(self.persona)
            if len(ids) + len(e_ids) > width or n >= self.PER_ROW:
                pad = width - len(ids)
                yield {"tokens": ids + [config.PAD_ID] * pad, "loss_mask": mask + [0] * pad}
                ids, mask, n = [], [], 0
            ids += e_ids
            mask += e_mask
            n += 1

    def sources(self) -> dict:
        return {**super().sources(), "persona": self._persona_rows()}
