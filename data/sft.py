"""Этап D1: ответ по источникам и честный отказ.

Вход — вопрос и пронумерованные документы, выход — короткий ответ со ссылками
на документы или помощь вместо тупика, если ответа в них нет. Формат —
протокол поиска из SEARCH.md (раздел 6): документы приходят в том же блоке
`<|result|>`, что и выдача поисковика на этапе D2, поэтому D2 продолжает D1,
а не переучивает его.

    <|user|>When was Canberra named?
    <|result|>[1] History of Canberra
    The city was officially named on 12 March 1913 ...
    [2] Australian Capital Territory
    ...<|/result|>
    <|assistant|>It's **12 March 1913** <|cite|>1<|/cite|>

    The city was officially named on 12 March 1913 … <|cite|>1<|/cite|><|eot|>

Подача ответа (прямой ответ + факты из документов, или «I couldn't find…» с
ближайшим найденным и предложением искать иначе) — `present`, 11.11.2026.

Лосс — только на том, что пишет модель: от `<|assistant|>` (не включая) до
`<|eot|>` (включая). Вопрос и документы маскируются.

Источники (SEARCH.md, раздел 5; лицензии — LICENSES.md):

    squad      rajpurkar/squad_v2        CC-BY-SA-4.0   один абзац + отвлекающие,
                                                         треть вопросов без ответа
    hotpot     hotpotqa/hotpot_qa        CC-BY-SA-4.0   десять абзацев, ответ
               (distractor)                              из двух
    musique    dgslibisey/MuSiQue        CC-BY-4.0      двадцать абзацев,
                                                         2–4 шага

Natural Questions в первую версию не вошёл: в датасете сырой HTML, 287
шардов, и вытащить из него короткий абзац с ответом — отдельная работа.

Порча контекста: у части вопросов HotpotQA и MuSiQue выбрасываются все
абзацы с ответом, и правильный ответ становится отказом. Без этого модель
учится, что ответ в документах есть всегда, и отказывать не умеет.

    python data/sft.py show          # пять примеров и счёт токенов
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# Отказ начинается всегда одинаково — «I couldn't find» — по этим словам его
# узнают замеры (evals/d1.py, data/calib.py). Но дальше он не тупик: что
# нашлось близкого и что можно сделать. Голое «The sources don't say.»
# пользователь видит как сломанный поиск и уходит (11.11.2026).
REFUSAL = "I couldn't find that in the sources."

FILES = {
    "squad": ("rajpurkar/squad_v2", {
        "train": ["squad_v2/train-00000-of-00001.parquet"],
        "valid": ["squad_v2/validation-00000-of-00001.parquet"]}),
    "hotpot": ("hotpotqa/hotpot_qa", {
        "train": ["distractor/train-00000-of-00002.parquet",
                  "distractor/train-00001-of-00002.parquet"],
        "valid": ["distractor/validation-00000-of-00001.parquet"]}),
    "musique": ("dgslibisey/MuSiQue", {
        "train": ["musique_ans_v1.0_train.jsonl"],
        "valid": ["musique_ans_v1.0_dev.jsonl"]}),
}

# Доля вопросов HotpotQA и MuSiQue, у которых выбрасываются абзацы с ответом.
CORRUPT = 0.15
# Сколько чужих абзацев подмешивать к абзацу SQuAD: модель должна выбрать
# нужный, а не отвечать по единственному.
SQUAD_DISTRACTORS = (0, 3)


def _download(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download
    token = config.key("HF_TOKEN", required=False) or None
    return hf_hub_download(repo, path, repo_type="dataset", token=token)


def load_rows(source: str, split: str) -> list[dict]:
    repo, files = FILES[source]
    rows: list[dict] = []
    for path in files[split]:
        local = _download(repo, path)
        if local.endswith(".parquet"):
            import pyarrow.parquet as pq
            rows.extend(pq.read_table(local).to_pylist())
        else:
            with open(local, encoding="utf-8") as fh:
                rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


# --- Примеры: (документы, вопрос, ответ) ------------------------------------

def _cites(numbers) -> str:
    return "".join(f"<|cite|>{n}<|/cite|>" for n in numbers)


def _answer(text: str, numbers) -> str:
    text = text.strip()
    if text.lower() in ("yes", "no"):
        text = text.capitalize()
    return f"{text} {_cites(numbers)}"


# --- Подача ответа: как у поисковика, а не обрывок ---------------------------
#
# Пользователь вводит запрос и ждёт сводку, как у Google: сначала прямой
# ответ, потом пара фактов со ссылками, откуда это. Факты — предложения из
# тех самых документов (выбираются по пересечению с вопросом и ответом), то
# есть подача не добавляет ничего, чего нет в источниках.

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of in on at to for and or is are was were be by with from as "
            "that this which who whom whose what when where how why did does do its it "
            "into than then there their they he she his her has have had been not".split())
CITE_RE = re.compile(r"<\|cite\|>\s*\d+\s*<\|/cite\|>")


def _content(s: str) -> set[str]:
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP}


def best_sentence(text: str, query: str, limit: int = 220, prefer: str = "") -> str:
    """Предложение текста, больше всего пересекающееся с запросом; если задан
    prefer (ответ) — сначала те, где он есть: факт должен подтверждать ответ."""
    q = _content(query)
    sents = [x.strip() for x in _SENT.split(text.strip()) if x.strip()]
    if not sents:
        return ""
    want = prefer.strip().lower()
    best = max(sents, key=lambda x: (bool(want) and want in x.lower(),
                                     len(q & _content(x)), -len(x)))
    if len(best) > limit:
        cut = best[:limit]
        best = cut[:cut.rfind(" ")] + "…" if " " in cut else cut
    return best


def keywordize(question: str, rng: random.Random | None = None) -> str:
    """Вопрос → запрос, как его набирают в поисковике: без служебных слов."""
    words = [w for w in re.findall(r"[\w'’.-]+", question.lower())
             if w not in _STOP and w not in ("much", "many", "name", "what's")]
    return " ".join(words[:10]) or question


def lead(text: str) -> str:
    """Первая строка ответа — сам ответ; по ней и судят, верно ли."""
    return text.strip().split("\n")[0][:300]


def overview(answer: str, facts, rng: random.Random) -> str:
    """Прямой ответ + до двух фактов со ссылками. facts — [(предложение, N)]."""
    answer = answer.strip()
    if answer.lower() in ("yes", "no"):
        answer = answer.capitalize()
    nums = list(dict.fromkeys(n for _, n in facts))
    facts = [(f, n) for f, n in facts if f and _content(f) - _content(answer)][:2]
    cites = _cites(nums)
    head = rng.choice([
        f"**{answer}** {cites}", f"It's **{answer}** {cites}",
        f"Short answer: **{answer}** {cites}", f"**{answer}**! {rng.choice(['✨', '🙂', '💡', ''])} {cites}",
        f"Looks like it's **{answer}** {cites}"]).replace("  ", " ")
    if not facts or rng.random() < 0.15:
        return head
    if rng.random() < 0.5:
        body = "\n".join(f"- {f} {_cites([n])}" for f, n in facts)
        return f"{head}\n\n{body}"
    body = " ".join(f"{f} {_cites([n])}" for f, n in facts)
    return f"{head}\n\n{body}"


def fallback(facts, rng: random.Random, tried: list[str] | None = None) -> str:
    """Не нашлось — но не тупик: что искала, что близкого нашлось, что дальше."""
    head = rng.choice([
        "I couldn't find a clear answer to that in the sources 🤔",
        "I couldn't find that exactly, sorry 😕",
        "I couldn't find a direct answer here 🤔",
        "I couldn't find it in what I searched 🙁"])
    parts = [head]
    if tried:
        parts.append(f"I tried {' and '.join(repr(t) for t in tried[-2:])}.")
    facts = [(f, n) for f, n in facts if f][:1]
    if facts:
        f, n = facts[0]
        parts.append(rng.choice(["The closest thing I found:", "Here's what I did find:",
                                 "What I did find:"]) + f" {f} {_cites([n])}")
    parts.append(rng.choice([
        "Want me to try different keywords?", "I can search again with other words if you like.",
        "Maybe try rephrasing it? 🙂", "If you tell me a bit more, I'll dig further 🔎", ""]))
    return " ".join(p for p in parts if p)


def present(docs, question: str, ref: str, ok: bool, rng: random.Random) -> str:
    """Эталон D1 («ответ <|cite|>N<|/cite|>» или REFUSAL) → подача для обучения."""
    if ok:
        nums = [int(x) for x in re.findall(r"<\|cite\|>\s*(\d+)", ref)]
        answer = CITE_RE.sub("", ref).strip()
        facts = [(best_sentence(docs[n - 1][1], question + " " + answer, prefer=answer), n)
                 for n in nums if 0 < n <= len(docs)]
        return overview(answer, facts or [("", n) for n in nums], rng)
    if not docs:
        return fallback([], rng)
    k = max(range(len(docs)), key=lambda i: len(_content(question) & _content(docs[i][1])))
    return fallback([(best_sentence(docs[k][1], question), k + 1)], rng)


def squad_example(row: dict, pool: list[tuple[str, str]], rng: random.Random):
    gold = (row["title"].replace("_", " "), row["context"])
    others = [p for p in rng.sample(pool, k=min(len(pool), 8)) if p[0] != gold[0]]
    docs = [gold] + others[:rng.randint(*SQUAD_DISTRACTORS)]
    rng.shuffle(docs)
    answers = [a for a in row["answers"]["text"] if a.strip()]
    if not answers:
        return docs, row["question"], REFUSAL, False
    return docs, row["question"], _answer(answers[0], [docs.index(gold) + 1]), True


def hotpot_example(row: dict, rng: random.Random):
    ctx = row["context"]
    docs = [(t, "".join(s).strip()) for t, s in zip(ctx["title"], ctx["sentences"])]
    support = list(dict.fromkeys(row["supporting_facts"]["title"]))
    if rng.random() < CORRUPT:
        docs = [d for d in docs if d[0] not in support]
        rng.shuffle(docs)
        return docs, row["question"], REFUSAL, False
    rng.shuffle(docs)
    titles = [d[0] for d in docs]
    nums = sorted(titles.index(t) + 1 for t in support if t in titles)
    return docs, row["question"], _answer(row["answer"], nums), True


def musique_example(row: dict, rng: random.Random):
    paras = [(p["title"], p["paragraph_text"].strip(), p["is_supporting"])
             for p in row["paragraphs"]]
    if rng.random() < CORRUPT:
        docs = [(t, x) for t, x, s in paras if not s]
        rng.shuffle(docs)
        return docs, row["question"], REFUSAL, False
    rng.shuffle(paras)
    nums = [i + 1 for i, p in enumerate(paras) if p[2]]
    docs = [(t, x) for t, x, _ in paras]
    return docs, row["question"], _answer(row["answer"], nums), True


def prompt_text(docs, question: str) -> tuple[str, str, str]:
    """Три куска текста между служебными токенами: вопрос, документы."""
    body = "\n".join(f"[{i}] {t}\n{x}" for i, (t, x) in enumerate(docs, 1))
    return question.strip() + "\n", body, "\n"


# --- Токенизация и упаковка -------------------------------------------------

class Encoder:
    """Пример → (ids, маска лосса). Служебные токены вставляются по id."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.special = {t: tokenizer.token_to_id(t) for t in config.SPECIAL_TOKENS
                        if not t.startswith("<|reserved")}
        missing = [t for t, i in self.special.items() if i is None]
        if missing:
            raise SystemExit(f"в словаре нет служебных токенов: {missing}")

    def _text(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False).ids if s else []

    def _answer(self, s: str) -> list[int]:
        """Ответ с <|cite|>N<|/cite|>: служебные — по id, остальное — словарём."""
        out: list[int] = []
        rest = s
        while rest:
            start = rest.find("<|cite|>")
            if start < 0:
                out += self._text(rest)
                break
            out += self._text(rest[:start])
            end = rest.index("<|/cite|>", start)
            out += [self.special["<|cite|>"]]
            out += self._text(rest[start + len("<|cite|>"):end])
            out += [self.special["<|/cite|>"]]
            rest = rest[end + len("<|/cite|>"):]
        return out

    def encode(self, docs, question: str, answer: str):
        q, body, tail = prompt_text(docs, question)
        sp = self.special
        prompt = ([sp["<|user|>"]] + self._text(q) + [sp["<|result|>"]]
                  + self._text(body) + [sp["<|/result|>"]] + self._text(tail)
                  + [sp["<|assistant|>"]])
        reply = self._answer(answer) + [config.EOT_ID]
        ids = prompt + reply
        mask = [0] * len(prompt) + [1] * len(reply)
        return ids, mask


class SFTStream:
    """Бесконечный поток упакованных примеров D1: tokens и loss_mask [B, T+1].

    Пример целиком в одной строке: если не влезает в остаток, строка добивается
    `<|pad|>` (маска 0) и пример начинает следующую. Примеры длиннее строки
    выбрасываются. Эпоха — перестановка всех примеров с seed+эпоха; позиция
    (эпоха, номер) лежит в снимке.
    """

    def __init__(self, seq_len: int, split: str = "train", seed: int = 0,
                 tokenizer=None, sources=("squad", "hotpot", "musique"),
                 limit: int | None = None, repeat: dict | None = None):
        if tokenizer is None:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(config.TOKENIZER_OUT / "synth1.json"))
        self.enc = Encoder(tokenizer)
        self.seq_len = seq_len
        self.seed = seed
        self.split = split
        self.rows: list[tuple[str, dict]] = []
        for src in sources:
            got = load_rows(src, split)
            if limit:
                got = got[:limit]
            # repeat: сколько раз брать источник за эпоху. D2 удваивает
            # SQuAD — единственный источник, где отказ трудный (STATUS.md).
            self.rows += [(src, r) for r in got] * (repeat or {}).get(src, 1)
            print(f"  D1 {split}: {src} — {len(got):,} вопросов", flush=True)
        self.pool = [(r["title"].replace("_", " "), r["context"])
                     for s, r in self.rows if s == "squad"]
        # SQuAD повторяет один абзац на пять вопросов; отвлекающие берутся
        # из уникальных абзацев.
        self.pool = list(dict.fromkeys(self.pool))
        self.epoch = 0
        self.index = 0
        self.tokens_emitted = 0
        self.docs = 0
        self.dropped = 0

    def describe(self) -> str:
        counts: dict[str, int] = {}
        for s, _ in self.rows:
            counts[s] = counts.get(s, 0) + 1
        parts = ", ".join(f"{k} {v:,}" for k, v in counts.items())
        return f"смесь D1 ({self.split}): {parts}; длина {self.seq_len}"

    def state(self) -> dict:
        return {"sft": "d1", "epoch": self.epoch, "index": self.index,
                "seed": self.seed, "seq_len": self.seq_len,
                "tokens_emitted": self.tokens_emitted, "docs": self.docs}

    def restore(self, state: dict) -> None:
        if state.get("sft") != "d1":
            raise SystemExit("снимок не от этапа D1 — нужен --new-stage")
        self.epoch, self.index = state["epoch"], state["index"]
        self.tokens_emitted = state.get("tokens_emitted", 0)
        self.docs = state.get("docs", 0)

    def carry_over(self, state: dict) -> list[str]:
        """Новый этап: позиция в корпусе претрейна к D1 отношения не имеет."""
        self.tokens_emitted = state.get("tokens_emitted", 0)
        return []

    def example(self, i: int, epoch: int):
        rng = random.Random((self.seed * 1_000_003 + epoch) * 10_000_019 + i)
        src, row = self.rows[i]
        if src == "squad":
            return squad_example(row, self.pool, rng)
        if src == "hotpot":
            return hotpot_example(row, rng)
        return musique_example(row, rng)

    def _order(self, epoch: int) -> list[int]:
        order = list(range(len(self.rows)))
        random.Random(self.seed * 7919 + epoch).shuffle(order)
        return order

    def examples(self):
        """(ids, mask) по порядку эпохи, начиная с сохранённой позиции."""
        while True:
            order = self._order(self.epoch)
            while self.index < len(order):
                i = order[self.index]
                self.index += 1
                docs, q, a, ok = self.example(i, self.epoch)
                rng = random.Random((self.seed * 31 + self.epoch) * 1_000_003 + i)
                # Каждый пятый вопрос — как набирают в поисковике.
                if rng.random() < 0.2:
                    q = keywordize(q)
                yield self.enc.encode(docs, q, present(docs, q, a, ok, rng))
            self.epoch += 1
            self.index = 0
            print(f"  D1: эпоха {self.epoch} начата", flush=True)

    def rows_packed(self):
        width = self.seq_len + 1
        ids: list[int] = []
        mask: list[int] = []
        for e_ids, e_mask in self.examples():
            if len(e_ids) > width:
                self.dropped += 1
                continue
            if len(ids) + len(e_ids) > width:
                pad = width - len(ids)
                yield ids + [config.PAD_ID] * pad, mask + [0] * pad
                ids, mask = [], []
            ids += e_ids
            mask += e_mask
            self.docs += 1

    def batches(self, batch_size: int):
        rows = self.rows_packed()
        while True:
            chunk = [next(rows) for _ in range(batch_size)]
            tokens = np.asarray([c[0] for c in chunk], dtype=np.int32)
            mask = np.asarray([c[1] for c in chunk], dtype=np.float32)
            self.tokens_emitted += tokens.size
            yield {"tokens": tokens, "loss_mask": mask}


def holdout(seq_len: int, batch_size: int, n_batches: int, tokenizer=None,
            limit: int | None = None):
    """Отложенный набор D1: валидационные части, тот же формат. Без перемешивания
    по эпохам — первые n_batches батчей фиксированного порядка."""
    stream = SFTStream(seq_len, split="valid", seed=777, tokenizer=tokenizer,
                       limit=limit)
    it = stream.batches(batch_size)
    got = [next(it) for _ in range(n_batches)]
    return (np.stack([g["tokens"] for g in got]),
            np.stack([g["loss_mask"] for g in got]))


if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "valid"
    s = SFTStream(8192, split=split)
    print(s.describe())
    for i in range(5):
        docs, q, a, ok = s.example(i * 997 % len(s.rows), 0)
        print("-" * 60)
        print(f"Q: {q}\n{len(docs)} документов; ответ: {a!r}")
    lens = []
    for k, (ids, m) in enumerate(s.examples()):
        lens.append((len(ids), sum(m)))
        if k >= 2000:
            break
    n = np.asarray(lens)
    print(f"\n2000 примеров: длина медиана {np.median(n[:, 0]):.0f}, "
          f"95% {np.percentile(n[:, 0], 95):.0f}, макс {n[:, 0].max()}; "
          f"токенов ответа в среднем {n[:, 1].mean():.1f}")
