"""Поиск: траектории, собранные программно из открытых датасетов вопросов.

Учителя нет. Траектории OpenResearcher (data/search_sft.py) писала чужая модель
в чужой среде, и наша модель копировала её повадки — открывала `example.com`,
рассуждала о «default source» (STATUS.md, 05.11.2026). Здесь каждая траектория
собрана из разметки человеческого датасета и проверена в той же среде, где
идёт замер (data/search_env.py: BM25, пять результатов, окно страницы).

Источники — все с коммерческой лицензией (LICENSES.md):

    hotpot    HotpotQA, CC-BY-SA 4.0. Два шага по двум статьям: поиск по
              названию статьи, мост (bridge) или сравнение (comparison).
    musique   MuSiQue, CC-BY 4.0. 2–4 шага. Запрос — подвопрос из разметки
              разложения, человеческий: «When was #1 founded?» с подставленным
              ответом прошлого шага → «When was Houston Baptist University
              founded?». Это и есть умение «переспросить».
    squad     SQuAD, CC-BY-SA 4.0. Один шаг: запрос — сам вопрос.

Корпус каждого источника — абзацы всех вопросов выборки (train), как у замера
(validation). Траектория выбрасывается, если поиск не нашёл нужный абзац в
пятёрке или окно страницы не показало ответ: учим только тому, что в этой
среде работает. 10% — без ответа: на последнем шаге нужный абзац убран из
выдачи; модель пробует ещё раз другими словами и, не найдя, говорит, что
нашлось близкого, и предлагает искать иначе — не тупик.

Ответ — сводка, как у поисковика: прямой ответ и факты со ссылками из
открытых страниц (sft.overview). 30% вопросов — ключевыми словами.

    <|think|>First, find which institute owned The Collegian.<|/think|>
    <|search|>The Collegian owned by<|/search|>
    <|result|>[1] The Collegian (Houston Baptist University) - en.wikipedia.org - …<|/result|>
    <|open|>1<|/open|><|page|>…<|/page|>
    <|think|>It is Houston Baptist University. Next: when was it founded?<|/think|>
    <|search|>When was Houston Baptist University founded?<|/search|> …
    1960 <|cite|>1<|/cite|><|cite|>6<|/cite|><|eot|>

    python data/wiki_search.py        # по 2,000 вопросов, сколько проходит
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
from data import search_env as ENV  # noqa: E402
from data import search_sft as SS  # noqa: E402
from data import sft as SFT  # noqa: E402

CACHE = pathlib.Path(config.ROOT) / "cache" / "search_train.json"
NEGATIVE = 0.10
# Сколько вопросов брать с источника: MuSiQue меньше всех (~20 тыс.) и ценнее
# всех — в нём настоящие подвопросы.
SIZES = {"hotpot": 30_000, "musique": 19_000, "squad": 20_000}
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _overlap(a: str, b: str) -> int:
    return len(set(ENV.words(a)) & set(ENV.words(b)))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _page(title: str, text_or_sents, key: str | None = None) -> dict:
    sents = (text_or_sents if isinstance(text_or_sents, list)
             else SENT.split(text_or_sents.strip()))
    url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
    return {"title": title, "url": url, "domain": "en.wikipedia.org",
            "lines": {i: s.strip() for i, s in enumerate(sents) if s.strip()},
            "snippets": [], "key": key or title}


# --- Корпуса ---------------------------------------------------------------

def pages_for(source: str, rows) -> dict[str, dict]:
    """Абзацы источника: ключ → страница. Ключ однозначен внутри корпуса."""
    pages: dict[str, dict] = {}
    if source == "hotpot":
        return ENV.wiki_pages(rows)
    if source == "musique":
        for r in rows:
            for p in r["paragraphs"]:
                key = f"{p['title']}::{_norm(p['paragraph_text'])[:60]}"
                pages.setdefault(key, _page(p["title"], p["paragraph_text"], key))
        return pages
    for r in rows:                                   # squad
        key = f"{r['title']}::{_norm(r['context'])[:60]}"
        pages.setdefault(key, _page(r["title"].replace("_", " "), r["context"], key))
    return pages


def make_env(source: str, rows) -> ENV.Env:
    env = ENV.Env(pages_for(source, rows))
    env.by_key = {k: i for i, k in enumerate(env.urls)}
    return env


# --- Траектории ------------------------------------------------------------

class Builder:
    """Сегменты протокола: нумерация источников сквозная, как в замере.

    Вопрос пользователя в 30% случаев — ключевыми словами, как его набирают в
    поисковике («jean loring comic publisher»); рассуждения и запросы при этом
    те же, ответ — тот же."""

    def __init__(self, env: ENV.Env, question: str, rng: random.Random):
        self.env, self.q, self.rng = env, question, rng
        shown = SFT.keywordize(question) if rng.random() < 0.3 else question
        self.segs: list[tuple[str, str]] = [("user", shown)]
        self.n_src = 0
        self.views: dict[int, str] = {}
        self.tried: list[str] = []

    def hop(self, think: str, query: str, key: str, need: str,
            negative: bool = False):
        """Поиск → открыть нужное. Номер источника, 0 — «не нашлось» или None —
        шаг в этой среде не проходит."""
        env = self.env
        idx = env.by_key.get(key)
        hits = env.search(query)
        if negative:
            hits = [i for i in hits if i != idx][:SS.TOP_RESULTS]
        elif idx not in hits:
            return None
        lines, num = [], 0
        for i in hits:
            self.n_src += 1
            lines.append(env.result_line(self.n_src, i, query))
            if i == idx:
                num = self.n_src
        self.segs += [("think", think), ("search", query),
                      ("result", "\n".join(lines) or "No results.")]
        self.tried.append(query)
        if negative:
            return 0
        view = env.view(idx, self.q + " " + query)
        if need and _norm(need)[:50] not in _norm(view):
            return None
        self.segs += [("open", str(num)), ("page", view)]
        self.views[num] = view
        return num

    def finish(self, think: str, answer: str, nums: list[int]):
        """Ответ — сводка: прямой ответ и факты из открытых страниц."""
        facts = [(SFT.best_sentence(self.views.get(n, ""), self.q + " " + answer,
                                    prefer=answer), n) for n in nums]
        self.segs += [("think", think),
                      ("answer", SFT.overview(answer, facts, self.rng))]
        return self.segs

    def give_up(self, what: str, key: str):
        """Не нашлось: ещё одна попытка другими словами, потом — не тупик, а
        что нашлось близкого и предложение искать иначе."""
        alt = SFT.keywordize(self.q)
        if alt.lower() not in (t.lower() for t in self.tried):
            self.hop(self.rng.choice([f"Nothing about {what}. Let me try other words.",
                                      "Not there. Different keywords.",
                                      "No luck. Rephrasing the search."]),
                     alt, key, "", negative=True)
        facts = [(SFT.best_sentence(v, self.q), n) for n, v in self.views.items()]
        self.segs += [("think", self.rng.choice(["Still not there.", "Can't find it."])),
                      ("answer", SFT.fallback(facts[-1:], self.rng, self.tried))]
        return self.segs


def hotpot(row, env, rng):
    gold = list(dict.fromkeys(row["supporting_facts"]["title"]))
    if len(gold) != 2:
        return None
    q = row["question"].strip()
    ctx = dict(zip(row["context"]["title"], row["context"]["sentences"]))
    need = {}
    for t, sid in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
        if t in ctx and sid < len(ctx[t]):
            need.setdefault(t, ctx[t][sid].strip())
    comparison = row.get("type") == "comparison"
    order = gold if comparison else sorted(gold, key=lambda t: -_overlap(t, q))
    b = Builder(env, q, rng)
    negative = rng.random() < NEGATIVE
    nums = []
    for hop, title in enumerate(order):
        if hop == 0:
            think = rng.choice([
                f"I need to compare {order[0]} and {order[1]}. Let me look up {order[0]}.",
                f"Two things to check: {order[0]} and {order[1]}. Starting with {order[0]}.",
            ]) if comparison else rng.choice([
                f"Let me look up {title} first.", f"Start with {title}.",
                f"First I need {title}."])
        else:
            think = (rng.choice([f"Now {title}.", f"Next, {title}."]) if comparison
                     else rng.choice([f"The page mentions {title}. Let me search it.",
                                      f"That points to {title}. Searching for it."]))
        num = b.hop(think, title, title, need.get(title, ""),
                    negative=negative and hop == 1)
        if num is None:
            return None
        if num == 0:
            return b.give_up(title, title)
        nums.append(num)
    return b.finish(rng.choice(["Both pages together answer it.",
                                "Got both pieces.", "That settles it."]),
                    row["answer"], nums)


def _subquery(text: str, answers: list[str]) -> str:
    for k, a in enumerate(answers, 1):
        text = text.replace(f"#{k}", a)
    text = text.replace(" >> ", " ").replace(">>", " ")
    return re.sub(r"\s+", " ", text).strip()


def musique(row, env, rng):
    steps = row["question_decomposition"]
    if not (2 <= len(steps) <= 4) or not row.get("answerable", True):
        return None
    paras = row["paragraphs"]
    b = Builder(env, row["question"].strip(), rng)
    negative = rng.random() < NEGATIVE
    answers, nums = [], []
    for k, st in enumerate(steps):
        p = paras[st["paragraph_support_idx"]]
        key = f"{p['title']}::{_norm(p['paragraph_text'])[:60]}"
        query = _subquery(st["question"], answers)
        if k == 0:
            think = rng.choice([f"Break it down. First: {query}",
                                f"Step one: {query}", f"Let me start with: {query}"])
        else:
            think = rng.choice([f"It's {answers[-1]}. Next: {query}",
                                f"So, {answers[-1]}. Now: {query}",
                                f"Got {answers[-1]}. Then: {query}"])
        last = k == len(steps) - 1
        num = b.hop(think, query, key, st["answer"], negative=negative and last)
        if num is None:
            return None
        if num == 0:
            return b.give_up(query.rstrip("?"), key)
        answers.append(st["answer"])
        nums.append(num)
    return b.finish(rng.choice(["That's the answer.", "Done — that answers it.",
                                "Found it."]),
                    row["answer"], nums)


def squad(row, env, rng):
    answers = [a for a in row["answers"]["text"] if a.strip()]
    if not answers:
        return None
    q = row["question"].strip()
    key = f"{row['title']}::{_norm(row['context'])[:60]}"
    b = Builder(env, q, rng)
    negative = rng.random() < NEGATIVE
    num = b.hop(rng.choice(["Let me search for this.", "Searching.", "Let me look it up."]),
                q, key, answers[0], negative=negative)
    if num is None:
        return None
    if num == 0:
        return b.give_up(q.rstrip("?"), key)
    return b.finish(rng.choice(["Found it.", "Here it is.", "The page says it."]),
                    answers[0], [num])


MAKERS = {"hotpot": hotpot, "musique": musique, "squad": squad}


def rows_for(source: str, split: str):
    rows = SFT.load_rows(source, split)
    if source == "squad":
        rows = [r for r in rows if any(a.strip() for a in r["answers"]["text"])]
    return rows


def trajectories(source: str, split: str, n: int, seed: int = 0):
    """[сегменты] для n вопросов источника; корпус — абзацы этих же вопросов."""
    rows = rows_for(source, split)
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    env = make_env(source, rows)
    rng = random.Random(seed + 1)
    out = []
    for r in rows:
        segs = MAKERS[source](r, env, rng)
        if segs is not None:
            out.append(segs)
    return out, len(rows), env.n


def build(sizes: dict | None = None, seed: int = 0, split: str = "train",
          cache: pathlib.Path = CACHE):
    """[(ids, mask)] со всех источников. С кэшем."""
    if cache.exists():
        data = json.loads(cache.read_text())
        print(f"  Поиск: из кэша {cache} — {len(data):,} траекторий", flush=True)
        return data
    from tokenizers import Tokenizer
    enc = SS.Encoder(Tokenizer.from_file(str(config.TOKENIZER_OUT / "synth1.json")))
    data = []
    for source, n in (sizes or SIZES).items():
        got, asked, n_pages = trajectories(source, split, n, seed)
        kept = neg = 0
        for segs in got:
            fit = SS.fit(enc, segs, SS.BUDGET)
            if fit is None:
                continue
            kept += 1
            neg += segs[-1][1].startswith("I couldn't find")
            data.append([fit[0], fit[1]])
        print(f"  Поиск {source}: {asked:,} вопросов, {n_pages:,} абзацев, "
              f"траекторий {kept:,} (без ответа {neg:,})", flush=True)
    random.Random(seed).shuffle(data)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def holdout(seq_len: int, batch_size: int, n_batches: int, limit: int | None = None):
    """Отложенный набор: траектории с валидационных частей, тот же формат."""
    cache = CACHE.with_name("search_valid.json")
    data = build({"hotpot": 300, "musique": 300, "squad": 300}, seed=777,
                 split="valid", cache=cache)
    stream = SearchMixStream.__new__(SearchMixStream)
    stream.seq_len, stream.seed, stream.docs = seq_len, 777, 0
    stream.wiki, stream.w_epoch, stream.w_index = data, 0, 0
    it = stream._wiki_rows()
    got = [[next(it) for _ in range(batch_size)] for _ in range(n_batches)]
    return (np.asarray([[r[0] for r in b] for b in got], dtype=np.int32),
            np.asarray([[r[1] for r in b] for b in got], dtype=np.float32))


class SearchMixStream:
    """Этап D2 без учителя: поиск (HotpotQA, MuSiQue, SQuAD), D1 и претрейн."""

    SHARES = {"wiki": 0.5, "d1": 0.3, "pretrain": 0.2}

    def __init__(self, seq_len: int, seed: int = 0, split: str = "train",
                 limit: int | None = None):
        from synth1 import data as D
        self.seq_len, self.seed = seq_len, seed
        self.wiki = build({k: 600 for k in SIZES} if limit else None)
        self.d1 = SFT.SFTStream(seq_len, split=split, seed=seed, limit=limit,
                                repeat={"squad": 2})
        self.text = D.TokenStream("anneal", seq_len=seq_len, seed=seed,
                                  split="train" if split == "train" else "valid")
        self.rng = random.Random(seed ^ 0x5EA)
        self.w_epoch = self.w_index = 0
        self.tokens_emitted = 0
        self.docs = 0

    def describe(self) -> str:
        return (f"смесь поиска: {len(self.wiki):,} траекторий (HotpotQA, MuSiQue, "
                f"SQuAD), D1 и претрейн; доли {self.SHARES}; длина {self.seq_len}")

    def state(self) -> dict:
        return {"sft": "d2w", "w_epoch": self.w_epoch, "w_index": self.w_index,
                "seed": self.seed, "seq_len": self.seq_len, "rng": self.rng.random(),
                "tokens_emitted": self.tokens_emitted, "docs": self.docs}

    def restore(self, state: dict) -> None:
        if state.get("sft") != "d2w":
            raise SystemExit("снимок не от этапа поиска — нужен --new-stage")
        self.w_epoch, self.w_index = state["w_epoch"], state["w_index"]
        self.tokens_emitted = state.get("tokens_emitted", 0)
        self.rng = random.Random(state.get("rng", 0))

    def carry_over(self, state: dict) -> list[str]:
        self.tokens_emitted = state.get("tokens_emitted", 0)
        return []

    def _wiki_rows(self):
        width = self.seq_len + 1
        ids: list[int] = []
        mask: list[int] = []
        fits = False
        while True:
            order = list(range(len(self.wiki)))
            random.Random(self.seed * 53 + self.w_epoch).shuffle(order)
            while self.w_index < len(order):
                e_ids, e_mask = self.wiki[order[self.w_index]]
                self.w_index += 1
                if len(e_ids) > width:
                    continue
                fits = True
                if len(ids) + len(e_ids) > width:
                    pad = width - len(ids)
                    yield ids + [config.PAD_ID] * pad, mask + [0] * pad
                    ids, mask = [], []
                ids += e_ids
                mask += e_mask
                self.docs += 1
            if not fits:
                raise SystemExit(f"ни одна траектория не влезает в строку {width} — "
                                 f"нужна длина больше")
            self.w_epoch += 1
            self.w_index = 0

    def batches(self, batch_size: int):
        gens = {"wiki": self._wiki_rows(), "d1": self.d1.rows_packed()}
        text = self.text.batches(1)
        names = [n for n in self.SHARES if self.SHARES[n] > 0]
        weights = [self.SHARES[n] for n in names]
        while True:
            rows, masks = [], []
            for _ in range(batch_size):
                pick = self.rng.choices(names, weights)[0]
                if pick == "pretrain":
                    r = list(next(text)["tokens"][0])
                    m = [1] * len(r)
                else:
                    r, m = next(gens[pick])
                rows.append(r)
                masks.append(m)
            tokens = np.asarray(rows, dtype=np.int32)
            self.tokens_emitted += tokens.size
            yield {"tokens": tokens, "loss_mask": np.asarray(masks, dtype=np.float32)}


if __name__ == "__main__":
    for source in SIZES:
        got, asked, n_pages = trajectories(source, "train", 2000)
        print(f"{source}: прошли {len(got)} из {asked}, абзацев {n_pages:,}")
        for k, t in got[1]:
            print(f"    {k:7} {t[:110]!r}")
