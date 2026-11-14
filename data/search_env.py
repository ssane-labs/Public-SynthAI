"""Офлайн-поисковик для замера D2: мини-корпус из того, что видел учитель.

Полный корпус OpenResearcher — 14.9M страниц (SEARCH.md, раздел 5), в сессию
Kaggle он не помещается. Для замера хватает меньшего: на каждый вопрос
учитель сделал 16 прогонов и в них нашёл и открыл десятки страниц. Всё это
вместе — по всем отложенным вопросам сразу — и есть корпус: на свой вопрос
модель найдёт нужное, если спросит правильно, а страницы чужих вопросов —
отвлекающие.

    страница      всё, что учитель видел по адресу: склеенные окна строк из
                  browser.open (L0, L1, …) плюс сниппеты из выдачи
    поиск         BM25 по заголовку и тексту, пять лучших, сниппет — лучшее
                  окно в 200 знаков
    открыть       строки страницы, больше всего пересекающиеся с вопросом и
                  последним запросом, с соседями, не длиннее 900 знаков —
                  то же окно, что в обучении (search_sft.page_text), только
                  выбранное без подсказки, куда смотрел ответ
    найти         строки текущей страницы со строкой поиска

Оговорка: корпус легче настоящего — нужная страница в нём точно есть. Замер
отвечает на вопрос «умеет ли модель пользоваться поиском», а не «найдёт ли в
интернете».
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import sys
import zlib
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data import search_sft as SS  # noqa: E402

WORD = re.compile(r"[a-z0-9]+")
STOP = set("a an the of in on at to for and or is are was were be by with from as "
           "that this which who what when where how why did does do its it".split())
HEAD_RE = re.compile(r"^\[\d+\]\s*(.*?)\s*\((\S+)\)\s*$")
DOC_RE = re.compile(r"【\d+†(Doc \d+)?†?(.*?)】")


def words(s: str) -> list[str]:
    return [w for w in WORD.findall(s.lower()) if w not in STOP]


def _scan_file(args):
    """Из одного файла траекторий: страницы и вопросы для нужных qid."""
    path, want = args
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    pages: dict[str, dict] = {}
    questions: dict[int, dict] = {}
    for g in range(pf.num_row_groups):
        qids = pf.read_row_group(g, columns=["qid"]).column(0).to_pylist()
        if not any(int(q) in want for q in qids):
            continue
        rows = pf.read_row_group(
            g, columns=["qid", "question", "answer", "messages"]).to_pylist()
        for row in rows:
            qid = int(row["qid"])
            if qid not in want:
                continue
            q = re.split(r"\n\s*You should follow the format", row["question"])[0].strip()
            questions.setdefault(qid, {"question": q, "answer": row.get("answer") or ""})
            pending = None
            for m in row["messages"]:
                to = m.get("recipient") or ""
                if m.get("role") == "assistant" and to.startswith("browser."):
                    pending = to
                elif m.get("role") == "tool" and pending:
                    body = SS._text(m)
                    if pending == "browser.open" and not body.startswith("Error"):
                        _add_page(pages, body)
                    elif pending == "browser.search":
                        _add_results(pages, body)
                    pending = None
    return pages, questions


def _add_page(pages: dict, body: str) -> None:
    first = body.split("\n", 1)[0]
    hit = HEAD_RE.match(first)
    if not hit:
        return
    # Ключ — номер документа корпуса («Doc 123»): под ним же страница лежит
    # в выдаче, и сниппет с полным текстом сливаются в одну страницу.
    key = hit.group(1) if hit.group(1).startswith("Doc ") else hit.group(2)
    lines = SS._lines(body)
    page = pages.setdefault(key, _empty())
    page["url"] = page.get("url") or hit.group(2)
    for k, v in lines.items():
        if not v.startswith("URL:"):
            page["lines"].setdefault(k, v)
    if not page["title"]:
        head = next((v for k, v in sorted(lines.items())
                     if v.strip() and not v.startswith("URL:")), "")
        page["title"] = head.strip()[:80]


def _empty() -> dict:
    return {"title": "", "url": "", "domain": "", "lines": {}, "snippets": []}


def _add_results(pages: dict, body: str) -> None:
    lines = SS._lines(body)
    text = " ".join(lines[k] for k in sorted(lines))
    for part in re.split(r"\*\s*(?=【\d+†)", text):
        hit = SS.ITEM_RE.match(part.strip())
        if not hit:
            continue
        label, domain = hit.group(2), hit.group(3)
        snippet = re.sub(r"\s+", " ", part.strip()[hit.end():].strip())[:SS.SNIPPET]
        key = label if label.startswith("Doc ") else \
            f"snippet://{domain}/{zlib.crc32(snippet[:80].encode())}"
        page = pages.setdefault(key, _empty())
        page["domain"] = page["domain"] or domain
        if not label.startswith("Doc ") and not page["title"]:
            page["title"] = label[:80]
        if snippet and snippet not in page["snippets"]:
            page["snippets"].append(snippet)


class Env:
    """Мини-корпус + BM25. Один на все вопросы замера."""

    def __init__(self, pages: dict[str, dict]):
        self.urls = list(pages)
        self.pages = [pages[u] for u in self.urls]
        self.by_url = {p.get("url") or u: i for i, (u, p) in
                       enumerate(zip(self.urls, self.pages))}
        self.texts = []
        for u, p in zip(self.urls, self.pages):
            body = " ".join(v for _, v in sorted(p["lines"].items()))
            body = body or " ".join(p["snippets"])
            self.texts.append(body)
        self.df = Counter()
        self.tf = []
        for p, t in zip(self.pages, self.texts):
            c = Counter(words(p["title"] + " " + t))
            self.tf.append(c)
            self.df.update(c.keys())
        self.n = len(self.texts)
        self.size = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.size) / max(self.n, 1)
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for w in c:
                self.post[w].append(i)

    def domain(self, i: int) -> str:
        p = self.pages[i]
        if p.get("domain"):
            return p["domain"]
        return (p.get("url") or self.urls[i]).split("://", 1)[-1].split("/", 1)[0]

    def search(self, query: str, top: int = SS.TOP_RESULTS) -> list[int]:
        q = words(query)
        scores: dict[int, float] = defaultdict(float)
        k1, b = 1.2, 0.75
        for w in set(q):
            idf = math.log(1 + (self.n - self.df[w] + 0.5) / (self.df[w] + 0.5))
            for i in self.post.get(w, ()):
                f = self.tf[i][w]
                scores[i] += idf * f * (k1 + 1) / (
                    f + k1 * (1 - b + b * self.size[i] / self.avg))
        return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:top]]

    def snippet(self, i: int, query: str) -> str:
        t = re.sub(r"\s+", " ", self.texts[i])
        q = set(words(query))
        best, at = -1, 0
        for start in range(0, max(1, len(t) - SS.SNIPPET + 1), 40):
            got = len(q & set(words(t[start:start + SS.SNIPPET])))
            if got > best:
                best, at = got, start
        return t[at:at + SS.SNIPPET]

    def result_line(self, num: int, i: int, query: str) -> str:
        title = self.pages[i]["title"] or self.snippet(i, query)[:60]
        return f"[{num}] {title} - {self.domain(i)} - {self.snippet(i, query)}"

    def view(self, i: int, focus: str) -> str:
        """Окно страницы под вопрос: строки с наибольшим пересечением + соседи."""
        lines = {k: v for k, v in self.pages[i]["lines"].items() if v.strip()}
        if not lines:
            return " ".join(self.pages[i]["snippets"])[:SS.PAGE_CHARS]
        q = set(words(focus))
        ranked = sorted(lines, key=lambda k: (-len(q & set(words(lines[k]))), k))
        keep = {k for k in ranked[:4] if q & set(words(lines[k]))}
        if not keep:
            return SS.page_text("\n".join(f"L{k}: {v}" for k, v in lines.items()), set())
        body = "\n".join(f"L{k}: {v}" for k, v in sorted(lines.items()))
        return SS.page_text(body, keep)

    def find(self, i: int, pattern: str) -> str:
        p = pattern.lower().strip()
        hits = [v for _, v in sorted(self.pages[i]["lines"].items()) if p and p in v.lower()]
        return " ".join(hits)[:400]


def build(qids: set[int], workers: int | None = None):
    """Страницы и вопросы по всем прогонам учителя для qids."""
    import os
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    from huggingface_hub import hf_hub_download
    token = config.key("HF_TOKEN", required=False) or None
    files = SS.list_files()
    with ThreadPoolExecutor(max_workers=8) as io:
        paths = list(io.map(lambda f: hf_hub_download(
            SS.REPO, f, repo_type="dataset", token=token), files))
    pages: dict[str, dict] = {}
    questions: dict[int, dict] = {}
    workers = workers or min(len(paths), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for p, q in pool.map(_scan_file, [(path, qids) for path in paths]):
            for url, page in p.items():
                have = pages.setdefault(url, _empty())
                for f in ("url", "domain"):
                    have[f] = have[f] or page[f]
                for k, v in page["lines"].items():
                    have["lines"].setdefault(k, v)
                have["snippets"] += [s for s in page["snippets"] if s not in have["snippets"]]
                have["title"] = have["title"] or page["title"]
            for k, v in q.items():
                questions.setdefault(k, v)
    return pages, questions


def held_out(cache: pathlib.Path | None = None) -> tuple[list[int], list[int]]:
    """(qid отложенных 2%, qid всех обученных) — по тому же разрезу, что MixStream."""
    import random
    cache = cache or pathlib.Path(config.ROOT) / "cache" / "d2_train.json"
    qids = json.loads(SS.qids_path(cache).read_text())
    order = list(range(len(qids)))
    random.Random(1234).shuffle(order)
    cut = max(1, len(order) // 50)
    return [qids[i] for i in order[:cut]], [qids[i] for i in order[cut:]]


def wiki_pages(rows) -> dict[str, dict]:
    """Абзацы Википедии из контекстов HotpotQA: заголовок → страница."""
    pages: dict[str, dict] = {}
    for r in rows:
        ctx = r["context"]
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            if title in pages:
                continue
            url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
            pages[title] = {"title": title, "url": url, "domain": "en.wikipedia.org",
                            "lines": {i: s.strip() for i, s in enumerate(sents)},
                            "snippets": []}
    return pages


def hotpot(n: int, seed: int = 0):
    """Второй корпус, полегче: абзацы Википедии из валидации HotpotQA (все
    вопросы, ~67 тыс. страниц) и n её вопросов. Вопрос — в два шага по двум
    статьям; без поиска модель их почти не знает, с поиском — должна."""
    import random

    from data import sft as SFT
    rows = SFT.load_rows("hotpot", "valid")
    pages = wiki_pages(rows)
    random.Random(seed).shuffle(rows)
    items = [{"question": r["question"], "gold": r["answer"]} for r in rows[:n]]
    return pages, items
