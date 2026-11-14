"""Этап D2: поиск. Траектории OpenResearcher → наш протокол.

Учитель (gpt-oss-120b, Apache-2.0) ищет в офлайн-поисковике по корпусу
OpenResearcher инструментами `browser.search` / `browser.open` /
`browser.find` и отвечает со ссылками вида 【курсор†L5-L8】. Траектории
рассчитаны на окно 128k: медиана ~100 тыс. символов и ~30 вызовов. У нас
8192 токена, поэтому переложение — это прежде всего сжатие (SEARCH.md,
раздел 5):

1. Берём только успешные прогоны с ответом, совпадающим с эталоном.
2. Из прогонов одного вопроса (16 seed'ов) — самый короткий после сжатия.
3. Инструменты учителя → наш протокол (SEARCH.md, раздел 6):

       <|user|>вопрос
       <|assistant|><|think|>коротко<|/think|>
       <|search|>запрос<|/search|>
       <|result|>[1] заголовок - домен - сниппет ...<|/result|>
       <|open|>1<|/open|>
       <|page|>нужные строки страницы<|/page|>
       <|find|>строка<|/find|>
       <|page|>найденные строки<|/page|>
       ответ <|cite|>1<|/cite|><|eot|>

   Источники нумеруются сквозь всю траекторию: [N] в выдаче, <|open|>N и
   <|cite|>N — один и тот же номер.
4. Выдача — до пяти результатов (плюс те, что потом открывались), сниппет
   до 200 символов. Страница — строки, на которые ссылается финальный ответ,
   и то, что нашёл find, с соседями; не больше 900 символов.
5. Размышления учителя — первые ~240 символов каждого куска.
6. Не влезло в окно — старые страницы сворачиваются в «[page folded]»;
   не влезло и так — траектория выбрасывается.

Лосс — только на том, что пишет модель: think, search, open, find, ответ.
Выдача, страницы и вопрос маскируются.

    python data/search_sft.py stats <файл.parquet>   # сколько выживает
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data import sft as SFT  # noqa: E402

REPO = "OpenResearcher/OpenResearcher-Dataset"

TOP_RESULTS = 5
SNIPPET = 200
PAGE_CHARS = 900
THINK_CHARS = 240

CITE_RE = re.compile(r"【(\d+)†L(\d+)(?:-L(\d+))?】")
ITEM_RE = re.compile(r"【(\d+)†(.*?)†(.*?)】")
LINE_RE = re.compile(r"^L(\d+): ?(.*)$")


# --- Разбор вывода инструментов ---------------------------------------------

def _text(m: dict) -> str:
    return "\n".join((c.get("text") or "") for c in (m.get("content") or []))


def _lines(body: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw in body.splitlines():
        hit = LINE_RE.match(raw)
        if hit:
            out[int(hit.group(1))] = hit.group(2)
    return out


def _cursor(body: str) -> int | None:
    hit = re.match(r"\s*\[(\d+)\]", body)
    return int(hit.group(1)) if hit else None


def parse_search(body: str) -> list[tuple[int, str, str, str]]:
    """Выдача: [(id, заголовок, домен, сниппет)]."""
    lines = _lines(body)
    text = " ".join(lines[k] for k in sorted(lines))
    out = []
    parts = re.split(r"\*\s*(?=【\d+†)", text)
    for part in parts:
        hit = ITEM_RE.match(part.strip())
        if not hit:
            continue
        rest = part.strip()[hit.end():].strip()
        title_snip = rest
        title = hit.group(2)
        if title.startswith("Doc "):
            title = ""
        snippet = re.sub(r"\s+", " ", title_snip)[:SNIPPET]
        out.append((int(hit.group(1)), title or snippet[:60], hit.group(3), snippet))
    return out


def page_text(body: str, keep: set[int]) -> str:
    """Строки страницы: отмеченные (с соседями) или начало, не длиннее PAGE_CHARS."""
    lines = _lines(body)
    content = {k: v for k, v in lines.items() if not v.startswith("URL:")}
    if not content:
        return ""
    want: list[int] = []
    for k in sorted(keep):
        for j in (k - 1, k, k + 1):
            if j in content and j not in want:
                want.append(j)
    if not want:
        want = sorted(content)[:12]
    want.sort()
    out, used = [], 0
    for k in want:
        piece = content[k].strip()
        if not piece:
            continue
        if used + len(piece) > PAGE_CHARS:
            break
        out.append(piece)
        used += len(piece) + 1
    return " ".join(out)


def _short(t: str, n: int = THINK_CHARS) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= n:
        return t
    cut = t[:n]
    dot = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:dot + 1] if dot > n // 3 else cut.rstrip() + "…"


# --- Правильность ответа ------------------------------------------------------

def _unbox(s: str) -> str:
    hit = re.search(r"\\boxed\{(.*)\}", s, re.S)
    return (hit.group(1) if hit else s).strip()


def _clean(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u202f", " ").replace("\u2011", "-").replace("‑", "-")
    return s.replace("**", "").strip()


def _norm(s: str) -> str:
    s = _clean(_unbox(s)).lower()
    s = re.sub(r"\\[a-z]+|[{}$\\]", " ", s)
    s = re.sub(r"[^\w\s.%-]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def final_answer(text: str) -> str:
    """Строка после «Exact Answer:»; если она пустая или только **, — следующая."""
    hit = re.search(r"Exact Answer:\s*(.*)", text)
    if not hit:
        return ""
    first = _clean(hit.group(1))
    if not first:
        rest = text[hit.end():].strip().splitlines()
        first = _clean(rest[0]) if rest else ""
    return _clean(_unbox(first))


def correct(pred: str, gold: str) -> bool:
    """Совпадение с эталоном. Короткий эталон — должен содержаться в ответе;
    длинный (эталоны бывают целыми фразами) — пословный F1 не меньше 0.5."""
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return False
    if p == g or g in p or (len(p) > 3 and p in g):
        return True
    pt, gt = p.split(), g.split()
    if len(gt) <= 3:
        return False
    common = sum(min(pt.count(w), gt.count(w)) for w in set(gt))
    if not common:
        return False
    prec, rec = common / len(pt), common / len(gt)
    return 2 * prec * rec / (prec + rec) >= 0.5


# --- Переложение -----------------------------------------------------------

def convert(row: dict):
    """Траектория → список сегментов [(вид, текст)] или None.

    Виды: user, think, search, result, open, page, find, answer.
    """
    q = row["question"]
    if "file '" in q or "associated with this task" in q:
        return None            # вопрос про файл, которого у нас нет
    q = re.split(r"\n\s*You should follow the format", q)[0].strip()
    msgs = row["messages"]
    final = next((m for m in reversed(msgs)
                  if m.get("role") == "assistant" and m.get("channel") == "final"), None)
    if final is None:
        return None
    ans = final_answer(_text(final))
    if not correct(ans, row.get("answer") or ""):
        return None

    # Какие строки каких страниц цитирует ответ, и что открывалось.
    cited: dict[int, set[int]] = {}
    for c, a, b in CITE_RE.findall(_text(final)):
        rng = range(int(a), int(b or a) + 1)
        cited.setdefault(int(c), set()).update(rng)
    opened_ids: dict[int, set[int]] = {}
    pending = None
    for m in msgs:
        to = m.get("recipient") or ""
        if m.get("role") == "assistant" and to == "browser.open":
            try:
                args = json.loads(_text(m))
            except ValueError:
                args = {}
            if isinstance(args.get("id"), int) and "cursor" in args:
                opened_ids.setdefault(int(args["cursor"]), set()).add(args["id"])

    segs: list[tuple[str, str]] = [("user", q)]
    src_of: dict[tuple[int, int], int] = {}   # (курсор выдачи, id) → номер источника
    page_src: dict[int, int] = {}             # курсор страницы → номер источника
    n_src = 0
    last_search_cursor = None
    finds: dict[int, set[int]] = {}
    for m in msgs:
        role, to = m.get("role"), (m.get("recipient") or "")
        body = _text(m)
        if role == "assistant" and m.get("channel") == "final":
            break
        if role == "assistant" and not to:
            if body.strip():
                segs.append(("think", _short(body)))
        elif role == "assistant" and to.startswith("browser."):
            try:
                args = json.loads(body)
            except ValueError:
                args = {}
            pending = (to, args)
        elif role == "tool" and pending is not None:
            tool, args = pending
            pending = None
            cur = _cursor(body)
            if tool == "browser.search":
                items = parse_search(body)
                if not items:
                    continue
                want = opened_ids.get(cur, set())
                keep = [it for it in items if it[0] in want]
                keep += [it for it in items if it[0] not in want][:max(0, TOP_RESULTS - len(keep))]
                keep.sort(key=lambda it: it[0])
                lines = []
                for it in keep:
                    n_src += 1
                    src_of[(cur, it[0])] = n_src
                    lines.append(f"[{n_src}] {it[1]} - {it[2]} - {it[3]}")
                segs.append(("search", str(args.get("query", ""))[:200]))
                segs.append(("result", "\n".join(lines)))
                last_search_cursor = cur
            elif tool == "browser.open":
                ref = args.get("id")
                from_cur = args.get("cursor", last_search_cursor)
                if isinstance(ref, int) and (from_cur, ref) in src_of:
                    num = src_of[(from_cur, ref)]
                elif isinstance(ref, str):
                    # Адрес по памяти. В нашей среде его нет, а модель, увидев
                    # такие траектории, открывала example.com и r.jina.ai вместо
                    # поиска (STATUS.md, 05.11.2026) — траекторию не берём.
                    return None
                else:
                    continue       # прокрутка уже открытой страницы — пропускаем
                if cur is not None:
                    page_src[cur] = num
                keep = cited.get(cur, set()) | finds.get(cur, set())
                segs.append(("open", str(num) if isinstance(ref, int) else ref))
                segs.append(("page", page_text(body, keep)))
            elif tool == "browser.find":
                pat = str(args.get("pattern", ""))[:80]
                hits = page_text(body, set())
                if pat and hits:
                    segs.append(("find", pat))
                    segs.append(("page", hits[:400]))

    cites = sorted({page_src[c] for c in cited if c in page_src})
    segs.append(("answer", SFT._answer(ans, cites) if cites else ans))
    return tidy(segs)


# Рассуждения учителя о самой среде: «tool might have a bug», «default
# source», «use jina.ai», «search engine is not working». У нас другая среда,
# и модель после D2 крутила такие мысли вместо действий (STATUS.md, 23.10).
META = re.compile(
    r"\b(tool|tools|search engine|engine|corpus|indexed|bug|error|not working|"
    r"failing|fails|failed|blocked|restricted|cache|jina|bing|topn|cursor|"
    r"default source|news source|web source|source\s*[:=]|\"source\"|"
    r"'source'|source default|default search|default\b|generic web search|"
    r"try again|try a different|different approach|example\.com)\b", re.I)


def tidy(segs):
    """Перед каждым действием и ответом — не больше одной мысли.

    У учителя в среднем 35 кусков рассуждения на 29 действий, часто по
    нескольку подряд; модель выучила цепочки «<|think|>…<|think|>…» без
    действий. Оставляем последнюю мысль перед действием — она его и
    объясняет, — а мысли о самой среде (META) выбрасываем.
    """
    out, pending = [], []
    for kind, text in segs:
        if kind == "think":
            pending.append(text)
            continue
        keep = [t for t in pending if not META.search(t)]
        if keep and kind != "user":
            out.append(("think", keep[-1]))
        pending = []
        out.append((kind, text))
    return out


class Encoder(SFT.Encoder):
    WRAP = {"think": ("<|think|>", "<|/think|>"), "search": ("<|search|>", "<|/search|>"),
            "result": ("<|result|>", "<|/result|>"), "open": ("<|open|>", "<|/open|>"),
            "page": ("<|page|>", "<|/page|>"), "find": ("<|find|>", "<|/find|>")}
    MASKED = {"user", "result", "page"}

    def encode_part(self, kind: str, text: str) -> list[int]:
        sp = self.special
        if kind == "user":
            return ([sp["<|user|>"]] + self._text(text.strip() + "\n")
                    + [sp["<|assistant|>"]])
        if kind == "answer":
            return self._answer(text) + [config.EOT_ID]
        a, b = self.WRAP[kind]
        return [sp[a]] + self._text(text) + [sp[b]]

    def join(self, segs, parts):
        ids: list[int] = []
        mask: list[int] = []
        for (kind, _), part in zip(segs, parts):
            ids += part
            mask += [0 if kind in self.MASKED else 1] * len(part)
        return ids, mask

    def encode_segments(self, segs):
        return self.join(segs, [self.encode_part(k, t) for k, t in segs])


def fit(enc: Encoder, segs, budget: int):
    """Кодирует; если длиннее budget — сворачивает страницы с самой старой.

    Каждый кусок кодируется один раз: перекодировать всю траекторию после
    каждой свёрнутой страницы выходило ~8 проходов на строку и полчаса на
    сборку набора на Kaggle.
    """
    parts = [enc.encode_part(k, t) for k, t in segs]
    size = sum(map(len, parts))
    pages = [i for i, (k, _) in enumerate(segs) if k == "page"]
    if size > budget and pages:
        folded = enc.encode_part("page", "[page folded]")
        segs = list(segs)
        while size > budget and pages:
            i = pages.pop(0)
            size += len(folded) - len(parts[i])
            segs[i], parts[i] = ("page", "[page folded]"), folded
    return enc.join(segs, parts) if size <= budget else None


# --- Сборка набора ------------------------------------------------------------

def list_files() -> list[str]:
    from huggingface_hub import HfApi
    token = config.key("HF_TOKEN", required=False) or None
    files = HfApi(token=token).list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if f.startswith("seed_") and f.endswith(".parquet"))


def _convert_file(path: str):
    """Все траектории файла → [(qid, ids, mask)] для влезающих в окно."""
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    enc = Encoder(Tokenizer.from_file(str(config.TOKENIZER_OUT / "synth1.json")))
    pf = pq.ParquetFile(path)
    out, seen, kept = [], 0, 0
    for g in range(pf.num_row_groups):
        rows = pf.read_row_group(
            g, columns=["qid", "question", "answer", "messages", "status"]).to_pylist()
        for row in rows:
            seen += 1
            if row.get("status") != "success":
                continue
            segs = convert(row)
            if segs is None:
                continue
            got = fit(enc, segs, BUDGET)
            if got is None:
                continue
            kept += 1
            out.append((int(row["qid"]), got[0], got[1]))
    return out, seen, kept


BUDGET = 8192 + 1


def qids_path(cache: pathlib.Path) -> pathlib.Path:
    return cache.with_name(cache.stem + "_qids.json")


def build(cache: pathlib.Path, workers: int | None = None,
          limit_files: int | None = None):
    """Скачать, переложить, оставить по самому короткому на вопрос. С кэшем."""
    if cache.exists():
        data = json.loads(cache.read_text())
        print(f"  D2: из кэша {cache} — {len(data):,} траекторий", flush=True)
        return data
    from concurrent.futures import ProcessPoolExecutor
    from huggingface_hub import hf_hub_download
    token = config.key("HF_TOKEN", required=False) or None
    files = list_files()[:limit_files] if limit_files else list_files()
    # По одному файлу 7.4 ГБ шли 10.6 минут (21.10) — дольше самой перекладки.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as io:
        paths = list(io.map(lambda f: hf_hub_download(
            REPO, f, repo_type="dataset", token=token), files))
    best: dict[int, tuple[list[int], list[int]]] = {}
    seen = kept = 0
    # Файлов 48; на TPU-машине Kaggle 96 ядер — все файлы разом.
    workers = workers or min(len(paths), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for out, s, k in pool.map(_convert_file, paths):
            seen += s
            kept += k
            for qid, ids, mask in out:
                if qid not in best or len(ids) < len(best[qid][0]):
                    best[qid] = (ids, mask)
    data = [[ids, mask] for _, (ids, mask) in sorted(best.items())]
    # Номера вопросов в том же порядке — по ним замер с поисковиком
    # (data/search_env.py) находит отложенные вопросы.
    qids_path(cache).parent.mkdir(parents=True, exist_ok=True)
    qids_path(cache).write_text(json.dumps(sorted(best)))
    print(f"  D2: {len(files)} файлов, траекторий {seen:,}, влезло и верно {kept:,}, "
          f"вопросов {len(data):,}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


# --- Поток с примесью ---------------------------------------------------------

class MixStream:
    """Строки по 8192+1: поиск, D1 и претрейн вперемешку, по долям.

    D1 держит ответы со ссылками и отказы, претрейн — свободный текст: после
    D1 модель его подзабыла (STATUS.md, 20.10.2026). Каждая строка — из одного
    источника; у претрейна маска лосса — единицы.
    """

    SHARES = {"search": 0.5, "d1": 0.3, "pretrain": 0.2}

    def __init__(self, seq_len: int, seed: int = 0, split: str = "train",
                 limit: int | None = None):
        from synth1 import data as D
        self.seq_len = seq_len
        self.seed = seed
        cache = pathlib.Path(config.ROOT) / "cache" / f"d2_{split}.json"
        data = build(cache, limit_files=1 if limit else None)
        rng = random.Random(1234)
        rng.shuffle(data)
        cut = max(1, len(data) // 50)          # 2% — отложенные
        self.search = data[cut:] if split == "train" else data[:cut]
        self.d1 = SFT.SFTStream(seq_len, split=split, seed=seed, limit=limit,
                                repeat={"squad": 2})
        self.text = D.TokenStream("anneal", seq_len=seq_len, seed=seed,
                                  split="train" if split == "train" else "valid")
        self.rng = random.Random(seed ^ 0xD2)
        self.epoch = self.index = 0
        self.tokens_emitted = 0
        self.docs = 0

    def describe(self) -> str:
        return (f"смесь D2: поиск {len(self.search):,} траекторий, D1 и претрейн; "
                f"доли {self.SHARES}; длина {self.seq_len}")

    def state(self) -> dict:
        return {"sft": "d2", "epoch": self.epoch, "index": self.index,
                "seed": self.seed, "seq_len": self.seq_len, "rng": self.rng.random(),
                "tokens_emitted": self.tokens_emitted, "docs": self.docs}

    def restore(self, state: dict) -> None:
        if state.get("sft") != "d2":
            raise SystemExit("снимок не от этапа D2 — нужен --new-stage")
        self.epoch, self.index = state["epoch"], state["index"]
        self.tokens_emitted = state.get("tokens_emitted", 0)
        self.rng = random.Random(state.get("rng", 0))

    def carry_over(self, state: dict) -> list[str]:
        self.tokens_emitted = state.get("tokens_emitted", 0)
        return []

    def _search_rows(self):
        width = self.seq_len + 1
        ids: list[int] = []
        mask: list[int] = []
        fits = False
        while True:
            order = list(range(len(self.search)))
            random.Random(self.seed * 31 + self.epoch).shuffle(order)
            while self.index < len(order):
                e_ids, e_mask = self.search[order[self.index]]
                self.index += 1
                if len(e_ids) > width:
                    continue        # длиннее строки (короткий seq_len) — мимо
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
            self.epoch += 1
            self.index = 0

    def batches(self, batch_size: int):
        search = self._search_rows()
        d1 = self.d1.rows_packed()
        text = self.text.batches(1)
        names = list(self.SHARES)
        weights = [self.SHARES[n] for n in names]
        while True:
            rows, masks = [], []
            for _ in range(batch_size):
                pick = self.rng.choices(names, weights)[0]
                if pick == "search":
                    r, m = next(search)
                elif pick == "d1":
                    r, m = next(d1)
                else:
                    r = list(next(text)["tokens"][0])
                    m = [1] * len(r)
                rows.append(r)
                masks.append(m)
            tokens = np.asarray(rows, dtype=np.int32)
            self.tokens_emitted += tokens.size
            yield {"tokens": tokens, "loss_mask": np.asarray(masks, dtype=np.float32)}


def holdout(seq_len: int, batch_size: int, n_batches: int, limit: int | None = None):
    """Отложенный набор D2: только поисковые строки отложенных 2% вопросов."""
    stream = MixStream.__new__(MixStream)
    stream.seq_len, stream.seed, stream.epoch, stream.index = seq_len, 777, 0, 0
    stream.docs = 0
    cache = pathlib.Path(config.ROOT) / "cache" / "d2_train.json"
    data = build(cache, limit_files=1 if limit else None)
    rng = random.Random(1234)
    rng.shuffle(data)
    stream.search = data[:max(1, len(data) // 50)]
    it = stream._search_rows()
    got = [[next(it) for _ in range(batch_size)] for _ in range(n_batches)]
    tokens = np.asarray([[r[0] for r in b] for b in got], dtype=np.int32)
    masks = np.asarray([[r[1] for r in b] for b in got], dtype=np.float32)
    return tokens, masks


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "stats":
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer
        enc = Encoder(Tokenizer.from_file(str(config.TOKENIZER_OUT / "synth1.json")))
        rows = pq.ParquetFile(sys.argv[2]).read_row_group(
            0, columns=["qid", "question", "answer", "messages", "status"]).to_pylist()
        why = {"файл в вопросе": 0, "ответ неверный": 0, "не влезло": 0, "влезло": 0}
        lens, shown = [], 0
        for row in rows:
            q = row["question"]
            if "file '" in q or "associated with this task" in q:
                why["файл в вопросе"] += 1
                continue
            segs = convert(row)
            if segs is None:
                why["ответ неверный"] += 1
                continue
            got = fit(enc, segs, BUDGET)
            if got is None:
                why["не влезло"] += 1
                continue
            why["влезло"] += 1
            lens.append(len(got[0]))
            if shown < 1:
                shown += 1
                print(enc.tok.decode(got[0], skip_special_tokens=False)[:3000])
        print(why)
        if lens:
            print(f"длина: медиана {np.median(lens):.0f}, 90% {np.percentile(lens, 90):.0f}")
