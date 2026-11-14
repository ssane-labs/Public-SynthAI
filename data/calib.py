"""Этап E: калибровка уверенности. Без ручной разметки (PLAN.md, этап E).

Модель после D2 сама отвечает на вопросы, ответы сверяются с эталоном, и из
этого получаются цели для головки ответа `u_answer` (model.py) и примеры
поведения:

    канал 0, на <|assistant|>   доля неверных из N ответов на этот вопрос:
                                «ошибусь ли» — до ответа, решает, искать ли
    канал 1, перед <|eot|>      неверен ли именно этот ответ: «ошиблась ли» —
                                после ответа, то, что видит пользователь

Вопросы двух видов:

    d1   вопрос + документы (SQuAD 2.0, HotpotQA, MuSiQue, с порчей контекста,
         data/sft.py). Верно — эталон в ответе, а на неотвечаемом — отказ.
         Ошибка здесь — ровно «уверенная выдумка», главная метрика продукта.
    cb   вопрос без документов (NQ-open, CC-BY-SA-3.0, та же лицензия, что у
         Natural Questions). Отвечает по памяти; поиск запрещён.

Поведение (лосс языка):
    — верные ответы — учим (отбор отказом: модель учится на своих же удачах);
    — неверные — только головка, язык не трогаем;
    — cb, где верных меньше половины, — учим начинать с <|search|>: не знаешь —
      ищи. Лосс на одном токене <|search|>, запрос — вопрос как есть, без лосса.

Генерация идёт на TPU отдельным процессом до обучения (synth1/sample.py):

    python data/calib.py build --init d2        # cache/e_train.json + Hub

Результат кладётся и на Hub (датасет synth-1-e-data): сессия, продолжившая
этап после обрыва, скачает его, а не будет генерировать заново.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data import search_sft as SS  # noqa: E402
from data import sft as SFT  # noqa: E402

NQ = ("google-research-datasets/nq_open", {
    "train": "nq_open/train-00000-of-00001.parquet",
    "valid": "nq_open/validation-00000-of-00001.parquet"})

# Отвечать сразу: рассуждение и поиск запрещены при генерации — нам нужен
# именно ответ, а не план поиска.
BANNED = ("<|think|>", "<|/think|>", "<|search|>", "<|/search|>", "<|open|>",
          "<|/open|>", "<|find|>", "<|/find|>", "<|result|>", "<|page|>",
          "<|user|>", "<|assistant|>")

CITE = re.compile(r"<\|cite\|>\s*\d+\s*<\|/cite\|>")
HUB_DATA = "synth-1-e-data"
CACHE = pathlib.Path(config.ROOT) / "cache" / "e_train.json"


def normalize(s: str) -> str:
    import string
    s = CITE.sub(" ", s).lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def refused(text: str) -> bool:
    return SFT.REFUSAL.lower()[:16] in text.lower()


def judge(item: dict, text: str) -> bool:
    """Верен ли ответ. d1: эталон в ответе или отказ на неотвечаемом.
    cb: один из эталонов в ответе, а ответ не простыня (иначе «угадать»
    можно перечислением)."""
    if item["kind"] == "d1" and not item["answerable"]:
        return refused(text)
    if refused(text):
        return False
    if item["kind"] == "d1":
        # Эталон D1 бывает длиннее ответа («Fulgencio Batista y Zaldívar»
        # при ответе «Fulgencio Batista»): включение в обе стороны или F1
        # не ниже 0.5 — тот же судья, что у траекторий D2 (STATUS.md, 23.10).
        return any(SS.correct(CITE.sub(" ", SFT.lead(text)), g) for g in item["golds"])
    got = normalize(text)
    for g in item["golds"]:
        g = normalize(g)
        if g and g in got and len(got.split()) <= 3 * len(g.split()) + 6:
            return True
    return False


# --- Вопросы -------------------------------------------------------------------

def load_nq(split: str) -> list[dict]:
    import pyarrow.parquet as pq
    path = SFT._download(NQ[0], NQ[1][split])
    return pq.read_table(path).to_pylist()


def cb_items(enc, n: int, split: str, seed: int) -> list[dict]:
    rows = load_nq(split)
    random.Random(seed).shuffle(rows)
    sp = enc.special
    out = []
    for r in rows[:n]:
        q = r["question"].strip()
        q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
        prompt = [sp["<|user|>"]] + enc._text(q + "\n") + [sp["<|assistant|>"]]
        out.append({"kind": "cb", "src": "nq", "question": q, "prompt": prompt,
                    "golds": list(r["answer"]), "answerable": True})
    return out


def d1_items(n: int, split: str, seed: int, length: int = 4608,
             exclude: set | None = None) -> list[dict]:
    """n вопросов с каждого источника D1, в формате и с порчей D1.
    exclude — вопросы, которые брать нельзя (их берут замеры)."""
    stream = SFT.SFTStream(length, split=split, seed=seed)
    enc = stream.enc
    picked = {s: 0 for s in SFT.FILES}
    out = []
    for i in stream._order(0):
        src, _ = stream.rows[i]
        if picked[src] >= n:
            continue
        docs, q, a, ok = stream.example(i, 0)
        if exclude and q in exclude:
            continue
        ids, mask = enc.encode(docs, q, a)
        prompt = ids[:mask.index(1)]
        if len(prompt) + 64 > length:
            continue
        gold = CITE.sub("", a).strip()
        out.append({"kind": "d1", "src": src, "question": q, "prompt": prompt,
                    "golds": [gold] if ok else [], "answerable": ok, "ref": a})
        picked[src] += 1
        if all(v >= n for v in picked.values()):
            break
    return out, enc


# --- Генерация -----------------------------------------------------------------

def load_run(run: str):
    import jax

    from synth1 import checkpoint as C
    from synth1 import model as M
    from synth1 import optimizer as O
    from synth1.config import DEFAULT
    template = M.init_model(jax.random.key(0), DEFAULT)
    state = O.init_state(template, O.OptConfig())
    local = C.pull(run)
    params, _, meta = C.load(local, template, state)
    return params, DEFAULT, meta


def answer_all(gen, enc, items: list[dict], samples: int, max_new: int,
               temperature: float = 1.0, seed: int = 0) -> None:
    """items[k]["outs"] — samples ответов (первый жадный), ["ok"] — верны ли."""
    sp = enc.special
    ban = [sp[t] for t in BANNED]
    prompts, temps, owner = [], [], []
    for k, it in enumerate(items):
        for s in range(samples):
            prompts.append(it["prompt"])
            temps.append(0.0 if s == 0 else temperature)
            owner.append(k)
    got = gen.generate(prompts, max_new=max_new, temps=temps, ban=ban,
                       seed=seed, log_every=2000)
    from synth1 import sample as G
    special = {i: t for t, i in sp.items()}
    for it in items:
        it["outs"], it["ok"], it["u_first"], it["u_last"] = [], [], [], []
        it["texts"] = []
    for k, g in zip(owner, got):
        it = items[k]
        text = G.decode(enc.tok, g["ids"], special)
        it["outs"].append(g["ids"])
        it["ok"].append(bool(judge(it, text)))
        it["u_first"].append(g["u_first"])
        it["u_last"].append(g["u_last"])
        it["texts"].append(text)


def build(init: str, n_d1: int, n_cb: int, samples_d1: int, samples_cb: int,
          out: pathlib.Path = CACHE, push: bool = True) -> None:
    from tokenizers import Tokenizer

    from synth1 import sample as G
    began = time.time()
    params, cfg, meta = load_run(init)
    print(f"E-данные: снимок {init!r}, шаг {meta['step']:,}", flush=True)
    gen = G.Generator(params, cfg)
    # Вопросы D1 — из валидации, которую модель не видела: на вопросах
    # обучения (две эпохи D1) она права в 75%, на новых — в ~43%, и головка,
    # выученная на первых, на вторых самоуверенна (ECE 0.27, STATUS.md,
    # 23.10). Вопросы замеров (evals/d1.py, e.py, g.py) исключены.
    from evals import d1 as D1
    held = {it["question"] for it in
            d1_items(128, "valid", seed=777)[0] + d1_items(64, "valid", seed=31)[0]
            + D1.build(128, 4608)[0]}
    d1, enc = d1_items(n_d1, "valid", seed=4242, exclude=held)
    print(f"  D1 — валидация без {len(held):,} вопросов замеров", flush=True)
    cb = cb_items(SS.Encoder(Tokenizer.from_file(
        str(config.TOKENIZER_OUT / "synth1.json"))), n_cb, "train", seed=4242)
    print(f"  вопросов: d1 {len(d1):,}, cb {len(cb):,}", flush=True)
    answer_all(gen, enc, cb, samples_cb, max_new=24, seed=1)
    answer_all(gen, enc, d1, samples_d1, max_new=48, seed=2)
    for name, part in (("d1", d1), ("cb", cb)):
        acc = np.mean([o for it in part for o in it["ok"]])
        print(f"  {name}: верных ответов {acc:.1%}", flush=True)
    keep = ("kind", "src", "prompt", "outs", "ok", "answerable", "ref")
    data = {"init": init, "step": meta["step"],
            "examples": [{k: it.get(k) for k in keep} for it in d1 + cb]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data))
    print(f"  записано {out} ({out.stat().st_size / 1e6:.0f} МБ) "
          f"за {(time.time() - began) / 60:.1f} мин", flush=True)
    for it in (d1[:3] + cb[:3]):
        print(f"  [{it['src']}] {it['question'][:70]!r} → {it['texts'][0][:80]!r} "
              f"{'верно' if it['ok'][0] else 'неверно'}")
    if push:
        _push(out)


def _hub():
    from huggingface_hub import HfApi
    return HfApi(token=config.key("HF_TOKEN", required=False) or None), \
        f"{config.HF_NAMESPACE}/{HUB_DATA}"


def _push(path: pathlib.Path) -> None:
    try:
        api, repo = _hub()
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                        repo_id=repo, repo_type="dataset")
        print(f"  на Hub: {repo}", flush=True)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  на Hub не ушло ({type(exc).__name__}: {exc}) — следующей "
              f"сессии придётся генерировать заново", flush=True)


def fetch(path: pathlib.Path = CACHE) -> bool:
    """Есть ли данные локально или на Hub (тогда скачать)."""
    if path.exists():
        return True
    try:
        from huggingface_hub import hf_hub_download
        api, repo = _hub()
        got = hf_hub_download(repo, path.name, repo_type="dataset",
                              token=api.token)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pathlib.Path(got).read_bytes())
        print(f"E-данные: скачаны с Hub ({repo})", flush=True)
        return True
    except Exception:                                          # noqa: BLE001
        return False


# --- Строки для обучения -----------------------------------------------------

def examples(data: dict, enc) -> list[tuple[list[int], list[int], dict, dict]]:
    """(ids, маска лосса, цели канала 0, цели канала 1) — позиции по ids.

    Позиция цели — та, из которой предсказывается следующий токен: канал 0 —
    позиция <|assistant|> (последний токен промпта), канал 1 — последний токен
    ответа (из него предсказывается <|eot|>).
    """
    sp = enc.special
    eot = config.EOT_ID
    out = []
    for ex in data["examples"]:
        prompt, outs, ok = ex["prompt"], ex["outs"], ex["ok"]
        wrong = 1.0 - float(np.mean(ok))
        a = len(prompt) - 1
        seen = set()
        for ids_, good in zip(outs, ok):
            key = tuple(ids_)
            if key in seen:
                continue
            seen.add(key)
            # Простыня без <|eot|> — обрезана генерацией; сам ответ неверен
            # (judge), а <|eot|> дописываем, чтобы позиция цели была.
            ids = prompt + list(ids_) + [eot]
            learn = good and not (ex["kind"] == "cb" and wrong > 0.5)
            mask = [0] * len(prompt) + [1 if learn else 0] * (len(ids_) + 1)
            out.append((ids, mask, {a: wrong}, {len(ids) - 2: 0.0 if good else 1.0}))
        if ex["kind"] == "cb" and wrong > 0.5:
            # Не знает — ищет. Запрос — сам вопрос. Лосс на всём действии:
            # с лоссом на одном <|search|> модель после E писала
            # «<|search|><|search|>…» — запрос ей никто не показал.
            q_ids = prompt[1:-1]
            ids = prompt + [sp["<|search|>"]] + q_ids + [sp["<|/search|>"]]
            mask = [0] * len(prompt) + [1] * (len(q_ids) + 2)
            out.append((ids, mask, {a: wrong}, {}))
    return out


class CalibStream:
    """Строки этапа E вперемешку с D2 (поиск, D1, претрейн): tokens, loss_mask,
    epi_q, epi_a [B, T+1]. В epi_* −1 — «цели нет»."""

    # Википедия (data/wiki_search.py) — поиск в той же среде, что у замера;
    # без неё поиск после E/F/G расползался (STATUS.md, 05.11.2026).
    SHARES = {"calib": 0.5, "wiki": 0.25, "d1": 0.1, "pretrain": 0.15}

    def __init__(self, seq_len: int, seed: int = 0, limit: int | None = None):
        from tokenizers import Tokenizer
        if not fetch():
            raise SystemExit("нет данных этапа E: сначала "
                             "python data/calib.py build --init d2")
        data = json.loads(CACHE.read_text())
        enc = SS.Encoder(Tokenizer.from_file(str(config.TOKENIZER_OUT / "synth1.json")))
        self.rows = examples(data, enc)
        if limit:
            self.rows = self.rows[:limit]
        self.init = data.get("init")
        self.seq_len = seq_len
        self.seed = seed
        from data import wiki_search as WS
        self.mix = WS.SearchMixStream(seq_len, seed=seed, limit=limit)
        self.rng = random.Random(seed ^ 0xE)
        self.epoch = self.index = 0
        self.tokens_emitted = 0
        self.docs = 0

    def describe(self) -> str:
        n_q = sum(1 for r in self.rows if r[2])
        return (f"смесь E: {len(self.rows):,} строк калибровки (данные со снимка "
                f"{self.init!r}, целей канала 0 — {n_q:,}), D2-смесь; доли "
                f"{self.SHARES}; длина {self.seq_len}")

    def state(self) -> dict:
        return {"sft": "e", "epoch": self.epoch, "index": self.index,
                "seed": self.seed, "seq_len": self.seq_len,
                "tokens_emitted": self.tokens_emitted, "docs": self.docs,
                "rng": self.rng.random()}

    def restore(self, state: dict) -> None:
        if state.get("sft") != "e":
            raise SystemExit("снимок не от этапа E — нужен --new-stage")
        self.epoch, self.index = state["epoch"], state["index"]
        self.tokens_emitted = state.get("tokens_emitted", 0)
        self.rng = random.Random(state.get("rng", 0))

    def carry_over(self, state: dict) -> list[str]:
        self.tokens_emitted = state.get("tokens_emitted", 0)
        return []

    def _calib_rows(self):
        width = self.seq_len + 1
        ids, mask, eq, ea = [], [], [], []

        def flush():
            pad = width - len(ids)
            return (ids + [config.PAD_ID] * pad, mask + [0] * pad,
                    eq + [-1.0] * pad, ea + [-1.0] * pad)

        while True:
            order = list(range(len(self.rows)))
            random.Random(self.seed * 131 + self.epoch).shuffle(order)
            while self.index < len(order):
                e_ids, e_mask, e_q, e_a = self.rows[order[self.index]]
                self.index += 1
                if len(e_ids) > width:
                    continue
                if len(ids) + len(e_ids) > width:
                    yield flush()
                    ids, mask, eq, ea = [], [], [], []
                base = len(ids)
                ids += e_ids
                mask += e_mask
                eq += [-1.0] * len(e_ids)
                ea += [-1.0] * len(e_ids)
                for p, v in e_q.items():
                    eq[base + p] = v
                for p, v in e_a.items():
                    ea[base + p] = v
                self.docs += 1
            self.epoch += 1
            self.index = 0
            print(f"  E: эпоха {self.epoch} начата", flush=True)

    # Ключи батча и их значение «нет цели». Наследники (F, G) добавляют свои.
    KEYS = {"loss_mask": 0.0, "epi_q": -1.0, "epi_a": -1.0}

    def sources(self) -> dict:
        """Имя → генератор строк-словарей. Недостающие ключи заполнит batches."""
        def calib():
            for r, m, q, a in self._calib_rows():
                yield {"tokens": r, "loss_mask": m, "epi_q": q, "epi_a": a}

        def plain(rows):
            for r, m in rows:
                yield {"tokens": r, "loss_mask": m}

        def text():
            for b in self.mix.text.batches(1):
                r = list(b["tokens"][0])
                yield {"tokens": r, "loss_mask": [1] * len(r)}

        return {"calib": calib(), "wiki": plain(self.mix._wiki_rows()),
                "d1": plain(self.mix.d1.rows_packed()), "pretrain": text()}

    def batches(self, batch_size: int):
        gens = self.sources()
        width = self.seq_len + 1
        names = [n for n in self.SHARES if self.SHARES[n] > 0]
        weights = [self.SHARES[n] for n in names]
        while True:
            got = {k: [] for k in ("tokens", *self.KEYS)}
            for _ in range(batch_size):
                row = next(gens[self.rng.choices(names, weights)[0]])
                got["tokens"].append(row["tokens"])
                for k, fill in self.KEYS.items():
                    got[k].append(row.get(k) or [fill] * width)
            tokens = np.asarray(got["tokens"], dtype=np.int32)
            self.tokens_emitted += tokens.size
            yield {"tokens": tokens,
                   **{k: np.asarray(got[k], dtype=np.float32) for k in self.KEYS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--init", default="d2", help="снимок, который отвечает")
    ap.add_argument("--n-d1", type=int, default=1500, help="вопросов с источника D1")
    ap.add_argument("--n-cb", type=int, default=8000, help="вопросов NQ-open")
    ap.add_argument("--samples-d1", type=int, default=2)
    ap.add_argument("--samples-cb", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="генерировать, даже если есть")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    if not args.force and fetch():
        data = json.loads(CACHE.read_text())
        if data.get("init") == args.init:
            print(f"E-данные уже есть ({len(data['examples']):,} вопросов, "
                  f"снимок {data['init']!r}) — генерировать не нужно", flush=True)
            return
    build(args.init, args.n_d1, args.n_cb, args.samples_d1, args.samples_cb,
          push=not args.no_push)


if __name__ == "__main__":
    main()
