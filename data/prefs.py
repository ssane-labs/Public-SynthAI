"""Этап F: предпочтения (ORPO). Пары — из ответов, собранных для этапа E.

Отдельной генерации нет: data/calib.py уже записал по каждому вопросу
несколько ответов модели и их проверку. Из них пары «лучше / хуже» на один
промпт (лосс — model.loss_fn, ветка pref):

    d1, неотвечаемый   отказ  >  выдуманный ответ        (если модель хоть раз
                                                          выдумала; отказ —
                                                          эталонная строка)
    d1, отвечаемый     верный ответ  >  неверный или ложный отказ
    cb, знает          верный  >  неверный
    cb, не знает       <|search|>вопрос<|/search|>  >  выдуманный ответ

То есть F учит ровно то, что продукт обещает: не выдумывать — отказаться или
пойти искать. Ответы — снимка d2 (до E): пары чуть «устаревшие», зато не
стоят ещё получаса генерации на TPU.

Пара целиком лежит в одной строке: [промпт + лучше + eot][промпт + хуже + eot].
Так она не разъедется по чипам. Не влезла — выбрасывается.
"""

from __future__ import annotations

import json
import random

from data import calib as CB
from data import sft as SFT

# Пары «искать > выдумать» многочисленны (NQ по памяти почти всегда мимо) —
# не больше этой доли, чтобы F не свёлся к одному поиску.
MAX_SEARCH_SHARE = 0.4


def pairs(data: dict, enc) -> list[tuple[list[int], list[int], list[int], int]]:
    """[(промпт, лучше, хуже, вид)] — id без <|eot|>."""
    sp = enc.special
    # Отказ — не голый, а с предложением искать иначе (sft.fallback).
    refusal = enc._answer(SFT.fallback([], random.Random(len(data["examples"]))))
    out, search = [], []
    rng = random.Random(77)
    for ex in data["examples"]:
        prompt, outs, ok = ex["prompt"], ex["outs"], ex["ok"]
        good = [o for o, k in zip(outs, ok) if k]
        bad = [o for o, k in zip(outs, ok) if not k]
        if not bad:
            continue
        if ex["kind"] == "d1" and not ex["answerable"]:
            out.append((prompt, good[0] if good else refusal, rng.choice(bad), 0))
        elif good:
            out.append((prompt, rng.choice(good), rng.choice(bad), 1))
        elif ex["kind"] == "d1" and ex.get("ref"):
            # Ответ был в документах, а модель ни разу не ответила верно —
            # эталон лучше. Без этих пар F учил только «откажись» и ложные
            # отказы на SQuAD доросли до 64% (STATUS.md, 05.11.2026).
            out.append((prompt, enc._answer(ex["ref"]), rng.choice(bad), 3))
        elif ex["kind"] == "cb":
            q_ids = prompt[1:-1]
            act = [sp["<|search|>"]] + q_ids + [sp["<|/search|>"]]
            search.append((prompt, act, rng.choice(bad), 2))
    rng.shuffle(search)
    cap = int(len(out) * MAX_SEARCH_SHARE / (1 - MAX_SEARCH_SHARE))
    out += search[:cap]
    rng.shuffle(out)
    return out


class PrefStream(CB.CalibStream):
    """Пары предпочтений + всё, что было на E (калибровка, поиск, D1, претрейн)."""

    SHARES = {"pref": 0.45, "calib": 0.2, "wiki": 0.2, "d1": 0.05,
              "pretrain": 0.1}
    KEYS = {**CB.CalibStream.KEYS, "pref": 0.0}

    def __init__(self, seq_len: int, seed: int = 0, limit: int | None = None):
        super().__init__(seq_len, seed=seed, limit=limit)
        from tokenizers import Tokenizer

        import config
        from data import search_sft as SS
        enc = SS.Encoder(Tokenizer.from_file(str(config.TOKENIZER_OUT / "synth1.json")))
        data = json.loads(CB.CACHE.read_text())
        self.pairs = pairs(data, enc)
        if limit:
            self.pairs = self.pairs[:limit]
        self.p_epoch = self.p_index = 0

    def describe(self) -> str:
        kinds = [0, 0, 0, 0]
        for p in self.pairs:
            kinds[p[3]] += 1
        return (f"смесь F: пар {len(self.pairs):,} (отказ>выдумка {kinds[0]:,}, "
                f"верно>неверно {kinds[1]:,}, поиск>выдумка {kinds[2]:,}, "
                f"эталон>неверно {kinds[3]:,}); "
                + super().describe())

    def state(self) -> dict:
        return {**super().state(), "sft": "f", "p_epoch": self.p_epoch,
                "p_index": self.p_index}

    def restore(self, state: dict) -> None:
        if state.get("sft") != "f":
            raise SystemExit("снимок не от этапа F — нужен --new-stage")
        super().restore({**state, "sft": "e"})
        self.p_epoch, self.p_index = state.get("p_epoch", 0), state.get("p_index", 0)

    def _pref_rows(self):
        import config
        width = self.seq_len + 1
        eot = config.EOT_ID
        ids, mask, pref, k = [], [], [], 0
        while True:
            order = list(range(len(self.pairs)))
            random.Random(self.seed * 17 + self.p_epoch).shuffle(order)
            while self.p_index < len(order):
                prompt, good, bad, _ = self.pairs[order[self.p_index]]
                self.p_index += 1
                a = prompt + good + [eot]
                b = prompt + bad + [eot]
                if len(a) + len(b) > width:
                    continue
                if len(ids) + len(a) + len(b) > width or k >= self.cfg_pairs():
                    pad = width - len(ids)
                    yield {"tokens": ids + [config.PAD_ID] * pad,
                           "loss_mask": mask + [0] * pad,
                           "pref": pref + [0.0] * pad}
                    ids, mask, pref, k = [], [], [], 0
                k += 1
                tail_a = len(good) + 1
                tail_b = len(bad) + 1
                ids += a + b
                # Язык учится только на лучшем; ORPO видит оба.
                mask += [0] * len(prompt) + [1] * tail_a + [0] * len(prompt) + [0] * tail_b
                pref += ([0.0] * len(prompt) + [float(k)] * tail_a
                         + [0.0] * len(prompt) + [float(-k)] * tail_b)
                self.docs += 1
            self.p_epoch += 1
            self.p_index = 0

    @staticmethod
    def cfg_pairs() -> int:
        from synth1.config import DEFAULT
        return DEFAULT.pref_pairs

    def sources(self) -> dict:
        return {**super().sources(), "pref": self._pref_rows()}
