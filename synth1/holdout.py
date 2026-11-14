"""Отложенный набор: одни и те же токены на каждом замере.

Зачем он нужен отдельно от лосса обучения. Обучающий лосс считается на случайном
батче из смеси шести источников, и русская страница, питоновский файл и школьная
задача дают на одной и той же модели разный лосс. Разброс между соседними
замерами выходит около ±0.25, а улучшение за две тысячи шагов на этом этапе —
сотые. То есть по обучающему лоссу нельзя отличить «учится медленно» от «не
учится вовсе», а это разные диагнозы с разным лечением.

Здесь набор фиксируется: те же документы, те же куски, тот же порядок, тот же
масштаб. Разброс от данных исчезает целиком, и разница в 0.01 становится
видимой сразу.

Набор берётся из `split="valid"` — по два шарда каждого источника, которых
обучение не видит никогда, — и собирается детерминированно от фиксированного
семени. Поэтому число сравнимо не только внутри сессии, но и между сессиями:
шаг 20,000 сегодня и шаг 40,000 через неделю меряются одной линейкой.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from synth1 import data as D
from synth1 import model as M
from synth1.layers import ACCUM

# Семя набора. Менять его нельзя: с другим семенем получится другой набор, и
# все прежние замеры станут несравнимы с новыми.
SEED = 777


def take(mixture: str, seq_len: int, batch_size: int, n_batches: int,
         seed: int = SEED) -> np.ndarray:
    """Собрать набор: [n_batches, batch_size, seq_len + 1]."""
    stream = D.TokenStream(mixture, seq_len=seq_len, split="valid", seed=seed)
    rows = []
    for batch in stream.batches(batch_size):
        rows.append(batch["tokens"])
        if len(rows) >= n_batches:
            break
    return np.stack(rows)


def evaluator(cfg):
    """Замерщик лосса на готовом наборе. Возвращает функцию (params, набор) → лосс.

    Логиты поднимаются кусками, как в обучении: целиком [B, T, V] не влезает в
    чип, и замер не должен быть единственным местом, где прогон падает по памяти.
    Обратного прохода нет, поэтому замер стоит примерно трети шага за батч.
    """

    @jax.jit
    def one(params, tokens, weights=None):
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        mask = (targets != 0).astype(ACCUM)
        if weights is not None:
            mask = mask * weights[:, 1:].astype(ACCUM)
        _, collect = M.forward(params, inputs, cfg, head=False)
        ce, _, _ = M.head_losses(
            collect["h_final"], M._unembed(params, cfg), targets, mask, cfg)
        # Суммы, а не средние: батчи с разным числом настоящих токенов иначе
        # вошли бы в ответ с разным весом.
        return ce, jnp.sum(mask)

    def run(params, batches, shard=None) -> tuple[float, float]:
        """batches — массив [n, B, T+1] или пара (токены, маски) того же вида:
        маска нужна этапам, где лосс считается только по ответу модели."""
        started = time.time()
        ce = tokens = 0.0
        if isinstance(batches, tuple):
            rows = zip(*batches)
        else:
            rows = ((row, None) for row in batches)
        for row, weights in rows:
            t = jnp.asarray(row)
            w = None if weights is None else jnp.asarray(weights)
            if shard is not None:
                t = shard(t)
                w = None if w is None else shard(w)
            c, m = one(params, t, w)
            ce += float(c)
            tokens += float(m)
        return ce / max(tokens, 1.0), time.time() - started

    return run
