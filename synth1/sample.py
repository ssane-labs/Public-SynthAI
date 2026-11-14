"""Генерация на всех чипах: короткие ответы для замеров и данных этапа E.

Кэша ключей и значений нет: каждый новый токен — полный прямой проход по
буферу. Ответы короткие (десятки токенов), а промпты разной длины, поэтому
выигрыш здесь не в кэше, а в том, чтобы считать на восьми чипах сразу и не
гонять паддинг: промпты сортируются по длине и режутся на корзины, у каждой
своя ширина буфера.

Веса лежат на каждом чипе целиком (579M в bf16 — 1.2 ГБ), батч режется по
образцам, и внутри shard_map чипы друг с другом не разговаривают: MoE и MoD
маршрутизируют только свои строки. Раскладку здесь не выводит XLA — по той
же причине, что и в обучении (synth1/fsdp.py).

Кроме токена, на каждом шаге снимается головка ответа `u_answer` в позиции,
из которой токен предсказан:

    u_first   [2] — на первом шаге, то есть на <|assistant|>: «ошибусь ли»
                    до ответа (этап E учит её канал 0);
    u_last    [2] — на шаге, где модель выдала <|eot|>: «ошиблась ли» после
                    ответа (канал 1).
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import config
from synth1 import model as M

AXIS = "gen"
WIDTHS = (256, 512, 1024, 2048, 3072, 4608, 8192)


def _mesh() -> Mesh:
    return Mesh(np.array(jax.devices()), (AXIS,))


class Generator:
    """generate(prompts, ...) → [{"ids", "u_first", "u_last"}] в порядке prompts."""

    def __init__(self, params, cfg, tokens_per_chip: int = 32_768):
        # Таблицы RoPE строятся на max_seq_len: корзине шире неё их не хватит.
        self.cfg = dataclasses.replace(
            cfg, max_seq_len=max(cfg.max_seq_len, WIDTHS[-1]))
        self.mesh = _mesh()
        self.n_dev = self.mesh.devices.size
        self.tokens_per_chip = tokens_per_chip
        self.params = jax.device_put(
            params, NamedSharding(self.mesh, P()))
        self._fns = {}

    def _fn(self):
        cfg = self.cfg
        if "step" in self._fns:
            return self._fns["step"]

        def local(params, rows, pos, temp, bias, key):
            _, collect = M.forward(params, rows, cfg, head=False)
            idx = jnp.arange(rows.shape[0])
            h = collect["h_final"][idx, pos]
            logits = (h @ M._unembed(params, cfg).astype(h.dtype).T).astype(jnp.float32)
            logits = logits + bias[None, :]
            key = jax.random.fold_in(key, jax.lax.axis_index(AXIS))
            sampled = jax.random.categorical(
                key, logits / jnp.maximum(temp, 1e-4)[:, None])
            tok = jnp.where(temp > 0, sampled, jnp.argmax(logits, -1))
            if "u_answer" in collect:
                u = collect["u_answer"][idx, pos].astype(jnp.float32)
            else:
                u = jnp.zeros((rows.shape[0], 2), jnp.float32)
            return tok.astype(jnp.int32), u

        fn = jax.jit(jax.shard_map(
            local, mesh=self.mesh,
            in_specs=(P(), P(AXIS, None), P(AXIS), P(AXIS), P(), P()),
            out_specs=(P(AXIS), P(AXIS, None)), check_vma=False))
        self._fns["step"] = fn
        return fn

    def generate(self, prompts: list[list[int]], max_new: int = 40,
                 temps=None, ban: tuple[int, ...] = (), seed: int = 0,
                 stop: tuple[int, ...] = (), log_every: int = 0):
        """prompts — списки id. temps — температура на промпт (0 — жадно).
        ban — токены, которые нельзя выдавать. Остановка — <|eot|> или stop."""
        temps = np.zeros(len(prompts), np.float32) if temps is None else \
            np.asarray(temps, np.float32)
        bias = np.zeros(self.cfg.vocab_size, np.float32)
        for t in ban:
            bias[t] = -1e9
        bias = jnp.asarray(bias)
        stops = {config.EOT_ID, *stop}
        fn = self._fn()
        key = jax.random.key(seed)
        out: list[dict | None] = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        done_n, calls = 0, 0
        i = 0
        while i < len(order):
            need = len(prompts[order[i]]) + max_new
            width = next((w for w in WIDTHS if w >= need), None)
            if width is None:
                raise ValueError(f"промпт длиннее {WIDTHS[-1] - max_new} токенов")
            per_chip = max(1, self.tokens_per_chip // width)
            batch = per_chip * self.n_dev
            chunk = []
            while i < len(order) and len(chunk) < batch and \
                    len(prompts[order[i]]) + max_new <= width:
                chunk.append(order[i])
                i += 1
            rows = np.zeros((batch, width), np.int32)
            pos = np.zeros(batch, np.int32)
            tmp = np.zeros(batch, np.float32)
            for b, j in enumerate(chunk):
                rows[b, :len(prompts[j])] = prompts[j]
                pos[b] = len(prompts[j]) - 1
                tmp[b] = temps[j]
            done = np.zeros(batch, bool)
            done[len(chunk):] = True
            new = [[] for _ in range(batch)]
            u_first = np.zeros((batch, 2), np.float32)
            u_last = np.full((batch, 2), np.nan, np.float32)
            for t in range(max_new):
                key, sub = jax.random.split(key)
                tok, u = fn(self.params, jnp.asarray(rows), jnp.asarray(pos),
                            jnp.asarray(tmp), bias, sub)
                tok, u = np.asarray(tok), np.asarray(u)
                calls += 1
                if t == 0:
                    u_first = u.copy()
                for b in range(batch):
                    if done[b]:
                        continue
                    if tok[b] in stops or pos[b] + 1 >= width:
                        done[b] = True
                        u_last[b] = u[b]
                        if tok[b] in stops and tok[b] != config.EOT_ID:
                            new[b].append(int(tok[b]))
                        continue
                    pos[b] += 1
                    rows[b, pos[b]] = tok[b]
                    new[b].append(int(tok[b]))
                if done.all():
                    break
            # Не дописал до <|eot|> за max_new — «после ответа» снимаем на
            # последнем выданном токене: простыня тоже ответ, и как раз её
            # головке важно не пропустить.
            unfinished = np.isnan(u_last[:, 0])
            u_last[unfinished] = u[unfinished]
            for b, j in enumerate(chunk):
                out[j] = {"ids": new[b], "u_first": u_first[b].tolist(),
                          "u_last": u_last[b].tolist()}
            done_n += len(chunk)
            if log_every and (done_n // log_every) != ((done_n - len(chunk)) // log_every):
                print(f"  генерация: {done_n:,}/{len(prompts):,}, ширина {width}, "
                      f"вызовов {calls:,}", flush=True)
        return out


def decode(tokenizer, ids: list[int], special: dict[int, str]) -> str:
    """Текст с служебными токенами: словарь их выбрасывает, вставляем руками."""
    text, run = "", []
    for t in ids:
        if t in special:
            text += tokenizer.decode(run) + special[t]
            run = []
        else:
            run.append(t)
    return text + tokenizer.decode(run)


def auroc(scores, labels) -> float:
    """Площадь под ROC: вероятность, что ошибка получит u выше верного ответа."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return math.nan
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    # Ничьи — средним рангом.
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        hit = allv == v
        if hit.sum() > 1:
            ranks[hit] = ranks[hit].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def ece(probs, labels, bins: int = 10) -> float:
    """Ожидаемая ошибка калибровки: |доля ошибок − средний u| по корзинам."""
    p = np.clip(np.asarray(probs, float), 0, 1)
    y = np.asarray(labels, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        hit = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if hit.any():
            total += hit.mean() * abs(y[hit].mean() - p[hit].mean())
    return float(total)
