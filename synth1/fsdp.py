"""Лосс на восьми чипах: FSDP руками, через shard_map.

Почему не автоматически. Раньше лосс шёл под обычным jit, а раскладку XLA
выводил сам — из того, что параметры разрезаны (sharding.py), а батч лежит
по оси `data`. Для внимания и плотных матмулей вывод получался верным. Для
MoE — нет, и это стоило почти всего MFU.

Разбор HLO шага 04.10.2026 (батч 32 × 2048, восемь устройств):

    all-gather  [32,4096,640]   7.0 ГБ   весь батч назначений на каждый чип
    all-reduce  [32,5112,640]   9.2 ГБ   буфер слотов экспертов, f32
    all-reduce  [12,640,32,426] 8.4 ГБ   выход эксперта, частичные суммы
    ...                         ~49 ГБ коллективов на шаг

Scatter в слоты (`moe._assign_slots`) индексирует батч массивом `rows`, и
XLA не узнаёт в нём батчевую ось: проще собрать батч целиком на каждом чипе.
А веса экспертов разрезаны по оси d_ff, которая в down-проекции
сворачивается, — и XLA оставляет веса разрезанными, а частичные суммы
активаций складывает all-reduce'ом. Итог: каждый чип считал MoE для всего
батча, а не для своей восьмой части, и ещё гонял десятки гигабайт по ICI.

Здесь то же самое без догадок. Внутри shard_map каждый чип видит только свой
кусок батча, а веса слоя собираются all-gather'ом в fp32 прямо перед его
счётом, внутри пересчитываемого блока (`model._GATHER`): полные веса живут
один слой, а при обратном проходе собираются заново — ровно то, что обещал
FSDP. Обратный проход all-gather'а — это
reduce-scatter, и градиенты сами ложатся разрезанными так же, как параметры:
оптимизатору менять нечего.

Математика та же, что у `model.loss_fn` на всём батче, с одной оговоркой:
лосс и метрики усредняются по чипам, а не по токенам всего батча. Число
токенов на чип одинаковое, маска режет только паддинг, так что разница — в
долях процента на батчах с паддингом.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from synth1 import model as M
from synth1 import sharding as S


def _spec(sharding) -> P:
    return sharding.spec


def _gather(p, spec: P):
    """Собрать параметр целиком. Неразрезанный (норма, роутер) — как есть."""
    for axis, name in enumerate(spec):
        if name == S.AXIS:
            return jax.lax.all_gather(p, S.AXIS, axis=axis, tiled=True)
    return p


def sharded_loss(cfg, mesh, params_like):
    """Функция (params, batch) → (loss, metrics) с явной раскладкой.

    params_like — дерево той же формы, что параметры (массивы или
    ShapeDtypeStruct): по нему строится раскладка, ровно та, что у
    `sharding.param_sharding`, то есть та, в которой параметры уже лежат.
    """
    specs = jax.tree.map(_spec, S.param_sharding(params_like, mesh))

    # Раскладка слоя зависит только от формы его тензоров, а форм слоёв две:
    # с MoD и без (MTP-слой устроен как слой без MoD). Поэтому раскладку
    # слоя достаточно найти по его структуре.
    layer_specs = {}
    for layer_spec in [*specs["layers"], specs.get("mtp", {}).get("layer")]:
        if layer_spec is not None:
            layer_specs.setdefault(jax.tree.structure(layer_spec), layer_spec)

    def gather_layer(layer_params):
        spec = layer_specs[jax.tree.structure(layer_params)]
        return jax.tree.map(_gather, layer_params, spec)

    def local(params, batch):
        # Всё, кроме слоёв, — словарь, головы, MTP-проекция — собирается
        # сразу: это 0.1 ГБ. Слои собираются каждый у себя (model._GATHER),
        # внутри пересчитываемого блока: полные веса живут один слой.
        full = {k: (v if k == "layers" else jax.tree.map(_gather, v, specs[k]))
                for k, v in params.items()}
        if "mtp" in params:
            full["mtp"] = {k: (v if k == "layer" else
                               jax.tree.map(_gather, v, specs["mtp"][k]))
                           for k, v in params["mtp"].items()}
        previous = M._GATHER
        M._GATHER = gather_layer
        try:
            loss, metrics = M.loss_fn(full, batch, cfg)
        finally:
            M._GATHER = previous
        loss = jax.lax.pmean(loss, S.AXIS)
        metrics = jax.tree.map(lambda x: jax.lax.pmean(x, S.AXIS), metrics)
        return loss, metrics

    def run(params, batch):
        # Раскладка батча — по его ключам: tokens всегда, loss_mask на этапах
        # с маской (data/sft.py). Всё режется по образцам.
        batch_specs = {k: P(S.AXIS, None) for k in batch}
        return jax.shard_map(
            local, mesh=mesh, in_specs=(specs, batch_specs),
            out_specs=(P(), P()))(params, batch)

    return run
