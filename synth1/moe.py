"""Слой экспертов с воротами по неуверенности.

Обычный MoE выбирает фиксированные top-k экспертов на каждый токен. Наш выбирает
одного, когда токен лёгкий, и двух, когда трудный — решает это сигнал `u` от
эпистемической головки (ARCHITECTURE.md, раздел 3.2).

При обучении считаются оба: форма графа на TPU обязана быть постоянной. Экономия
достаётся инференсу, где второго эксперта можно просто не звать, если его ворота
закрыты. Это не отговорка, а обычное устройство разреженных слоёв: обучение
платит за гибкость, генерация её тратит.

Балансировка — без вспомогательного лосса. Вместо штрафа в градиенте у роутера
есть обучаемый смещением bias, который двигается вне градиента по фактической
загрузке экспертов. Aux-loss тянет роутер к равномерности прямо в градиенте и
портит качество там, где выбор должен быть решительным.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from synth1.layers import ACCUM, init_ffn, init_linear, swiglu


def init_moe(key, cfg) -> dict:
    keys = jax.random.split(key, cfg.n_experts + cfg.n_shared_experts + 1)
    params = {
        # Роутер маленький, но инициализация у него важна: слишком большая — и
        # на первых шагах все токены уедут к одному эксперту, а вытащить их
        # обратно нечем, балансировка работает медленнее, чем схлопывается выбор.
        "router": init_linear(keys[0], cfg.d_model, cfg.n_experts, scale=0.1),
        # Смещение балансировки. Не параметр в смысле градиента: обновляется
        # руками в шаге обучения, поэтому хранится отдельной веткой дерева.
        "router_bias": jnp.zeros((cfg.n_experts,)),
        # Эксперты пачкой: ось 0 — номер эксперта. Так их считает один einsum,
        # а не цикл на двенадцать матмулей.
        "experts": {
            name: jnp.stack([
                init_ffn(keys[1 + i], cfg.d_model, cfg.d_ff, cfg.n_layers)[name]
                for i in range(cfg.n_experts)
            ])
            for name in ("gate", "up", "down")
        },
    }
    if cfg.n_shared_experts:
        params["shared"] = [
            init_ffn(keys[1 + cfg.n_experts + i], cfg.d_model, cfg.d_ff, cfg.n_layers)
            for i in range(cfg.n_shared_experts)
        ]
    return params


def _assign_slots(top_idx: jax.Array, cfg, capacity: int):
    """Кто в какой слот какого эксперта попадёт.

    top_idx: [B, T, K] — выбранные эксперты. Возвращает (slot, valid, expert),
    где slot — плоский номер места в буфере всех экспертов, [B, T*K].

    Слоты нужны из-за постоянства формы: эксперт получает ровно capacity
    токенов, лишние отбрасываются — их вклад в выход ноль, токен идёт дальше
    только по residual. Это цена статического графа на TPU.

    Первая версия строила маску [B, T, K, E, C] и раскладывала через einsum.
    Это стоило 320 МБ на слой при батче 8 и длине 2048, то есть больше, чем
    сами активации; здесь тот же результат обходится в 63 МБ, потому что
    хранится не «кто куда» для всех пар, а один номер слота на назначение.
    """
    b, t, k = top_idx.shape
    e = cfg.n_experts

    onehot = jax.nn.one_hot(top_idx, e, dtype=jnp.int32)          # [B,T,K,E]
    flat = onehot.reshape(b, t * k, e)
    # Очередь к эксперту: сколько токенов заняли его слоты раньше нас. Порядок
    # обхода — по позиции в последовательности, затем по номеру выбора.
    queue = jnp.cumsum(flat, axis=1) - flat                       # [B,T*K,E]

    expert = top_idx.reshape(b, t * k)
    position = jnp.take_along_axis(queue, expert[..., None], axis=2)[..., 0]
    valid = position < capacity
    slot = expert * capacity + position

    # Невалидные назначения уводим за границу буфера: scatter с mode="drop" их
    # выбросит, gather с mode="fill" вернёт ноль. Так не нужен ни отдельный
    # мусорный слот, ни ветвление.
    return jnp.where(valid, slot, e * capacity), valid, expert


@jax.custom_vjp
def permute_rows(x: jax.Array, fwd: jax.Array, bwd: jax.Array) -> jax.Array:
    """y[b, i] = x[b, fwd[b, i]], ноль там, где fwd указывает за край.

    fwd и bwd — взаимно обратные частичные перестановки строк: если
    fwd[b, i] = j и j в пределах x, то bwd[b, j] = i, и наоборот. Тогда
    производная по x — это та же операция в обратную сторону, и обратный
    проход обходится gather'ом, а не scatter-add, которым JAX транспонирует
    gather по умолчанию. На TPU scatter строк — последовательная петля.
    """
    return _take_rows(x, fwd)


def _take_rows(x, idx):
    n = x.shape[1]
    safe = jnp.minimum(idx, n - 1)
    rows = jnp.take_along_axis(x, safe[..., None], axis=1)
    return jnp.where((idx < n)[..., None], rows, jnp.zeros((), x.dtype))


def _permute_fwd(x, fwd, bwd):
    return _take_rows(x, fwd), (fwd, bwd)


def _permute_bwd(res, g):
    fwd, bwd = res
    zero = lambda a: np.zeros(a.shape, jax.dtypes.float0)  # noqa: E731
    return _take_rows(g, bwd), zero(fwd), zero(bwd)


permute_rows.defvjp(_permute_fwd, _permute_bwd)


def inverse_rows(idx: jax.Array, size: int) -> jax.Array:
    """Обратная перестановка: для каждой из size строк — откуда она пришла.

    idx [B, N]: куда едет строка i (за край — никуда). Возвращает [B, size],
    где строка, в которую не приехал никто, указывает на N, то есть за край.
    Scatter здесь есть, но по одному целому на строку, а не по строке в 640
    каналов.
    """
    b, n = idx.shape
    rows = jnp.arange(b)[:, None]
    inv = jnp.full((b, size + 1), n, jnp.int32)
    inv = inv.at[rows, jnp.minimum(idx, size)].set(
        jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32), (b, n)))
    return inv[:, :size]


def moe(params: dict, x: jax.Array, u: jax.Array | None, cfg,
        capacity_factor: float = 1.25):
    """Разреженный FFN. Возвращает (выход, служебные величины для лосса).

    u: [B, T] — предсказанная неуверенность, значения в (0, 1). None выключает
    ворота и превращает слой в обычный top-k MoE.
    """
    b, t, d = x.shape
    e, k = cfg.n_experts, cfg.top_k
    capacity = max(1, int(capacity_factor * k * t / e))

    router_logits = jnp.einsum("btd,de->bte", x, params["router"].astype(x.dtype))
    router_logits = router_logits.astype(ACCUM)

    # Выбираем по смещённым логитам, а взвешиваем по чистым: смещение — это
    # регулировщик очереди, ему нечего делать в величине вклада эксперта.
    biased = router_logits + params["router_bias"].astype(ACCUM)
    _, top_idx = jax.lax.top_k(biased, k)                          # [B,T,K]

    probs = jax.nn.softmax(router_logits, axis=-1)                 # [B,T,E]
    weights = jnp.take_along_axis(probs, top_idx, axis=-1)         # [B,T,K]

    # Нормировка идёт до ворот, а не после: ворота обязаны убирать вклад
    # второго эксперта, а не перекладывать его на первого.
    if cfg.router_renorm:
        weights = weights / jnp.maximum(
            jnp.sum(weights, axis=-1, keepdims=True), 1e-9)

    # Ворота второго эксперта. При низкой неуверенности множитель уходит в ноль,
    # и второй эксперт перестаёт влиять на выход — на инференсе его не считаем.
    if cfg.entropy_gate and u is not None and k >= 2:
        gate2 = jax.nn.sigmoid(
            (u.astype(ACCUM) - cfg.entropy_gate_threshold) / cfg.entropy_gate_temp
        )                                                          # [B,T]
        gate = jnp.concatenate(
            [jnp.ones_like(weights[..., :1]),
             jnp.broadcast_to(gate2[..., None], weights[..., 1:].shape)],
            axis=-1,
        )
        weights = weights * gate
    else:
        gate2 = None

    slot, valid, expert = _assign_slots(top_idx, cfg, capacity)    # [B, T*K]
    rows = jnp.arange(b)[:, None]

    # Собрать входы экспертов: каждое назначение кладёт копию своего токена в
    # свой слот. Токен, выбранный дважды, попадает в два разных слота.
    src = jnp.broadcast_to(x[:, :, None, :], (b, t, k, d)).reshape(b, t * k, d)
    if cfg.moe_dispatch == "gather":
        owner = inverse_rows(slot, e * capacity)                   # [B, E*C]
        xe = permute_rows(src, owner, slot)
    else:
        xe = jnp.zeros((b, e * capacity, d), x.dtype).at[rows, slot].set(
            src, mode="drop")
    xe = xe.reshape(b, e, capacity, d)

    experts = params["experts"]
    if cfg.moe_fold_batch:
        # Батч сворачивается в ось слотов: [B,E,C,D] → [E, B*C, D]. Математика
        # та же, но у einsum'а остаётся одна батчевая ось (эксперт) вместо двух,
        # и веса эксперта не нужно размножать по батчу. Замер на CPU-HLO
        # 14.09.2026: трафик блока 4.37 → 3.51 ГБ, интенсивность 55.5 → 69.1
        # флоп/байт, расхождение результатов 0.002 (округление bf16).
        #
        # По умолчанию выключено, и вот почему: ось B — та самая, по которой
        # батч разрезан между чипами. Сворачивая её внутрь, мы просим XLA
        # переразложить массив, и на восьмёрке это может стоить больше, чем
        # экономит. На CPU такой цены не видно вовсе. Строка в `speed.py`
        # называется «эксперты со свёрткой батча» — сначала число, потом
        # умолчание.
        z = jnp.swapaxes(xe, 0, 1).reshape(e, b * capacity, d)
        gate_h = jnp.einsum("ekd,edf->ekf", z, experts["gate"].astype(x.dtype))
        up_h = jnp.einsum("ekd,edf->ekf", z, experts["up"].astype(x.dtype))
        y_folded = jnp.einsum("ekf,efd->ekd", jax.nn.silu(gate_h) * up_h,
                              experts["down"].astype(x.dtype))
        ye = jnp.swapaxes(y_folded.reshape(e, b, capacity, d), 0, 1)
    else:
        gate_h = jnp.einsum("becd,edf->becf", xe, experts["gate"].astype(x.dtype))
        up_h = jnp.einsum("becd,edf->becf", xe, experts["up"].astype(x.dtype))
        ye = jnp.einsum("becf,efd->becd", jax.nn.silu(gate_h) * up_h,
                        experts["down"].astype(x.dtype))

    # Разложить обратно и сложить вклады: выброшенные назначения дают ноль.
    ye_flat = ye.reshape(b, e * capacity, d)
    if cfg.moe_dispatch == "gather":
        taken = permute_rows(ye_flat, slot, owner)                 # [B,T*K,D]
    else:
        taken = ye_flat.at[rows, slot].get(mode="fill", fill_value=0.0)
    scale = (weights.reshape(b, t * k) * valid.astype(weights.dtype))
    y = (taken * scale[..., None].astype(taken.dtype)).reshape(b, t, k, d).sum(2)

    # Норма разреженной ветви снимается здесь, до общего эксперта: их отношение
    # и есть ответ на вопрос, работает ли MoE. Замер 02.09.2026 дал 22.3% при
    # 1.1B параметров из 1.23B в маршрутизируемых экспертах, и с обучением доля
    # не росла, а падала (27.4% на step-14000). Пока это самое дорогое число в
    # проекте, и держать его в разовом скрипте оказалось ошибкой: скрипт
    # потерялся вместе со scratchpad, а число нужно на каждом прогоне. Две
    # свёртки по [B,T,D] против einsum'ов экспертов не стоят ничего.
    routed_norm = jnp.mean(jnp.linalg.norm(y.astype(ACCUM), axis=-1))
    shared_norm = jnp.zeros((), ACCUM)
    if cfg.n_shared_experts:
        shared_out = sum(swiglu(shared, x) for shared in params["shared"])
        shared_norm = jnp.mean(jnp.linalg.norm(shared_out.astype(ACCUM), axis=-1))
        y = y + shared_out

    # Служебное: z-loss держит логиты роутера в узде, load нужен шагу обучения
    # для сдвига router_bias, dropped — диагностика переполнения слотов.
    #
    # Загрузка считается по СПРОСУ, до отбрасывания по вместимости. Раньше
    # считалась по принятым, и это ломало балансировку именно там, где она
    # нужна: эксперт, которого выбрали втрое чаще положенного, принимает ровно
    # вместимость, а остальное отбрасывается — и в load он выглядит почти
    # нормально загруженным. Сигнал насыщается на потолке, смещение двигается
    # еле-еле, перекос не рассасывается никогда.
    #
    # Замер 02.09.2026 на снимке step-14000: в слое 0 один эксперт забирал 97%
    # токенов, сбрасывалось 64% назначений, а смещение балансировки уехало до
    # -9.8 и всё равно не помогло. По спросу такой перекос виден сразу.
    demand = jax.nn.one_hot(expert, e, dtype=ACCUM).sum(axis=(0, 1))   # [E]
    accepted = jax.nn.one_hot(expert, e, dtype=ACCUM) * valid[..., None]
    routed = accepted.sum(axis=(0, 1))                             # [E]
    assigned = jnp.array(b * t * k, dtype=ACCUM)
    aux = {
        "router_z_loss": jnp.mean(jax.nn.logsumexp(router_logits, axis=-1) ** 2),
        "load": demand / jnp.maximum(demand.sum(), 1.0),
        "dropped": 1.0 - routed.sum() / jnp.maximum(assigned, 1.0),
        "gate2_open": jnp.mean(gate2) if gate2 is not None else jnp.array(1.0),
        # Доля разреженной ветви в выходе слоя: |routed| / (|routed| + |shared|).
        # Без общего эксперта она равна единице по определению.
        "expert_share": routed_norm / jnp.maximum(routed_norm + shared_norm, 1e-9),
    }
    return y, aux


def update_router_bias(bias: jax.Array, load: jax.Array, lr: float) -> jax.Array:
    """Сдвиг балансировки. Вне градиента — это регулировщик, а не параметр.

    Перегруженному эксперту смещение уменьшаем, недогруженному увеличиваем, шаг
    фиксированный по знаку. По величине ошибки шагать нельзя: загрузка шумит от
    батча к батчу, и пропорциональный шаг раскачивает роутер.
    """
    target = 1.0 / load.shape[0]
    return bias - lr * jnp.sign(load - target)
