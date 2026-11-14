"""Оптимизатор: Muon для матриц, AdamW для всего остального.

Почему Muon. Его состояние — один momentum на параметр, тогда как AdamW держит
два. На нашей модели это 5 ГБ против 10 ГБ, а чекпоинт уезжает на Hub после
каждой сессии по каналу, который мы уже измерили. Плюс по данным он эффективнее
AdamW на шаг — на нём обучен Kimi K2.

Идея Muon в одну строку: обычный momentum, но перед применением матрица
обновления ортогонализуется. Градиент матрицы обычно почти вырожден — почти вся
норма сидит в нескольких направлениях, и шаг двигает их, а не остальные.
Ортогонализация выравнивает сингулярные числа, и шаг идёт по всем направлениям
разом.

Что чем оптимизируется:

    Muon    2D-матрицы: проекции внимания, эксперты, MTP
    AdamW   всё, у чего нет двух осей или чья геометрия не матричная:
            embedding (это таблица поиска, а не преобразование), нормы,
            роутеры, эпистемические головки

Роутеры отданы AdamW сознательно: матрица [d_model, n_experts] почти вектор,
ортогонализовать там нечего, а сходиться ему надо быстро.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from synth1.layers import ACCUM

# Коэффициенты пятишаговой итерации Ньютона–Шульца. Подобраны так, чтобы
# сингулярные числа сходились к единице за пять шагов из любого старта в (0,1].
NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5


@dataclasses.dataclass(frozen=True)
class OptConfig:
    # Пиковый LR понижен втрое 05.09.2026 — по замеру, не по вкусу. Проба на
    # step-50000: тысяча шагов с одного снимка, одни и те же батчи, менялся
    # только множитель. Отложенный лосс 2.3176 →
    #     множитель 1.0 (было 0.02):  2.2855, то есть -0.0321
    #     множитель 0.3 (стало):      2.0975, то есть -0.2202
    # Впятеро больше за те же шаги. Дело не только в скорости: на 1.0 лосс
    # ходил вверх-вниз (первые 500 шагов вообще вверх), на 0.3 спускался на
    # каждом из четырёх замеров. Это подпись слишком крупного шага, и под
    # Muon она резче обычного: Ньютон-Шульц выравнивает сингулярные числа,
    # поэтому длина шага почти не зависит от величины градиента, и плато
    # само задаёт уровень, на котором лосс встаёт.
    #
    # Оговорка, записанная вместе с числом: из -0.2202 первые -0.1682 пришлись
    # на первые 250 шагов. Похоже на выход из шумного режима, а не на
    # устойчиво более быстрый спуск. Держится ли выигрыш на длинной дистанции
    # — видно по отложенному лоссу следующей сессии, он печатается каждую
    # тысячу шагов.
    lr_muon: float = 0.006
    lr_adam: float = 9e-4
    momentum: float = 0.95
    nesterov: bool = True
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Ортогонализация — не бесплатная. На целевой конфигурации пять итераций
    # Ньютона–Шульца по всем матрицам экспертов стоят около 20 TFLOP за шаг при
    # 32 TFLOP самого шага, то есть больше половины сверху. Ручки здесь для
    # того, чтобы это можно было померить и решить, а не платить вслепую:
    # ns_steps укорачивает итерацию, use_muon=False возвращает всех под AdamW.
    ns_steps: int = 5
    use_muon: bool = True

    # WSD: warmup, плато, отжиг. Косинус требует знать полную длину заранее, а
    # у нас она набирается сессиями по девять часов и может кончиться раньше.
    # С WSD прогон продолжается с любого места, а если бюджет обрубят — запускаем
    # отжиг и получаем рабочую модель, а не полуфабрикат.
    warmup_steps: int = 2_000
    decay_frac: float = 0.12
    lr_floor: float = 0.01      # доля от пиковой, ниже которой не опускаемся
    # Шаг, с которого считается разогрев. 0 — с начала обучения. Новый этап
    # (pretrain.py --new-stage) стартует с отожжённого снимка и разогревает LR
    # заново от своего первого шага.
    warmup_from: int = 0


def newton_schulz(g: jax.Array, steps: int = NS_STEPS) -> jax.Array:
    """Ортогонализация матрицы обновления. Считается в bf16 — точности хватает."""
    a, b, c = NS_COEFFS
    x = g.astype(jnp.bfloat16)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.T

    x = x / (jnp.linalg.norm(x) + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * (s @ s)) @ x

    if transposed:
        x = x.T
    return x.astype(g.dtype)


def _orthogonalize_all(directions: dict, muon_paths: set, steps: int) -> dict:
    """Ньютон–Шульц для всех Muon-матриц разом, пакетами по форме.

    По одному тензору — как было до 04.10.2026 — это плохо ложится на восемь
    чипов. Эксперты разрезаны по оси d_ff, а итерация сворачивает именно её
    (x @ x.T), поэтому каждая из пяти итераций кончалась all-reduce'ом
    [12, 640, 640] в fp32: ~5 ГБ коллективов на шаг (разбор HLO 04.10).

    Здесь матрицы одной формы складываются в одну стопку, стопка режется по
    оси «номер матрицы» — каждый чип ортогонализует свои матрицы целиком, без
    единого обмена внутри итерации. Перекладка туда и обратно — два all-to-all
    по ~240 МБ на чип. Математика та же: каждая матрица нормируется и
    итерируется отдельно, как и прежде.
    """
    from synth1 import sharding as S

    flat, treedef = jax.tree_util.tree_flatten_with_path(directions)
    groups: dict[tuple, list[int]] = {}
    for i, (path, d) in enumerate(flat):
        if jax.tree_util.keystr(path) in muon_paths:
            r, c = d.shape[-2], d.shape[-1]
            groups.setdefault((min(r, c), max(r, c)), []).append(i)

    mesh = S.current_mesh()
    n = mesh.shape[S.AXIS] if mesh is not None else 1
    out = [d for _, d in flat]
    for (r, c), idx in groups.items():
        pieces = []
        for i in idx:
            d = flat[i][1]
            m = d.reshape(-1, d.shape[-2], d.shape[-1])
            if m.shape[-2] > m.shape[-1]:
                m = jnp.swapaxes(m, -1, -2)
            pieces.append(m)
        stack = jnp.concatenate(pieces, axis=0)
        total = stack.shape[0]
        pad = (-total) % n
        if pad:
            stack = jnp.concatenate([stack, jnp.zeros((pad, r, c), stack.dtype)])
        if n > 1:
            from jax.sharding import NamedSharding, PartitionSpec as P
            stack = jax.lax.with_sharding_constraint(
                stack, NamedSharding(mesh, P(S.AXIS, None, None)))
        done = jax.vmap(lambda mat: newton_schulz(mat, steps))(stack)[:total]
        at = 0
        for i, m in zip(idx, pieces):
            k = m.shape[0]
            part = done[at:at + k]
            at += k
            d = flat[i][1]
            if d.shape[-2] > d.shape[-1]:
                part = jnp.swapaxes(part, -1, -2)
            out[i] = part.reshape(d.shape)
    return jax.tree_util.tree_unflatten(treedef, out)


def _is_muon(path: tuple, param: jax.Array, opt: "OptConfig | None" = None) -> bool:
    """Кого ведём Muon'ом. Решение по форме и по имени, а не по одному из двух."""
    if opt is not None and not opt.use_muon:
        return False
    name = jax.tree_util.keystr(path)
    if any(tag in name for tag in ("embed", "router", "epi", "mod", "norm", "bias")):
        return False
    return param.ndim >= 2


def init_state(params: dict, opt: OptConfig) -> dict:
    """Состояние оптимизатора. Muon хранит один буфер, AdamW — два.

    Буферы лежат отдельными деревьями той же формы, что параметры, а не
    словарями в листьях: tree_map сопоставляет деревья по структуре, и словарь
    на месте листа она развернёт в поддерево, после чего структуры разъедутся.

    У Muon-параметров второй буфер — скаляр-заглушка. Структура так остаётся
    единой, а памяти это стоит одного числа на тензор.
    """
    return {
        "step": jnp.array(0, dtype=jnp.int32),
        "m": jax.tree.map(lambda p: jnp.zeros_like(p, dtype=ACCUM), params),
        "v": jax.tree_util.tree_map_with_path(
            lambda path, p: (jnp.zeros((), ACCUM) if _is_muon(path, p, opt)
                             else jnp.zeros_like(p, dtype=ACCUM)),
            params,
        ),
    }


def lr_schedule(step: jax.Array, total_steps: int, opt: OptConfig) -> jax.Array:
    """Warmup — плато — отжиг. Возвращает множитель к пиковому LR."""
    step = step.astype(ACCUM)
    warmup = jnp.clip((step - opt.warmup_from) / max(opt.warmup_steps, 1), 0.0, 1.0)

    decay_start = total_steps * (1.0 - opt.decay_frac)
    # 1 - sqrt(progress): спускается медленнее линейного в начале отжига и
    # быстрее в конце. По замерам WSD это работает лучше линейного спуска.
    progress = jnp.clip((step - decay_start) / max(total_steps - decay_start, 1), 0, 1)
    decay = 1.0 - jnp.sqrt(progress)

    return warmup * jnp.maximum(decay, opt.lr_floor)


def global_norm(tree) -> jax.Array:
    """Норма градиента, устойчивая к выбросам.

    Наивное sum(x²) переполняется раньше, чем сам градиент: элемент 1e20 ещё
    представим в fp32, а его квадрат — уже inf. Дальше norm=inf даёт clip=0,
    и шаг молча становится нулевым; хуже — 0·inf даёт nan, и тогда nan
    расходится по всем весам за один шаг.

    Поэтому делим на максимум перед возведением в квадрат: результат тот же,
    а промежуточные значения не выходят за единицу.
    """
    leaves = jax.tree.leaves(tree)
    peak = jnp.max(jnp.stack([jnp.max(jnp.abs(x.astype(ACCUM))) for x in leaves]))
    safe = jnp.where(peak > 0, peak, 1.0)
    total = sum(jnp.sum(jnp.square(x.astype(ACCUM) / safe)) for x in leaves)
    return safe * jnp.sqrt(total)


def apply_updates(params: dict, grads: dict, state: dict,
                  total_steps: int, opt: OptConfig) -> tuple[dict, dict, dict]:
    """Один шаг. Возвращает (новые параметры, новое состояние, метрики)."""
    step = state["step"] + 1
    scale = lr_schedule(step, total_steps, opt)

    gnorm = global_norm(grads)
    # Клип по глобальной норме: спайк в одном тензоре иначе утащит весь шаг.
    clip = jnp.minimum(1.0, opt.grad_clip / (gnorm + 1e-6))

    # Плохой батч не должен стоить прогона. Если градиент разошёлся, шаг
    # пропускается целиком: веса остаются прежними, буферы momentum тоже.
    # Иначе один nan расползается по всем тензорам и убивает сессию — а сессия
    # у нас девять часов и берётся из недельной квоты.
    healthy = jnp.isfinite(gnorm)
    clip = jnp.where(healthy, clip, 0.0)

    # Первый проход: momentum и направление. Ортогонализация Muon отложена —
    # она идёт ниже одним пакетом на все матрицы сразу (`_orthogonalize_all`).
    def first(path, p, g, m_prev, v_prev):
        g = g.astype(ACCUM) * clip
        if _is_muon(path, p, opt):
            m = opt.momentum * m_prev + g
            direction = g + opt.momentum * m if opt.nesterov else m
            return direction, m, v_prev
        m = opt.b1 * m_prev + (1 - opt.b1) * g
        v = opt.b2 * v_prev + (1 - opt.b2) * jnp.square(g)
        mhat = m / (1 - opt.b1 ** step.astype(ACCUM))
        vhat = v / (1 - opt.b2 ** step.astype(ACCUM))
        return mhat / (jnp.sqrt(vhat) + opt.eps) * opt.lr_adam, m, v

    is_triple = lambda x: isinstance(x, tuple) and len(x) == 3  # noqa: E731
    triples = jax.tree_util.tree_map_with_path(
        first, params, grads, state["m"], state["v"])
    raw = jax.tree.map(lambda x: x[0], triples, is_leaf=is_triple)
    next_m = jax.tree.map(lambda x: x[1], triples, is_leaf=is_triple)
    next_v = jax.tree.map(lambda x: x[2], triples, is_leaf=is_triple)

    muon_paths = {jax.tree_util.keystr(path)
                  for path, p in jax.tree_util.tree_flatten_with_path(params)[0]
                  if _is_muon(path, p, opt)}
    ortho = _orthogonalize_all(raw, muon_paths, opt.ns_steps)

    def finish(path, p, d):
        if jax.tree_util.keystr(path) in muon_paths:
            # Поправка на форму: у широкой матрицы норма обновления после
            # ортогонализации иначе зависела бы от размеров.
            rows, cols = d.shape[-2], d.shape[-1]
            upd = d * jnp.sqrt(jnp.maximum(1.0, rows / cols)) * (opt.lr_muon * scale)
        else:
            upd = d * scale
        # Weight decay не трогает нормы и одномерное: там он тянет масштаб к
        # нулю и ломает нормализацию, а регуляризовать нечего.
        if p.ndim >= 2:
            upd = upd + opt.weight_decay * opt.lr_adam * scale * p.astype(ACCUM)
        return (p.astype(ACCUM) - upd).astype(p.dtype)

    stepped = jax.tree_util.tree_map_with_path(finish, params, ortho)

    # Второе условие здоровья — уже по результату, а не по градиенту. Между
    # конечным градиентом и конечными весами стоит Muon: Newton–Шульц считается
    # в bf16 и на вырожденной матрице способен выдать nan там, где на входе
    # всё было конечно. Один такой шаг отравляет веса навсегда — дальше лосс
    # nan при любом батче, и прогон стоит уже не шага, а сессии.
    healthy = healthy & jnp.all(jnp.stack(
        [jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(stepped)]))

    # На больном шаге возвращаем всё как было — и веса, и буферы. Иначе
    # momentum запомнит выброс и протащит его в следующие шаги.
    keep = lambda new, old: jnp.where(healthy, new, old)  # noqa: E731
    new_params = jax.tree.map(keep, stepped, params)
    new_m = jax.tree.map(keep, next_m, state["m"])
    new_v = jax.tree.map(keep, next_v, state["v"])

    metrics = {"grad_norm": gnorm, "lr_scale": scale, "clipped": 1.0 - clip,
               "skipped": 1.0 - healthy.astype(ACCUM)}
    return new_params, {"step": step, "m": new_m, "v": new_v}, metrics


def count_state_bytes(params: dict, opt: OptConfig) -> dict[str, float]:
    """Сколько весит состояние. Ради этого числа Muon и выбран."""
    muon_p = adam_p = 0
    for path, p in jax.tree_util.tree_flatten_with_path(params)[0]:
        if _is_muon(path, p, opt):
            muon_p += p.size
        else:
            adam_p += p.size
    total = muon_p + adam_p
    return {
        "params_M": total / 1e6,
        "muon_M": muon_p / 1e6,
        "adam_M": adam_p / 1e6,
        # bf16 веса + fp32 master + состояние
        "state_GB": (total * 2 + total * 4 + (muon_p * 4 + adam_p * 8)) / 1024**3,
        "state_GB_if_all_adam": (total * 2 + total * 4 + total * 8) / 1024**3,
    }
