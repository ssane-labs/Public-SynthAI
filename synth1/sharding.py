"""Раскладка модели и данных по восьми чипам TPU.

Схема — FSDP: параметры и состояние оптимизатора разрезаны между чипами, каждый
хранит свою восьмую часть. Перед вычислением XLA сам собирает нужный кусок
(all-gather), после — раскладывает градиенты обратно (reduce-scatter). Писать
эти коллективы руками не нужно: достаточно объявить, что где лежит.

Почему не чистая репликация. Состояние целевой модели — 11.6 ГБ, чип держит 16.
Реплика влезает, но на активации и на батч не остаётся почти ничего, а размер
батча — это прямо количество увиденных токенов за сессию. При FSDP на чип
приходится 1.45 ГБ, и оставшиеся четырнадцать уходят на работу.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

AXIS = "data"


_CURRENT: Mesh | None = None


def current_mesh() -> Mesh | None:
    """Раскладка текущего прогона, если она уже создана.

    Нужна одному месту — сплавленному вниманию. Ядро Pallas XLA разрезать
    между чипами сам не умеет («Mosaic kernels cannot be automatically
    partitioned»), его надо обернуть в shard_map, а для этого нужна сетка.
    Тащить её параметром через каждый слой ради одного вызова не стоит:
    сетка на прогон одна и создаётся раньше модели.
    """
    return _CURRENT


def make_mesh() -> Mesh:
    try:
        devices = jax.devices()
    except RuntimeError as exc:
        # TPU достаётся ровно одному процессу. Сообщение самого JAX про
        # /dev/vfio занимает пол-экрана и не говорит, что делать, а причина
        # почти всегда одна и та же.
        if "vfio" in str(exc) or "Device or resource busy" in str(exc):
            raise SystemExit(
                "TPU занят другим процессом.\n"
                "\n"
                "Чаще всего это само ядро ноутбука: стоит выполнить в нём\n"
                "`import jax` и тронуть устройства, как чип оказывается\n"
                "захвачен, и отдельный процесс обучения его уже не получит.\n"
                "\n"
                "Что делать:\n"
                "  1. Run → Restart session — это освободит чип.\n"
                "  2. Пройти заново ячейки с окружением и кодом проекта.\n"
                "  3. Не выполнять `import jax` в ячейках: проверка числа\n"
                "     чипов в ноутбуке идёт отдельным процессом ровно поэтому.\n"
                "\n"
                "Если ядро ни при чём — значит висит прошлый прогон:\n"
                "  pkill -f train/pretrain.py\n"
            ) from exc
        raise
    global _CURRENT
    _CURRENT = Mesh(devices, (AXIS,))
    return _CURRENT


def _shard_axis(shape: tuple[int, ...], n: int) -> int | None:
    """По какой оси резать тензор.

    Берём самую длинную ось, которая делится на число чипов нацело. Не делится
    ни одна — тензор реплицируется; для норм и головок это и правильно, они
    крошечные.
    """
    candidates = [(dim, i) for i, dim in enumerate(shape) if dim % n == 0 and dim >= n]
    if not candidates:
        return None
    return max(candidates)[1]


def param_sharding(tree, mesh: Mesh):
    """Раскладка дерева параметров. Возвращает дерево NamedSharding."""
    n = mesh.shape[AXIS]

    def spec(p):
        axis = _shard_axis(p.shape, n)
        if axis is None:
            return NamedSharding(mesh, P())
        return NamedSharding(mesh, P(*([None] * axis + [AXIS])))

    return jax.tree.map(spec, tree)


def batch_sharding(mesh: Mesh) -> NamedSharding:
    """Батч режется по образцам: это и есть data parallel."""
    return NamedSharding(mesh, P(AXIS, None))


def check_batch(batch_size: int, mesh: Mesh) -> None:
    """Делится ли батч между чипами. Проверяется до всего остального.

    Иначе ошибка вылезает на первом device_put — то есть после инициализации
    модели, компиляции и предполётной проверки. На TPU это четыре минуты, и
    все они уходят впустую из-за числа, которое видно сразу.
    """
    n = mesh.shape[AXIS]
    if batch_size % n == 0:
        return
    lower = max(n, (batch_size // n) * n)
    upper = ((batch_size // n) + 1) * n
    raise SystemExit(
        f"Батч {batch_size} не делится на {n} чипов — каждому должно достаться "
        f"поровну.\n"
        f"Ближайшие подходящие: --batch-size {lower} или {upper}."
        + (f"\nМеньше {n} взять нельзя: это по одному образцу на чип."
           if batch_size < n else "")
    )


def distribute(tree, mesh: Mesh):
    """Разложить уже созданное дерево по чипам."""
    shardings = param_sharding(tree, mesh)
    return jax.tree.map(lambda x, s: jax.device_put(x, s), tree, shardings)


def report(tree, mesh: Mesh, label: str = "параметры") -> str:
    """Сколько чего досталось одному чипу. Печатается один раз при старте."""
    n = mesh.shape[AXIS]
    sharded = replicated = 0
    for leaf in jax.tree.leaves(tree):
        if _shard_axis(leaf.shape, n) is None:
            replicated += leaf.size
        else:
            sharded += leaf.size
    total = sharded + replicated
    per_chip = sharded / n + replicated
    return (f"{label}: {total / 1e6:.0f}M, "
            f"разрезано {sharded / total:.0%}, "
            f"на чип {per_chip / 1e6:.0f}M "
            f"({per_chip * 4 / 1024**3:.2f} ГБ в fp32)")
