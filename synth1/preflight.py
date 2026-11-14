"""Предполётная проверка: один шаг под микроскопом, до начала прогона.

Смысл в том, чтобы не платить сессией за диагностику. Когда лосс на TPU стал
nan, из лога видно только «nan» — а вопрос «где именно» стоит ещё одного
запуска. Здесь один шаг проходит с проверкой всех промежуточных величин, и если
что-то разошлось, называется первое место по порядку вычисления: слой, тензор,
компонента лосса.

Проверка стоит секунды и запускается по умолчанию. Отключается `--no-preflight`,
когда конфигурация уже отработана и хочется сразу в дело.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from synth1 import model as M
from synth1.layers import rms_norm, rope_tables


def _finite(x) -> bool:
    return bool(jnp.all(jnp.isfinite(x)))


def _stats(x) -> str:
    x32 = x.astype(jnp.float32)
    return (f"min={float(jnp.min(x32)):+.3e} max={float(jnp.max(x32)):+.3e} "
            f"std={float(jnp.std(x32)):.3e}")


def trace_forward(params: dict, tokens: jax.Array, cfg) -> list[str]:
    """Прогон по слоям вручную, без remat, с проверкой после каждого.

    Дублирует forward намеренно: нужен доступ к состоянию между слоями, а
    рабочий forward его не отдаёт — он обёрнут в checkpoint и обязан быть чистым.
    """
    problems = []
    sin, cos = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
    x = jnp.take(params["embed"], tokens, axis=0).astype(jnp.bfloat16)

    if not _finite(x):
        problems.append("embedding: nan сразу после выборки")
        return problems

    for i, layer_params in enumerate(params["layers"]):
        x, aux = M.apply_layer(layer_params, x, i, sin, cos, cfg)
        kind = "глобальный" if M.is_global_layer(i, cfg) else f"окно {cfg.window}"
        mod = ", MoD" if M.uses_mod(i, cfg) else ""
        if not _finite(x):
            problems.append(
                f"слой {i} ({kind}{mod}): активации разошлись, {_stats(x)}")
            return problems
        if aux["u"] is not None and not _finite(aux["u"]):
            problems.append(f"слой {i}: неуверенность u стала nan")
            return problems
        if not _finite(aux["moe"]["router_z_loss"]):
            problems.append(f"слой {i}: логиты роутера разошлись")
            return problems

    h = rms_norm(x, params["norm_out"])
    if not _finite(h):
        problems.append(f"финальная норма: nan, вход {_stats(x)}")
    return problems


def run(params: dict, state: dict, batch: dict, cfg, opt, total_steps: int,
        loss_fn=M.loss_fn, apply_updates=None) -> None:
    """Полная проверка. Печатает отчёт; при беде завершает прогон с диагнозом."""
    print("предполётная проверка:", flush=True)

    # Проверка идёт по срезу батча, а не по всему. Она считается без jit, в три
    # прохода (форвард, лосс, градиент), и на полном батче держит в памяти
    # столько, сколько сам шаг обучения никогда не держит, — то есть падает по
    # памяти там, где обучение прошло бы. Строк берём по числу устройств:
    # меньше нельзя, шардинг батча этого не переживёт. nan виден на любом срезе.
    tokens = batch["tokens"]
    full = tokens.shape[0]
    slim = len(jax.devices())
    if full > slim:
        tokens = tokens[:slim]
        batch = {k: v[:slim] for k, v in batch.items()}
        print(f"  батч урезан до {slim} (из {full})")

    # 1. Прямой проход целиком.
    logits, collect = M.forward(params, tokens[:, :-1], cfg)

    if not _finite(logits):
        print("  логиты разошлись — ищу слой", flush=True)
        for line in trace_forward(params, tokens[:, :-1], cfg):
            print(f"    {line}")
        raise SystemExit(
            "Прямой проход даёт nan. Обучение не начато — чинить модель, "
            "а не расписание."
        )
    print(f"  логиты: {_stats(logits)}")

    if collect.get("u_layers"):
        u = jnp.stack([jnp.mean(v) for v in collect["u_layers"]])
        print(f"  неуверенность по слоям: {float(jnp.min(u)):.3f}"
              f"…{float(jnp.max(u)):.3f}")

    # 2. Лосс покомпонентно: агрегат скрывает, какая часть сломалась.
    loss, metrics = loss_fn(params, batch, cfg)
    broken = [k for k, v in metrics.items()
              if jnp.ndim(v) == 0 and not _finite(v)]
    parts = " ".join(
        f"{k}={float(v):.3f}" for k, v in sorted(metrics.items())
        if jnp.ndim(v) == 0 and k.startswith("loss")
    )
    print(f"  {parts}")
    if broken:
        raise SystemExit(f"Компоненты лосса разошлись: {', '.join(broken)}")

    # 3. Градиенты: какой тензор первым.
    grads = jax.grad(loss_fn, has_aux=True)(params, batch, cfg)[0]
    bad = [jax.tree_util.keystr(p)
           for p, g in jax.tree_util.tree_flatten_with_path(grads)[0]
           if not _finite(g)]
    if bad:
        print(f"  nan в градиентах {len(bad)} тензоров, первые:")
        for name in bad[:5]:
            print(f"    {name}")
        raise SystemExit("Градиенты разошлись при конечном лоссе — "
                         "смотреть на обратный проход, не на данные.")

    gnorm = jnp.sqrt(sum(jnp.sum(jnp.square(g.astype(jnp.float32)))
                         for g in jax.tree.leaves(grads)))
    print(f"  норма градиента: {float(gnorm):.3f}")

    # 4. Шаг оптимизатора: Muon ортогонализует, и это отдельный источник бед.
    if apply_updates is not None:
        new_params, _, _ = apply_updates(params, grads, state, total_steps, opt)
        bad = [jax.tree_util.keystr(p)
               for p, w in jax.tree_util.tree_flatten_with_path(new_params)[0]
               if not _finite(w)]
        if bad:
            print(f"  после шага nan в {len(bad)} тензорах, первые:")
            for name in bad[:5]:
                print(f"    {name}")
            raise SystemExit(
                "Лосс и градиенты конечны, а веса после шага — нет. "
                "Виноват оптимизатор: смотреть Muon и weight decay."
            )
        print("  шаг оптимизатора: веса конечны")

    print("  всё сходится, начинаем", flush=True)
