"""Проверки на нескольких устройствах. Отдельный файл — из-за одной строки.

Восемь поддельных устройств включаются флагом XLA, а он читается при первом
импорте jax и позже не меняется. Поэтому эти проверки нельзя дописать в
`test_model.py`: там jax уже поднят на одном устройстве.

Зачем вообще. Сессия 15.09.2026 потеряла ускорение ровно здесь: сплавленное
внимание собиралось на одном чипе и падало на восьми —

    NotImplementedError: Mosaic kernels cannot be automatically partitioned.
    Please wrap the call in a shard_map.

Проба перед обучением это поймала и увела прогон на блочный путь, так что
сессия не пропала. Но поймать такое должна была проверка на этой машине, а не
чип: разница между «одно устройство» и «восемь» — это разная компиляция, и
дешёвых способов увидеть её ровно два, флаг XLA и дисциплина его применять.

    py -3 synth1/test_sharded.py
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys

# ДО импорта jax. Иначе устройство останется одно и проверка станет пустой.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + " --xla_force_host_platform_device_count=8").strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from synth1 import layers as L  # noqa: E402
from synth1 import model as M  # noqa: E402
from synth1 import sharding as S  # noqa: E402
from synth1.config import ModelConfig  # noqa: E402

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = ""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'ok  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))


# Формы взяты так, чтобы ядро согласилось считать: блок 128 дорожек — его
# минимум, длина обязана делиться на блок.
CFG = ModelConfig(
    vocab_size=256, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=32,
    n_experts=4, n_shared_experts=1, d_ff=128, top_k=2,
    max_seq_len=512, window=128, global_every=2, mod_every=2, mod_block=64,
    attn_impl="splash", splash_interpret=True, splash_block=128,
)


def main():
    mesh = S.make_mesh()
    n = mesh.shape[S.AXIS]
    print(f"\nустройств: {len(jax.devices())}, сетка {dict(mesh.shape)}")
    check("сетка поднялась на восьми устройствах", n == 8, str(n))
    check("сетка доступна вниманию", S.current_mesh() is not None)

    blocked = dataclasses.replace(CFG, attn_impl="blocked")
    b, t = 8, 256
    key = jax.random.key(0)
    params = L.init_attention(key, CFG)
    x = jax.random.normal(jax.random.fold_in(key, 1), (b, t, CFG.d_model),
                          jnp.float32) * 0.3
    x = jax.device_put(x, S.batch_sharding(mesh))
    sin, cos = L.rope_tables(CFG.head_dim, CFG.max_seq_len, CFG.rope_base)

    print("\n1. Сплавленное внимание с разрезанным батчем")
    for label, win in (("оконный слой", CFG.window), ("глобальный слой", None)):
        f_b = jax.jit(lambda p, xx, w=win: L.attention(p, xx, sin, cos, w, blocked))
        f_s = jax.jit(lambda p, xx, w=win: L.attention(p, xx, sin, cos, w, CFG))
        try:
            a, c = f_b(params, x), f_s(params, x)
        except Exception as exc:                       # noqa: BLE001
            check(f"собралось: {label}", False, f"{type(exc).__name__}: {str(exc)[:90]}")
            continue
        check(f"собралось: {label}", True)
        gap = float(jnp.max(jnp.abs(a - c)))
        scale = float(jnp.max(jnp.abs(a)))
        check(f"выход совпал с блочным: {label}", gap < 1e-4 * max(scale, 1e-3),
              f"расхождение {gap:.2e} при масштабе {scale:.3f}")
        check(f"раскладка выхода не поехала: {label}", a.sharding == c.sharding,
              str(c.sharding))

        gb = jax.jit(jax.grad(lambda p, w=win: jnp.sum(
            L.attention(p, x, sin, cos, w, blocked) ** 2)))(params)
        gs = jax.jit(jax.grad(lambda p, w=win: jnp.sum(
            L.attention(p, x, sin, cos, w, CFG) ** 2)))(params)
        worst = max(float(jnp.max(jnp.abs(gb[k] - gs[k]))
                          / jnp.maximum(jnp.max(jnp.abs(gb[k])), 1e-9))
                    for k in ("wq", "wk", "wv", "wo"))
        check(f"градиенты совпали: {label}", worst < 1e-3, f"относительно {worst:.2e}")

    print("\n2. Шаг целиком считается и сходится с блочным")
    # Главное здесь не числа, а то, что вся модель компилируется на восьми
    # устройствах: внимание внутри shard_map, эксперты и головы — снаружи.
    mp = M.init_model(jax.random.key(3), CFG)
    mp = S.distribute(mp, mesh)
    tokens = jax.random.randint(jax.random.key(4), (b, t + 1), 1, CFG.vocab_size)
    tokens = jax.device_put(tokens, S.batch_sharding(mesh))
    try:
        loss_s = jax.jit(lambda p, tk: M.loss_fn(p, {"tokens": tk}, CFG)[0])(mp, tokens)
        loss_b = jax.jit(lambda p, tk: M.loss_fn(p, {"tokens": tk}, blocked)[0])(mp, tokens)
        check("шаг собрался со сплавленным вниманием", bool(jnp.isfinite(loss_s)),
              f"loss={float(loss_s):.4f}")
        check("лосс тот же, что на блочном пути",
              abs(float(loss_s) - float(loss_b)) < 1e-2,
              f"{float(loss_s):.4f} против {float(loss_b):.4f}")
    except Exception as exc:                           # noqa: BLE001
        check("шаг собрался со сплавленным вниманием", False,
              f"{type(exc).__name__}: {str(exc)[:90]}")

    print("\n3. Лосс под shard_map (synth1/fsdp.py) совпадает с обычным jit")
    # Батч без паддинга: тогда среднее по чипам и среднее по токенам — одно и
    # то же, и расходиться может только округление.
    from synth1 import fsdp as F
    loss_plain = lambda p, bt: M.loss_fn(p, bt, blocked)  # noqa: E731
    loss_fsdp = F.sharded_loss(blocked, mesh, mp)
    batch = {"tokens": tokens}
    (lp, mp_), gp = jax.jit(jax.value_and_grad(loss_plain, has_aux=True))(mp, batch)
    (lf, mf_), gf = jax.jit(jax.value_and_grad(loss_fsdp, has_aux=True))(mp, batch)
    check("лосс совпал", abs(float(lp) - float(lf)) < 1e-3,
          f"{float(lf):.5f} против {float(lp):.5f}")
    check("загрузка экспертов совпала",
          float(jnp.max(jnp.abs(mp_["load"] - mf_["load"]))) < 1e-5)

    # Эталон — тот же шаг на одном устройстве. Сравнивать два восьмичиповых
    # пути между собой бессмысленно: оба отходят от эталона на шум bf16 и на
    # смену выбора в top-k (MoD, роутер), и мера здесь — не отойти дальше,
    # чем отходит прежний путь.
    host = jax.device_get(mp)
    g1 = jax.jit(jax.grad(lambda p, bt: loss_plain(p, bt)[0]))(
        host, jax.device_get(batch))
    errs_p, errs_f = [], []
    for a, c, ref in zip(jax.tree.leaves(gp), jax.tree.leaves(gf),
                         jax.tree.leaves(g1)):
        scale = float(jnp.max(jnp.abs(ref))) + 1e-12
        errs_p.append(float(jnp.max(jnp.abs(jnp.asarray(a) - ref))) / scale)
        errs_f.append(float(jnp.max(jnp.abs(jnp.asarray(c) - ref))) / scale)
    mean_p = sum(errs_p) / len(errs_p)
    mean_f = sum(errs_f) / len(errs_f)
    check("градиенты не дальше от эталона, чем у прежнего пути",
          mean_f < 1.5 * mean_p + 1e-3,
          f"среднее отклонение {mean_f:.2e} против {mean_p:.2e}")
    want = S.param_sharding(mp, mesh)
    # Головка ответа до этапа E в лосс не входит: её градиент — константный
    # ноль, и раскладку XLA ему выбирает какую хочет.
    same_layout = all(g.sharding.spec == w.spec for g, w in
                      zip(jax.tree.leaves(gf), jax.tree.leaves(want))
                      if bool(jnp.any(g != 0)))
    check("градиенты разрезаны так же, как параметры", same_layout)

    print(f"\n{'=' * 52}")
    print(f"пройдено {len(PASSED)}, провалено {len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print(f"  FAIL {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
