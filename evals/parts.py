"""Разрез шага по частям модели: куда уходят миллисекунды и байты.

`speed.py` меряет шаг целиком и отвечает на вопрос «сколько стоит механизм»,
выключая его. Этот скрипт отвечает на другой вопрос — «из чего сложены 900 мс»,
— и не требует выключать ничего: каждая часть собирается отдельно, на тех же
формах, что видит один чип из восьми.

Зачем отдельный инструмент. Замер 14.09.2026 дал MFU 4.3%, и перебор по
`--sweep` показал, что ни одна ручка конфигурации не объясняет больше 15%:
дорого не что-то одно, дорого всё сразу. При этом ни один предел железа рядом
не стоит — шаг 915 мс против 59 мс по счёту и 73–122 мс по памяти (прикидка по
формам). Значит, теряется не на арифметике и не на полосе, а между ними: на
сплавлении операций, на последовательных мелких ядрах и на пересчёте. Найти
такое можно только разрезом по частям на живом чипе.

Две величины на каждую часть:

* **флоп/байт** — арифметическая интенсивность. У v5e пик 197 TFLOPS bf16 при
  полосе HBM 819 ГБ/с, то есть перелом на 240 флоп/байт. Всё, что ниже, стоит
  ровно столько, сколько весит его трафик, и ускорять там счёт бессмысленно.
* **мс** — время, если под рукой настоящий чип. На CPU печатаются только
  флопы и байты: они из скомпилированного HLO и от железа не зависят.

Запуск:

    python evals/parts.py                  # как на чипе: батч 4 на устройство
    python evals/parts.py --batch-size 8   # столько же, но вдвое больше батч
    python evals/parts.py --no-time        # только флопы и байты, без прогонов
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from synth1 import model as M
from synth1 import moe as MoE
from synth1.config import ABLATION, DEFAULT
from synth1.layers import attention, init_attention, rope_tables

PRESETS = {"default": DEFAULT, "ablation": ABLATION}

# Пик и полоса одного чипа. Нужны, чтобы перевести флопы и байты в проценты от
# железа; если чип другой, число приедет из --peak-tflops / --hbm.
V5E_PEAK_TFLOPS = 197.0
V5E_HBM_GBS = 819.0


def cost(fn, *args) -> tuple[float, float]:
    """Флопы и трафик памяти скомпилированной функции. Из HLO, не с устройства."""
    compiled = jax.jit(fn).lower(*args).compile()
    analysis = compiled.cost_analysis()
    if isinstance(analysis, (list, tuple)):
        analysis = analysis[0]
    return float(analysis.get("flops", 0)), float(analysis.get("bytes accessed", 0))


def timed(fn, *args, iters: int = 5) -> float:
    """Секунды на вызов. Первый прогон — компиляция, он не считается."""
    f = jax.jit(fn)
    out = f(*args)
    jax.block_until_ready(out)
    start = time.perf_counter()
    for _ in range(iters):
        out = f(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / iters


def grad_of(fn):
    """Часть модели так, как её видит обучение: прямой проход плюс обратный."""
    def scalar(*args):
        out = fn(*args)
        out = out[0] if isinstance(out, tuple) else out
        return jnp.sum(out.astype(jnp.float32))
    return jax.value_and_grad(scalar)


def _experts_only(params, xe, cfg):
    """Три einsum'а эксперта без роутера, раскладки и сборки — ровно та же
    математика, что в `moe.py`, и обе укладки, чтобы их можно было сравнить."""
    experts = params["experts"]
    b, e, c, d = xe.shape
    if cfg.moe_fold_batch:
        z = jnp.swapaxes(xe, 0, 1).reshape(e, b * c, d)
        g = jnp.einsum("ekd,edf->ekf", z, experts["gate"].astype(z.dtype))
        up = jnp.einsum("ekd,edf->ekf", z, experts["up"].astype(z.dtype))
        y = jnp.einsum("ekf,efd->ekd", jax.nn.silu(g) * up,
                       experts["down"].astype(z.dtype))
        return jnp.swapaxes(y.reshape(e, b, c, d), 0, 1)
    g = jnp.einsum("becd,edf->becf", xe, experts["gate"].astype(xe.dtype))
    up = jnp.einsum("becd,edf->becf", xe, experts["up"].astype(xe.dtype))
    return jnp.einsum("becf,efd->becd", jax.nn.silu(g) * up,
                      experts["down"].astype(xe.dtype))


def build(cfg, batch: int, seq: int):
    """Части модели с уже подставленными параметрами. Возвращает список
    (имя, сколько таких в модели, функция, аргументы)."""
    key = jax.random.PRNGKey(0)
    k_attn, k_layer, k_mod, k_emb = jax.random.split(key, 4)
    d = cfg.d_model

    x = jnp.zeros((batch, seq, d), jnp.bfloat16)
    u = jnp.full((batch, seq), 0.5, jnp.bfloat16)
    sin, cos = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)

    attn_p = init_attention(k_attn, cfg)
    layer_local = M.init_layer(k_layer, 1, cfg)      # нечётный: без MoD
    layer_mod = M.init_layer(k_mod, 0, cfg)          # чётный: с MoD
    unembed = jax.random.normal(k_emb, (cfg.vocab_size, d), jnp.float32) * 0.02
    targets = jnp.ones((batch, seq), jnp.int32)
    mask = jnp.ones((batch, seq), jnp.float32)

    n_global = cfg.n_layers // cfg.global_every
    n_local = cfg.n_layers - n_global
    n_mod = sum(1 for i in range(cfg.n_layers) if M.uses_mod(i, cfg))

    blocked_cfg = dataclasses.replace(cfg, attn_impl="blocked")
    parts = [
        ("внимание, оконный слой", n_local,
         grad_of(lambda p, h: attention(p, h, sin, cos, cfg.window, cfg)),
         (attn_p, x)),
        ("внимание, глобальный слой", n_global,
         grad_of(lambda p, h: attention(p, h, sin, cos, None, cfg)),
         (attn_p, x)),
        # Прежний путь рядом, чтобы выигрыш сплавленного ядра был виден прямо
        # здесь, а не только в перебор `speed.py`. В сумму шага не идёт.
        ("  для сравнения: оконный блочно", n_local,
         grad_of(lambda p, h: attention(p, h, sin, cos, cfg.window, blocked_cfg)),
         (attn_p, x)),
        ("  для сравнения: глобальный блочно", n_global,
         grad_of(lambda p, h: attention(p, h, sin, cos, None, blocked_cfg)),
         (attn_p, x)),
        ("эксперты, слой без MoD", cfg.n_layers - n_mod,
         grad_of(lambda p, h, uu: MoE.moe(p["moe"], h, uu, cfg)[0]),
         (layer_local, x, u)),
        ("эксперты, слой с MoD", n_mod,
         grad_of(lambda p, h, uu: M._moe_with_mod(p, h, uu, cfg)[0]),
         (layer_mod, x, u)),
        ("слой целиком, оконный", n_local,
         grad_of(lambda p, h: M.apply_layer(p, h, 1, sin, cos, cfg)[0]),
         (layer_local, x)),
        ("голова: логиты и CE", 1,
         grad_of(lambda h, w: M.head_losses(h, w, targets, mask, cfg)[0]),
         (x, unembed)),
    ]

    # Внутренности блока экспертов. Считаются отдельно, потому что байты и
    # время здесь расходятся сильнее всего: scatter в слоты весит треть
    # гигабайта, а на TPU может оказаться развёрнутым циклом и стоить в разы
    # больше, чем весит. По HLO этого не видно — видно только на чипе.
    capacity = max(1, int(1.25 * cfg.top_k * seq / cfg.n_experts))
    e, k = cfg.n_experts, cfg.top_k
    rows_idx = jnp.arange(batch)[:, None]
    slot = jnp.zeros((batch, seq * k), jnp.int32)
    xe = jnp.zeros((batch, e, capacity, d), jnp.bfloat16)
    ye = jnp.zeros((batch, e * capacity, d), jnp.bfloat16)
    top_idx = jnp.zeros((batch, seq, k), jnp.int32)

    def scatter(h, sl):
        src = jnp.broadcast_to(h[:, :, None, :], (batch, seq, k, d))
        src = src.reshape(batch, seq * k, d)
        return jnp.zeros((batch, e * capacity, d), h.dtype).at[rows_idx, sl].set(
            src, mode="drop")

    parts += [
        ("  из них: очередь в слоты", cfg.n_layers,
         lambda ti: MoE._assign_slots(ti, cfg, capacity)[0], (top_idx,)),
        ("  из них: scatter в слоты", cfg.n_layers, scatter, (x, slot)),
        ("  из них: einsum'ы экспертов", cfg.n_layers,
         grad_of(functools.partial(_experts_only, cfg=cfg)),
         (layer_local["moe"], xe)),
        ("  из них: gather из слотов", cfg.n_layers,
         lambda y, sl: y.at[rows_idx, sl].get(mode="fill", fill_value=0.0),
         (ye, slot)),
    ]
    k_full = jax.random.fold_in(key, 7)
    full = jax.eval_shape(lambda kk: M.init_model(kk, cfg), k_full)
    if cfg.mtp:
        # Настоящие веса нужны только двум маленьким частям MTP-хвоста;
        # всю модель ради них не поднимаем.
        k_m = jax.random.split(k_full, 3)
        mtp_params = {
            "embed": jax.random.normal(k_m[0], (cfg.vocab_size, d), jnp.float32) * 0.02,
            "mtp": {"proj": jax.random.normal(k_m[1], (2 * d, d), jnp.float32) * 0.02,
                    "norm_h": jnp.ones((d,)), "norm_e": jnp.ones((d,)),
                    "layer": M.init_layer(k_m[2], cfg.n_layers - 1, cfg)},
        }
        parts.append(
            ("MTP-хвост", 1,
             grad_of(lambda p, h: M._mtp_hidden(p, h, targets, sin, cos, cfg)),
             (mtp_params, x)))

    # Шаг целиком — не для разреза, а для сверки: сумма частей должна объяснять
    # его большую часть, иначе разрез врёт и искать надо не там.
    parts.append(
        ("ВЕСЬ ШАГ (для сверки)", 1,
         jax.value_and_grad(lambda p, t_: M.loss_fn(p, {"tokens": t_}, cfg)[0]),
         (full, jax.ShapeDtypeStruct((batch, seq + 1), jnp.int32))))
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="default", choices=list(PRESETS))
    ap.add_argument("--batch-size", type=int, default=4,
                    help="батч НА ОДИН ЧИП: при обучении с --batch-size 32 на "
                         "восьмёрке чип видит 4")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--no-time", action="store_true",
                    help="только флопы и байты: на CPU прогоны идут минутами и "
                         "ничего не говорят о TPU")
    ap.add_argument("--peak-tflops", type=float, default=V5E_PEAK_TFLOPS)
    ap.add_argument("--hbm", type=float, default=V5E_HBM_GBS, help="ГБ/с на чип")
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    cfg = type(cfg)(**{**cfg.__dict__,
                       "max_seq_len": max(cfg.max_seq_len, args.seq_len)})
    on_tpu = jax.devices()[0].platform == "tpu"
    measure_time = not args.no_time

    print(f"устройство: {jax.devices()[0].device_kind}, "
          f"батч на чип {args.batch_size} × {args.seq_len} = "
          f"{args.batch_size * args.seq_len:,} токенов")
    print(f"перелом счёт/память: {args.peak_tflops * 1e12 / (args.hbm * 1e9):.0f} "
          f"флоп/байт — ниже этого часть упирается в память, а не в чип\n")

    head = (f"{'часть (прямой + обратный)':28} {'шт':>3} {'ГФЛОП':>7} {'ГБ':>6} "
            f"{'флоп/байт':>10} {'предел':>8}")
    if measure_time:
        head += f" {'мс':>7} {'вклад':>7}"
    print(head)
    print("-" * (len(head) + 2))

    rows = []
    for name, count, fn, fn_args in build(cfg, args.batch_size, args.seq_len):
        flops, byts = cost(fn, *fn_args)
        intensity = flops / max(byts, 1)
        # Нижняя граница времени: что медленнее — прочитать байты или посчитать.
        floor = max(flops / (args.peak_tflops * 1e12), byts / (args.hbm * 1e9)) * 1e3
        line = (f"{name:28} {count:>3} {flops/1e9:7.1f} {byts/1e9:6.2f} "
                f"{intensity:10.1f} {floor:7.1f}м")
        ms = None
        abstract = any(isinstance(a, jax.ShapeDtypeStruct)
                       or any(isinstance(l, jax.ShapeDtypeStruct)
                              for l in jax.tree.leaves(a))
                       for a in fn_args)
        if measure_time and not abstract:
            ms = timed(fn, *fn_args, iters=args.iters) * 1e3
            line += f" {ms:7.1f}"
        elif measure_time:
            # Шаг целиком поднимать с настоящими весами незачем: его время
            # меряет speed.py, а здесь нужны только флопы и байты.
            line += f" {'—':>7}"
        rows.append((name, count, flops, byts, floor, ms))
        print(line, flush=True)

    # Сборка шага из частей. «Слой целиком» в сумму не идёт — он дублирует
    # внимание и экспертов и стоит здесь только для сверки: его число должно
    # примерно равняться сумме своих слагаемых.
    def get(name):
        for r in rows:
            if r[0] == name:
                return r
        return None

    parts_of_step = ["внимание, оконный слой", "внимание, глобальный слой",
                     "эксперты, слой без MoD", "эксперты, слой с MoD",
                     "голова: логиты и CE", "MTP-хвост"]
    sum_flops = sum_bytes = sum_ms = 0.0
    for name in parts_of_step:
        r = get(name)
        if r is None:
            continue
        sum_flops += r[2] * r[1]
        sum_bytes += r[3] * r[1]
        if r[5] is not None:
            sum_ms += r[5] * r[1]

    active = cfg.n_params()["active"]
    tokens = args.batch_size * args.seq_len
    whole = get("ВЕСЬ ШАГ (для сверки)")
    print("\nшаг на один чип:")
    print(f"  сумма частей:       {sum_flops/1e12:6.2f} TFLOP, {sum_bytes/1e9:6.1f} ГБ")
    if whole:
        print(f"  шаг целиком:        {whole[2]/1e12:6.2f} TFLOP, {whole[3]/1e9:6.1f} ГБ"
              f"  (части объясняют {sum_bytes/max(whole[3],1):.0%} трафика)")
    ref_flops = whole[2] if whole else sum_flops
    ref_bytes = whole[3] if whole else sum_bytes
    print(f"  6·N·T, по которому считается MFU: {6*active*tokens/1e12:6.2f} TFLOP — "
          f"{6*active*tokens/max(ref_flops,1):.0%} от того, что чип считает на самом деле")
    print(f"  интенсивность шага: {ref_flops/max(ref_bytes,1):5.0f} флоп/байт "
          f"при переломе {args.peak_tflops*1e12/(args.hbm*1e9):.0f}")
    mem_ms = ref_bytes / (args.hbm * 1e9) * 1e3
    cmp_ms = ref_flops / (args.peak_tflops * 1e12) * 1e3
    print(f"  предел по памяти:   {mem_ms:6.0f} мс   предел по счёту: {cmp_ms:.0f} мс")
    if measure_time:
        print(f"  сумма замеров:      {sum_ms:6.0f} мс")

    if not on_tpu:
        print(
            "\n  ЧИТАТЬ С ПОПРАВКОЙ: это не TPU.\n"
            "  Флопы от бэкенда почти не зависят, а БАЙТЫ зависят сильно: XLA на\n"
            "  CPU сплавляет операции куда хуже и пишет в память почти каждый\n"
            "  промежуточный тензор. Для рабочей конфигурации прикидка по формам\n"
            "  (веса, матрицы внимания, активации) даёт втрое-вшестеро меньше,\n"
            "  чем показывает здешний HLO. Поэтому сравнивать варианты между\n"
            "  собой тут можно, а объявлять по этим числам потолок MFU — нельзя:\n"
            "  за ним надо идти на чип."
        )


if __name__ == "__main__":
    main()
