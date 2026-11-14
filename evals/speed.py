"""Куда уходит время шага. Замер, а не догадки.

Первый прогон на TPU дал 0.49 с на шаг в 16K токенов — это 3–4% от пика чипа,
тогда как MoE-модели обычно живут на 30–40%. Разница между этими числами — вся
разница между «претрейн за пять недель» и «претрейн за два года», поэтому
вопрос стоит отдельной программы.

Программа отвечает на два вопроса, и только на них:

    сколько стоит шаг     мс, токенов в секунду, TFLOPS, MFU
    из чего он состоит    что меняется, если убрать по одному подозреваемому

Второй вопрос важнее. Абсолютное число говорит «плохо», а разложение говорит
«вот здесь». Каждый вариант — та же модель с одним выключенным механизмом:
разница в мс и есть цена механизма.

    py -3 evals/speed.py --preset default --batch-size 8
    py -3 evals/speed.py --preset default --batch-size 8 --sweep

Считается на fake-данных: вопрос про железо, а не про корпус. Токены рождаются
на устройстве, поток с Hub в замер не попадает вовсе.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from synth1 import fsdp as F  # noqa: E402
from synth1 import layers as L  # noqa: E402
from synth1 import model as M  # noqa: E402
from synth1 import optimizer as O  # noqa: E402
from synth1 import sharding as S  # noqa: E402
from synth1.config import ABLATION, ABLATION_DENSE, DEFAULT, ModelConfig  # noqa: E402

PRESETS = {"default": DEFAULT, "ablation": ABLATION, "ablation-dense": ABLATION_DENSE}

# Пик bf16 на чип. Число нужно только для MFU — процента от того, что железо
# способно выдать в принципе.
PEAK_TFLOPS = {
    "TPU v5 lite": 197.0,
    "TPU v5e": 197.0,
    "TPU v4": 275.0,
    "TPU v3": 123.0,
}


def peak_tflops(default: float = 197.0) -> float:
    kind = jax.devices()[0].device_kind
    for name, value in PEAK_TFLOPS.items():
        if name.lower() in kind.lower():
            return value * len(jax.devices())
    return default * len(jax.devices())


def flops_per_step(cfg: ModelConfig, tokens: int) -> float:
    """Счёт по `C = 6ND` — прямой и обратный проход на активных параметрах.

    Это нижняя оценка: внимание, роутер и выходная проекция сюда не входят
    полностью. Зато она не зависит от того, что XLA решил сделать с графом, и
    сравнима между конфигурациями — а сравнивать нам и надо.
    """
    return 6.0 * cfg.n_params()["active"] * tokens


def build_step(cfg: ModelConfig, opt: O.OptConfig, total_steps: int,
               mesh=None, fsdp: bool = False):
    def step(params, state, batch):
        if fsdp and mesh is not None and mesh.shape[S.AXIS] > 1:
            loss_fn = F.sharded_loss(cfg, mesh, params)
        else:
            loss_fn = lambda p, b: M.loss_fn(p, b, cfg)  # noqa: E731
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch)
        params, state, opt_metrics = O.apply_updates(
            params, grads, state, total_steps, opt)
        return params, state, {**metrics, **opt_metrics}

    return jax.jit(step, donate_argnums=(0, 1))


def build_forward(cfg: ModelConfig):
    return jax.jit(lambda p, b: M.loss_fn(p, b, cfg)[0])


def steady(times: list[float]) -> tuple[float, str]:
    """Устоявшееся время шага из нескольких кругов плюс предупреждение.

    Замер, в который попала компиляция, даёт круги разной длины: разовая
    примесь делится на число шагов, поэтому короткий круг выглядит длиннее.
    Ровно так 05.09.2026 один и тот же конфиг дал 11.1 с на двадцати шагах и
    20.6 с на десяти — из чего следует 189 с примеси и 1.7 с настоящего шага.
    Напечатанный тогда MFU 0.4% был занижен в четырнадцать раз.

    Поэтому кругов несколько, берётся наименьший, а расхождение между ними
    печатается: молча выдать число, в котором сидит компиляция, — это хуже,
    чем не мерить вовсе, потому что по такому числу принимают решения.
    """
    best = min(times)
    spread = (max(times) - best) / max(best, 1e-9)
    if spread > 0.15:
        return best, f'круги разошлись на {spread:.0%}, в таймере не только шаг'
    return best, ''


def measure_step(fn, params, state, batch, steps: int,
                 rounds: int = 2) -> list[float]:
    """Время шага обучения, кругами. Возвращает время каждого круга.

    Состояние переносится из круга в круг, как в настоящем цикле:
    donate_argnums забирает буферы себе, и повторный вызов с теми же —
    обращение к удалённому.

    Прогрев идёт двумя вызовами, а не одним. Одного не хватало: donate меняет
    раскладку выходных буферов, второй вызов приходит с другой раскладкой
    входов и компилируется заново — и этот второй компиляционный проход
    попадал ровно в таймер.

    Блокировка ждёт и веса, и состояние оптимизатора: ждать только веса
    значит мерить неполный шаг.
    """
    def go(params, state, n):
        started = time.perf_counter()
        for _ in range(n):
            params, state, _ = fn(params, state, batch)
        jax.block_until_ready((params, state))
        return (time.perf_counter() - started) / n, params, state

    for _ in range(2):
        _, params, state = go(params, state, 1)

    times = []
    for _ in range(rounds):
        seconds, params, state = go(params, state, steps)
        times.append(seconds)
    return times


def measure(fn, args, steps: int) -> float:
    """То же для функции без состояния — прямого прохода."""
    out = fn(*args)
    jax.block_until_ready(out)

    started = time.perf_counter()
    for _ in range(steps):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - started) / steps


def run_variant(name: str, cfg: ModelConfig, opt: O.OptConfig, args, mesh,
                peak: float, full_step: bool = True) -> dict:
    params = M.init_model(jax.random.key(0), cfg)
    state = O.init_state(params, opt)
    params = S.distribute(params, mesh)
    state = {**state, "m": S.distribute(state["m"], mesh),
             "v": S.distribute(state["v"], mesh)}

    tokens = jax.random.randint(
        jax.random.key(1), (args.batch_size, args.seq_len + 1), 1,
        cfg.vocab_size, dtype=jnp.int32)
    batch = {"tokens": jax.device_put(tokens, S.batch_sharding(mesh))}

    if full_step:
        fn = build_step(cfg, opt, 200_000, mesh, fsdp=args.fsdp)
        times = measure_step(fn, params, state, batch, args.steps)
        seconds, warning = steady(times)
    else:
        fn = build_forward(cfg)
        seconds = measure(fn, (params, batch), args.steps)
        warning = ''

    n_tokens = args.batch_size * args.seq_len
    flops = flops_per_step(cfg, n_tokens)
    if not full_step:
        flops = flops / 3.0            # только прямой проход — треть счёта
    tflops = flops / seconds / 1e12

    row = {
        "имя": name,
        "мс/шаг": seconds * 1000,
        "токенов/с": n_tokens / seconds,
        "TFLOPS": tflops,
        "MFU %": 100.0 * tflops / peak,
    }
    print(f"  {name:<22} {row['мс/шаг']:>8.1f} мс  "
          f"{row['токенов/с']:>9,.0f} ток/с  "
          f"{tflops:>7.1f} TFLOPS  MFU {row['MFU %']:>5.1f}%"
          + (f'  <- {warning}' if warning else ''), flush=True)
    return row


def pick(plan, only: str):
    """Оставить варианты, чьё имя содержит одну из подстрок.

    Полный перебор — тринадцать вариантов и полтора часа, потому что каждый
    заново компилируется. Когда подозреваемый уже назван, платить за всех
    незачем: «как есть» остаётся всегда, иначе не с чем сравнивать.
    """
    if not only:
        return plan
    # Совпадение по подстроке, поэтому имена вариантов подобраны с общим
    # корнем: «внимание целиком», «внимание блоками 512». Иначе фильтр
    # «внимание» не совпал бы с «блок внимания» из-за склонения и молча
    # отдал бы неполный набор — а это полтора часа сессии впустую.
    marks = [w.strip().lower() for w in only.split(",") if w.strip()]
    kept = [row for row in plan
            if row[0] == "как есть"
            or any(mark in row[0].lower() for mark in marks)]
    if len(kept) < 2:
        raise SystemExit(
            f"по фильтру {only!r} нечего мерить. Есть: "
            + ", ".join(row[0] for row in plan)
        )
    return kept


def variants(base: ModelConfig,
             base_opt: O.OptConfig) -> list[tuple[str, ModelConfig, O.OptConfig]]:
    """Подозреваемые. Каждый выключается по одному — разница и есть его цена."""
    def m(name, **kw):
        return (name, dataclasses.replace(base, **kw), base_opt)

    def o(name, **kw):
        return (name, base, dataclasses.replace(base_opt, **kw))

    out = [("как есть", base, base_opt)]
    if base.attn_impl == "splash":
        # Главный вопрос этого перебора: сколько даёт сплавленное ядро против
        # блочного пути, который считал все прошлые сессии.
        out.append(m("внимание блочное", attn_impl="blocked"))
        for size in (256, 512):
            if size <= base.max_seq_len:
                out.append(m(f"splash блоком {size}", splash_block=size))
    if base.mod:
        out.append(m("без MoD", mod=False))
    if base.entropy_gate:
        out.append(m("без ворот", entropy_gate=False))
    if base.epistemic_head:
        out.append(m("без эпи-головки", epistemic_head=False))
    if base.mtp:
        out.append(m("без MTP", mtp=False))
    if base.attn_chunk and base.attn_impl == "blocked":
        # Размер блока трогает только блочный путь: при splash его не смотрит
        # никто, и такой вариант померил бы ровно ничего.
        out.append(m("внимание целиком", attn_chunk=0))
        # Внимание было главным подозреваемым в низком MFU и подозреваемым
        # быть перестало. Замер 07.09.2026, батч 32, длина 2048:
        #     как есть (блоки 256)   1404.7 мс
        #     внимание целиком       1507.3 мс   -7.3%
        #     блоками 512            1474.8 мс   -5.0%
        #     блоками 1024           1562.0 мс  -11.2%
        # То есть 256 — лучший из четырёх, а не худший: убрать блочность или
        # укрупнить блок только замедляет. Splash attention отсюда ничего не
        # выиграет, и MFU 5.9% объясняется чем-то другим.
        #
        # Дописано 14.09.2026: вывод про splash оказался неверным. Он написан
        # до того, как стало видно, что дело не в размере блока, а в том, что
        # блочный путь вообще выносит матрицу [C,S] в память. Сплавленное ядро
        # её не выносит, и сравнение с ним — строка «внимание блочное» выше.
        for size in (128, 512, 1024):
            if size < base.max_seq_len:
                out.append(m(f"внимание блоками {size}", attn_chunk=size))
    if not base.moe_fold_batch:
        # Свёртка батча в ось слотов: та же математика, на одну батчевую ось в
        # einsum'е меньше. По HLO на CPU экономит пятую часть трафика блока
        # экспертов, но может стоить переразложения массива между чипами —
        # ровно это и надо померить (moe.py, cfg.moe_fold_batch).
        out.append(m("эксперты со свёрткой батча", moe_fold_batch=True))
    out.append(m("логиты целиком", loss_chunk=0))
    out.append(m("логиты по 1024", loss_chunk=1024))
    if base.remat:
        out.append(m("без пересчёта", remat=False))
    if base.n_experts > 1:
        out.append(m("роутер top-1", top_k=1))

    # Оптимизатор — такой же подозреваемый, как и модель. По прикидке пять
    # итераций Ньютона–Шульца по матрицам экспертов стоят около 20 TFLOP при
    # 32 TFLOP самого шага: если это подтвердится, ортогонализация окажется
    # самой дорогой частью обучения, о которой нигде не написано.
    if base_opt.use_muon:
        out.append(o("Ньютон–Шульц 3", ns_steps=3))
        out.append(o("Ньютон–Шульц 1", ns_steps=1))
        out.append(o("без Muon, чистый AdamW", use_muon=False))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="default", choices=list(PRESETS))
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10, help="шагов в замере")
    ap.add_argument("--sweep", action="store_true",
                    help="перебрать подозреваемых, выключая по одному")
    ap.add_argument("--only", default="",
                    help="мерить только варианты, чьё имя содержит эти "
                         "подстроки через запятую. «как есть» остаётся "
                         "всегда. Полный перебор — полтора часа, потому что "
                         "каждый вариант компилируется заново; когда "
                         "подозреваемый назван, платить за всех незачем")
    ap.add_argument("--auto-sharding", dest="fsdp", action="store_false",
                    help="лосс под обычным jit, без shard_map (как до 04.10)")
    ap.add_argument("--forward-only", action="store_true",
                    help="мерить только прямой проход")
    ap.add_argument("--peak-tflops", type=float, default=None,
                    help="пик bf16 на чип, если устройство неизвестно")
    args = ap.parse_args()

    base = dataclasses.replace(
        PRESETS[args.preset],
        max_seq_len=max(PRESETS[args.preset].max_seq_len, args.seq_len))

    base_opt = O.OptConfig()
    mesh = S.make_mesh()
    if base.attn_impl == "splash" and not L.probe_splash(
            base, args.seq_len, S.batch_sharding(mesh), args.batch_size):
        base = dataclasses.replace(base, attn_impl="blocked")
    S.check_batch(args.batch_size, mesh)
    peak = (args.peak_tflops * len(jax.devices())
            if args.peak_tflops else peak_tflops())

    print(f"устройств: {len(jax.devices())} × {jax.devices()[0].device_kind}, "
          f"пик {peak:.0f} TFLOPS bf16")
    print(f"{base.summary()}")
    print(f"батч {args.batch_size} × длина {args.seq_len} = "
          f"{args.batch_size * args.seq_len:,} токенов на шаг, "
          f"{args.steps} шагов в замере\n")

    rows = []
    plan = (pick(variants(base, base_opt), args.only) if args.sweep
            else [("как есть", base, base_opt)])
    for name, cfg, opt in plan:
        try:
            rows.append(run_variant(name, cfg, opt, args, mesh, peak,
                                    full_step=not args.forward_only))
        except Exception as exc:                          # noqa: BLE001
            # Вариант может не влезть в память — это тоже результат, и он не
            # должен обрывать остальные замеры.
            print(f"  {name:<22} не измерен: {str(exc)[:60]}", flush=True)

    if len(rows) > 1:
        base_ms = rows[0]["мс/шаг"]
        print("\nцена механизма (сколько шаг теряет из-за него):")
        for row in rows[1:]:
            delta = base_ms - row["мс/шаг"]
            share = 100.0 * delta / base_ms
            print(f"  {row['имя']:<22} {delta:>+8.1f} мс  ({share:>+5.1f}%)")


if __name__ == "__main__":
    main()
