"""Проверки оптимизатора и шага обучения.

Главный тест здесь — «запоминание»: модель гоняется на одном и том же батче
несколько десятков шагов и обязана его выучить. Если лосс не падает, дальше идти
незачем, и узнать это надо на ноутбуке за минуту, а не на TPU за час.

    py -3 synth1/test_train.py
"""

from __future__ import annotations

import dataclasses

import pathlib
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from synth1 import model as M  # noqa: E402
from synth1 import optimizer as O  # noqa: E402
from synth1.config import DEFAULT, ModelConfig  # noqa: E402
from synth1.moe import update_router_bias  # noqa: E402
from synth1.test_model import TINY, check, FAILED, PASSED  # noqa: E402


def main():
    print("\n1. Ньютон–Шульц ортогонализует")
    spread = lambda m: float(  # noqa: E731
        (lambda s: s.max() / (s.min() + 1e-9))(
            jnp.linalg.svd(m.astype(jnp.float32), compute_uv=False)))

    # Настоящий градиент матрицы почти вырожден: почти вся норма сидит в
    # нескольких направлениях. Ради этого Muon и затевался.
    g = jax.random.normal(jax.random.key(0), (64, 32))
    u, _, vt = jnp.linalg.svd(g, full_matrices=False)
    skewed = (u * jnp.concatenate([jnp.array([10.0, 5.0]),
                                   jnp.full((30,), 0.05)])) @ vt
    before, after = spread(skewed), spread(O.newton_schulz(skewed))
    check("скошенный градиент выровнялся", after < before / 20,
          f"разброс {before:.0f} → {after:.2f}")

    # На обычной матрице разброс и так небольшой, и после пяти шагов в bf16
    # он садится примерно к 1.7 — точной ортогонализации здесь не бывает и не
    # требуется. Muon нужно приближение: направления должны стать сравнимы по
    # величине, а не строго равны. Гнаться за единицей — это лишние итерации
    # Ньютона–Шульца на каждом шаге обучения, и они не окупаются.
    plain = spread(O.newton_schulz(g))
    check("обычная матрица подтянулась к ортогональной", plain < 2.0,
          f"разброс {spread(g):.2f} → {plain:.2f}")

    o = O.newton_schulz(skewed)
    check("форма сохранилась", o.shape == skewed.shape, str(o.shape))
    check("широкая матрица тоже обрабатывается",
          O.newton_schulz(g.T).shape == (32, 64))

    print("\n2. Расписание LR")
    opt = O.OptConfig(warmup_steps=100)
    total = 1000
    at = lambda s: float(O.lr_schedule(jnp.array(s), total, opt))  # noqa: E731
    check("на старте почти ноль", at(1) < 0.02, f"{at(1):.4f}")
    check("после разогрева единица", abs(at(100) - 1.0) < 1e-6, f"{at(100):.4f}")
    check("на плато держится", abs(at(500) - 1.0) < 1e-6, f"{at(500):.4f}")
    check("к концу отжигается", at(999) < 0.2, f"{at(999):.4f}")
    check("ниже пола не падает", at(total) >= opt.lr_floor - 1e-9, f"{at(total):.4f}")

    print("\n3. Кого ведёт Muon")
    key = jax.random.key(0)
    params = M.init_model(key, TINY)
    muon, adam = [], []
    for path, p in jax.tree_util.tree_flatten_with_path(params)[0]:
        (muon if O._is_muon(path, p) else adam).append(jax.tree_util.keystr(path))
    check("проекции внимания — Muon", any("wq" in n for n in muon))
    check("эксперты — Muon", any("experts" in n for n in muon))
    check("embedding — AdamW", any("embed" in n for n in adam))
    check("нормы — AdamW", any("norm" in n for n in adam))
    check("роутеры — AdamW", any("router" in n for n in adam))

    print("\n4. Состояние оптимизатора")
    state = O.init_state(params, opt)
    check("структура состояния совпала с параметрами",
          jax.tree.structure(state["m"]) == jax.tree.structure(params))
    sizes = O.count_state_bytes(params, opt)
    check("заглушки не занимают память",
          sum(v.size for v in jax.tree.leaves(state["v"]))
          < sum(p.size for p in jax.tree.leaves(params)),
          "второй буфер меньше числа параметров")

    print("\n5. Шаг обучения не ломает модель")
    b, t = 2, 32
    batch = {"tokens": jax.random.randint(jax.random.key(1), (b, t + 1), 1,
                                          TINY.vocab_size)}
    (loss0, _), grads = jax.value_and_grad(M.loss_fn, has_aux=True)(params, batch, TINY)
    new_params, new_state, metrics = O.apply_updates(params, grads, state, 1000, opt)
    check("параметры конечны после шага",
          all(bool(jnp.all(jnp.isfinite(p))) for p in jax.tree.leaves(new_params)))
    check("параметры сдвинулись",
          float(jnp.max(jnp.abs(new_params["layers"][0]["attn"]["wq"]
                                - params["layers"][0]["attn"]["wq"]))) > 0)
    check("шаг посчитан", int(new_state["step"]) == 1)
    check("норма градиента конечна", bool(jnp.isfinite(metrics["grad_norm"])),
          f"{float(metrics['grad_norm']):.3f}")

    print("\n6. Запоминание: один батч, тридцать шагов")
    # Настоящая проверка: модель обязана выучить конкретный батч наизусть.
    # Не сходится здесь — не сойдётся нигде.
    cfg = TINY
    params = M.init_model(jax.random.key(2), cfg)
    state = O.init_state(params, opt)
    opt_fast = O.OptConfig(warmup_steps=5, lr_muon=0.02, lr_adam=3e-3)

    @jax.jit
    def train_step(params, state, batch):
        (loss, metrics), grads = jax.value_and_grad(M.loss_fn, has_aux=True)(
            params, batch, cfg)
        params, state, opt_metrics = O.apply_updates(
            params, grads, state, 200, opt_fast)
        return params, state, loss, metrics, opt_metrics

    started = time.time()
    losses = []
    for step in range(30):
        params, state, loss, metrics, om = train_step(params, state, batch)
        losses.append(float(loss))
        if step % 10 == 0 or step == 29:
            print(f"    шаг {step:>2}  loss {float(loss):>7.3f}  "
                  f"ppl {float(metrics['perplexity']):>9.1f}  "
                  f"|g| {float(om['grad_norm']):>6.2f}  "
                  f"u {float(metrics['u_mean']):.3f}→{float(metrics['u_target']):.3f}")

    check("лосс падает", losses[-1] < losses[0] - 1.0,
          f"{losses[0]:.3f} → {losses[-1]:.3f}")
    check("падает монотонно в целом",
          sum(losses[i + 1] < losses[i] for i in range(len(losses) - 1)) > 20,
          f"{sum(losses[i + 1] < losses[i] for i in range(len(losses) - 1))}/29 шагов вниз")
    check("не разошлось", all(l == l and l < 100 for l in losses))
    print(f"    {30 / (time.time() - started):.1f} шагов/с на CPU")

    print("\n7. Балансировка выравнивает загрузку экспертов")
    bias = jnp.zeros(TINY.n_experts)
    load = jnp.array([0.6, 0.3, 0.05, 0.05])
    for _ in range(50):
        bias = update_router_bias(bias, load, lr=0.01)
    check("смещение развело перегруженных и отстающих",
          float(bias[0]) < float(bias[2]),
          f"перегруженный {float(bias[0]):.3f}, отстающий {float(bias[2]):.3f}")

    print("\n8. Память состояния на целевой модели")
    full = M.init_model(jax.random.key(3), DEFAULT)
    sizes = O.count_state_bytes(full, opt)
    print(f"    параметров {sizes['params_M']:.0f}M "
          f"(Muon {sizes['muon_M']:.0f}M, AdamW {sizes['adam_M']:.0f}M)")
    print(f"    состояние {sizes['state_GB']:.1f} ГБ "
          f"(было бы {sizes['state_GB_if_all_adam']:.1f} ГБ на чистом AdamW)")
    check("состояние влезает в 8 чипов по 16 ГБ",
          sizes["state_GB"] / 8 < 12.0,
          f"{sizes['state_GB'] / 8:.2f} ГБ на чип при FSDP")
    check("Muon экономит заметно",
          sizes["state_GB"] < sizes["state_GB_if_all_adam"] * 0.85,
          f"экономия {1 - sizes['state_GB'] / sizes['state_GB_if_all_adam']:.0%}")

    print("\n9. Больной шаг пропускается, а не портит веса")
    # Прогон длиной в девять часов не должен умирать от одного плохого батча.
    small = M.init_model(jax.random.key(4), TINY)
    st = O.init_state(small, opt)
    good = {"tokens": jax.random.randint(jax.random.key(5), (2, 33), 1,
                                         TINY.vocab_size)}
    _, grads = jax.value_and_grad(M.loss_fn, has_aux=True)(small, good, TINY)

    poisoned = dict(jax.tree.map(lambda g: g, grads))
    poisoned["norm_out"] = poisoned["norm_out"].at[0].set(jnp.nan)
    after, st2, met = O.apply_updates(small, poisoned, st, 1000, opt)

    check("шаг помечен как пропущенный", float(met["skipped"]) == 1.0)
    same = all(bool(jnp.array_equal(a, b)) for a, b in
               zip(jax.tree.leaves(small), jax.tree.leaves(after)))
    check("веса не тронуты", same)
    check("буферы momentum не тронуты",
          all(bool(jnp.array_equal(a, b)) for a, b in
              zip(jax.tree.leaves(st["m"]), jax.tree.leaves(st2["m"]))))
    check("здоровый шаг по-прежнему меняет веса",
          not all(bool(jnp.array_equal(a, b)) for a, b in zip(
              jax.tree.leaves(small),
              jax.tree.leaves(O.apply_updates(small, grads, st, 1000, opt)[0]))))

    # Между конечным градиентом и конечными весами стоит Muon: Ньютон–Шульц
    # считается в bf16 и на вырожденной матрице способен выдать nan там, где на
    # входе всё было конечно. Проверка по градиенту такой шаг пропускает, и
    # веса травятся навсегда — дальше лосс nan при любом батче. Подменяем
    # ортогонализацию на заведомо больную и смотрим, ловится ли это.
    healthy_ns = O.newton_schulz
    O.newton_schulz = lambda g, steps=O.NS_STEPS: jnp.full_like(g, jnp.nan)
    try:
        after_ns, st_ns, met_ns = O.apply_updates(small, grads, st, 1000, opt)
    finally:
        O.newton_schulz = healthy_ns

    check("nan из оптимизатора тоже ловится", float(met_ns["skipped"]) == 1.0)
    check("веса пережили больной Ньютон–Шульц",
          all(bool(jnp.array_equal(a, b)) for a, b in
              zip(jax.tree.leaves(small), jax.tree.leaves(after_ns))))
    check("буферы пережили тоже",
          all(bool(jnp.array_equal(a, b)) for a, b in
              zip(jax.tree.leaves(st["m"]), jax.tree.leaves(st_ns["m"]))))

    # Норма не должна переполняться там, где сам градиент ещё представим.
    huge = jax.tree.map(lambda g: jnp.full_like(g, 1e20), grads)
    n = O.global_norm(huge)
    check("норма не переполняется на больших градиентах",
          bool(jnp.isfinite(n)) and float(n) > 1e20, f"|g|={float(n):.3e}")

    print("\n10. Батч проверяется до дорогих шагов")
    from synth1 import sharding as S
    mesh = S.make_mesh()
    n = mesh.shape[S.AXIS]
    ok = True
    try:
        S.check_batch(n * 2, mesh)
    except SystemExit:
        ok = False
    check("кратный батч проходит", ok, f"{n * 2} при {n} устройствах")

    if n > 1:
        caught = ""
        try:
            S.check_batch(n + 1, mesh)
        except SystemExit as exc:
            caught = str(exc)
        check("некратный батч отвергается с подсказкой",
              "batch-size" in caught, caught.splitlines()[0] if caught else "молча прошёл")

    print("\nНовый этап: разогрев от своего первого шага и спад к концу этапа")
    start, n, warm = 161_888, 20_000, 500
    stage = dataclasses.replace(O.OptConfig(), warmup_from=start,
                                warmup_steps=warm, decay_frac=(n - warm) / (start + n))
    lr = lambda s: float(O.lr_schedule(jnp.array(s), start + n, stage))  # noqa: E731
    check("на первом шаге этапа LR почти ноль", lr(start + 1) < 0.01, f"{lr(start + 1):.4f}")
    check("к концу разогрева — пик", abs(lr(start + warm) - 1.0) < 1e-6)
    check("к концу этапа — пол", abs(lr(start + n) - stage.lr_floor) < 1e-6)
    check("по умолчанию разогрев с нуля, как раньше",
          abs(float(O.lr_schedule(jnp.array(1000), 200_000, O.OptConfig())) - 0.5) < 1e-6)

    print(f"\n{'=' * 52}")
    print(f"пройдено {len(PASSED)}, провалено {len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print(f"  FAIL {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
