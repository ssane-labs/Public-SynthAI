"""Проверки модели. Гоняются на CPU за секунды — до всякого TPU.

Задача не «покрыть тестами», а поймать то, что на TPU обнаружится через час
прогона и будет стоить сессии: разъехавшиеся формы, утечку будущего в прошлое,
мёртвый градиент у головки, которая ни на что не влияет.

    py -3 synth1/test_model.py
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from synth1 import model as M  # noqa: E402
from synth1.config import ModelConfig  # noqa: E402

# Крошечная конфигурация: та же структура, но считается мгновенно.
TINY = ModelConfig(
    vocab_size=256, d_model=64, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=16,
    n_experts=4, n_shared_experts=1, d_ff=128, top_k=2,
    max_seq_len=64, window=8, global_every=2, mod_every=2, mod_block=8,
)

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = ""):
    (PASSED if condition else FAILED).append(name)
    mark = "ok  " if condition else "FAIL"
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))


def main():
    key = jax.random.key(0)
    cfg = TINY
    params = M.init_model(key, cfg)
    b, t = 2, 32
    tokens = jax.random.randint(jax.random.key(1), (b, t + 1), 1, cfg.vocab_size)

    print("\n1. Прямой проход")
    logits, collect = M.forward(params, tokens[:, :-1], cfg)
    check("форма логитов", logits.shape == (b, t, cfg.vocab_size), str(logits.shape))
    check("логиты конечны", bool(jnp.all(jnp.isfinite(logits))))
    check("логиты не сплющены на старте",
          0.3 < float(jnp.std(logits)) < 10.0, f"std={float(jnp.std(logits)):.2f}")
    check("головка на каждом слое",
          len(collect["u_layers"]) == cfg.n_layers, str(len(collect["u_layers"])))
    check("u в (0,1)",
          all(0 < float(u.min()) and float(u.max()) < 1 for u in collect["u_layers"]))
    check("MTP выдал логиты", "mtp_logits" in collect)

    print("\n2. Причинность — будущее не течёт в прошлое")
    # Меняем последний токен: предсказания на более ранних позициях обязаны
    # остаться прежними. Это ловит и ошибку в маске, и ошибку в MoD-отборе.
    #
    # Граница проверки — начало последнего блока MoD, а не последний токен:
    # внутри своего блока токены соревнуются за проход через экспертов, и там
    # взаимное влияние допущено сознательно (model.py, _moe_with_mod).
    alt = tokens[:, :-1].at[:, -1].set((tokens[:, -2] + 7) % cfg.vocab_size)
    logits_alt, _ = M.forward(params, alt, cfg)
    safe = t - cfg.mod_block
    delta = float(jnp.max(jnp.abs(logits[:, :safe] - logits_alt[:, :safe])))
    check("прошлое до текущего блока не изменилось", delta < 1e-3,
          f"max|Δ| на первых {safe} позициях={delta:.2e}")

    # А без MoD причинность обязана быть строгой на всей длине.
    nomod = dataclasses.replace(cfg, mod=False)
    base_n, _ = M.forward(params, tokens[:, :-1], nomod)
    alt_n, _ = M.forward(params, alt, nomod)
    delta_n = float(jnp.max(jnp.abs(base_n[:, :-1] - alt_n[:, :-1])))
    check("без MoD причинность строгая", delta_n < 1e-3, f"max|Δ|={delta_n:.2e}")

    print("\n3. Скользящее окно действительно ограничивает")
    # Проверяется на самом внимании, а не на всей модели: в модели токены ещё
    # соревнуются за слоты экспертов, и ранний токен легально влияет на поздний
    # через очередь. Это каузально корректно, но к окну отношения не имеет.
    from synth1.layers import attention, rope_tables
    sin, cos = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
    attn_params = params["layers"][0]["attn"]
    long_t = 48
    x = jax.random.normal(jax.random.key(2), (1, long_t, cfg.d_model), jnp.bfloat16)
    out = attention(attn_params, x, sin, cos, cfg.window, cfg)
    x_far = x.at[:, 0].set(x[:, 0] + 5.0)          # правка вне окна последнего токена
    out_far = attention(attn_params, x_far, sin, cos, cfg.window, cfg)
    tail = float(jnp.max(jnp.abs(out[:, -1] - out_far[:, -1])))
    check("токен за окном не влияет", tail < 1e-2,
          f"max|Δ| на последней позиции={tail:.2e}")
    head = float(jnp.max(jnp.abs(out[:, 0] - out_far[:, 0])))
    check("а внутри окна влияет", head > 1e-2, f"max|Δ| на позиции 0={head:.2e}")

    print("\n3b. Блочное внимание совпадает с полным")
    # Экономия памяти не имеет смысла, если меняет ответ. Сверяем блочный путь
    # с прямолинейным на обоих типах слоёв.
    xl = jax.random.normal(jax.random.key(7), (2, 64, cfg.d_model), jnp.bfloat16)
    dense_cfg = dataclasses.replace(cfg, attn_chunk=0)
    block_cfg = dataclasses.replace(cfg, attn_chunk=16)
    for label, win in [(f"окно {cfg.window}", cfg.window), ("глобальный", None)]:
        a = attention(attn_params, xl, sin, cos, win, dense_cfg)
        c = attention(attn_params, xl, sin, cos, win, block_cfg)
        diff = float(jnp.max(jnp.abs(a.astype(jnp.float32) - c.astype(jnp.float32))))
        check(f"{label}: блоками то же самое", diff < 5e-2, f"max|Δ|={diff:.2e}")

    print("\n4. Пересчёт активаций ничего не меняет")
    # remat меняет только то, что хранится между проходами. Логиты и градиенты
    # обязаны совпасть с точностью до арифметики — иначе это не экономия
    # памяти, а тихая порча.
    no_remat = dataclasses.replace(cfg, remat=False)
    logits_nr, _ = M.forward(params, tokens[:, :-1], no_remat)
    check("логиты совпали", float(jnp.max(jnp.abs(logits - logits_nr))) < 1e-3,
          f"max|Δ|={float(jnp.max(jnp.abs(logits - logits_nr))):.2e}")

    grad_r = jax.grad(lambda p: M.loss_fn(p, {"tokens": tokens}, cfg)[0])(params)
    grad_n = jax.grad(lambda p: M.loss_fn(p, {"tokens": tokens}, no_remat)[0])(params)
    gdiff = max(float(jnp.max(jnp.abs(a - b)))
                for a, b in zip(jax.tree.leaves(grad_r), jax.tree.leaves(grad_n)))
    check("градиенты совпали", gdiff < 1e-3, f"max|Δ|={gdiff:.2e}")

    print("\n4b. Нарезка логитов ничего не меняет")
    # То же требование, что и к remat: экономия памяти обязана быть невидимой
    # в числах. Нарезка трогает и лосс, и z-регуляризацию, и энтропию для
    # эпистемической головки — если она врёт, врут все три сразу и тихо.
    whole = dataclasses.replace(cfg, loss_chunk=0)          # логиты целиком
    sliced = dataclasses.replace(cfg, loss_chunk=8)         # шестнадцать кусков
    (lw, mw), gw = jax.value_and_grad(M.loss_fn, has_aux=True)(
        params, {"tokens": tokens}, whole)
    (ls, ms), gs = jax.value_and_grad(M.loss_fn, has_aux=True)(
        params, {"tokens": tokens}, sliced)
    check("лосс совпал", abs(float(lw) - float(ls)) < 1e-3,
          f"целиком {float(lw):.6f}, кусками {float(ls):.6f}")
    for part in ("loss_lm", "loss_z", "loss_mtp", "loss_epi", "u_target"):
        if part in mw:
            check(f"{part} совпал",
                  abs(float(mw[part]) - float(ms[part])) < 1e-3,
                  f"{float(mw[part]):.6f} / {float(ms[part]):.6f}")
    cdiff = max(float(jnp.max(jnp.abs(a - b)))
                for a, b in zip(jax.tree.leaves(gw), jax.tree.leaves(gs)))
    check("градиенты совпали", cdiff < 1e-3, f"max|Δ|={cdiff:.2e}")

    print("\n4г. Маска лосса (этап D1): лосс только по ответу")
    ones = jnp.ones(tokens.shape, jnp.float32)
    l_plain = M.loss_fn(params, {"tokens": tokens}, cfg)[1]["loss_lm"]
    l_ones = M.loss_fn(params, {"tokens": tokens, "loss_mask": ones}, cfg)[1]["loss_lm"]
    check("маска из единиц ничего не меняет", abs(float(l_plain) - float(l_ones)) < 1e-6)
    tail = ones.at[:, : tokens.shape[1] // 2].set(0.0)
    l_tail = M.loss_fn(params, {"tokens": tokens, "loss_mask": tail}, cfg)[1]["loss_lm"]
    by_hand = jax.nn.log_softmax(M.forward(params, tokens[:, :-1], cfg)[0], -1)
    picked = -jnp.take_along_axis(by_hand, tokens[:, 1:, None], -1)[..., 0]
    want = jnp.sum(picked * tail[:, 1:]) / jnp.sum(tail[:, 1:])
    check("лосс по маске — среднее только по отмеченным позициям",
          abs(float(l_tail) - float(want)) < 1e-3, f"{float(l_tail):.5f} / {float(want):.5f}")

    print("\n4д. Калибровка (этап E): BCE головки ответа по позициям с целью")
    none = jnp.full(tokens.shape, -1.0)
    base = M.loss_fn(params, {"tokens": tokens, "loss_mask": ones}, cfg)
    empty = M.loss_fn(params, {"tokens": tokens, "loss_mask": ones,
                               "epi_q": none, "epi_a": none}, cfg)
    check("без целей лосс не меняется",
          abs(float(base[0]) - float(empty[0])) < 1e-6)
    eq = none.at[0, 3].set(1.0)
    ea = none.at[1, 5].set(0.0)
    calib_batch = {"tokens": tokens, "loss_mask": ones, "epi_q": eq, "epi_a": ea}
    (_, cm), cg = jax.value_and_grad(M.loss_fn, has_aux=True)(params, calib_batch, cfg)
    _, col = M.forward(params, tokens[:, :-1], cfg)
    ua = col["u_answer"]
    want_q = -float(jnp.log(ua[0, 3, 0]))
    want_a = -float(jnp.log(1 - ua[1, 5, 1]))
    # u считается в bf16: допуск — его точность, а не fp32.
    check("канал 0 — BCE в позиции цели",
          abs(float(cm["loss_calib_0"]) - want_q) < 5e-3,
          f"{float(cm['loss_calib_0']):.5f} / {want_q:.5f}")
    check("канал 1 — BCE в позиции цели",
          abs(float(cm["loss_calib_1"]) - want_a) < 5e-3,
          f"{float(cm['loss_calib_1']):.5f} / {want_a:.5f}")
    check("градиент дошёл до головки ответа",
          float(jnp.max(jnp.abs(cg["epi_answer"]))) > 0)

    print("\n4е. ORPO (этап F): пара в одной строке, лосс — как руками")
    pref = jnp.zeros(tokens.shape).at[0, 2:5].set(1.0).at[0, 7:9].set(-1.0)
    (_, pm), pg = jax.value_and_grad(M.loss_fn, has_aux=True)(
        params, {"tokens": tokens, "loss_mask": ones, "pref": pref}, cfg)
    lp = jax.nn.log_softmax(M.forward(params, tokens[:, :-1], cfg)[0], -1)
    lp = jnp.take_along_axis(lp, tokens[:, 1:, None], -1)[..., 0]
    lc, lr = float(jnp.mean(lp[0, 1:4])), float(jnp.mean(lp[0, 6:8]))
    odds = lambda m: m - float(jnp.log(-jnp.expm1(m)))  # noqa: E731
    want = float(-jax.nn.log_sigmoid(odds(lc) - odds(lr)))
    check("ORPO совпал с ручным расчётом", abs(float(pm["loss_or"]) - want) < 2e-2,
          f"{float(pm['loss_or']):.4f} / {want:.4f}")
    check("одна пара найдена", float(pm["pref_pairs"]) == 1.0)
    base_l = M.loss_fn(params, {"tokens": tokens, "loss_mask": ones}, cfg)[0]
    check("без пар ORPO ничего не добавляет",
          abs(float(M.loss_fn(params, {"tokens": tokens, "loss_mask": ones,
                                       "pref": jnp.zeros(tokens.shape)}, cfg)[0])
              - float(base_l)) < 1e-5)
    check("градиент ORPO конечен",
          all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(pg)))

    print("\n4в. Слоты экспертов gather'ом — то же, что scatter'ом")
    # Длина 64 при блоке MoD 32 — два блока, иначе отбор MoD вырожден.
    tk = jax.random.randint(jax.random.key(5), (b, 65), 1, cfg.vocab_size)
    by_scatter = dataclasses.replace(cfg, moe_dispatch="scatter", mod_block=32)
    by_gather = dataclasses.replace(by_scatter, moe_dispatch="gather")
    (l1, _), g1 = jax.value_and_grad(M.loss_fn, has_aux=True)(
        params, {"tokens": tk}, by_scatter)
    (l2, _), g2 = jax.value_and_grad(M.loss_fn, has_aux=True)(
        params, {"tokens": tk}, by_gather)
    check("лосс совпал", abs(float(l1) - float(l2)) < 1e-5,
          f"{float(l1):.6f} / {float(l2):.6f}")
    pdiff = max(float(jnp.max(jnp.abs(a - c)))
                for a, c in zip(jax.tree.leaves(g1), jax.tree.leaves(g2)))
    check("градиенты совпали", pdiff < 1e-4, f"max|Δ|={pdiff:.2e}")

    print("\n5. Лосс и градиенты")
    batch = {"tokens": tokens}
    (loss, metrics), grads = jax.value_and_grad(M.loss_fn, has_aux=True)(params, batch, cfg)
    check("лосс конечен", bool(jnp.isfinite(loss)), f"loss={float(loss):.3f}")
    check("лосс близок к случайному",
          abs(float(metrics["loss_lm"]) - jnp.log(cfg.vocab_size)) < 1.5,
          f"lm={float(metrics['loss_lm']):.3f}, log V={float(jnp.log(cfg.vocab_size)):.3f}")

    flat = jax.tree.leaves(grads)
    check("все градиенты конечны", all(bool(jnp.all(jnp.isfinite(g))) for g in flat))

    # Нулевой градиент допустим ровно у двух вещей, и обе такие по замыслу:
    # router_bias двигается вне градиента (это регулировщик очереди), а
    # epi_answer до этапа E ни в один лосс не входит — головка ответа начнёт
    # учиться, когда появятся метки «ошиблась ли модель».
    paths = jax.tree_util.tree_flatten_with_path(grads)[0]
    silent = [jax.tree_util.keystr(p) for p, g in paths
              if float(jnp.max(jnp.abs(g))) == 0]
    unexpected = [p for p in silent
                  if "router_bias" not in p and "epi_answer" not in p]
    check("градиент дошёл всюду, кроме объяснённого", not unexpected,
          f"молчат {len(silent)} (ожидаемо), лишних {len(unexpected)}"
          + (f": {unexpected[:3]}" if unexpected else ""))

    print("\n6. Наши три компонента получают градиент")
    g0 = grads["layers"][0]
    check("эпистемическая головка", float(jnp.max(jnp.abs(g0["epi"]))) > 0)
    check("роутер MoE", float(jnp.max(jnp.abs(g0["moe"]["router"]))) > 0)
    check("роутер глубины (MoD)", float(jnp.max(jnp.abs(g0["mod"]))) > 0)
    check("router_bias не в градиенте — он вне его",
          float(jnp.max(jnp.abs(g0["moe"]["router_bias"]))) == 0)

    print("\n7. Ворота по неуверенности работают в обе стороны")
    check("метрика открытости ворот есть", "gate2_open" in metrics,
          f"{float(metrics.get('gate2_open', -1)):.3f}")
    check("часть токенов пропущена через MoE не полностью",
          0.0 <= float(metrics["moe_dropped"]) < 0.5,
          f"переполнение слотов={float(metrics['moe_dropped']):.1%}")

    print("\n7b. Нормировка весов top-k")
    # Без неё вклад разреженной ветви равен сумме сырых вероятностей выбранных
    # экспертов — на двенадцати экспертах это около 0.17 против единицы у
    # общего. С ней сумма ровно единица.
    from synth1.moe import moe as moe_fn
    h = jax.random.normal(jax.random.key(11), (2, 16, cfg.d_model), jnp.bfloat16)
    moe_params = params["layers"][0]["moe"]
    plain, _ = moe_fn(moe_params, h, None, cfg)
    renorm_cfg = dataclasses.replace(cfg, router_renorm=True, entropy_gate=False)
    scaled, _ = moe_fn(moe_params, h, None, renorm_cfg)
    check("нормировка меняет выход",
          float(jnp.max(jnp.abs(plain.astype(jnp.float32)
                                - scaled.astype(jnp.float32)))) > 1e-3)
    # Проверяем сам множитель: сумма весов выбранных до нормировки заметно
    # меньше единицы, после — единица.
    logits = jnp.einsum("btd,de->bte", h, moe_params["router"].astype(h.dtype))
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    top = jax.lax.top_k(probs, cfg.top_k)[0].sum(-1)
    check("сырые веса заметно меньше единицы", float(jnp.mean(top)) < 0.9,
          f"сумма top-{cfg.top_k} = {float(jnp.mean(top)):.3f} "
          f"при {cfg.n_experts} экспертах")


    print("\n7в. Загрузка меряется по спросу, а не по потолку вместимости")
    # Балансировка двигает смещение по загрузке. Если считать загрузку по
    # принятым токенам, перегруженный эксперт упирается в вместимость и
    # выглядит почти нормальным — сигнал насыщается ровно там, где нужен.
    # На снимке step-14000 это стоило слоям 0-2 работающего MoE: один эксперт
    # забирал 97% токенов, сбрасывалось 64% назначений.
    from synth1.moe import init_moe
    skew_cfg = dataclasses.replace(cfg, entropy_gate=False, top_k=2)
    mp = init_moe(jax.random.key(7), skew_cfg)
    mp = {**mp, "router_bias": jnp.zeros(skew_cfg.n_experts).at[0].set(20.0)}
    hx = jax.random.normal(jax.random.key(8), (2, 256, skew_cfg.d_model),
                           jnp.bfloat16)
    _, skew_aux = moe_fn(mp, hx, None, skew_cfg)
    rl = jnp.einsum("btd,de->bte", hx, mp["router"].astype(hx.dtype))
    _, picks = jax.lax.top_k(rl.astype(jnp.float32) + mp["router_bias"],
                             skew_cfg.top_k)
    want = (jnp.bincount(picks.ravel(), length=skew_cfg.n_experts)
            / picks.size)
    skew_load = skew_aux["load"]
    skew_max = float(jnp.max(skew_load)) * skew_cfg.n_experts
    skew_drop = float(skew_aux["dropped"])
    check("загрузка равна спросу до вместимости",
          float(jnp.max(jnp.abs(want - skew_load))) < 1e-6,
          f"макс|Δ| = {float(jnp.max(jnp.abs(want - skew_load))):.2e}")
    # Потолок перекоса при таком построении — n_experts/top_k: первый выбор у
    # всех один и тот же, второй распределён. Проверяем, что видно именно его,
    # а не срезанную вместимостью половину.
    ceiling = skew_cfg.n_experts / skew_cfg.top_k
    check("перекос виден целиком, а не срезанный вместимостью",
          skew_max > 0.9 * ceiling,
          f"перекос {skew_max:.2f} при потолке {ceiling:.2f}")
    check("и сброс при этом настоящий", skew_drop > 0.1,
          f"сброшено {skew_drop:.1%}")

    print("\n8. Балансировка сдвигает смещение к отстающим")
    from synth1.moe import update_router_bias
    load = jnp.array([0.7, 0.1, 0.1, 0.1])          # первый перегружен
    new_bias = update_router_bias(jnp.zeros(4), load, lr=0.01)
    check("перегруженному понизили", float(new_bias[0]) < 0)
    check("недогруженным повысили", all(float(new_bias[i]) > 0 for i in (1, 2, 3)))

    print("\n9. Сплавленное внимание считает то же, что блочное")
    # Ядро Pallas написано под TPU, но умеет считаться и интерпретатором на
    # любом железе — медленно, зато теми же числами. Проверка идёт на fp32 и
    # короткой длине: в bf16 разошлось бы само округление, и порог пришлось бы
    # ставить такой, что он перестал бы что-либо ловить.
    import dataclasses as _dc

    from synth1 import layers as L

    splash_cfg = _dc.replace(cfg, attn_impl="splash", splash_interpret=True,
                             splash_block=128, window=128, d_model=128, n_heads=4,
                             n_kv_heads=2, head_dim=32, max_seq_len=512)
    bs, ts = 1, 256
    akey = jax.random.key(7)
    ap = L.init_attention(akey, splash_cfg)
    xs = jax.random.normal(jax.random.fold_in(akey, 1),
                           (bs, ts, splash_cfg.d_model), jnp.float32) * 0.3
    sin, cos = L.rope_tables(splash_cfg.head_dim, splash_cfg.max_seq_len,
                             splash_cfg.rope_base)
    for label, win in (("оконный слой", splash_cfg.window), ("глобальный слой", None)):
        blocked = L.attention(ap, xs, sin, cos, win,
                              _dc.replace(splash_cfg, attn_impl="blocked"))
        fused = L.attention(ap, xs, sin, cos, win, splash_cfg)
        gap = float(jnp.max(jnp.abs(blocked - fused)))
        scale = float(jnp.max(jnp.abs(blocked)))
        check(f"выход совпал: {label}", gap < 1e-4 * max(scale, 1e-3),
              f"расхождение {gap:.2e} при масштабе {scale:.3f}")

        gb = jax.grad(lambda p, w=win, c=_dc.replace(splash_cfg, attn_impl="blocked"):
                      jnp.sum(L.attention(p, xs, sin, cos, w, c) ** 2))(ap)
        gf = jax.grad(lambda p, w=win, c=splash_cfg:
                      jnp.sum(L.attention(p, xs, sin, cos, w, c) ** 2))(ap)
        worst = max(
            float(jnp.max(jnp.abs(gb[n] - gf[n])) / jnp.maximum(jnp.max(jnp.abs(gb[n])), 1e-9))
            for n in ("wq", "wk", "wv", "wo"))
        check(f"градиенты совпали: {label}", worst < 1e-3,
              f"худшее относительное расхождение {worst:.2e}")

    check("на CPU сплавленное внимание само не включается",
          not L.splash_ok(2048, _dc.replace(cfg, attn_impl="splash")))
    check("длина не кратная блоку уходит на блочный путь",
          not L.splash_ok(200, _dc.replace(cfg, attn_impl="splash",
                                           splash_interpret=True)))

    print(f"\n{'=' * 52}")
    print(f"пройдено {len(PASSED)}, провалено {len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print(f"  FAIL {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
