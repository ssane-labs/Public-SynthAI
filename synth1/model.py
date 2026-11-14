"""Модель Synth-1: блок, адаптивная глубина, эпистемическая головка, MTP.

Читать вместе с ARCHITECTURE.md — здесь код того, что там описано словами.

Одно уточнение против исходной спецификации. Адаптивная глубина применяется к
блоку экспертов, а не ко всему слою целиком. Причина: на выбранном подмножестве
токенов внимание пришлось бы считать по их исходным позициям, с маской по
настоящему порядку, — это лишний источник ошибок в причинности, а выигрыш
маленький. В нашей конфигурации на MoE приходится 118M активных параметров из
138M (без словаря), то есть 86% счёта; пропуская только его, мы забираем почти
всю экономию и не трогаем внимание.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from synth1.layers import (ACCUM, attention, init_attention, init_linear,
                           normal, rms_norm, rope_tables)
from synth1.moe import init_moe, inverse_rows, moe, permute_rows


def is_global_layer(layer: int, cfg) -> bool:
    """Каждый n-й слой смотрит на весь контекст, остальные — в своё окно."""
    return (layer + 1) % cfg.global_every == 0


def uses_mod(layer: int, cfg) -> bool:
    return cfg.mod and (layer % cfg.mod_every == 0)


# --- Инициализация ---------------------------------------------------------

def init_layer(key, layer: int, cfg) -> dict:
    ka, km, ke, kd = jax.random.split(key, 4)
    params = {
        "attn": init_attention(ka, cfg),
        "moe": init_moe(km, cfg),
        "norm_attn": jnp.ones((cfg.d_model,)),
        "norm_moe": jnp.ones((cfg.d_model,)),
        "post_attn": jnp.ones((cfg.d_model,)),
        "post_moe": jnp.ones((cfg.d_model,)),
    }
    if cfg.epistemic_head:
        # Вся эпистемическая головка слоя — один вектор на 768 чисел.
        # Она не «дополнительная модель», а термометр, воткнутый в поток.
        params["epi"] = init_linear(ke, cfg.d_model, 1, scale=0.1)
    if uses_mod(layer, cfg):
        params["mod"] = init_linear(kd, cfg.d_model, 1, scale=0.1)
    return params


def init_model(key, cfg) -> dict:
    keys = jax.random.split(key, cfg.n_layers + 3)
    params = {
        # Матрица связанная: она же выходная проекция. Отсюда и масштаб —
        # std = 1/sqrt(d_model), чтобы логиты на старте были порядка единицы.
        # Считать её от размера словаря (как обычный линейный слой) нельзя:
        # получится std 0.0045, логиты сплющатся, и первые тысячи шагов уйдут
        # на то, чтобы разложить распределение обратно.
        "embed": normal(keys[0], (cfg.vocab_size, cfg.d_model),
                        std=1.0 / math.sqrt(cfg.d_model)),
        "layers": [init_layer(keys[1 + i], i, cfg) for i in range(cfg.n_layers)],
        "norm_out": jnp.ones((cfg.d_model,)),
    }
    if not cfg.tie_embeddings:
        params["unembed"] = init_linear(keys[-2], cfg.d_model, cfg.vocab_size)
    if cfg.epistemic_head:
        # Головка ответа: два числа — «ошибусь ли» и «нужен ли инструмент».
        # До этапа E обучается только первое, второе стоит нулём.
        params["epi_answer"] = init_linear(keys[-1], cfg.d_model, 2, scale=0.1)
    if cfg.mtp:
        kmtp = jax.random.fold_in(keys[-1], 1)
        k1, k2 = jax.random.split(kmtp)
        params["mtp"] = {
            "proj": init_linear(k1, 2 * cfg.d_model, cfg.d_model),
            "norm_h": jnp.ones((cfg.d_model,)),
            "norm_e": jnp.ones((cfg.d_model,)),
            "layer": init_layer(k2, cfg.n_layers - 1, cfg),
        }
    return params


# --- Блок ------------------------------------------------------------------

# Сборка весов слоя перед счётом. По умолчанию — ничего: веса уже целые.
# Под shard_map (synth1/fsdp.py) сюда ставится all-gather, и тогда он идёт
# внутри пересчитываемого слоя: собранные веса живут один слой, а не весь
# шаг, и при обратном проходе собираются заново. Это и есть FSDP.
_GATHER = None


def apply_layer(params: dict, x: jax.Array, layer: int, sin, cos, cfg):
    """Один слой. Возвращает (выход, служебные величины).

    Побочных эффектов нет намеренно: слой оборачивается в jax.checkpoint, а тот
    прогоняет функцию дважды — при прямом проходе и заново при обратном. Всё,
    что писалось бы «мимо» возвращаемого значения, во второй раз запишется
    повторно, и списки метрик молча удвоятся.
    """
    window = None if is_global_layer(layer, cfg) else cfg.window
    if _GATHER is not None:
        params = _GATHER(params)

    h = rms_norm(x, params["norm_attn"])
    h = attention(params["attn"], h, sin, cos, window, cfg)
    x = x + rms_norm(h, params["post_attn"])

    # Неуверенность считается здесь, до экспертов: она управляет и тем, кого
    # звать, и тем, звать ли вообще.
    u = None
    if cfg.epistemic_head:
        u = jax.nn.sigmoid(
            jnp.einsum("btd,do->bto", x, params["epi"].astype(x.dtype))[..., 0]
        )

    h = rms_norm(x, params["norm_moe"])

    if uses_mod(layer, cfg):
        h, aux = _moe_with_mod(params, h, u, cfg)
    else:
        h, aux = moe(params["moe"], h, u, cfg)

    return x + rms_norm(h, params["post_moe"]), {"u": u, "moe": aux}


# Пересчёт вместо хранения. Внимание держит матрицу [B, H, T, T]: при батче 32
# и длине 2048 это 6.4 ГБ на слой, и двадцать четыре слоя таких активаций не
# влезут никуда. С checkpoint хранится только вход слоя, а внутренности
# считаются заново при обратном проходе — примерно треть лишнего счёта в обмен
# на возможность вообще запуститься с большим батчем.
_checkpointed_layer = jax.checkpoint(apply_layer, static_argnums=(2, 5))


def _moe_with_mod(params: dict, h: jax.Array, u, cfg):
    """Через экспертов проходит только часть токенов, остальные идут мимо.

    Отбор — по обучаемому скору плюс неуверенность.

    Больное место классического MoD: `top-k` по всей последовательности
    некаузален. Токены соревнуются между собой, поэтому изменение последнего
    токена меняет состав отобранных — и задним числом меняет предсказания на
    ранних позициях. Утечка будущего в прошлое, ровно то, чего в языковой
    модели быть не должно; наш тест причинности ловит это немедленно.

    Отбор идёт внутри блоков по `mod_block` токенов. Токен соревнуется только
    со своим блоком, поэтому влияние ограничено его границами: правка в блоке N
    не трогает ничего в блоках до N. Остаточная утечка внутри блока — один бит
    на токен, и она про «трудный ли токен», а не про то, какой он; MoD-статья
    работает именно на этом допущении.

    Второй выигрыш от блоков: на инференсе отбор становится локальным, и `u`
    служит порогом честно — ему не нужно знать распределение по всей
    последовательности.
    """
    b, t, d = h.shape

    score = jnp.einsum("btd,do->bto", h, params["mod"].astype(h.dtype))[..., 0]
    if u is not None:
        score = score + cfg.mod_entropy_alpha * u.astype(score.dtype)

    # Блок должен делить длину нацело — иначе форма графа поедет. Короткая
    # последовательность (абляции, тесты) отбирается целиком одним блоком.
    block = cfg.mod_block if t % cfg.mod_block == 0 else t
    n_blocks = t // block
    keep_per_block = max(1, int(cfg.mod_capacity * block))

    blocked = score.astype(ACCUM).reshape(b, n_blocks, block)
    _, local_idx = jax.lax.top_k(blocked, keep_per_block)      # [B, nb, keep]
    offsets = (jnp.arange(n_blocks) * block)[None, :, None]
    idx = (local_idx + offsets).reshape(b, n_blocks * keep_per_block)
    keep = idx.shape[1]
    if cfg.moe_dispatch == "gather":
        # Отобранные позиции — частичная перестановка, обратная к ней
        # выписывается явно, и оба направления идут gather'ом (moe.py).
        back = inverse_rows(idx, t)                                 # [B, T]
        gather = permute_rows(h, idx, back)                         # [B, keep, D]
    else:
        gather = jnp.take_along_axis(h, idx[..., None], axis=1)     # [B, keep, D]
    u_sel = jnp.take_along_axis(u, idx, axis=1) if u is not None else None

    y_sel, aux = moe(params["moe"], gather, u_sel, cfg)

    # Умножение на ворота отбора — единственный путь градиента к w_mod: без него
    # скор ни на что не влияет и остаётся тем, чем был при инициализации.
    gate = jax.nn.sigmoid(jnp.take_along_axis(score, idx, axis=1))
    y_sel = y_sel * gate[..., None].astype(y_sel.dtype)

    if cfg.moe_dispatch == "gather":
        out = permute_rows(y_sel, back, idx)
    else:
        rows = jnp.arange(b)[:, None]
        out = jnp.zeros_like(h).at[rows, idx].set(y_sel)

    aux = {**aux, "mod_kept": jnp.array(keep / t, dtype=ACCUM)}
    return out, aux


# --- Прямой проход ---------------------------------------------------------

def _unembed(params: dict, cfg) -> jax.Array:
    return params["embed"] if cfg.tie_embeddings else params["unembed"].T


def forward(params: dict, tokens: jax.Array, cfg,
            head: bool = True) -> tuple[jax.Array | None, dict]:
    """tokens: [B, T] целые. Возвращает логиты [B, T, V] и служебные величины.

    head=False останавливается перед выходной проекцией и отдаёт только
    скрытые состояния. Логиты — самый большой тензор в модели: при батче 32 и
    длине 2048 это 1.5 ГБ на чип, и обучению они целиком не нужны ни разу.
    Лосс считает их сам, кусками (см. `head_losses`).
    """
    sin, cos = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)

    embed = params["embed"]
    x = jnp.take(embed, tokens, axis=0).astype(jnp.bfloat16)

    layer_fn = _checkpointed_layer if cfg.remat else apply_layer
    collect: dict = {"u_layers": [], "moe_aux": []}
    for i, layer_params in enumerate(params["layers"]):
        x, aux = layer_fn(layer_params, x, i, sin, cos, cfg)
        if aux["u"] is not None:
            collect["u_layers"].append(aux["u"])
        collect["moe_aux"].append(aux["moe"])

    h_final = rms_norm(x, params["norm_out"])
    collect["h_final"] = h_final

    if cfg.epistemic_head:
        collect["u_answer"] = jax.nn.sigmoid(
            jnp.einsum("btd,do->bto", h_final, params["epi_answer"].astype(h_final.dtype))
        )

    unembed = _unembed(params, cfg)
    if cfg.mtp:
        mtp_h = _mtp_hidden(params, h_final, tokens, sin, cos, cfg)
        collect["mtp_h"] = mtp_h
        if head:
            collect["mtp_logits"] = jnp.einsum(
                "btd,vd->btv", mtp_h, unembed.astype(mtp_h.dtype)).astype(ACCUM)

    if not head:
        return None, collect

    logits = jnp.einsum("btd,vd->btv", h_final, unembed.astype(h_final.dtype))
    return logits.astype(ACCUM), collect


def _mtp_hidden(params: dict, h: jax.Array, tokens: jax.Array, sin, cos, cfg):
    """Предсказание токена через один: черновик для спекулятивного декодирования.

    Модуль видит состояние на позиции t и настоящий токен t+1, а предсказывает
    t+2. При генерации настоящего t+1 ещё нет — берётся только что выданный, и
    если основная модель его подтвердит, два токена уходят за один проход.
    """
    mtp = params["mtp"]
    embed = params["embed"]

    nxt = jnp.roll(tokens, shift=-1, axis=1)
    e = jnp.take(embed, nxt, axis=0).astype(h.dtype)

    joined = jnp.concatenate(
        [rms_norm(h, mtp["norm_h"]), rms_norm(e, mtp["norm_e"])], axis=-1
    )
    z = jnp.einsum("btd,de->bte", joined, mtp["proj"].astype(h.dtype))
    z, _ = apply_layer(mtp["layer"], z, cfg.n_layers - 1, sin, cos, cfg)
    return z


# --- Лосс ------------------------------------------------------------------

def cross_entropy(logits: jax.Array, targets: jax.Array,
                  mask: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits.astype(ACCUM), axis=-1)
    picked = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return -jnp.sum(picked * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _logit_chunk(h: jax.Array, unembed: jax.Array, targets: jax.Array,
                 mask: jax.Array, entropy: bool):
    """Всё, что нужно от логитов куска, за один их подъём в память.

    Три величины считаются из одного и того же log_softmax: сам лосс,
    z-регуляризация и энтропия для эпистемической головки. Отдельными вызовами
    это стоило бы трёх материализаций [B, T, V] вместо одной.
    """
    logits = jnp.einsum("btd,vd->btv", h, unembed.astype(h.dtype)).astype(ACCUM)
    logp = jax.nn.log_softmax(logits, axis=-1)

    picked = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    ce = -jnp.sum(picked * mask)

    # logp = logits - logsumexp(logits), поэтому logsumexp достаётся вычитанием,
    # а не вторым проходом по всему словарю.
    z = logits[..., 0] - logp[..., 0]
    z2 = jnp.sum(jnp.square(z) * mask)

    ent = (-jnp.sum(jnp.exp(logp) * logp, axis=-1) if entropy
           else jnp.zeros_like(picked))
    return ce, z2, ent, picked


_logit_chunk_remat = jax.checkpoint(_logit_chunk, static_argnums=(4,))


def head_losses(h: jax.Array, unembed: jax.Array, targets: jax.Array,
                mask: jax.Array, cfg, entropy: bool = False,
                per_token: bool = False):
    """Лосс по выходной проекции, не поднимая [B, T, V] целиком.

    Логиты — самый большой тензор обучения: 32×2048×32768 в fp32 это 8.6 ГБ,
    полтора на чип, и живут они не в одном экземпляре — log_softmax, z-loss и
    энтропия делают ещё по копии, а обратный проход держит их все. Ровно на
    этом прогон и упирался в HBM.

    Поэтому длина режется на куски, каждый кусок обёрнут в remat: вперёд
    логиты куска поднимаются и умирают, назад — пересчитываются. Пик падает
    во столько раз, сколько кусков, ценой одного лишнего умножения на словарь.

    Возвращает (сумма CE, сумма z², энтропия [B, T]) — суммы, а не средние:
    делить на маску должен тот, кто знает свою маску. per_token=True
    добавляет четвёртым log p цели в каждой позиции [B, T] (этап F, ORPO).
    """
    t = h.shape[1]
    chunk = cfg.loss_chunk or t
    ce = jnp.zeros((), ACCUM)
    z2 = jnp.zeros((), ACCUM)
    ents, picks = [], []
    for start in range(0, t, chunk):
        s = slice(start, min(start + chunk, t))
        c_ce, c_z2, c_ent, c_pick = _logit_chunk_remat(
            h[:, s], unembed, targets[:, s], mask[:, s], entropy)
        ce = ce + c_ce
        z2 = z2 + c_z2
        ents.append(c_ent)
        picks.append(c_pick)
    if per_token:
        return ce, z2, jnp.concatenate(ents, axis=1), jnp.concatenate(picks, axis=1)
    return ce, z2, jnp.concatenate(ents, axis=1)


def loss_fn(params: dict, batch: dict, cfg) -> tuple[jax.Array, dict]:
    """Полный лосс: язык, MTP, эпистемическая головка и две регуляризации.

    batch: {"tokens": [B, T+1]} — вход и цель со сдвигом на один.
    """
    tokens = batch["tokens"]
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    mask = (targets != 0).astype(ACCUM)   # 0 — это <|pad|>
    # Этап D1 и дальше: лосс только на том, что пишет модель (data/sft.py).
    # Маска приходит выровненной по tokens, цель — со сдвигом на один.
    if "loss_mask" in batch:
        mask = mask * batch["loss_mask"][:, 1:].astype(ACCUM)

    # head=False: логиты поднимаются кусками внутри head_losses, целиком они
    # не нужны никому — см. её описание.
    _, collect = forward(params, inputs, cfg, head=False)
    unembed = _unembed(params, cfg)
    want_entropy = bool(cfg.epistemic_head and collect.get("u_layers"))

    denom = jnp.maximum(jnp.sum(mask), 1.0)
    want_pref = "pref" in batch
    got = head_losses(collect["h_final"], unembed, targets, mask, cfg,
                      entropy=want_entropy, per_token=want_pref)
    ce_sum, z2_sum, entropy = got[:3]

    loss_lm = ce_sum / denom
    total = loss_lm
    metrics = {"loss_lm": loss_lm}

    # z-loss: держит модуль логитов от расползания. Без него bf16 на длинном
    # прогоне даёт спайки, которые видно только по внезапному NaN.
    if cfg.z_loss:
        loss_z = z2_sum / denom
        total = total + cfg.z_loss * loss_z
        metrics["loss_z"] = loss_z

    if cfg.mtp and "mtp_h" in collect:
        # Цель модуля — токен через один, поэтому сдвиг ещё на единицу.
        t2 = targets[:, 1:]
        m2 = mask[:, 1:]
        mtp_ce, _, _ = head_losses(
            collect["mtp_h"][:, :-1], unembed, t2, m2, cfg)
        loss_mtp = mtp_ce / jnp.maximum(jnp.sum(m2), 1.0)
        total = total + cfg.mtp_weight * loss_mtp
        metrics["loss_mtp"] = loss_mtp

    # Эпистемический лосс. Цель — собственная энтропия модели на этом токене,
    # отцепленная от градиента: головка учится её предсказывать, а не влиять на
    # неё. Нормировка на log V делает величину сравнимой между словарями.
    if want_entropy:
        target_u = jax.lax.stop_gradient(entropy / jnp.log(cfg.vocab_size))

        errs = [
            jnp.sum(jnp.square(u.astype(ACCUM) - target_u) * mask)
            / jnp.maximum(jnp.sum(mask), 1.0)
            for u in collect["u_layers"]
        ]
        loss_epi = jnp.mean(jnp.stack(errs))
        total = total + cfg.epistemic_weight * loss_epi
        metrics["loss_epi"] = loss_epi
        metrics["u_mean"] = jnp.mean(jnp.stack(
            [jnp.mean(u) for u in collect["u_layers"]]))
        metrics["u_target"] = jnp.mean(target_u)

    # Этап E: головка ответа учится предсказывать собственную ошибку
    # (data/calib.py). Канал 0 — на <|assistant|>, «ошибусь ли»; канал 1 —
    # на последнем токене ответа, «ошиблась ли». Цели выровнены по tokens,
    # −1 — цели нет. Бинарная кросс-энтропия, среднее по позициям с целью.
    if "epi_q" in batch and "u_answer" in collect:
        ua = jnp.clip(collect["u_answer"].astype(ACCUM), 1e-6, 1 - 1e-6)
        loss_calib = jnp.zeros((), ACCUM)
        for ch, name in enumerate(("epi_q", "epi_a")):
            y = batch[name][:, :-1].astype(ACCUM)
            has = (y >= 0).astype(ACCUM)
            y = jnp.maximum(y, 0.0)
            bce = -(y * jnp.log(ua[..., ch]) + (1 - y) * jnp.log(1 - ua[..., ch]))
            part = jnp.sum(bce * has) / jnp.maximum(jnp.sum(has), 1.0)
            metrics[f"loss_calib_{ch}"] = part
            loss_calib = loss_calib + part
        total = total + cfg.calib_weight * loss_calib
        metrics["loss_calib"] = loss_calib

    # Этап F: ORPO (Hong et al., 2024) — без референс-модели. Пара «лучше /
    # хуже» на один промпт лежит в одной строке; pref = +k на токенах
    # выбранного ответа пары k, −k — на отвергнутом (data/prefs.py). Лосс
    # языка на выбранном уже идёт через loss_mask; здесь — только
    # −log σ(log odds(лучше) − log odds(хуже)), odds по средней log p токена.
    if want_pref:
        k_max = cfg.pref_pairs
        pref = batch["pref"][:, 1:].astype(jnp.int32)
        logp_tok = got[3]
        stats = []
        for sign in (1, -1):
            idx = jnp.where(pref * sign > 0, jnp.abs(pref) - 1, k_max)
            member = jax.nn.one_hot(idx, k_max, dtype=ACCUM)        # [B,T,K]
            total_lp = jnp.einsum("bt,btk->bk", logp_tok, member)
            count = jnp.sum(member, axis=1)
            mean_lp = total_lp / jnp.maximum(count, 1.0)
            mean_lp = jnp.minimum(mean_lp, -1e-4)
            log_odds = mean_lp - jnp.log(-jnp.expm1(mean_lp))
            stats.append((log_odds, count > 0))
        (odds_c, has_c), (odds_r, has_r) = stats
        valid = (has_c & has_r).astype(ACCUM)
        gap = odds_c - odds_r
        n_pairs = jnp.maximum(jnp.sum(valid), 1.0)
        loss_or = jnp.sum(-jax.nn.log_sigmoid(gap) * valid) / n_pairs
        total = total + cfg.orpo_weight * loss_or
        metrics["loss_or"] = loss_or
        metrics["pref_acc"] = jnp.sum((gap > 0) * valid) / n_pairs
        metrics["pref_pairs"] = jnp.sum(valid)

    if collect.get("moe_aux"):
        aux = collect["moe_aux"]
        loss_router_z = jnp.mean(jnp.stack([a["router_z_loss"] for a in aux]))
        total = total + cfg.router_z_loss * loss_router_z
        metrics["loss_router_z"] = loss_router_z
        metrics["moe_dropped"] = jnp.mean(jnp.stack([a["dropped"] for a in aux]))
        metrics["gate2_open"] = jnp.mean(jnp.stack([a["gate2_open"] for a in aux]))
        # Среднее по слоям, как в замере 02.09.2026 (evals/ablations.md).
        # Разброс между слоями там был от 0.3% в слое 0 до 47% в слое 5, так
        # что среднее — это индикатор тренда, а не портрет. Послойный разрез
        # лежит рядом и печатается в лог на каждом --show-every.
        metrics["expert_share"] = jnp.mean(jnp.stack(
            [a["expert_share"] for a in aux]))
        metrics["expert_share_layers"] = jnp.stack(
            [a["expert_share"] for a in aux])
        # Загрузка нужна снаружи: по ней шаг обучения двигает router_bias.
        metrics["load"] = jnp.stack([a["load"] for a in aux])

    metrics["loss"] = total
    metrics["perplexity"] = jnp.exp(loss_lm)
    return total, metrics
