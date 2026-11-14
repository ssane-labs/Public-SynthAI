"""Базовые слои. Чистый JAX, никаких Flax и Haiku.

Параметры — вложенные словари массивов, то есть обычный pytree. Этого хватает:
jax.jit, jax.grad и jax.sharding работают с ним напрямую, а чекпоинт становится
плоским словарём без магии фреймворка. Цена — инициализацию пишем руками, зато
понятно, что где лежит, и шардить можно по имени параметра.

Соглашение по осям:
    x    [B, T, D]        батч, позиция, канал
    q    [B, T, H, Dh]    головы
    k, v [B, T, Hkv, Dh]  голов меньше — это GQA
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp

# Вычисления идут в bf16, накопления в softmax и нормализациях — в fp32.
# Смешивать приходится вручную: на TPU bf16-суммирование длинных рядов уводит
# статистику, и первым это ловит именно softmax.
COMPUTE = jnp.bfloat16
ACCUM = jnp.float32


# --- Инициализация ---------------------------------------------------------

def normal(key, shape, std: float, dtype=jnp.float32) -> jax.Array:
    return jax.random.normal(key, shape, dtype) * std


def init_linear(key, fan_in: int, *out_shape: int, scale: float = 1.0) -> jax.Array:
    """Матрица линейного слоя. std = scale/sqrt(fan_in).

    Выходные проекции блока (o_proj и down_proj эксперта) инициализируются с
    scale = 1/sqrt(2*n_layers): сумма вкладов всех слоёв в residual тогда не
    растёт с глубиной, и первые шаги не сносит.
    """
    return normal(key, (fan_in, *out_shape), std=scale / math.sqrt(fan_in))


# --- Нормализация ----------------------------------------------------------

def rms_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-6) -> jax.Array:
    """RMSNorm. Считается в fp32 независимо от типа входа."""
    x32 = x.astype(ACCUM)
    scale = jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + eps)
    return (x32 * scale).astype(x.dtype) * weight.astype(x.dtype)


# --- Позиции ---------------------------------------------------------------

def rope_tables(head_dim: int, max_len: int, base: float) -> tuple[jax.Array, jax.Array]:
    """Таблицы синусов и косинусов. Считаются один раз и живут в fp32.

    В bf16 угол на позиции 30 000 округляется до соседнего, и дальние токены
    начинают путать порядок — на 32k это уже не мелочь.
    """
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=ACCUM) / head_dim))
    t = jnp.arange(max_len, dtype=ACCUM)
    angles = jnp.outer(t, inv_freq)                 # [T, Dh/2]
    return jnp.sin(angles), jnp.cos(angles)


def apply_rope(x: jax.Array, sin: jax.Array, cos: jax.Array,
               positions: jax.Array | None = None) -> jax.Array:
    """Поворот пар каналов. x: [B, T, H, Dh].

    positions нужен при генерации, когда шаг короче кэша: там позиция токена не
    совпадает с его индексом в массиве.
    """
    if positions is not None:
        sin = jnp.take(sin, positions, axis=0)
        cos = jnp.take(cos, positions, axis=0)
    else:
        sin, cos = sin[: x.shape[1]], cos[: x.shape[1]]
    sin = sin[None, :, None, :]
    cos = cos[None, :, None, :]

    x1, x2 = jnp.split(x.astype(ACCUM), 2, axis=-1)
    out = jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return out.astype(x.dtype)


# --- Маски -----------------------------------------------------------------

def causal_window_mask(q_len: int, kv_len: int, window: int | None,
                       offset: int = 0) -> jax.Array:
    """True там, где смотреть можно. [T_q, T_kv].

    window=None — глобальный слой, видит всё прошлое. Иначе видит последние
    `window` позиций: именно это делает KV-кэш постоянным по длине.
    """
    q_pos = jnp.arange(q_len)[:, None] + offset
    kv_pos = jnp.arange(kv_len)[None, :]
    mask = kv_pos <= q_pos
    if window is not None:
        mask &= q_pos - kv_pos < window
    return mask


# --- Внимание --------------------------------------------------------------

def init_attention(key, cfg) -> dict:
    kq, kk, kv, ko = jax.random.split(key, 4)
    d, h, hkv, dh = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    params = {
        "wq": init_linear(kq, d, h, dh),
        "wk": init_linear(kk, d, hkv, dh),
        "wv": init_linear(kv, d, hkv, dh),
        "wo": init_linear(ko, h * dh, d, scale=1.0 / math.sqrt(2 * cfg.n_layers)),
    }
    if cfg.qk_norm:
        params["q_norm"] = jnp.ones((dh,))
        params["k_norm"] = jnp.ones((dh,))
    return params


def _attend(q: jax.Array, k: jax.Array, v: jax.Array, q_offset: int | jax.Array,
            kv_offset: int | jax.Array, window: int | None) -> jax.Array:
    """Ядро внимания на паре блоков. q: [B,C,Hkv,G,Dh], k/v: [B,S,Hkv,Dh].

    Позиции задаются смещениями, а не индексами в массиве: блок запросов знает,
    какие абсолютные позиции он занимает, и маска строится по ним.
    """
    dh = q.shape[-1]
    scale = 1.0 / math.sqrt(dh)
    logits = jnp.einsum("bcngk,bsnk->bngcs", q, k).astype(ACCUM) * scale

    q_pos = q_offset + jnp.arange(q.shape[1])[:, None]
    kv_pos = kv_offset + jnp.arange(k.shape[1])[None, :]
    mask = kv_pos <= q_pos
    if window is not None:
        mask &= q_pos - kv_pos < window

    logits = jnp.where(mask, logits, jnp.finfo(ACCUM).min)
    probs = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
    return jnp.einsum("bngcs,bsnk->bcngk", probs, v)


def _blocked_attention(q: jax.Array, k: jax.Array, v: jax.Array,
                       window: int | None, cfg) -> jax.Array:
    """Внимание блоками запросов. Полная матрица [T,T] не материализуется.

    Прямолинейная реализация считает [B, H, T, T] целиком: при батче 8, длине
    2048 и двенадцати головах это 1.5 ГБ на слой в fp32 — TPU v5e отдал ровно
    такой отказ на первом же прогоне рабочей длины.

    Здесь запросы идут блоками по `attn_chunk`, и каждому блоку нужен не весь
    ключевой ряд, а лишь та часть, которую он вправе видеть. Для оконных слоёв
    это `window + chunk` позиций независимо от длины последовательности — вот
    почему растяжка до 32k почти ничего не стоит: 120 МБ и на 2k, и на 32k.
    Глобальные слои берут ряд целиком, но их всего четыре из двадцати четырёх.

    Тело блока обёрнуто в checkpoint: иначе обратный проход сохранил бы
    промежуточные величины всех блоков разом, и экономия бы обнулилась.
    """
    b, t, h, dh = q.shape
    hkv = k.shape[2]
    g = h // hkv
    chunk = min(cfg.attn_chunk, t)
    n_blocks = t // chunk

    kv_len = t if window is None else min(t, window + chunk)
    q = q.reshape(b, t, hkv, g, dh)

    @jax.checkpoint
    def one_block(i):
        q_start = i * chunk
        q_blk = jax.lax.dynamic_slice(
            q, (0, q_start, 0, 0, 0), (b, chunk, hkv, g, dh))

        if window is None:
            kv_start = 0
        else:
            # Блоку нужны позиции от (начало - окно + 1) до (конец блока).
            # Клампим, чтобы срез не выходил за края: смещение всё равно
            # передаётся в маску, и лишние позиции она отсечёт.
            kv_start = jnp.clip(q_start - window + 1, 0, max(t - kv_len, 0))

        k_blk = jax.lax.dynamic_slice(k, (0, kv_start, 0, 0), (b, kv_len, hkv, dh))
        v_blk = jax.lax.dynamic_slice(v, (0, kv_start, 0, 0), (b, kv_len, hkv, dh))
        return _attend(q_blk, k_blk, v_blk, q_start, kv_start, window)

    def step(carry, i):
        return carry, one_block(i)

    _, blocks = jax.lax.scan(step, None, jnp.arange(n_blocks))
    # [n_blocks, B, C, Hkv, G, Dh] → [B, T, H*Dh]
    out = jnp.moveaxis(blocks, 0, 1).reshape(b, t, h * dh)
    return out


@functools.lru_cache(maxsize=64)
def _splash_mask(seq_len: int, window: int | None, groups: int):
    """Маска для ядра. Она ленивая и чисто numpy, поэтому кэшируется без
    опаски — в отличие от самого ядра, см. ниже."""
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_mask as sm)

    one = (sm.LocalMask((seq_len, seq_len), (window - 1, 0), 0) if window
           else sm.CausalMask((seq_len, seq_len)))
    return sm.MultiHeadMask([one] * groups)


def _splash_kernel(seq_len: int, window: int | None, groups: int, block: int,
                   interpret: bool):
    """Сплавленное внимание одним ядром Pallas.

    Почему оно вообще нужно. Блочная реализация ниже материализует матрицу
    [C, S] в памяти: einsum её пишет, маска читает и пишет, softmax читает
    дважды, и только потом она умножается на V. На оконном слое это 419 МБ на
    слой при батче 4 — и всё это гоняется через HBM трижды за шаг (прямой
    проход, пересчёт, обратный). Ядро держит блок матрицы в VMEM и наружу его
    не отдаёт вовсе.

    Маска задаётся лениво и по блокам, поэтому окно достаётся бесплатно: блоки
    ключей вне окна ядро просто не считает.

    Объект ядра НЕ кэшируется между вызовами, хотя и стоил бы дёшево. Внутри он
    держит массивы, созданные в момент постройки; постройка идёт во время
    трассировки, и экземпляр, переживший её, роняет следующую трассировку с
    UnexpectedTracerError. Кэшируется только маска — она чисто numpy.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk)

    # Все блоки одного размера: без явных обратных блоков ядро отказывается
    # дифференцироваться («Need to specify backward blocks»).
    sizes = sk.BlockSizes(
        block_q=block, block_kv=block, block_kv_compute=block,
        block_q_dkv=block, block_kv_dkv=block, block_kv_dkv_compute=block,
        block_q_dq=block, block_kv_dq=block)
    return sk.make_splash_mqa_single_device(
        _splash_mask(seq_len, window, groups), block_sizes=sizes,
        interpret=interpret)


def splash_ok(seq_len: int, cfg) -> bool:
    """Годится ли форма под ядро.

    Три условия. Внимание должно быть заказано как splash; длина должна нацело
    делиться на блок; и мы должны быть на TPU — либо явно попросить interpret,
    как делает проверка совпадения чисел. Без последнего условия все 132
    проверки на CPU уехали бы в интерпретатор Pallas и шли бы минутами.
    """
    if getattr(cfg, "attn_impl", "blocked") != "splash":
        return False
    # Ядро считает блоками по 128 дорожек и меньший блок не принимает
    # («bkv_compute must be a multiple of 128»), поэтому короткие длины и
    # некратные блоки уходят на блочный путь.
    block = cfg.splash_block
    if seq_len < block or block % 128 or seq_len % block:
        return False
    return (jax.devices()[0].platform == "tpu"
            or getattr(cfg, "splash_interpret", False))


def probe_splash(cfg, seq_len: int | None = None, sharding=None,
                 batch_size: int = 1) -> bool:
    """Собирается ли ядро на этом железе. Две секунды до обучения вместо
    восьми потерянных часов.

    Проверяется именно компиляция: Mosaic отказывает не на трассировке, а
    позже, и обычный try вокруг вызова такую беду не ловит.
    """
    if getattr(cfg, "attn_impl", "blocked") != "splash":
        return False
    if jax.devices()[0].platform != "tpu":
        # Вне TPU ядро и так не включается (см. splash_ok), а гонять его
        # интерпретатором ради пробы — только пугать лог.
        return False
    # Проба идёт на той длине, на которой пойдёт обучение: ядро проверяет
    # формы, и «собралось на 256» ничего не говорит про 2048.
    t = seq_len or max(256, cfg.splash_block * 2)
    if not splash_ok(t, cfg):
        return False
    # Батч и раскладка — как в обучении. Проба на одном чипе не поймала бы
    # отказ из-за шардинга: ядро компилируется под разрезанный батч, и это
    # другая компиляция.
    b, hkv, dh = max(batch_size, 1), cfg.n_kv_heads, cfg.head_dim
    h = cfg.n_heads
    try:
        q = jnp.zeros((b, t, h, dh), COMPUTE)
        k = jnp.zeros((b, t, hkv, dh), COMPUTE)
        if sharding is not None:
            q = jax.device_put(q, sharding)
            k = jax.device_put(k, sharding)
        fn = jax.jit(lambda a, b_, c: jnp.sum(
            _splash_attention(a, b_, c, cfg.window, cfg).astype(ACCUM)))
        jax.block_until_ready(fn(q, k, k))
        grad = jax.jit(jax.grad(lambda a, b_, c: jnp.sum(
            _splash_attention(a, b_, c, cfg.window, cfg).astype(ACCUM))))
        jax.block_until_ready(grad(q, k, k))
        return True
    except Exception as exc:                       # noqa: BLE001
        print(f"  сплавленное внимание не собралось ({type(exc).__name__}: "
              f"{str(exc)[:120]}) — идём блочным путём", flush=True)
        return False


def _splash_attention(q: jax.Array, k: jax.Array, v: jax.Array,
                      window: int | None, cfg) -> jax.Array:
    """q: [B,T,H,Dh], k/v: [B,T,Hkv,Dh] → [B,T,H*Dh].

    GQA укладывается в MQA: каждая KV-голова со своей группой Q-голов — это
    ровно multi-query, и vmap по батчу и по KV-голове даёт нужную форму без
    копий k и v.
    """
    b, t, h, dh = q.shape
    hkv = k.shape[2]
    g = h // hkv
    if k.shape[1] != t:
        # Маска строится квадратной, по длине запросов. Для генерации с
        # KV-кэшем, где запрос один, а ключей целый кэш, нужна другая маска —
        # и молчать об этом нельзя: посчитается не то, что нужно.
        raise ValueError(
            f"сплавленное внимание ждёт одинаковых длин, а пришли "
            f"q={t} и k={k.shape[1]}; для KV-кэша нужен блочный путь")
    block = cfg.splash_block
    interpret = jax.devices()[0].platform != "tpu"
    kernel = _splash_kernel(t, window, g, block, interpret)

    # Ядро не масштабирует логиты само — 1/sqrt(Dh) уезжает в q заранее.
    qs = jnp.transpose(q.reshape(b, t, hkv, g, dh), (0, 2, 3, 1, 4))
    qs = qs * jnp.asarray(1.0 / math.sqrt(dh), qs.dtype)
    ks = jnp.transpose(k, (0, 2, 1, 3))
    vs = jnp.transpose(v, (0, 2, 1, 3))

    def run(qq, kk, vv):
        return jax.vmap(jax.vmap(kernel, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))(
            qq, kk, vv)

    # Ядро Mosaic XLA между чипами сам не разрезает: «Mosaic kernels cannot be
    # automatically partitioned. Please wrap the call in a shard_map». Так и
    # делаем — батч у нас разрезан по оси data, и каждый чип считает своё,
    # ничего ни у кого не спрашивая. check_vma=False потому, что доказать
    # раскладку выхода ядра JAX не может: за неё отвечает out_specs.
    mesh = _mesh_for_shard_map()
    if mesh is not None:
        from jax.sharding import PartitionSpec as P
        from synth1.sharding import AXIS
        run = jax.shard_map(run, mesh=mesh, in_specs=(P(AXIS), P(AXIS), P(AXIS)),
                            out_specs=P(AXIS), check_vma=False)

    out = run(qs, ks, vs)
    out = jnp.transpose(out, (0, 3, 1, 2, 4))          # [B,T,Hkv,G,Dh]
    return out.reshape(b, t, h * dh).astype(q.dtype)


def _mesh_for_shard_map():
    """Сетка, если чипов больше одного. На одном устройстве shard_map не нужен
    и только мешает — например, в `parts.py`, который меряет один чип."""
    from synth1 import sharding as S

    mesh = S.current_mesh()
    if mesh is None or mesh.shape[S.AXIS] <= 1:
        return None
    return mesh


def attention(params: dict, x: jax.Array, sin: jax.Array, cos: jax.Array,
              window: int | None, cfg, use_rope: bool = True) -> jax.Array:
    b, t, _ = x.shape
    h, hkv, dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim

    q = jnp.einsum("btd,dhk->bthk", x, params["wq"].astype(x.dtype))
    k = jnp.einsum("btd,dhk->bthk", x, params["wk"].astype(x.dtype))
    v = jnp.einsum("btd,dhk->bthk", x, params["wv"].astype(x.dtype))

    # QK-norm: без него маленькая модель на большом LR срывается в спайки
    # логитов внимания — норма q растёт, softmax вырождается в argmax, градиент
    # исчезает. Стоит две нормализации, спасает весь прогон.
    if cfg.qk_norm:
        q = rms_norm(q, params["q_norm"])
        k = rms_norm(k, params["k_norm"])

    if use_rope:
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

    # GQA: группы Q-голов делят одну KV-голову. Разворачиваем в явную ось групп,
    # чтобы не материализовать копии k и v.
    g = h // hkv

    if splash_ok(t, cfg):
        return jnp.einsum("btd,de->bte", _splash_attention(q, k, v, window, cfg),
                          params["wo"].astype(x.dtype))

    if cfg.attn_chunk and t > cfg.attn_chunk:
        out = _blocked_attention(q, k, v, window, cfg)
    else:
        # Короткая последовательность: блоки только мешают, матрица и так мала.
        out = _attend(q.reshape(b, t, hkv, g, dh), k, v, 0, 0, window)
        out = out.reshape(b, t, h * dh)

    return jnp.einsum("btd,de->bte", out, params["wo"].astype(x.dtype))


# --- SwiGLU ----------------------------------------------------------------

def init_ffn(key, d_model: int, d_ff: int, n_layers: int) -> dict:
    kg, ku, kd = jax.random.split(key, 3)
    return {
        "gate": init_linear(kg, d_model, d_ff),
        "up": init_linear(ku, d_model, d_ff),
        "down": init_linear(kd, d_ff, d_model, scale=1.0 / math.sqrt(2 * n_layers)),
    }


def swiglu(params: dict, x: jax.Array) -> jax.Array:
    gate = jnp.einsum("...d,df->...f", x, params["gate"].astype(x.dtype))
    up = jnp.einsum("...d,df->...f", x, params["up"].astype(x.dtype))
    return jnp.einsum("...f,fd->...d", jax.nn.silu(gate) * up,
                      params["down"].astype(x.dtype))
