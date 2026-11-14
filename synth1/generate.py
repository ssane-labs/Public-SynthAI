"""Генерация текста.

Реализация нарочно простая: на каждом шаге прогоняется весь префикс заново, без
KV-кэша. Для того, ради чего она нужна сейчас — посмотреть, что модель говорит
после обучения, — этого достаточно, а кэш и спекуляция по MTP относятся к этапу
H и к движку инференса, который будет писаться под CUDA отдельно.

Заводить сложное кэширование раньше времени значит отлаживать его на модели,
которая ещё ничего не выучила.

    py -3 synth1/generate.py --run pretrain --prompt "The key idea is"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from synth1 import checkpoint as C  # noqa: E402
from synth1 import model as M  # noqa: E402
from synth1 import optimizer as O  # noqa: E402
from synth1.config import ModelConfig  # noqa: E402


def sample_next(logits: jax.Array, key, temperature: float, top_p: float):
    """Выбор следующего токена. temperature=0 — жадный выбор."""
    if temperature <= 0:
        return jnp.argmax(logits, axis=-1)

    logits = logits / temperature
    probs = jax.nn.softmax(logits, axis=-1)

    if top_p < 1.0:
        order = jnp.argsort(probs)[::-1]
        sorted_probs = probs[order]
        cumulative = jnp.cumsum(sorted_probs)
        # Оставляем наименьшее множество токенов, покрывающее top_p массы.
        # Сдвиг на единицу нужен, чтобы первый токен не отсекался, даже когда
        # он один уже покрывает порог.
        keep = cumulative - sorted_probs < top_p
        sorted_probs = jnp.where(keep, sorted_probs, 0.0)
        probs = jnp.zeros_like(probs).at[order].set(sorted_probs)
        probs = probs / jnp.sum(probs)

    return jax.random.categorical(key, jnp.log(probs + 1e-9))


_STEP_CACHE: dict = {}


def _forward_fixed(cfg, length: int):
    """Скомпилированный проход на буфере постоянной длины.

    Наивная генерация подаёт всё более длинный список токенов, и JAX
    перекомпилирует граф на каждом шаге: сорок токенов — сорок компиляций, что
    дороже самой генерации в сотни раз. Поэтому вход всегда одной длины,
    добитый `<|pad|>` справа, а логиты берутся с позиции последнего настоящего
    токена. Причинная маска гарантирует, что добивка ни на что не влияет.
    """
    key = (id(cfg), length)
    if key not in _STEP_CACHE:
        _STEP_CACHE[key] = jax.jit(lambda p, t: M.forward(p, t, cfg))
    return _STEP_CACHE[key]


def generate(params, cfg, tokenizer, prompt: str, max_tokens: int = 100,
             temperature: float = 0.8, top_p: float = 0.95, seed: int = 0,
             show_uncertainty: bool = False, buffer: int | None = None):
    """Возвращает (текст, список неуверенностей по шагам)."""
    ids = list(tokenizer.encode(prompt).ids)
    key = jax.random.key(seed)
    uncertainties: list[float] = []

    # Буфер с запасом на всю генерацию, но не длиннее окна модели.
    length = min(buffer or (len(ids) + max_tokens + 1), cfg.max_seq_len)
    forward = _forward_fixed(cfg, length)

    for _ in range(max_tokens):
        window = ids[-length:]
        pos = len(window) - 1
        padded = window + [config.PAD_ID] * (length - len(window))
        logits, collect = forward(params, jnp.asarray([padded], dtype=jnp.int32))

        if collect.get("u_layers"):
            uncertainties.append(float(collect["u_layers"][-1][0, pos]))

        key, subkey = jax.random.split(key)
        nxt = int(sample_next(logits[0, pos], subkey, temperature, top_p))
        if nxt == config.EOT_ID:
            break
        ids.append(nxt)

    return tokenizer.decode(ids), uncertainties


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="прогон на Hub")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--local", help="путь к локальному снимку")
    ap.add_argument("--prompt", default="The most important idea in")
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from synth1.data import load_tokenizer
    tokenizer = load_tokenizer()

    if not (args.run or args.local):
        raise SystemExit(
            "Нужен снимок: --run <имя> или --local <путь>.\n"
            "Без обученных весов генерация выдаёт шум, и смотреть на него незачем."
        )

    directory = (pathlib.Path(args.local) if args.local
                 else C.pull(args.run, args.step))
    meta = json.loads((directory / C.META).read_text(encoding="utf-8"))
    cfg = ModelConfig(**meta["config"])
    params = M.init_model(jax.random.key(0), cfg)
    params, _, meta = C.load(directory, params, O.init_state(params, O.OptConfig()))

    print(f"{meta['run']}, шаг {meta['step']:,}\n")
    text, uncertainty = generate(
        params, cfg, tokenizer, args.prompt,
        max_tokens=args.max_tokens, temperature=args.temperature,
        top_p=args.top_p, seed=args.seed,
    )
    print(text)

    if uncertainty:
        u = np.asarray(uncertainty)
        print(f"\nнеуверенность: средняя {u.mean():.3f}, "
              f"размах {u.min():.3f}–{u.max():.3f}")


if __name__ == "__main__":
    main()
