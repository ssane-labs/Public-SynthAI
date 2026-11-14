"""Проверки генерации на нескольких чипах (synth1/sample.py) и метрик калибровки.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \\
        python synth1/test_sample.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from synth1 import model as M  # noqa: E402
from synth1 import sample as G  # noqa: E402
from synth1.config import ABLATION  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = ""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'ok  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))


def main():
    cfg = ABLATION
    params = M.init_model(jax.random.key(3), cfg)
    rng = np.random.default_rng(0)
    prompts = [list(rng.integers(100, 5000, size=n)) for n in (5, 40, 300, 17, 9)]

    print("1. Первый токен и головка — как у прямого прохода")
    gen = G.Generator(params, cfg, tokens_per_chip=4096)
    out = gen.generate(prompts, max_new=4)
    check("ответ на каждый промпт, по порядку", len(out) == len(prompts))
    for k, p in enumerate(prompts):
        # Ширина буфера та же, что выберет генератор: от неё зависит ёмкость
        # экспертов, а значит, и сам ответ.
        width = next(w for w in G.WIDTHS if w >= len(p) + 4)
        rows = jnp.zeros((1, width), jnp.int32).at[0, :len(p)].set(jnp.asarray(p))
        logits, col = M.forward(params, rows, cfg)
        want = int(jnp.argmax(logits[0, len(p) - 1]))
        u = np.asarray(col["u_answer"][0, len(p) - 1])
        if k in (0, 2):
            check(f"промпт {len(p)}: первый токен", out[k]["ids"][0] == want,
                  f"{out[k]['ids'][0]} / {want}")
            check(f"промпт {len(p)}: u до ответа",
                  np.allclose(out[k]["u_first"], u, atol=2e-2),
                  f"{out[k]['u_first']} / {u.tolist()}")

    print("\n2. Запреты, остановка, температура")
    banned = out[0]["ids"][0]
    again = gen.generate(prompts[:1], max_new=4, ban=(banned,))
    check("запрещённый токен не выдаётся", banned not in again[0]["ids"])
    stop = gen.generate(prompts[:1], max_new=4, stop=(banned,))
    check("стоп-токен останавливает и остаётся в ответе",
          stop[0]["ids"] == [banned] and not np.isnan(stop[0]["u_last"][0]))
    hot = gen.generate(prompts * 4, max_new=6, temps=[2.0] * 20, seed=7)
    check("с температурой ответы разные",
          len({tuple(h["ids"]) for h in hot}) > len(prompts))

    print("\n3. AUROC и ECE")
    check("AUROC: идеальное разделение — 1", G.auroc([0.9, 0.8, 0.1], [1, 1, 0]) == 1.0)
    check("AUROC: ничьи — 0.5", G.auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5)
    check("AUROC: обратное — 0", G.auroc([0.1, 0.9], [1, 0]) == 0.0)
    check("ECE: точная калибровка — 0",
          abs(G.ece([0.25] * 4, [1, 0, 0, 0])) < 1e-9)
    check("ECE: всё уверенно и всё неверно — 1",
          abs(G.ece([0.0] * 4, [1, 1, 1, 1]) - 1.0) < 1e-9)

    print(f"\n{'=' * 52}")
    print(f"пройдено {len(PASSED)}, провалено {len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print(f"  FAIL {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
