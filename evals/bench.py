"""Батарея замеров. Всё, что здесь считается, — число, а не впечатление.

Четыре группы:

    язык        перплексия на отложенных шардах, отдельно по источникам
    калибровка  сдержала ли эпистемическая головка обещание: связана ли
                предсказанная неуверенность с настоящей ошибкой
    разрежённость  сколько экспертов реально звалось, сколько токенов прошло
                мимо блока — то есть настоящая ли экономия
    скорость    токенов в секунду и вес KV-кэша

Вторая группа — главная. Весь проект держится на утверждении, что модель
способна оценивать своё незнание. Если корреляция около нуля, эпистемическая
головка — украшение, и это надо знать раньше, чем позже.

    py -3 evals/bench.py --run pretrain --step 20000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from synth1 import checkpoint as C  # noqa: E402
from synth1 import data as D  # noqa: E402
from synth1 import model as M  # noqa: E402
from synth1 import optimizer as O  # noqa: E402
from synth1.config import ModelConfig  # noqa: E402


def perplexity(params, cfg, mixture: str, seq_len: int, batches: int,
               batch_size: int = 4) -> dict:
    """Перплексия на отложенных шардах — их обучение не видело никогда."""
    stream = D.TokenStream(mixture, seq_len=seq_len, split="valid", seed=777)

    @jax.jit
    def evaluate(params, tokens):
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        mask = (targets != 0).astype(jnp.float32)
        logits, collect = M.forward(params, inputs, cfg)
        logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        picked = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]

        # Вероятность, назначенная верному токену, и предсказанная неуверенность
        # на той же позиции — сырьё для проверки калибровки.
        correct = (jnp.argmax(logits, axis=-1) == targets).astype(jnp.float32)
        u = (collect["u_layers"][-1] if collect.get("u_layers")
             else jnp.zeros_like(correct))
        return -jnp.sum(picked * mask) / jnp.sum(mask), correct, u, mask

    losses, corrects, us, masks = [], [], [], []
    for i, batch in enumerate(stream.batches(batch_size)):
        loss, correct, u, mask = evaluate(params, jnp.asarray(batch["tokens"]))
        losses.append(float(loss))
        corrects.append(np.asarray(correct))
        us.append(np.asarray(u))
        masks.append(np.asarray(mask))
        if i + 1 >= batches:
            break

    mean_loss = float(np.mean(losses))
    return {
        "loss": mean_loss,
        "perplexity": float(np.exp(mean_loss)),
        "accuracy": float(np.concatenate([c.ravel() for c in corrects]).mean()),
        "_correct": np.concatenate([c.ravel() for c in corrects]),
        "_u": np.concatenate([u.ravel() for u in us]),
        "_mask": np.concatenate([m.ravel() for m in masks]),
    }


def calibration(correct: np.ndarray, u: np.ndarray, mask: np.ndarray,
                bins: int = 10) -> dict:
    """Держит ли головка обещание.

    Проверяем две вещи. Корреляция: растёт ли предсказанная неуверенность там,
    где модель действительно ошибается. И разбиение по корзинам: в корзине с
    высоким u доля ошибок обязана быть выше, чем в корзине с низким.

    Корреляция около нуля означает, что головка выучила среднее по больнице и
    ни на что не смотрит. Это не «чуть хуже» — это отсутствие того, ради чего
    проект затевался.
    """
    keep = mask > 0
    correct, u = correct[keep], u[keep]
    error = 1.0 - correct

    if u.std() < 1e-6:
        return {"correlation": 0.0, "note": "головка выдаёт константу",
                "bins": []}

    corr = float(np.corrcoef(u, error)[0, 1])

    edges = np.quantile(u, np.linspace(0, 1, bins + 1))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (u >= lo) & (u <= hi)
        if sel.sum() < 10:
            continue
        rows.append({"u": float(u[sel].mean()),
                     "error_rate": float(error[sel].mean()),
                     "n": int(sel.sum())})

    monotone = all(rows[i]["error_rate"] <= rows[i + 1]["error_rate"] + 0.02
                   for i in range(len(rows) - 1)) if len(rows) > 1 else False
    spread = (rows[-1]["error_rate"] - rows[0]["error_rate"]) if rows else 0.0

    return {"correlation": corr, "monotone": monotone,
            "spread": float(spread), "bins": rows}


def sparsity(params, cfg, seq_len: int) -> dict:
    """Настоящая ли экономия: сколько экспертов звалось, сколько токенов прошло."""
    tokens = jax.random.randint(jax.random.key(0), (2, seq_len + 1), 1,
                                cfg.vocab_size)
    _, metrics = M.loss_fn(params, {"tokens": tokens}, cfg)

    out = {
        "gate2_open": float(metrics.get("gate2_open", 1.0)),
        "moe_dropped": float(metrics.get("moe_dropped", 0.0)),
        "u_mean": float(metrics.get("u_mean", 0.0)),
    }
    if "load" in metrics:
        load = np.asarray(metrics["load"])
        even = 1.0 / load.shape[-1]
        out["expert_imbalance"] = float(np.abs(load - even).mean() / even)
        out["expert_max_share"] = float(load.max())

    # Активные параметры с учётом того, что ворота часть времени закрыты.
    n = cfg.n_params()
    expert = 3 * cfg.d_model * cfg.d_ff
    saved = cfg.n_layers * expert * (1 - out["gate2_open"])
    if cfg.mod:
        mod_layers = cfg.n_layers // cfg.mod_every
        saved += mod_layers * expert * (cfg.n_shared_experts + cfg.top_k) * (
            1 - cfg.mod_capacity)
    out["active_nominal_M"] = n["active"] / 1e6
    out["active_effective_M"] = (n["active"] - saved) / 1e6
    out["saved_share"] = float(saved / n["active"])
    return out


def speed(params, cfg, seq_len: int, batch_size: int = 4,
          repeats: int = 5) -> dict:
    tokens = jax.random.randint(jax.random.key(1), (batch_size, seq_len + 1), 1,
                                cfg.vocab_size)
    fwd = jax.jit(lambda p, t: M.forward(p, t, cfg)[0])

    fwd(params, tokens[:, :-1]).block_until_ready()          # прогрев компиляции
    started = time.time()
    for _ in range(repeats):
        fwd(params, tokens[:, :-1]).block_until_ready()
    dt = (time.time() - started) / repeats

    return {
        "forward_ms": dt * 1000,
        "tokens_per_s": batch_size * seq_len / dt,
        "kv_cache_MB_8k": cfg.kv_cache_bytes(8192) / 1024**2,
        "kv_cache_MB_32k": cfg.kv_cache_bytes(32768) / 1024**2,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="прогон на Hub; без него меряется свежая модель")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--local", help="путь к локальному снимку")
    ap.add_argument("--preset", default="default")
    ap.add_argument("--mixture", default="pretrain")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--skip-perplexity", action="store_true",
                    help="без корпуса: только скорость и разрежённость")
    ap.add_argument("--out", default=None, help="куда положить JSON с результатом")
    args = ap.parse_args()

    from synth1.config import ABLATION, ABLATION_DENSE, DEFAULT
    presets = {"default": DEFAULT, "ablation": ABLATION,
               "ablation-dense": ABLATION_DENSE}
    cfg = presets[args.preset]

    params = M.init_model(jax.random.key(0), cfg)
    label = "свежая модель"

    if args.local or args.run:
        directory = (pathlib.Path(args.local) if args.local
                     else C.pull(args.run, args.step))
        meta = json.loads((directory / C.META).read_text(encoding="utf-8"))
        cfg = ModelConfig(**meta["config"])
        params = M.init_model(jax.random.key(0), cfg)
        state = O.init_state(params, O.OptConfig())
        params, _, meta = C.load(directory, params, state)
        label = f"{meta['run']}, шаг {meta['step']:,}"

    print(f"\n{label}: {cfg.summary()}\n")
    result: dict = {"label": label, "config": args.preset}

    print("скорость")
    result["speed"] = speed(params, cfg, args.seq_len)
    print(f"  прямой проход {result['speed']['forward_ms']:.0f} мс, "
          f"{result['speed']['tokens_per_s']:,.0f} токенов/с")
    print(f"  KV-кэш: {result['speed']['kv_cache_MB_8k']:.0f} МБ на 8k, "
          f"{result['speed']['kv_cache_MB_32k']:.0f} МБ на 32k")

    print("\nразрежённость")
    result["sparsity"] = sparsity(params, cfg, args.seq_len)
    s = result["sparsity"]
    print(f"  ворота второго эксперта открыты на {s['gate2_open']:.1%} токенов")
    print(f"  активно {s['active_nominal_M']:.0f}M номинально, "
          f"{s['active_effective_M']:.0f}M фактически "
          f"(экономия {s['saved_share']:.1%})")
    if "expert_imbalance" in s:
        print(f"  перекос загрузки экспертов {s['expert_imbalance']:.2f}, "
              f"самый нагруженный держит {s['expert_max_share']:.1%}")

    if not args.skip_perplexity:
        print("\nязык (отложенные шарды, обучение их не видело)")
        ppl = perplexity(params, cfg, args.mixture, args.seq_len, args.batches)
        correct, u, mask = ppl.pop("_correct"), ppl.pop("_u"), ppl.pop("_mask")
        result["language"] = ppl
        print(f"  перплексия {ppl['perplexity']:.2f}, "
              f"точность следующего токена {ppl['accuracy']:.1%}")

        print("\nкалибровка эпистемической головки")
        cal = calibration(correct, u, mask)
        result["calibration"] = cal
        if cal.get("note"):
            print(f"  {cal['note']}")
        else:
            print(f"  корреляция неуверенности с ошибкой {cal['correlation']:+.3f}")
            print(f"  монотонность по корзинам: "
                  f"{'да' if cal['monotone'] else 'нет'}, "
                  f"разброс ошибки {cal['spread']:.1%}")
            for row in cal["bins"][:: max(1, len(cal["bins"]) // 5)]:
                print(f"    u≈{row['u']:.3f} → ошибка {row['error_rate']:.1%} "
                      f"({row['n']:,} токенов)")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"\nрезультат: {out}")


if __name__ == "__main__":
    main()
