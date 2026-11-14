"""Профиль шага на чипе: какие операции съедают время.

`speed.py` отвечает «сколько стоит шаг», `parts.py` — «сколько стоит часть
модели в отдельности». Этот файл снимает трассу настоящего шага обучения,
целиком и с той же раскладкой по чипам, — то, на чём поймано 04.10.2026, что
MoE считался на всём батче на каждом чипе.

    python evals/profile.py --batch-size 32 --out /kaggle/working/prof

Трасса ложится в --out (формат TensorBoard/xprof). Разбирается где угодно:

    pip install xprof
    python evals/profile.py --read /путь/к/prof
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import glob
import gzip
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def record(args) -> None:
    import jax
    import jax.numpy as jnp

    from evals import speed as SP
    from synth1 import model as M
    from synth1 import optimizer as O
    from synth1 import sharding as S
    base = SP.PRESETS[args.preset]
    cfg = dataclasses.replace(base, max_seq_len=max(base.max_seq_len, args.seq_len))
    for pair in args.set:
        name, _, raw = pair.partition("=")
        cur = getattr(cfg, name)
        val = (raw.lower() in ("1", "true")) if isinstance(cur, bool) else type(cur)(raw)
        cfg = dataclasses.replace(cfg, **{name: val})
    opt = O.OptConfig()
    if args.no_muon:
        opt = dataclasses.replace(opt, use_muon=False)
    mesh = S.make_mesh()
    params = S.distribute(M.init_model(jax.random.key(0), cfg), mesh)
    state = O.init_state(params, opt)
    state = {**state, "m": S.distribute(state["m"], mesh),
             "v": S.distribute(state["v"], mesh)}
    tokens = jax.random.randint(jax.random.key(1),
                                (args.batch_size, args.seq_len + 1), 1,
                                cfg.vocab_size, dtype=jnp.int32)
    batch = {"tokens": jax.device_put(tokens, S.batch_sharding(mesh))}
    fn = SP.build_step(cfg, opt, 200_000, mesh, fsdp=not args.auto_sharding)

    t0 = time.perf_counter()
    for _ in range(3):
        params, state, _ = fn(params, state, batch)
    jax.block_until_ready((params, state))
    print(f"компиляция и прогрев: {time.perf_counter() - t0:.0f} с", flush=True)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        params, state, _ = fn(params, state, batch)
    jax.block_until_ready((params, state))
    sec = (time.perf_counter() - t0) / args.steps
    n_tok = args.batch_size * args.seq_len
    peak = SP.peak_tflops()
    tflops = SP.flops_per_step(cfg, n_tok) / sec / 1e12
    mem = jax.devices()[0].memory_stats() or {}
    print(f"шаг {sec * 1000:.1f} мс, {n_tok / sec:,.0f} ток/с, "
          f"{tflops:.1f} TFLOPS, MFU {100 * tflops / peak:.1f}% (6·N·T), "
          f"пик памяти чипа {mem.get('peak_bytes_in_use', 0) / 1024**3:.2f} ГБ",
          flush=True)

    if args.out:
        jax.profiler.start_trace(args.out)
        for _ in range(3):
            params, state, _ = fn(params, state, batch)
        jax.block_until_ready((params, state))
        jax.profiler.stop_trace()
        print(f"трасса: {args.out}", flush=True)


def read(path: str, top: int = 40) -> None:
    """Сводка по операциям одного чипа из трассы xprof."""
    from xprof.convert import raw_to_tool_data as R

    files = glob.glob(f"{path}/**/*.xplane.pb", recursive=True)
    if not files:
        raise SystemExit(f"в {path} нет *.xplane.pb")
    data, _ = R.xspace_to_tool_data(files[:1], "trace_viewer@", {})
    if isinstance(data, bytes) and data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    trace = json.loads(data)
    names = {p["pid"]: p["args"]["name"] for p in trace.get("traceEvents", [])
             if p.get("ph") == "M" and p.get("name") == "process_name"}
    # На чипе процесс трассы называется /device:TPU:0; на CPU своих устройств
    # в трассе нет, и берётся хост — для проверки самого разбора этого хватит.
    ordered = sorted(names.items())
    chip = next((pid for pid, n in ordered if "TPU:0" in n),
                next((pid for pid, n in ordered if "device" in n.lower()),
                     ordered[0][0]))
    print(f"процесс трассы: {names[chip]}")
    ops = collections.Counter()
    for ev in trace["traceEvents"]:
        if ev.get("pid") == chip and ev.get("ph") == "X":
            ops[ev["name"]] += ev.get("dur", 0)
    total = sum(ops.values())
    print(f"сумма по операциям {total / 1000 / 3:.1f} мс на шаг")
    for name, us in ops.most_common(top):
        print(f"  {us / 1000 / 3:8.2f} мс  {100 * us / total:5.1f}%  {name[:90]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="default")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--auto-sharding", action="store_true")
    ap.add_argument("--no-muon", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--read", default="")
    args = ap.parse_args()
    if args.read:
        read(args.read)
    else:
        record(args)


if __name__ == "__main__":
    main()
