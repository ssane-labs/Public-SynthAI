"""Проверка чекпоинтов: снимок обязан восстанавливаться до последнего бита.

Тест здесь один и он главный: обучаем несколько шагов, снимаем снимок, обучаем
ещё, восстанавливаемся из снимка и повторяем те же шаги. Результат обязан
совпасть точно. Расхождение означает, что после перезапуска сессии обучение
пойдёт другим путём — а это ровно то, ради чего чекпоинты и делаются.

    py -3 synth1/test_checkpoint.py
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from synth1 import checkpoint as C  # noqa: E402
from synth1 import model as M  # noqa: E402
from synth1 import optimizer as O  # noqa: E402
from synth1.test_model import TINY, FAILED, PASSED, check  # noqa: E402


def main():
    cfg, opt = TINY, O.OptConfig(warmup_steps=5)
    key = jax.random.key(0)
    params = M.init_model(key, cfg)
    state = O.init_state(params, opt)
    batch = {"tokens": jax.random.randint(jax.random.key(1), (2, 33), 1,
                                          cfg.vocab_size)}

    @jax.jit
    def step(params, state, batch):
        (loss, _), grads = jax.value_and_grad(M.loss_fn, has_aux=True)(
            params, batch, cfg)
        params, state, _ = O.apply_updates(params, grads, state, 100, opt)
        return params, state, loss

    print("\n1. Пять шагов, снимок")
    for _ in range(5):
        params, state, loss = step(params, state, batch)
    loss_at_5 = float(loss)
    print(f"    шаг 5: loss {loss_at_5:.4f}")

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="synth1-ckpt-"))
    try:
        meta = {"run": "test", "corpus": {"reader": "заглушка"}, "config": "TINY"}
        C.save(tmp, params, state, meta)
        check("файлы на месте",
              (tmp / C.TENSORS).exists() and (tmp / C.META).exists())

        print("\n2. Ещё пять шагов — эталон")
        ref_params, ref_state = params, state
        for _ in range(5):
            ref_params, ref_state, ref_loss = step(ref_params, ref_state, batch)
        print(f"    шаг 10: loss {float(ref_loss):.4f}")

        print("\n3. Восстановление и те же пять шагов")
        # Образцы структуры берём свежие — как это и будет в новой сессии,
        # где ничего, кроме конфигурации, ещё не существует.
        fresh = M.init_model(jax.random.key(99), cfg)
        fresh_state = O.init_state(fresh, opt)
        got_params, got_state, got_meta = C.load(tmp, fresh, fresh_state)

        check("шаг восстановлен", int(got_state["step"]) == 5,
              str(int(got_state["step"])))
        check("метаданные на месте", got_meta["run"] == "test")

        max_diff = max(
            float(jnp.max(jnp.abs(a - b)))
            for a, b in zip(jax.tree.leaves(got_params), jax.tree.leaves(params))
        )
        check("веса совпали бит в бит", max_diff == 0.0, f"max|Δ|={max_diff:.2e}")

        m_diff = max(
            float(jnp.max(jnp.abs(a - b)))
            for a, b in zip(jax.tree.leaves(got_state["m"]),
                            jax.tree.leaves(state["m"]))
        )
        check("momentum совпал бит в бит", m_diff == 0.0, f"max|Δ|={m_diff:.2e}")

        for _ in range(5):
            got_params, got_state, got_loss = step(got_params, got_state, batch)
        print(f"    шаг 10 после восстановления: loss {float(got_loss):.4f}")

        check("обучение пошло тем же путём",
              abs(float(got_loss) - float(ref_loss)) < 1e-6,
              f"эталон {float(ref_loss):.6f}, после восстановления "
              f"{float(got_loss):.6f}")

        final_diff = max(
            float(jnp.max(jnp.abs(a - b)))
            for a, b in zip(jax.tree.leaves(got_params),
                            jax.tree.leaves(ref_params))
        )
        check("и веса сошлись точно", final_diff == 0.0, f"max|Δ|={final_diff:.2e}")

        print("\n3б. Только веса (конец этапа SFT): вдвое легче, оптимизатор с нуля")
        light = tmp.parent / (tmp.name + "-light")
        C.save(light, params, state, meta, weights_only=True)
        full_size = (tmp / C.TENSORS).stat().st_size
        light_size = (light / C.TENSORS).stat().st_size
        check("снимок легче полного", light_size < 0.6 * full_size,
              f"{light_size / 1e6:.1f} / {full_size / 1e6:.1f} МБ")
        lp, ls, lmeta = C.load(light, fresh, fresh_state)
        check("веса те же",
              max(float(jnp.max(jnp.abs(a - b))) for a, b in
                  zip(jax.tree.leaves(lp), jax.tree.leaves(params))) == 0.0)
        check("состояние — из образца, шаг — из снимка",
              int(ls["step"]) == 5 and lmeta.get("weights_only") is True and
              all(bool(jnp.all(a == b)) for a, b in
                  zip(jax.tree.leaves(ls["m"]), jax.tree.leaves(fresh_state["m"]))))
        shutil.rmtree(light, ignore_errors=True)

        print("\n4. Несовпадение конфигурации ловится внятно")
        from synth1.config import ModelConfig
        other = M.init_model(jax.random.key(0),
                             ModelConfig(**{**TINY.__dict__, "n_layers": 6}))
        try:
            C.load(tmp, other, O.init_state(other, opt))
            check("чужой чекпоинт отклонён", False, "загрузился молча")
        except SystemExit as exc:
            check("чужой чекпоинт отклонён", "нет" in str(exc).lower(),
                  str(exc).split("\n")[0][:60])

        print("\n4b. Загрузка не собирает модель на одном чипе")
        # Тензор возвращается туда, где ему жить, а не на нулевое устройство.
        # На одном чипе разницы не видно, на восьми она стоила сессии: снимок
        # приезжал репликой рядом с уже разложенной моделью, и шестнадцать
        # гигабайт HBM кончались на аллокации в 54 МБ.
        from synth1 import sharding as S
        mesh = S.make_mesh()
        spread = S.distribute(M.init_model(jax.random.key(3), cfg), mesh)
        spread_state = O.init_state(spread, opt)
        spread_state = {**spread_state,
                        "m": S.distribute(spread_state["m"], mesh),
                        "v": S.distribute(spread_state["v"], mesh)}
        back, back_state, _ = C.load(tmp, spread, spread_state)

        check("раскладка весов сохранена",
              all(a.sharding == b.sharding for a, b in
                  zip(jax.tree.leaves(back), jax.tree.leaves(spread))))
        check("раскладка momentum сохранена",
              all(a.sharding == b.sharding for a, b in
                  zip(jax.tree.leaves(back_state["m"]),
                      jax.tree.leaves(spread_state["m"]))))

        print("\n5. Место на диске")
        check("снимок оценивается до записи",
              C.size_gb(params, state["m"], state["v"]) > 0)
        check("свободное место видно", C.free_gb(tmp) > 0)

        # Порог задран выше любого диска: так проверяется отказ писать, а не
        # сам диск. Полупустой папки после отказа остаться не должно.
        tight = tmp / "step-999"
        try:
            C.save(tight, params, state, meta, margin_gb=10.0**9)
            check("без места снимок не пишется", False, "записался")
        except OSError as exc:
            check("без места снимок не пишется", not tight.exists(),
                  str(exc)[:60])

        print("\n6. Ротация локальных снимков")
        room = tmp / "rotate"
        for n in (1, 2, 3):
            (room / f"step-{n}").mkdir(parents=True)
        C.rotate(room, keep=1)
        check("keep=1 оставляет последний",
              [d.name for d in room.iterdir()] == ["step-3"])
        C.rotate(room, keep=0)
        check("keep=0 сносит всё", not list(room.iterdir()))

        # Ротация на Hub. Сеть здесь не нужна: проверяется выбор жертв, а он
        # чистая функция. Цена ошибки — снесённый последний снимок, то есть
        # весь прогон, поэтому проверяется отдельно от заливки.
        check("на Hub остаются свежие",
              C.steps_to_prune([2000, 4000, 6000], keep=1) == [2000, 4000])
        check("keep=2 оставляет два",
              C.steps_to_prune([2000, 4000, 6000], keep=2) == [2000])
        check("сносить нечего, когда снимков меньше keep",
              C.steps_to_prune([6000], keep=2) == [])
        check("порядок не по алфавиту, а по числу",
              C.steps_to_prune([9000, 10000, 11000], keep=1) == [9000, 10000])
        check("квота отличается от сети",
              C._is_storage_full(Exception("Private repository storage limit reached"))
              and not C._is_storage_full(Exception("Connection reset by peer")))

        # Поиск снимка в output ноутбука. Сессия 29.08.2026 оставила
        # step-33477 в /kaggle/working, и это единственный путь его вернуть;
        # ошибка здесь означает поднятый не тот шаг или половинный снимок.
        out = tmp / "output"
        for step in (14000, 33477, 6000):
            d = out / "synth-1" / "checkpoints" / "pretrain" / f"step-{step}"
            d.mkdir(parents=True)
            (d / C.META).write_text("{}", encoding="utf-8")
            (d / C.TENSORS).write_bytes(b"x")
        check("берётся самый свежий, а не первый попавшийся",
              C.find_snapshot(out).name == "step-33477",
              C.find_snapshot(out).name)
        # Половинный снимок выглядит как снимок и разваливается на загрузке —
        # в следующей сессии, когда чинить уже поздно.
        broken = out / "synth-1" / "checkpoints" / "pretrain" / "step-40000"
        broken.mkdir(parents=True)
        (broken / C.META).write_text("{}", encoding="utf-8")
        check("снимок без тензоров не берётся",
              C.find_snapshot(out).name == "step-33477",
              C.find_snapshot(out).name)
        check("пустой каталог — ничего", C.find_snapshot(tmp / "нет") is None)

        print("\nПерекладывание репозитория ради квоты")
        # Считается занятым заметно больше, чем лежит в HEAD, — значит в
        # истории висят блобы. Замер 08.09.2026: 65.08 ГБ при 9.3 в
        # снимках, то есть семь заливок одной сессии.
        check("мусор в истории виден", C.needs_compacting(65.08, 9.30))
        check("свежий репозиторий не трогаем", not C.needs_compacting(9.30, 9.30))
        # Счётчик Hub отстаёт на часы в обе стороны, поэтому порог с
        # запасом: лишнее перекладывание стоит десяти минут, пропущенное —
        # всей сессии.
        check("небольшой перекос терпим", not C.needs_compacting(13.0, 9.30))
        check("пустого репозитория нет", not C.needs_compacting(0.0, 0.0))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 52}")
    print(f"пройдено {len(PASSED)}, провалено {len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print(f"  FAIL {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
