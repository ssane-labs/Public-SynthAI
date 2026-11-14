"""Прогнать батарею абляций подряд и собрать из них таблицу.

Абляция — это `train/pretrain.py` с одной правкой конфигурации и коротким
бюджетом шагов. Руками это десять команд, десять логов и вычитывание чисел
глазами; здесь — одна команда, один бюджет времени и готовая таблица в конце.

Правила те же, что в `evals/ablations.md`, и они вшиты в код, а не в дисциплину:

* одно изменение за прогон — каждая строка списка меняет ровно одно поле;
* равные условия — длина, батч, шаги, смесь и seed одинаковы у всех;
* прогон, который не состоялся, попадает в таблицу с причиной, а не пропадает;
* строка без числа в таблицу не попадает.

**Бюджет времени главнее полноты списка.** Квота TPU кончается молча, и десять
начатых абляций хуже четырёх законченных: незаконченная не сравнима ни с чем.
Поэтому первый прогон замеряется, а дальше каждый следующий запускается только
если он целиком помещается в остаток с запасом.

    python evals/ablate.py --hours 2.5
    python evals/ablate.py --hours 0.5 --only router,MoD
    python evals/ablate.py --hours 3 --steps 3000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


@dataclasses.dataclass
class Run:
    """Одна строка батареи."""

    key: str                        # латиницей: имя файла и имя прогона
    name: str                       # как назовём в таблице
    question: str                   # на какой вопрос отвечает
    overrides: tuple[str, ...] = ()  # правки конфигурации: поле=значение
    preset: str = "ablation"
    extra: tuple[str, ...] = ()     # прочие аргументы pretrain.py


# Порядок — по убыванию подозрения, а не по красоте. Первым идёт база: без неё
# остальные строки не с чем сравнивать, и если времени хватит только на две,
# это должны быть база и главный подозреваемый.
BATTERY: list[Run] = [
    Run("base", "база",
        "с чем сравниваем: всё как в претрейне"),
    Run("lr-x3", "LR ×3",
        "батарея 19.09 дала -0.144 на LR ×2 — единственная строка за порогом. "
        "Оптимум надо взять в вилку, а не остановиться на первом улучшении",
        extra=("--lr-scale", "3.0")),
    Run("lr-x4", "LR ×4",
        "верхняя граница вилки: где повышение начнёт портить",
        extra=("--lr-scale", "4.0")),
    Run("base-seed1", "база, seed 1",
        "сколько шумит сама батарея: то же самое другим seed. Без этого числа "
        "порог 0.05 — догадка, а не порог",
        extra=("--seed", "1")),
    Run("lr-x2-seed1", "LR ×2, seed 1",
        "выигрыш ×2 — это механизм или везение одного seed",
        extra=("--lr-scale", "2.0", "--seed", "1")),
    Run("renorm", "router_renorm вкл",
        "нормировать ли веса top-k на их сумму — главный подозреваемый в плато: "
        "сейчас эксперт входит в выход с сырой вероятностью ~1/12, а общий — с "
        "единицей",
        overrides=("router_renorm=true",)),
    Run("no-mod", "без MoD",
        "адаптивная глубина: вклад экспертов проседает именно на слоях с ней "
        "(0.52 → 0.23 против 0.40 → 0.37 на слоях без неё)",
        overrides=("mod=false",)),
    Run("dense", "плотный",
        "оправдана ли разрежённость вообще: те же активные FLOPs без роутера, "
        "без ворот и без MoD",
        preset="ablation-dense"),
    Run("no-gate", "без ворот",
        "ворота второго эксперта по неуверенности — центральная идея проекта, "
        "и она ни разу не мерилась",
        overrides=("entropy_gate=false",)),
    Run("mod-75", "MoD 75%",
        "если MoD вредит, то вредит ли доля или сам механизм",
        overrides=("mod_capacity=0.75",)),
    Run("top1", "роутер top-1",
        "нужен ли второй эксперт вообще",
        overrides=("top_k=1",)),
    Run("lr-x2", "LR ×2",
        "для узкой модели оптимальный LR обычно выше; у Synth-2 понижение дало "
        "ступеньку, а повышение не пробовали",
        extra=("--lr-scale", "2.0")),
]


def run_one(spec: Run, args, out_dir: pathlib.Path, hours: float) -> dict:
    """Один прогон. Возвращает итог из JSON или запись о неудаче."""
    summary = out_dir / f"{spec.key}.json"
    cmd = [
        sys.executable, str(ROOT / "train" / "pretrain.py"),
        "--run", f"abl-{spec.key}",
        "--preset", spec.preset,
        "--mixture", args.mixture,
        "--seq-len", str(args.seq_len),
        "--batch-size", str(args.batch_size),
        "--steps", str(args.steps),
        # Расписание LR обязано быть одинаковым у всех строк, иначе сравнение
        # идёт не про механизм, а про то, кто где оказался на кривой.
        "--total-steps", str(args.steps),
        # Разогрев по умолчанию равен длине всей абляции: без этой строки
        # пикового LR не увидит ни один прогон.
        "--warmup", str(max(args.steps // 20, 10)),
        "--hours", f"{hours:.3f}",
        "--eval-every", str(args.eval_every),
        "--show-every", str(args.steps),
        "--log-every", str(max(args.steps // 20, 1)),
        "--no-checkpoint",
        "--summary", str(summary),
    ]
    for pair in spec.overrides:
        cmd += ["--set", pair]
    if args.fake_data:
        cmd.append("--fake-data")
    cmd += list(spec.extra)

    print(f"\n{'=' * 70}\n{spec.name}\n  {spec.question}", flush=True)
    print("$ " + " ".join(cmd[1:]), flush=True)
    started = time.time()
    code = subprocess.run(cmd, cwd=ROOT).returncode
    spent = time.time() - started

    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        data["wall"] = spent
        data["name"] = spec.name
        data["question"] = spec.question
        return data
    return {"name": spec.name, "question": spec.question, "wall": spent,
            "failed": f"код {code}, итога нет"}


def table(rows: list[dict], args) -> str:
    """Таблица в том виде, в каком её ждёт `evals/ablations.md`."""
    base = next((r for r in rows if r["name"] == "база" and "failed" not in r), None)
    out = [
        f"| Вариант | Отложенный лосс | Δ к базе | Эксперты | мс/шаг | Вердикт |",
        f"|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "failed" in r:
            out.append(f"| {r['name']} | — | — | — | — | не состоялся: {r['failed']} |")
            continue
        loss = r.get("holdout_last")
        share = r.get("expert_share")
        if loss is None:
            out.append(f"| {r['name']} | — | — | "
                       + (f"{share:.3f}" if share is not None else "—")
                       + f" | {r.get('ms_per_step', 0):.0f} | "
                       f"прошёл без отложенного набора — числа нет |")
            continue
        delta = (loss - base["holdout_last"]) if base and base.get("holdout_last") else None
        if r is base:
            verdict = "опорная строка"
        elif delta is None:
            verdict = "нет числа"
        elif abs(delta) < args.threshold:
            verdict = f"в пределах шума (порог {args.threshold})"
        else:
            verdict = "**лучше**" if delta < 0 else "хуже"
        out.append(
            f"| {r['name']} | {loss:.4f} | "
            + (f"{delta:+.4f}" if delta is not None else "—") + " | "
            + (f"{share:.3f}" if share is not None else "—") + " | "
            f"{r.get('ms_per_step', 0):.0f} | {verdict} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, required=True,
                    help="весь бюджет на батарею, включая замеры")
    ap.add_argument("--steps", type=int, default=2000,
                    help="шагов в каждой абляции; условие по умолчанию из "
                         "evals/ablations.md")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mixture", default="pretrain")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="разница отложенного лосса, ниже которой решение не "
                         "принимается (evals/ablations.md)")
    ap.add_argument("--reserve-min", type=float, default=6.0,
                    help="минут, которые не трогаем: на последний прогон, на "
                         "печать таблицы и на то, чтобы Kaggle не убил сессию "
                         "на полуслове")
    ap.add_argument("--only", default="",
                    help="только строки, чьё имя содержит эти подстроки через "
                         "запятую; «база» остаётся всегда")
    ap.add_argument("--fake-data", action="store_true",
                    help="случайные токены вместо корпуса: проверить, что вся "
                         "цепочка работает, не тратя ни минуты TPU на смысл. "
                         "Отложенного набора при этом нет, и таблица выйдет "
                         "пустой — это нормально")
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "evals" / "out" / "ablations")
    args = ap.parse_args()

    plan = BATTERY
    if args.only:
        wanted = [w.strip().lower() for w in args.only.split(",") if w.strip()]
        plan = [r for r in BATTERY
                if r.name == "база" or any(w in r.name.lower() for w in wanted)]

    args.out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.hours * 3600
    reserve = args.reserve_min * 60

    print(f"батарея: {len(plan)} строк, бюджет {args.hours:.2f} ч, "
          f"{args.steps} шагов на строку, батч {args.batch_size}×{args.seq_len}")
    print("порядок — по убыванию подозрения; строка запускается только если "
          "целиком помещается в остаток")

    rows: list[dict] = []
    longest = 0.0
    for i, spec in enumerate(plan):
        left = deadline - time.time() - reserve
        # Первому прогону даём всё, что есть: мы ещё не знаем, сколько он стоит.
        # Дальше требуем, чтобы поместился самый долгий из виденных, плюс 15%.
        need = longest * 1.15 if longest else 0.0
        if left <= 0 or (need and left < need):
            skipped = [s.name for s in plan[i:]]
            print(f"\nостаток {left / 60:.0f} мин — не хватает на прогон "
                  f"(нужно ~{need / 60:.0f} мин). Не запускались: "
                  f"{', '.join(skipped)}", flush=True)
            break
        row = run_one(spec, args, args.out, hours=max(left, 60) / 3600)
        rows.append(row)
        longest = max(longest, row["wall"])
        done, share = row.get("holdout_last"), row.get("expert_share")
        if "failed" in row:
            note = f"НЕ СОСТОЯЛСЯ: {row['failed']}"
        elif done is None:
            note = "прошёл, но отложенного набора не было — сравнивать нечем"
        else:
            note = f"отложенный {done:.4f}"
            if share is not None:
                note += f", эксперты {share:.3f}"
        print(f"  → {spec.name}: {note}, {row['wall'] / 60:.1f} мин", flush=True)

    print(f"\n{'=' * 70}\nИТОГ БАТАРЕИ\n")
    print(table(rows, args))
    print(f"\nУсловия: {args.steps} шагов, {args.batch_size}×{args.seq_len}, "
          f"смесь {args.mixture!r}, порог решения {args.threshold}.")
    print("Строки с вердиктом «в пределах шума» решения не дают: разница "
          "меньше порога — это не «одинаково», а «не померили».")
    (args.out / "table.md").write_text(table(rows, args) + "\n", encoding="utf-8")
    print(f"\nтаблица: {args.out / 'table.md'}")


if __name__ == "__main__":
    main()
