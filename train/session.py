"""Сессия целиком: запускает обучение и разбирается с бедами без человека.

Ноутбук жмут с телефона и смотрят лог потом. Между запуском и разбором может
пройти неделя, и попросить что-то поправить в промежутке нельзя — значит всё,
что можно починить на месте, должно чиниться на месте.

Что этот файл умеет, кроме как запустить `train/pretrain.py`:

    не влезло в память      батч вдвое меньше и заново
    сорвался лосс           откат к снимку и обход данных, на которых сорвались
    упало непонятно чем     ещё одна попытка, вдруг сеть
    кончилось время         выход; следующая сессия продолжит с Hub

Плюс `--probe-lr`: перед обучением два коротких прогона с одного снимка и
разными LR, чтобы ответить, не слишком ли высоко плато. Ответ нужен раньше,
чем будут потрачены восемь часов, а руками с телефона два прогона не запустить.

Почему отдельным процессом, а не циклом внутри pretrain: после нехватки памяти
на TPU процесс уже нельзя считать здоровым — часть буферов осталась занятой, и
следующая попытка меряется не с чистого листа. Свежий процесс дешевле любой
попытки прибраться.

    py -3 train/session.py --hours 8.2
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Что в хвосте лога означает какую беду. Строки — те самые, что печатают
# pretrain.py и XLA; менять их надо вместе.
OUT_OF_MEMORY = ("RESOURCE_EXHAUSTED", "Out of memory",
                 "RuntimeBufferAllocationFailure")
TIME_IS_UP = "лимит времени сессии"
# Нехватка памяти на загрузке снимка — это не про батч. Батч определяет размер
# активаций, а снимок приезжает до первого шага, и уменьшать тут нечего:
# 28.08.2026 сессия честно спустилась с 32 до 8 и упала все три раза на одной
# и той же строке.
LOADING = ("checkpoint.py", "_unflatten", "C.load")
STEPPED = "шаг "
LOSS_BROKE = "лосс перестал быть конечным"
NOWHERE = "снимок некуда девать"
DONE = "заданное число шагов"
# Локальный диск Kaggle (20 ГБ) кончился: скачивание снимка падает до первого
# шага, и повтор упирается в то же место — 29.10.2026 восемь попыток подряд.
DISK_FULL = ("No space left on device", "os error 28")


def decide(code: int, tail: str) -> str:
    """Что делать после прогона. Чистая функция — её и проверяют тесты.

    Порядок проверок важнее их содержания: время вышло — это успех, даже если
    рядом в логе есть слово об ошибке, а нехватка памяти важнее непонятного
    кода возврата, потому что объясняет его.
    """
    if TIME_IS_UP in tail:
        return "время вышло"
    if any(mark in tail for mark in OUT_OF_MEMORY):
        if any(mark in tail for mark in LOADING) and STEPPED not in tail:
            return "не влезло на загрузке"
        return "не влезло"
    if LOSS_BROKE in tail:
        return "лосс сорвался"
    # Место на Hub кончилось. Перезапуск не поможет: снимки будут уезжать в ту
    # же переполненную квоту, а локальный не переживёт контейнер. Дальше сессия
    # считала бы в никуда — так 29.08.2026 сгорели 19,477 шагов.
    if NOWHERE in tail:
        return "снимок некуда девать"
    if code != 0 and any(mark in tail for mark in DISK_FULL):
        return "диск полон"
    if code != 0:
        return "непонятная поломка"
    if DONE in tail:
        return "прогон закончен"
    return "прогон закончен"


# Строка итога отложенного набора из pretrain.py. Меняется вместе с ней.
HOLDOUT = re.compile(
    r"отложенный лосс: ([\d.]+) \(шаг ([\d,]+)\) → ([\d.]+) \(шаг ([\d,]+)\), "
    r"([+-][\d.]+)"
)


def read_holdout(tail: str) -> tuple[float, float, float] | None:
    """(первый, последний, сдвиг) из хвоста лога прогона. Чистая — её проверяют тесты.

    None означает, что прогон не дожил до второго замера или отложенный набор
    не собрался. Тогда сравнивать нечего, и врать числом нельзя.
    """
    found = HOLDOUT.findall(tail)
    if not found:
        return None
    first, _, last, _, move = found[-1]
    return float(first), float(last), float(move)


def compare_probes(results: dict[float, tuple[float, float, float] | None]) -> str:
    """Вердикт по пробам. Чистая функция ради теста: цена ошибки — неверный
    вывод про LR, а его потом закрепят на недели обучения."""
    measured = {k: v for k, v in results.items() if v is not None}
    if len(measured) < 2:
        return ("проба не удалась: отложенный лосс снят меньше чем на двух "
                "прогонах, сравнивать нечего")
    best = min(measured, key=lambda k: measured[k][2])
    moves = {k: v[2] for k, v in measured.items()}
    spread = max(moves.values()) - min(moves.values())
    # Порог: сдвиги отложенного лосса меряются сотыми, и разница меньше 0.01
    # между двумя тысячами шагов — это не разница, а тот же результат.
    if spread < 0.01:
        return ("плато ни при чём: все LR дали одинаковый сдвиг "
                f"(разброс {spread:.4f}). Следующий подозреваемый — "
                "router_renorm, и он проверяется абляцией с нуля")
    return (f"лучший LR — множитель {best}: сдвиг {moves[best]:+.4f} против "
            + ", ".join(f"{k} → {v:+.4f}" for k, v in sorted(moves.items())
                        if k != best))


def free_checkpoints(keep: str) -> list[str]:
    """Удалить локальные снимки всех прогонов, кроме keep. Они лежат на Hub."""
    import shutil
    root = ROOT / "checkpoints"
    gone = []
    for d in sorted(root.glob("*")) if root.exists() else []:
        if d.is_dir() and d.name != keep:
            shutil.rmtree(d, ignore_errors=True)
            gone.append(d.name)
    return gone


def run_live(cmd: list[str], tail_lines: int = 500) -> tuple[int, str]:
    """Запуск с выводом по мере появления. Возвращает (код, хвост).

    По мере появления — не роскошь: прогон идёт часами, и если копить вывод в
    памяти, то при обрыве сессии в логе не останется ни лосса, ни причины.
    """
    print("$", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=env, encoding="utf-8", errors="replace")
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        if len(tail) > tail_lines:
            tail.pop(0)
    proc.wait()
    return proc.returncode, "".join(tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=8.2,
                    help="сколько всего времени у сессии, считая уже потраченное")
    ap.add_argument("--run", default="pretrain")
    ap.add_argument("--preset", default="default")
    ap.add_argument("--mixture", default="pretrain")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-batch", type=int, default=8,
                    help="ниже не опускаемся: батч обязан делиться на число чипов")
    ap.add_argument("--total-steps", type=int, default=200_000)
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    # На диск снимок стоит десять секунд, на Hub — свои 9.3 ГБ квоты, и квота
    # эта не возвращается (замер 08.09.2026, checkpoint.prune_remote). Сессия
    # 07.09 залила семь снимков и встала на восьмом, не досчитав полутора
    # часов. Восемь тысяч шагов — это две заливки за сессию плюс последняя;
    # между ними страховка не на Hub, а локальная: снимок лежит в
    # /kaggle/working и переживает контейнер как output ноутбука.
    ap.add_argument("--push-every", type=int, default=8000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--show-every", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=1000,
                    help="каждые N шагов мерить лосс на отложенном наборе")
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="множитель к LR: ручка для проверки высоты плато")
    ap.add_argument("--probe-lr", type=float, default=None, metavar="МНОЖИТЕЛЬ",
                    help="перед обучением сравнить два LR на одном снимке: "
                         "текущий и этот. Оба прогона идут без записи снимков, "
                         "поэтому точка продолжения не сдвигается, и оба видят "
                         "одни и те же батчи — сравниваются решения, а не данные")
    ap.add_argument("--probe-only", action="store_true",
                    help="снять пробу и выйти, не начиная обучение. Нужно, когда сессия отдана замерам: продолжать претрейн на LR, который проба и проверяет, смысла нет")
    ap.add_argument("--probe-steps", type=int, default=1000,
                    help="шагов в каждой пробе")
    ap.add_argument("--probe-eval-every", type=int, default=250,
                    help="как часто мерить отложенный лосс внутри пробы")
    ap.add_argument("--skip-step", type=int, default=128,
                    help="на сколько батчей сдвигаться после сорвавшегося лосса")
    ap.add_argument("--max-tries", type=int, default=8)
    ap.add_argument("--anneal", type=int, default=None, metavar="ШАГОВ",
                    help="настоящий отжиг: передаётся в pretrain.py как есть. "
                         "Только под своим --run, не под 'pretrain'")
    ap.add_argument("--init-from", default=None, metavar="ПРОГОН",
                    help="первый старт своего --run со снимка этого прогона")
    ap.add_argument("--new-stage", action="store_true",
                    help="новый этап со снимка --init-from: передаётся в "
                         "pretrain.py как есть (длина, смесь, разогрев LR)")
    ap.add_argument("--warmup", type=int, default=None,
                    help="длина разогрева LR, передаётся как есть")
    ap.add_argument("--set", action="append", default=[], metavar="ПОЛЕ=ЗНАЧЕНИЕ",
                    dest="overrides",
                    help="правка конфигурации модели, передаётся как есть")
    ap.add_argument("--no-push", action="store_true",
                    help="не отправлять снимки на Hub — для проверки связки")
    ap.add_argument("--fake-data", action="store_true",
                    help="случайные токены: проверить, что железо считает")
    ap.add_argument("--reserve-hours", type=float, default=0.3,
                    help="запас на выгрузку последнего снимка")
    args = ap.parse_args()

    started = time.time()

    def left() -> float:
        return args.hours - (time.time() - started) / 3600

    batch = args.batch_size

    # --- Проба LR ---------------------------------------------------------
    # Два коротких прогона с одного и того же снимка. Оба с --no-push и
    # --no-checkpoint: точка продолжения не сдвигается, значит второй прогон
    # видит ровно те же веса и те же батчи, что и первый, — сравниваются
    # решения, а не данные.
    if args.probe_lr is not None and left() > args.reserve_hours + 0.5:
        print()
        print(f"проба LR: {args.probe_steps} шагов на множителях "
              f"{args.lr_scale} и {args.probe_lr}, с одного снимка, "
              f"без записи снимков", flush=True)
        print()
        results: dict[float, tuple[float, float, float] | None] = {}
        for scale in (args.lr_scale, args.probe_lr):
            if left() < args.reserve_hours + 0.3:
                print("на пробу времени не осталось — переходим к обучению",
                      flush=True)
                break
            cmd = [sys.executable, str(ROOT / "train" / "pretrain.py"),
                   "--run", args.run, "--preset", args.preset,
                   "--mixture", args.mixture, "--seq-len", str(args.seq_len),
                   "--batch-size", str(batch),
                   "--steps", str(args.probe_steps),
                   "--total-steps", str(args.total_steps),
                   "--hours", f"{max(left() - args.reserve_hours, 0.1):.2f}",
                   "--log-every", str(args.log_every),
                   "--eval-every", str(args.probe_eval_every),
                   "--lr-scale", str(scale),
                   "--show-every", "0",
                   "--resume", "--no-push", "--no-checkpoint"]
            code, tail = run_live(cmd)
            results[scale] = read_holdout(tail)
            got = results[scale]
            said = (f"отложенный лосс {got[0]:.4f} → {got[1]:.4f} "
                    f"({got[2]:+.4f})") if got else "числа нет"
            print(f"[проба {scale}: {said}, осталось {left():.2f} ч]",
                  flush=True)
        print()
        print("=" * 60)
        print("ИТОГ ПРОБЫ LR: " + compare_probes(results))
        print("=" * 60, flush=True)
        print()
        if args.probe_only:
            print("проба закончена, обучение не начинаем (--probe-only)",
                  flush=True)
            return
        # Дальше сессия считает на заданном LR в любом случае: менять его по
        # автоматическому выводу нельзя — это решение человека, а снимок под
        # новый LR ещё не учился.

    # Опускаться ниже заданного батча бессмысленно, но и объявлять минимум
    # больше стартового тоже: тогда первая же нехватка памяти выглядит как
    # тупик, хотя уменьшать просто нечего.
    min_batch = min(args.min_batch, args.batch_size)
    skip = 0
    tries = 0
    mystery = 0
    disk_cleaned = False

    while left() > args.reserve_hours + 0.1 and tries < args.max_tries:
        tries += 1
        cmd = [sys.executable, str(ROOT / "train" / "pretrain.py"),
               "--run", args.run, "--preset", args.preset,
               "--mixture", args.mixture, "--seq-len", str(args.seq_len),
               "--batch-size", str(batch),
               "--steps", "1000000", "--total-steps", str(args.total_steps),
               "--hours", f"{max(left() - args.reserve_hours, 0.1):.2f}",
               "--checkpoint-every", str(args.checkpoint_every),
               "--push-every", str(args.push_every),
               "--log-every", str(args.log_every),
               "--show-every", str(args.show_every),
               "--eval-every", str(args.eval_every),
               "--lr-scale", str(args.lr_scale), "--resume"]
        if args.anneal:
            cmd += ["--anneal", str(args.anneal)]
        if args.init_from:
            cmd += ["--init-from", args.init_from]
        if args.new_stage:
            cmd += ["--new-stage"]
        if args.warmup is not None:
            cmd += ["--warmup", str(args.warmup)]
        for pair in args.overrides:
            cmd += ["--set", pair]
        if skip:
            cmd += ["--skip-batches", str(skip)]
        if args.no_push:
            cmd += ["--no-push"]
        if args.fake_data:
            cmd += ["--fake-data"]

        code, tail = run_live(cmd)
        verdict = decide(code, tail)
        print(f"\n[попытка {tries}: {verdict}, осталось {left():.2f} ч]",
              flush=True)

        if verdict == "время вышло":
            break
        if verdict != "непонятная поломка":
            mystery = 0

        if verdict == "снимок некуда девать":
            # Ротацией это больше не лечится: удаление папок и схлопывание
            # истории оставляют блобы висеть до сборщика мусора Hub (замер
            # 08.09.2026, checkpoint.prune_remote). Квоту освобождает только
            # снос репозитория, а его можно делать лишь когда снимок лежит
            # где-то ещё — в output этой самой сессии.
            print("на Hub нет места — считать дальше значит жечь квоту впустую.\n"
                  "  Снимок остался в /kaggle/working и переживёт контейнер\n"
                  "  как output версии. Со следующего запуска подключить его\n"
                  "  входом (Add Input -> Your Work -> Notebook Output): ячейка\n"
                  "  6б снесёт репозиторий целиком и зальёт снимок заново.\n"
                  "  Что Hub считает занятым: py -3 synth1/checkpoint.py usage", flush=True)
            break
        if verdict == "диск полон":
            # Чужие снимки на диске — копии с Hub (их качали замеры и прошлые
            # этапы): удаляем все, кроме своего, и пробуем ещё раз. Не помогло
            # — дальше без нас: повторы место не освободят.
            if not disk_cleaned:
                disk_cleaned = True
                freed = free_checkpoints(keep=args.run)
                print(f"диск полон — удалены локальные снимки {freed}, "
                      f"пробуем ещё раз", flush=True)
                continue
            print("диск полон и после чистки — дальше без нас", flush=True)
            break
        if verdict == "не влезло на загрузке":
            print("снимок не помещается в память ещё до первого шага — "
                  "батч тут ни при чём, нужна правка кода", flush=True)
            break
        if verdict == "не влезло":
            if batch > min_batch:
                batch //= 2
                print(f"пробуем батч {batch}", flush=True)
                continue
            print("не влезает даже минимальный батч — дальше без нас",
                  flush=True)
            break
        if verdict == "лосс сорвался":
            # Позиция в корпусе лежит в снимке, поэтому перезапуск сам по себе
            # выдал бы тот же батч и тот же nan. Сдвигаемся.
            skip += args.skip_step
            print(f"откат к снимку, обходим {skip} батчей", flush=True)
            continue
        if verdict == "непонятная поломка":
            # Процесс, убитый нехваткой памяти самой машиной, не успевает
            # сказать ни слова: код возврата есть, объяснения нет. Один раз
            # это может быть сеть, два подряд — почти наверняка память, и
            # повторять с тем же батчем значит терять время сессии.
            mystery += 1
            if mystery >= 2 and batch > min_batch:
                batch //= 2
                mystery = 0
                print(f"вторая молчаливая смерть подряд — пробуем батч {batch}",
                      flush=True)
                continue
            print("пробуем ещё раз", flush=True)
            continue
        break

    print(f"\nсессия заняла {(time.time() - started) / 3600:.2f} ч, "
          f"попыток {tries}", flush=True)


if __name__ == "__main__":
    main()
