"""Генератор ноутбуков Kaggle.

Ноутбуки собираются скриптом, а не правятся руками. Причин две. JSON с
экранированными переводами строк невозможно ревьюить в диффе, а ещё в прототипе на MiniCPM
уже был потерян прогон из-за того, что `source` ячейки записали строкой вместо
списка строк — Kaggle тогда импортирует ячейку пустой и молча.

    py -3 notebooks/make_notebooks.py
"""

from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = "https://github.com/ssane-labs/synth-1.git"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]}


def code(text: str) -> dict:
    # source обязан быть списком строк. Kaggle принимает и строку, но при
    # импорте показывает пустую ячейку — потерянная сессия ищется потом часами.
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [line + "\n" for line in text.strip().split("\n")]}


NOTEBOOK = [
    md("""
# Synth-1 — обучение на TPU

Порядок: проверить железо → поставить окружение → прогон на случайных токенах →
настоящее обучение.

Перед запуском: в правой панели включить **TPU VM v5e-8** и добавить секрет
`HF_TOKEN` (Add-ons → Secrets). Без секрета не будет ни корпуса, ни чекпоинтов.
"""),

    md("## 1. Что за железо досталось"),
    code("""
import subprocess, os, sys
print(subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip(), "ядер CPU")
print(subprocess.run(["free", "-g"], capture_output=True, text=True).stdout)
print(subprocess.run(["df", "-h", "/kaggle/working"], capture_output=True, text=True).stdout)
"""),

    md("""
## 2. Окружение

JAX на образе Kaggle бывает старее нужного, поэтому ставим явно. Всё через
`subprocess`: под Save & Run ячейки с `!` ведут себя иначе, чем в интерактивном
запуске, и на этом в прототипе на MiniCPM уже спотыкались.
"""),
    code("""
def run(*cmd, check=True):
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout: print(p.stdout[-3000:])
    if p.returncode and check:
        print(p.stderr[-3000:]); raise SystemExit(f"упало: {' '.join(cmd)}")
    return p

run(sys.executable, "-m", "pip", "install", "-q", "-U",
    "jax[tpu]", "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html")
run(sys.executable, "-m", "pip", "install", "-q",
    "tokenizers", "safetensors", "huggingface_hub", "pyyaml")
"""),

    md("## 3. Код проекта"),
    code(f"""
WORK = "/kaggle/working/synth-1"
if not os.path.exists(WORK):
    run("git", "clone", "--depth", "1", "{REPO}", WORK)
else:
    run("git", "-C", WORK, "pull", "--ff-only")
os.chdir(WORK)
sys.path.insert(0, WORK)
run("git", "-C", WORK, "log", "--oneline", "-3")
"""),

    md("""
## 4. Чипы на месте?

Должно быть восемь устройств. Если их меньше — TPU не выдан, и дальше идти
незачем: прогон на одном чипе съест квоту и ничего не покажет.
"""),
    code("""
import jax
devs = jax.devices()
print(f"{len(devs)} устройств: {devs[0].device_kind}")
for d in devs: print(" ", d)
assert len(devs) == 8, f"ожидалось 8 чипов, видно {len(devs)} — проверь настройку TPU"
"""),

    md("""
## 5. Считает ли железо

Случайные токены вместо корпуса. Лосс здесь не падает и не должен — шум
предсказывать нечем, он держится около log(V). Смотрим на другое: компилируется
ли шаг, сколько миллисекунд он занимает и влезает ли модель.
"""),
    code("""
run(sys.executable, "train/pretrain.py",
    "--run", "smoke", "--preset", "default", "--fake-data",
    "--steps", "30", "--seq-len", "2048", "--batch-size", "8",
    "--log-every", "5", "--checkpoint-every", "1000000", "--no-push")
"""),

    md("""
## 6. Абляции

Каждая — сорок минут, модель поменьше, один и тот же корпус и seed. Сравниваем
разрежённую конфигурацию с плотной при сопоставимых активных FLOPs. Результаты
идут в `evals/ablations.md` числами, а не словами.
"""),
    code("""
COMMON = ["--mixture", "pretrain", "--seq-len", "2048", "--batch-size", "16",
          "--steps", "2000", "--total-steps", "2000", "--log-every", "50",
          "--checkpoint-every", "1000"]

run(sys.executable, "train/pretrain.py", "--run", "abl-dense",
    "--preset", "ablation-dense", *COMMON)
"""),
    code("""
run(sys.executable, "train/pretrain.py", "--run", "abl-moe",
    "--preset", "ablation", *COMMON)
"""),

    md("""
## 7. Претрейн

Запускать, когда абляции выбрали конфигурацию. `--resume` подхватывает последний
снимок с Hub вместе с позицией в корпусе, поэтому следующая сессия продолжает, а
не начинает заново.

`--hours 8.2` оставляет запас до лимита в девять часов: снимок на ~4.4 ГБ надо
успеть выгрузить.
"""),
    code("""
run(sys.executable, "train/pretrain.py",
    "--run", "pretrain", "--preset", "default", "--mixture", "pretrain",
    "--seq-len", "2048", "--batch-size", "32",
    "--steps", "1000000", "--total-steps", "200000",
    "--hours", "8.2", "--checkpoint-every", "2000", "--log-every", "50",
    "--resume")
"""),
]


def main():
    notebook = {
        "cells": NOTEBOOK,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = HERE / "kaggle_tpu.ipynb"
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False),
                   encoding="utf-8")

    bad = [i for i, c in enumerate(notebook["cells"])
           if not isinstance(c["source"], list)]
    if bad:
        raise SystemExit(f"ячейки {bad} записаны строкой, а не списком строк")

    print(f"{out} — {len(NOTEBOOK)} ячеек, "
          f"{out.stat().st_size / 1024:.1f} КБ")


if __name__ == "__main__":
    main()
