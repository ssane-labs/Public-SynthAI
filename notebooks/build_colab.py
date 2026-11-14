"""Сборка ноутбука для Colab. Запускать после правок:

    py -3 notebooks/build_colab.py

Colab — для отладки, Kaggle — для дела. Причина простая: в бесплатном Colab
один чип TPU, а не восемь, и целевая модель туда не помещается вместе с
состоянием оптимизатора. Зато нет очереди, и проверить «не падает ли код»
можно за минуту вместо часа ожидания.

Правило про jax то же, что и на Kaggle: ядро ноутбука не должно его
импортировать, иначе захватит устройство и отдельный процесс останется ни с чем.
"""

from __future__ import annotations

import ast
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# Synth-1 — отладка в Colab

Здесь проверяется, что код работает: очереди нет, ответ за минуту. Настоящие
абляции и претрейн идут на Kaggle, где восемь чипов.

Что выбрать в Runtime → Change runtime type:

| Ускоритель | Годится для |
|---|---|
| **TPU v5e-1** | ближе всего к боевому железу, один чип |
| **T4 / L4 GPU** | тоже подходит, JAX умеет CUDA |
| CPU | только синтаксис и тесты, обучение будет ползти |

Целевая модель (1.2B) на один чип не влезет вместе с состоянием оптимизатора —
для неё нужен Kaggle. Здесь работаем с конфигурацией `ablation` (94M).

Токен: слева на панели «ключ» (Secrets) → добавить `HF_TOKEN` и включить доступ
для этого ноутбука.
"""),

    md("## 1. Окружение и код"),
    code("""
import subprocess, os, sys

def run(*cmd, check=True):
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout: print(p.stdout[-4000:])
    if p.returncode and check:
        print(p.stderr[-4000:]); raise SystemExit(f"упало: {' '.join(cmd)}")
    return p

WORK = "/content/synth-1"
if not os.path.exists(WORK):
    run("git", "clone", "--depth", "1",
        "https://github.com/ssane-labs/synth-1.git", WORK)
else:
    run("git", "-C", WORK, "fetch", "--depth", "1", "origin", "main")
    run("git", "-C", WORK, "reset", "--hard", "origin/main")
os.chdir(WORK)
run(sys.executable, "-m", "pip", "install", "-q",
    "tokenizers", "safetensors", "huggingface_hub", "pyyaml")
"""),

    md("""
## 2. Токен

Colab хранит секреты отдельно от переменных окружения, поэтому переносим руками.
Если секрета нет, дальше пойдёт всё, кроме корпуса и выгрузки на Hub.
"""),
    code("""
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN подхвачен")
except Exception as exc:
    print(f"без токена ({exc}); работает только --fake-data")
"""),

    md("""
## 3. Что за устройство

Отдельным процессом — чтобы ядро не захватило ускоритель. На TPU нужен
`jax[tpu]`, на GPU — `jax[cuda12]`; ставим по факту.
"""),
    code("""
if os.path.exists("/dev/accel0") or "COLAB_TPU_ADDR" in os.environ:
    run(sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]",
        "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html")
else:
    run(sys.executable, "-m", "pip", "install", "-q", "-U", "jax[cuda12]")

run(sys.executable, "-c",
    "import jax; d = jax.devices(); print(len(d), 'устройств:', d[0].device_kind)")
"""),

    md("""
## 4. Тесты — то же, что гоняется локально

Пятьдесят с лишним проверок модели, обучения и чекпоинтов. Занимают пару минут
и ловят почти всё, что иначе всплывёт на третьем часу прогона.
"""),
    code("""
for t in ["synth1/test_model.py", "synth1/test_train.py", "synth1/test_checkpoint.py"]:
    run(sys.executable, t)
"""),

    md("""
## 5. Короткий прогон

Случайные токены, маленькая модель. Проверяем, что шаг компилируется и считает.
Перед первым шагом идёт предполётная проверка — она назовёт место, если что-то
разойдётся.
"""),
    code("""
# Батч должен делиться на число устройств. В Colab чип один, поэтому годится
# любой; на Kaggle с восемью чипами — только кратный восьми.
run(sys.executable, "train/pretrain.py",
    "--run", "colab-smoke", "--preset", "ablation", "--fake-data",
    "--steps", "20", "--seq-len", "512", "--batch-size", "4",
    "--log-every", "5", "--no-checkpoint")
"""),

    md("""
## 6. Прогон на настоящем корпусе

Здесь уже течёт текст с Hub. `--show-every` печатает примеры генерации: на
двухстах шагах это будет каша, и так и должно быть — смысл в том, чтобы видеть,
как она перестаёт быть кашей.
"""),
    code("""
run(sys.executable, "train/pretrain.py",
    "--run", "colab-try", "--preset", "ablation", "--mixture", "pretrain",
    "--steps", "200", "--total-steps", "200", "--seq-len", "512",
    "--batch-size", "4", "--log-every", "20", "--show-every", "100",
    "--no-checkpoint")
"""),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "TPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    for i, cell in enumerate(nb["cells"]):
        assert isinstance(cell["source"], list), f"ячейка {i}: source не список"
        if cell["cell_type"] == "code":
            # Ячейка, которая не разбирается питоном, падает на Kaggle
            # через минуту после старта и уносит сессию целиком. Здесь
            # это стоит миллисекунды: 31.08.2026 экранированные переводы
            # строк развернулись при сборке и порвали строковый литерал.
            try:
                ast.parse("".join(cell["source"]))
            except SyntaxError as exc:
                raise SystemExit(
                    f"ячейка {i} не разбирается питоном: {exc.msg}, "
                    f"строка {exc.lineno}"
                ) from exc
        if cell["cell_type"] == "code":
            body = "".join(cell["source"])
            assert "import jax" not in body or "sys.executable" in body, (
                f"ячейка {i} импортирует jax в ядре — она захватит устройство"
            )
    print(f"Собрано: {OUT} ({len(CELLS)} ячеек)")


if __name__ == "__main__":
    main()
