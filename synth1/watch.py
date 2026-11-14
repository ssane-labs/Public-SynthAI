"""Показ примеров генерации по ходу обучения.

Смысл — чтобы за цифрами лосса было видно живое. Перплексия падает одинаково
ровно и когда всё хорошо, и когда модель выучила выдавать частотный мусор;
разница видна только глазами. Никаких автопроверок здесь нет намеренно: это
окно, а не сторож.

Промпты подобраны так, чтобы по ним читался прогресс претрейна:

    язык        связная фраза вместо словесной каши — первое, что появляется
    факты       продолжение, где нужно знание, а не грамматика
    код         отступы, скобки, имена — ломается заметнее прозы
    выдача      продолжение сниппета поисковой выдачи — главный вход модели
    счёт        арифметика — то, ради чего цифры режутся по одной

Рядом с каждым примером печатается средняя неуверенность: у обученной модели
она должна быть низкой на шаблонном продолжении и высокой там, где нужен факт.
Расхождение между этими двумя — и есть то, ради чего затеян проект, и увидеть
его хочется как можно раньше.
"""

from __future__ import annotations

import time

PROMPTS = [
    ("язык", "The most important thing to understand about"),
    ("факты", "The capital of France is"),
    ("код", "def fibonacci(n):\n    if n < 2:"),
    ("выдача", "[1] Canberra - Wikipedia\nCanberra is the capital city of"),
    ("счёт", "12 + 7 = "),
]


def show(params, cfg, tokenizer, step: int, max_tokens: int = 40,
         temperature: float = 0.8, seed: int = 0) -> None:
    """Напечатать продолжения. Никогда не роняет обучение."""
    from synth1 import generate as G

    started = time.time()
    print(f"\n--- примеры на шаге {step:,} " + "-" * 34, flush=True)
    for label, prompt in PROMPTS:
        try:
            text, unc = G.generate(
                params, cfg, tokenizer, prompt,
                max_tokens=max_tokens, temperature=temperature, seed=seed + step,
            )
            tail = text[len(prompt):].replace("\n", " ⏎ ")
            mean_u = sum(unc) / len(unc) if unc else float("nan")
            print(f"  [{label:<8} u={mean_u:.2f}] {prompt}«{tail}»", flush=True)
        except Exception as exc:                              # noqa: BLE001
            # Генерация — украшение, а не работа. Сломалась — печатаем и живём
            # дальше: терять сессию из-за витрины нельзя.
            print(f"  [{label:<8}] не вышло: {type(exc).__name__}: {exc}",
                  flush=True)
    print(f"--- {time.time() - started:.0f}с " + "-" * 42 + "\n", flush=True)


def load_tokenizer():
    """Токенизатор из репозитория. None, если его нет — не повод падать."""
    import config
    try:
        from tokenizers import Tokenizer
        path = config.TOKENIZER_OUT / "synth1.json"
        if not path.exists():
            print(f"токенизатора нет ({path}), примеры показывать нечем")
            return None
        return Tokenizer.from_file(str(path))
    except Exception as exc:                                  # noqa: BLE001
        print(f"токенизатор не загрузился: {exc}")
        return None
