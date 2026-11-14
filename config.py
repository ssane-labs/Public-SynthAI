"""Пути, ключи и константы проекта. Одна точка входа, чтобы не искать их по коду.

Порядок поиска ключа:
    1. переменная окружения
    2. .env рядом с этим файлом
    3. Kaggle Secrets (когда код исполняется в ноутбуке)
"""

import os
import pathlib
import sys

# Вывод всегда в UTF-8, даже когда он перенаправлен в файл.
#
# На Windows перенаправленный поток берёт кодировку системы (у нас cp1251), и
# первая же стрелка в сообщении роняет процесс с UnicodeEncodeError — уже
# наступали на это в прототипе на MiniCPM, когда фоновый генератор падал на собственном
# сообщении о прогрессе.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"

DATA = ROOT / "data"
DATA_SAMPLE = DATA / "sample"          # сэмпл текста под обучение токенизатора
TOKENIZER_OUT = ROOT / "tokenizer" / "out"
EVALS = ROOT / "evals"

# --- Токенизатор -----------------------------------------------------------
# 32768 = 256 * 128. Кратность 128 нужна TPU: иначе последняя матмуль-плитка
# выходной проекции считается вхолостую.
#
# У Synth-2 было 49152, из них ~4k merges под кириллицу. Synth-1 только
# английская, и у модели на 159M активных параметров выходная проекция — заметная
# доля счёта: 32768 вместо 49152 делает её на треть дешевле.
VOCAB_SIZE = 32768

# Спецтокены объявляются все и сразу, включая те, что понадобятся только на SFT.
# Добавить токен потом — значит переучивать embedding, а он у нас связанный.
#
# Инструментов общего вида у Synth-1 нет — есть только поиск, и он встроен в
# словарь отдельными токенами, а не вызовом <|tool|> с JSON внутри. Короче на
# каждом ходе, не нужен разбор JSON, и модель не может «позвать» то, чего нет.
# Формат хода — SEARCH.md, раздел 6.
SPECIAL_TOKENS = [
    "<|pad|>",        # 0, он же id для маскирования лосса
    "<|eot|>",        # конец документа или хода
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|think|>",
    "<|/think|>",
    "<|search|>",     # поисковый запрос, внутри — текст запроса
    "<|/search|>",
    "<|result|>",     # выдача: пронумерованные [N] заголовок, адрес, сниппет
    "<|/result|>",
    "<|open|>",       # открыть страницу выдачи, внутри — номер [N] или адрес
    "<|/open|>",
    "<|page|>",       # текст открытой страницы (кусок, не вся)
    "<|/page|>",
    "<|find|>",       # найти строку на открытой странице
    "<|/find|>",
    "<|cite|>",       # ссылка на источник в ответе, внутри — номер
    "<|/cite|>",
    "<|unsure|>",     # маркер хеджирования; ставится на этапе E, не раньше
]

# Сколько протокольных токенов обязано встать на свои id подряд с нуля.
N_PROTOCOL_TOKENS = len(SPECIAL_TOKENS)

# Резерв, чтобы будущие протокольные токены не ломали словарь.
RESERVED_SLOTS = 256
SPECIAL_TOKENS += [
    f"<|reserved_{i}|>" for i in range(RESERVED_SLOTS - len(SPECIAL_TOKENS))
]

PAD_ID = 0
EOT_ID = 1

# --- Ключи -----------------------------------------------------------------
_WHERE = {
    "HF_TOKEN": "корпус, чекпоинты, токенизатор — всё через Hugging Face",
}


def _load_env_file() -> dict:
    if not ENV_FILE.exists():
        return {}
    out = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


_ENV_FILE_VALUES = _load_env_file()


def _kaggle_secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return None
    try:
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def key(name: str, required: bool = True) -> str:
    """Значение ключа. Падает внятно, а не KeyError посреди прогона на два часа."""
    value = (
        os.environ.get(name)
        or _ENV_FILE_VALUES.get(name)
        or _kaggle_secret(name)
    )
    if value:
        return value
    if not required:
        return ""
    where = _WHERE.get(name, "")
    raise SystemExit(
        f"Нет ключа {name}"
        + (f" — он нужен для: {where}" if where else "")
        + f"\nПоложи его в {ENV_FILE} (образец рядом, .env.example)"
        f" или в переменные окружения."
    )


HF_NAMESPACE = _ENV_FILE_VALUES.get("HF_NAMESPACE") or os.environ.get(
    "HF_NAMESPACE", "sekund0chka"
)

# Кэш HF: на Windows по умолчанию уезжает на C:, который у нас маленький.
_hf_home = _ENV_FILE_VALUES.get("HF_HOME") or os.environ.get("HF_HOME")
if _hf_home:
    os.environ["HF_HOME"] = _hf_home
