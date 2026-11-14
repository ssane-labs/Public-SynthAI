"""Чтение parquet-шардов с Hugging Face по HTTP Range.

Почему не datasets.streaming: на реальном канале fsspec рвёт соединение на
больших шардах и не восстанавливается — шестнадцать ретраев подряд и падение,
проверено. Здесь один поток, свои ретраи и блочный кэш; медленнее в идеальных
условиях, зато доходит до конца.

Второй довод важнее первого: позиция в корпусе тут явная — (шард, группа строк).
Её можно положить в чекпоинт и продолжить прогон ровно с того места, где сессия
Kaggle оборвалась. У datasets для этого есть state_dict(), но он непрозрачен и
привязан к версии библиотеки.
"""

from __future__ import annotations

import io
import os
import random
import time
from dataclasses import dataclass

import requests

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
BLOCK = 4 << 20          # гранулярность кэша при мелких чтениях
TIMEOUT = 120

# Ретраев много и паузы длинные: сумма ожидания — около получаса.
#
# Это не перестраховка. Сбор корпуса идёт часами, и за это время интернет
# успевает пропасть целиком — не «сервер занят», а DNS перестаёт резолвиться.
# Восьми попыток по тридцать секунд на такое не хватило, и прогон, прошедший
# больше половины пути, упал на ровном месте.
MAX_RETRIES = 20
MAX_BACKOFF = 120


def resolve_url(repo: str, path: str, revision: str = "main") -> str:
    return f"{HF_ENDPOINT}/datasets/{repo}/resolve/{revision}/{path}"


class RemoteFile(io.RawIOBase):
    """file-like над URL. Отдаётся в pyarrow вместо локального файла.

    pyarrow сначала читает футер parquet (мелкие чтения с конца), потом группы
    строк (крупные). Мелкие обслуживаются блочным кэшем, крупные идут одним
    Range-запросом — иначе на группу в 50 МБ уходило бы тринадцать round-trip'ов.
    """

    def __init__(self, url: str, token: str | None = None, size: int | None = None):
        self.url = url
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._pos = 0
        self._block_start = -1
        self._block = b""
        self.size = size if size is not None else self._head_size()

    def _head_size(self) -> int:
        r = self._request_with_retries("HEAD", None)
        return int(r.headers["Content-Length"])

    def _request_with_retries(self, method: str, rng: tuple[int, int] | None):
        headers = dict(self.headers)
        if rng is not None:
            headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
        last = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.request(
                    method, self.url, headers=headers,
                    allow_redirects=True, timeout=TIMEOUT,
                )
                if r.status_code in (200, 206):
                    return r
                # 416 — запрошен диапазон за концом файла, ретраить нечего.
                if r.status_code == 416:
                    raise EOFError(f"Range за концом файла: {rng}")
                # Клиентские ошибки не лечатся повтором: 401/403 — это «прими
                # условия датасета или дай токен», 404 — «такого файла нет».
                # Ждать по тридцать секунд восемь раз подряд бессмысленно.
                if r.status_code in (401, 403, 404, 405):
                    raise IOError(
                        f"HTTP {r.status_code} на {self.url}\n"
                        + {
                            401: "нужен HF_TOKEN",
                            403: "датасет закрыт: прими условия на его странице "
                                 "на huggingface.co или проверь права токена",
                            404: "файла нет — проверь путь в mixture.yaml",
                        }.get(r.status_code, "")
                    )
                last = f"HTTP {r.status_code}"
            except (requests.RequestException, ConnectionError) as exc:
                # Сюда попадает и «сервер не ответил», и «интернета нет вовсе».
                # Различать их бессмысленно: лечение одно — подождать подольше.
                last = type(exc).__name__
            sleep = min(2 ** attempt, MAX_BACKOFF)
            print(f"    сеть: {last}, повтор через {sleep}с "
                  f"({attempt + 1}/{MAX_RETRIES})", flush=True)
            time.sleep(sleep)
        raise IOError(f"Не смог прочитать {self.url}: {last}")

    def _fetch(self, start: int, length: int) -> bytes:
        end = min(start + length, self.size) - 1
        if end < start:
            return b""
        return self._request_with_retries("GET", (start, end)).content

    # --- интерфейс файла ---
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self._pos
        size = max(0, min(size, self.size - self._pos))
        if size == 0:
            return b""

        # Крупное чтение — одним запросом, мимо кэша.
        if size > BLOCK // 2:
            data = self._fetch(self._pos, size)
            self._pos += len(data)
            return data

        # Мелкое — через блок. Блок начинается ровно с текущей позиции, а не с
        # выровненной границы: при выравнивании остаток блока после _pos мог
        # оказаться короче запроса, и pyarrow получал недочитанный футер
        # («File too short»). Условие size <= BLOCK/2 гарантирует покрытие.
        if not (self._block_start <= self._pos
                and self._pos + size <= self._block_start + len(self._block)):
            self._block_start = self._pos
            self._block = self._fetch(self._block_start, BLOCK)
        off = self._pos - self._block_start
        data = self._block[off:off + size]
        self._pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


# --- Шарды -----------------------------------------------------------------

@dataclass(frozen=True)
class Shard:
    repo: str
    path: str
    size: int


def list_shards(repo: str, patterns: list[str], token: str | None = None,
                revision: str = "main") -> list[Shard]:
    """Файлы репозитория, подходящие под шаблоны путей из mixture.yaml."""
    import fnmatch

    url = f"{HF_ENDPOINT}/api/datasets/{repo}/tree/{revision}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    out: list[Shard] = []
    seen_dirs: set[str] = set()

    # Дерево обходится по нужным каталогам, а не рекурсивно от корня: у крупных
    # корпусов в корне десятки тысяч файлов, и recursive=1 отдаёт их минутами.
    dirs = sorted({p.rsplit("/", 1)[0] for p in patterns if "/" in p})
    for d in dirs or [""]:
        if d in seen_dirs:
            continue
        seen_dirs.add(d)
        cursor = f"{url}/{d}" if d else url
        page = f"{cursor}?limit=1000"
        while page:
            r = requests.get(page, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            for entry in r.json():
                if entry.get("type") != "file":
                    continue
                path = entry["path"]
                if any(fnmatch.fnmatch(path, pat) for pat in patterns):
                    out.append(Shard(repo, path, entry.get("size", 0)))
            link = r.headers.get("Link", "")
            page = None
            if 'rel="next"' in link:
                page = link.split(";")[0].strip("<> ")
    if not out:
        raise SystemExit(f"В {repo} нет файлов под {patterns}")
    return sorted(out, key=lambda s: s.path)


def iter_shard(shard: Shard, field: str, token: str | None = None,
               start_group: int = 0, meta_fields: list[str] | None = None):
    """Документы одного шарда. Отдаёт (номер группы строк, запись).

    Запись — словарь с ключом "text" и теми метаполями, которые попросили: их
    смотрит фильтр качества (путь файла выдаёт вендоренный код надёжнее, чем
    сам код).

    Читается группами: это единица и восстановления после обрыва, и позиции в
    чекпоинте. Внутри группы ничего не сохраняется — при возобновлении группа
    перечитывается целиком, и это дешевле, чем хранить смещение строки.
    """
    import pyarrow.parquet as pq

    fh = RemoteFile(resolve_url(shard.repo, shard.path), token=token,
                    size=shard.size or None)
    pf = pq.ParquetFile(fh)

    names = pf.schema_arrow.names
    if field not in names:
        candidates = [n for n in names if n in ("text", "content", "code", "raw_content")]
        if not candidates:
            raise SystemExit(f"В {shard.path} нет поля {field!r}. Есть: {names}")
        field = candidates[0]

    columns = [field] + [m for m in (meta_fields or []) if m in names and m != field]

    for gi in range(start_group, pf.num_row_groups):
        table = pf.read_row_group(gi, columns=columns)
        rows = table.to_pydict()
        texts = rows[field]
        for i, text in enumerate(texts):
            if not text:
                continue
            row = {"text": text}
            for m in columns[1:]:
                row[m] = rows[m][i]
            yield gi, row


def shard_order(shards: list[Shard], seed: int) -> list[Shard]:
    """Детерминированный порядок шардов: тот же seed — тот же порядок всегда."""
    order = list(shards)
    random.Random(seed).shuffle(order)
    return order
