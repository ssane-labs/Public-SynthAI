"""Корпус: чтение mixture.yaml и потоковая выдача текста.

Данные никогда не лежат на диске целиком. 300B токенов в uint16 — это 600 ГБ,
столько нет ни на Kaggle, ни здесь. Текст течёт с Hub (data/stream.py),
токенизируется на лету, а позиция в потоке кладётся в чекпоинт, чтобы прогон
переживал конец сессии.
"""

from __future__ import annotations

import pathlib
import random
import re
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data import filters, stream  # noqa: E402

MIXTURE_FILE = pathlib.Path(__file__).resolve().parent / "mixture.yaml"
_SPEC = yaml.safe_load(MIXTURE_FILE.read_text(encoding="utf-8"))


def mixture(name: str) -> dict[str, float]:
    """Доли источников. Падает сразу, если смесь не сходится в единицу."""
    if name not in _SPEC["mixtures"]:
        raise SystemExit(f"Нет смеси {name!r}. Есть: {', '.join(_SPEC['mixtures'])}")
    weights = _SPEC["mixtures"][name]
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"Смесь {name!r} суммируется в {total}, а должна в 1.0")
    return weights


def source_spec(name: str) -> dict:
    if name not in _SPEC["sources"]:
        raise SystemExit(f"Нет источника {name!r}. Есть: {', '.join(_SPEC['sources'])}")
    return _SPEC["sources"][name]


# --- Фильтр личности -------------------------------------------------------
# Обоснование — mixture.yaml. Регекс собирается один раз: он прогоняется по
# каждому документу корпуса, и на трёхстах миллиардах токенов это заметно.

def _build_identity_filter():
    spec = _SPEC.get("identity_filter", {})
    patterns = spec.get("patterns", [])
    if not patterns:
        return None
    flags = 0 if spec.get("case_sensitive") else re.IGNORECASE
    return re.compile("|".join(re.escape(p) for p in patterns), flags)


_IDENTITY_RE = _build_identity_filter()


def mentions_foreign_identity(text: str) -> bool:
    return _IDENTITY_RE is not None and _IDENTITY_RE.search(text) is not None


# --- Чтение источника ------------------------------------------------------

class SourceReader:
    """Один источник как бесконечный поток документов.

    Позиция — пара (номер шарда, номер группы строк). Обе величины сохраняются
    и восстанавливаются, так что прогон продолжается ровно оттуда, где сессия
    Kaggle кончилась, а не с начала корпуса.
    """

    # Сколько шардов каждого источника отложено под валидацию. Обучение их не
    # видит никогда, поэтому перплексия на них означает то, что должна.
    # Отбираются последние в перемешанном порядке — то есть детерминированно,
    # но не по алфавиту: первые шарды корпусов часто отличаются по содержимому
    # от остальных.
    HOLDOUT_SHARDS = 2

    def __init__(self, name: str, seed: int = 0, token: str | None = None,
                 split: str = "train"):
        self.name = name
        self.seed = seed
        self.split = split
        spec = source_spec(name)
        self.repo = spec["repo"]
        self.field = spec["field"]
        self.meta = spec.get("meta", [])
        self.lang = spec.get("lang", "")
        self.token = token
        self.seen = 0
        self.dropped = 0
        shards = stream.list_shards(self.repo, spec["paths"], token=token)
        ordered = stream.shard_order(shards, seed)
        if len(ordered) <= self.HOLDOUT_SHARDS:
            raise SystemExit(
                f"{name}: всего {len(ordered)} шардов, "
                f"отложить под валидацию {self.HOLDOUT_SHARDS} нельзя"
            )
        self.shards = (ordered[:-self.HOLDOUT_SHARDS] if split == "train"
                       else ordered[-self.HOLDOUT_SHARDS:])
        self.shard_idx = 0
        self.group_idx = 0
        self.epoch = 0

    def describe(self) -> str:
        total = sum(s.size for s in self.shards) / 1024**3
        return f"{self.name}: {len(self.shards)} шардов, {total:.1f} ГБ"

    def state(self) -> dict:
        return {"shard_idx": self.shard_idx, "group_idx": self.group_idx,
                "epoch": self.epoch, "seed": self.seed}

    def restore(self, state: dict) -> None:
        if state.get("seed") != self.seed:
            raise SystemExit(
                f"{self.name}: чекпоинт снят с seed={state.get('seed')}, "
                f"а читаем с seed={self.seed} — порядок шардов другой, "
                "продолжать нельзя."
            )
        self.shard_idx = state["shard_idx"]
        self.group_idx = state["group_idx"]
        self.epoch = state.get("epoch", 0)

    def __iter__(self):
        while True:
            shard = self.shards[self.shard_idx]
            for group_idx, row in stream.iter_shard(
                shard, self.field, token=self.token, start_group=self.group_idx,
                meta_fields=self.meta,
            ):
                self.group_idx = group_idx
                self.seen += 1
                # Нормализация до фильтра: чужие служебные метки влияют и на
                # длину, и на долю алфавитных символов, по которым идёт отсев.
                row = filters.normalise(row, self.lang)
                # Отсев качества — здесь, а не в общем цикле: правила у каждого
                # источника свои, и статистика выброшенного нужна поштучно.
                if not filters.keep(row, self.lang):
                    self.dropped += 1
                    continue
                yield row["text"]
            self.shard_idx += 1
            self.group_idx = 0
            if self.shard_idx >= len(self.shards):
                # Корпус кончился. На нашем бюджете это не должно случиться ни
                # разу; если случилось — идём по второму кругу и говорим об этом.
                self.shard_idx = 0
                self.epoch += 1
                print(f"  {self.name}: корпус пройден целиком, "
                      f"эпоха {self.epoch}", flush=True)


class MixtureReader:
    """Смесь источников с весами. Источник для каждого документа тянется жребием.

    Жребий детерминирован: его состояние тоже попадает в чекпоинт, иначе после
    перезапуска пропорции поедут.
    """

    def __init__(self, name: str, seed: int = 0, token: str | None = None,
                 split: str = "train"):
        self.name = name
        self.split = split
        self.weights = mixture(name)
        self.names = list(self.weights)
        self.readers = {n: SourceReader(n, seed=seed, token=token, split=split)
                        for n in self.names}
        self.iters = {n: iter(r) for n, r in self.readers.items()}
        self.rng = random.Random(seed ^ 0x5EED)
        self.counts = {n: 0 for n in self.names}

    def describe(self) -> str:
        lines = [f"смесь {self.name!r}:"]
        for n in self.names:
            lines.append(f"  {self.weights[n]:>5.0%}  {self.readers[n].describe()}")
        return "\n".join(lines)

    def state(self) -> dict:
        return {
            "mixture": self.name,
            "rng": self.rng.getstate(),
            "counts": self.counts,
            "sources": {n: r.state() for n, r in self.readers.items()},
        }

    def restore(self, state: dict) -> None:
        if state["mixture"] != self.name:
            raise SystemExit(
                f"Чекпоинт снят со смеси {state['mixture']!r}, а читаем {self.name!r}"
            )
        # getstate отдаёт кортеж, а json превращает его в список — чинить надо
        # рекурсивно, иначе setstate ругается на тип внутреннего элемента.
        rng_state = state["rng"]
        self.rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
        self.counts = dict(state["counts"])
        for n, s in state["sources"].items():
            self.readers[n].restore(s)

    def __iter__(self):
        population, weights = self.names, [self.weights[n] for n in self.names]
        while True:
            name = self.rng.choices(population, weights)[0]
            self.counts[name] += 1
            yield name, next(self.iters[name])


def iter_texts(
    mixture_name: str,
    seed: int = 0,
    limit: int | None = None,
    drop_foreign_identity: bool = True,
    min_chars: int = 200,
    token: str | None = None,
):
    """Документы смеси как строки. Здесь же — фильтр личности и отсев огрызков."""
    reader = MixtureReader(mixture_name, seed=seed,
                           token=token or config.key("HF_TOKEN", required=False) or None)
    print(reader.describe(), flush=True)
    kept = 0
    for _, text in reader:
        if len(text) < min_chars:
            continue
        if drop_foreign_identity and mentions_foreign_identity(text):
            continue
        yield text
        kept += 1
        if limit is not None and kept >= limit:
            return


def counted(iterable, every: int = 10_000, label: str = ""):
    """Прогресс без tqdm: на Kaggle он засоряет лог, а нам нужны редкие строки."""
    for i, item in enumerate(iterable, 1):
        if i % every == 0:
            print(f"  {label}{i:,}", flush=True)
        yield item


if __name__ == "__main__":
    # Проверка, что источники живы и отдают то, что обещано в mixture.yaml.
    # Запускать после любой правки долей, путей или имён репозиториев.
    name = sys.argv[1] if len(sys.argv) > 1 else "debug"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    reader = MixtureReader(name, seed=0,
                           token=config.key("HF_TOKEN", required=False) or None)
    print(reader.describe(), flush=True)

    shown = 0
    for src, text in reader:
        if len(text) < 200 or mentions_foreign_identity(text):
            continue
        head = text[:130].replace("\n", " ⏎ ")
        print(f"[{src:<17}] {len(text):>7,} симв | {head}...")
        shown += 1
        if shown >= limit:
            break

    print("\nотсев по качеству:")
    for n, r in reader.readers.items():
        if r.seen:
            print(f"  {n:<18} прочитано {r.seen:>6,}, "
                  f"выброшено {r.dropped:>6,} ({r.dropped / r.seen:>5.1%})")
    print("Источники отвечают.")
