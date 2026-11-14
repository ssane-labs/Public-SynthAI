"""От текста к батчам токенов.

Документы приходят потоком с Hub (data/sources.py), токенизируются, склеиваются
через <|eot|> и нарезаются на куски ровно нужной длины. Padding не используется
вовсе: документы упаковываются вплотную, и ни один токен батча не пропадает
впустую. На нашем бюджете это не оптимизация, а разница между 270B и 200B
увиденных токенов.

Токенизация идёт в отдельном потоке с очередью впереди: на TPU шаг обучения
занимает десятки миллисекунд, и однопоточный Python-токенизатор успел бы стать
узким местом там, где узким местом должна быть матмуль.
"""

from __future__ import annotations

import pathlib
import queue
import sys
import threading

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data import sources  # noqa: E402


def load_tokenizer(path: pathlib.Path | None = None):
    from tokenizers import Tokenizer

    path = path or (config.TOKENIZER_OUT / "synth1.json")
    if not path.exists():
        raise SystemExit(
            f"Нет токенизатора: {path}\n"
            "Собрать: py -3 tokenizer/train_tokenizer.py sample && ... train"
        )
    return Tokenizer.from_file(str(path))


class TokenStream:
    """Бесконечный поток кусков длины `seq_len + 1`.

    Плюс один — потому что вход и цель это один и тот же массив со сдвигом:
    tokens[:-1] предсказывает tokens[1:].
    """

    def __init__(self, mixture: str, seq_len: int, seed: int = 0,
                 tokenizer=None, batch_texts: int = 64, split: str = "train"):
        self.mixture = mixture
        self.seq_len = seq_len
        self.chunk = seq_len + 1
        self.tokenizer = tokenizer or load_tokenizer()
        self.batch_texts = batch_texts
        self.split = split
        self.reader = sources.MixtureReader(
            mixture, seed=seed, split=split,
            token=config.key("HF_TOKEN", required=False) or None,
        )
        self.buffer: list[int] = []
        self.docs = 0
        self.tokens_emitted = 0

    def describe(self) -> str:
        return self.reader.describe()

    def state(self) -> dict:
        """Позиция в корпусе. Буфер не сохраняем: он меньше одного куска, и
        потерять его дешевле, чем тащить в чекпоинт список из тысяч чисел."""
        return {"reader": self.reader.state(), "docs": self.docs,
                "tokens_emitted": self.tokens_emitted, "seq_len": self.seq_len}

    def carry_over(self, state: dict) -> None:
        """Перенести позицию в корпусе из снимка другого этапа.

        Новый этап меняет длину куска и смесь, поэтому точное продолжение
        невозможно: буфер и жребий смеси начинаются заново. Зато каждый
        источник, общий у двух смесей, продолжает с того шарда, где
        остановился, — этап не перечитывает то, что модель уже видела.
        """
        sources = state["reader"].get("sources", {})
        moved = []
        for name, reader in self.reader.readers.items():
            if name in sources:
                reader.restore(sources[name])
                moved.append(name)
        self.tokens_emitted = state.get("tokens_emitted", 0)
        self.docs = state.get("docs", 0)
        return moved

    def restore(self, state: dict) -> None:
        if state["seq_len"] != self.seq_len:
            raise SystemExit(
                f"Чекпоинт снят на длине {state['seq_len']}, а поток на "
                f"{self.seq_len}. Смена длины — это новый этап curriculum, "
                "позиция в корпусе от неё не зависит, но сказать об этом надо явно."
            )
        self.reader.restore(state["reader"])
        self.docs = state["docs"]
        self.tokens_emitted = state["tokens_emitted"]

    def _texts(self):
        """Документы пачками — batch-кодирование быстрее поштучного вчетверо."""
        batch = []
        for _, text in self.reader:
            if len(text) < 200 or sources.mentions_foreign_identity(text):
                continue
            batch.append(text)
            if len(batch) >= self.batch_texts:
                yield batch
                batch = []

    def chunks(self):
        """Куски по seq_len+1 токенов, склеенные вплотную."""
        eot = config.EOT_ID
        for texts in self._texts():
            for encoded in self.tokenizer.encode_batch(texts):
                self.buffer.extend(encoded.ids)
                self.buffer.append(eot)
                self.docs += 1

            while len(self.buffer) >= self.chunk:
                out = self.buffer[: self.chunk]
                # Сдвиг ровно на seq_len, а не на chunk: последний токен куска
                # служит целью для предпоследнего и входом для следующего.
                self.buffer = self.buffer[self.seq_len:]
                self.tokens_emitted += self.seq_len
                yield np.asarray(out, dtype=np.int32)

    def batches(self, batch_size: int):
        rows = []
        for chunk in self.chunks():
            rows.append(chunk)
            if len(rows) == batch_size:
                yield {"tokens": np.stack(rows)}
                rows = []


class Prefetcher:
    """Очередь батчей, наполняемая фоновым потоком.

    JAX всё равно ждёт данные на хосте, поэтому один поток-производитель
    закрывает разрыв полностью: пока TPU считает шаг, следующий батч уже
    токенизируется.
    """

    def __init__(self, iterator, depth: int = 4):
        self.queue: queue.Queue = queue.Queue(maxsize=depth)
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(iterator,),
                                       daemon=True)
        self.thread.start()

    def _run(self, iterator):
        try:
            for item in iterator:
                if self._stop.is_set():
                    return
                self.queue.put(item)
        except BaseException as exc:          # noqa: BLE001
            # Исключение фонового потока иначе исчезает бесследно, а обучение
            # просто зависает на пустой очереди — самый неприятный вид поломки.
            self.error = exc
            self.queue.put(None)

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                if self.error:
                    raise self.error
                return
            yield item

    def stop(self):
        self._stop.set()


def token_budget(seq_len: int, batch_size: int, steps: int) -> str:
    total = seq_len * batch_size * steps
    return (f"{total / 1e9:.1f}B токенов "
            f"({batch_size}×{seq_len} × {steps:,} шагов)")


if __name__ == "__main__":
    # Проверка сквозного пути: текст с Hub → токены → батч.
    import time

    mixture = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 512

    stream = TokenStream(mixture, seq_len=seq_len, batch_texts=16)
    print(stream.describe())

    started = time.time()
    for i, batch in enumerate(stream.batches(batch_size=4), 1):
        t = batch["tokens"]
        if i == 1:
            print(f"\nформа батча {t.shape}, тип {t.dtype}")
            head = stream.tokenizer.decode(t[0][:60].tolist())
            print(f"начало первого куска: {head[:150]!r}")
        if i >= 3:
            break

    dt = time.time() - started
    print(f"\n{stream.docs} документов → {stream.tokens_emitted:,} токенов "
          f"за {dt:.1f}с ({stream.tokens_emitted / dt / 1000:.1f}k токенов/с)")
    print(f"пустых мест в батчах нет: padding не используется")
