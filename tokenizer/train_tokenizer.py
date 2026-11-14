"""Свой BPE-токенизатор Synth-1.

Три команды, запускаются по порядку:

    py -3 tokenizer/train_tokenizer.py sample --gb 1.5
    py -3 tokenizer/train_tokenizer.py train
    py -3 tokenizer/train_tokenizer.py eval

Сэмпл кладётся на диск отдельным шагом, чтобы обучение можно было перезапускать
с другими параметрами, не выкачивая корпус заново.

Критерий приёмки (PLAN.md, этап 0): английский сжимается не хуже 4.2 символов на
токен, русский — не хуже 3.0. Ниже этого словарь можно считать рабочим.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data import sources  # noqa: E402

SAMPLE_DIR = config.DATA_SAMPLE / "tokenizer"
OUT_DIR = config.TOKENIZER_OUT
TOKENIZER_FILE = OUT_DIR / "synth1.json"

# Правило разрезания текста до BPE. Взято в форме, проверенной на больших
# токенизаторах, с одной правкой: цифры режутся по одной, а не группами до трёх.
#
# Группы дешевле по длине, но арифметика у маленькой модели от них разваливается —
# «127» и «1275» не имеют общих токенов, и складывать столбиком не на чем.
# Мы маленькие, нам важнее считать, чем экономить.
SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def cmd_sample(args):
    """Выкачать сэмпл текста по смеси `tokenizer` из mixture.yaml."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = SAMPLE_DIR / "sample.jsonl"
    state_file = SAMPLE_DIR / "sample.state.json"
    budget = int(args.gb * 1024**3)

    reader = sources.MixtureReader(
        "tokenizer", seed=args.seed,
        token=config.key("HF_TOKEN", required=False) or None,
    )
    print(reader.describe(), flush=True)
    print(f"Цель: {args.gb} ГБ текста в {out_file}")

    # Докачка. Сбор идёт часами по узкому каналу, и обрыв связи не должен
    # стоить всего пройденного пути: позиция в корпусе лежит рядом с данными и
    # обновляется вместе с отчётом о прогрессе.
    written = 0
    per_source: dict[str, int] = {}
    if not args.restart and state_file.exists() and out_file.exists():
        saved = json.loads(state_file.read_text(encoding="utf-8"))
        # Файл мог уйти вперёд состояния — то, что записано после последнего
        # сохранения позиции. Обрезаем до известной границы, иначе после
        # восстановления те же документы лягут вторым слоем.
        with out_file.open("r+b") as fh:
            fh.truncate(saved["written"])
        reader.restore(saved["reader"])
        written = saved["written"]
        per_source = saved["per_source"]
        print(f"Продолжаем с {written / 1024**3:.2f} ГБ "
              f"({sum(per_source.values()):,} документов уже собрано)")

    next_report = written + 64 * 1024**2
    started = time.time()
    start_bytes = written

    def save_state():
        state_file.write_text(json.dumps({
            "written": written, "per_source": per_source,
            "reader": reader.state(),
        }, ensure_ascii=False), encoding="utf-8")

    with out_file.open("a" if written else "w", encoding="utf-8") as fh:
        for src, text in reader:
            if len(text) < 200 or sources.mentions_foreign_identity(text):
                continue
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += len(text.encode("utf-8"))
            per_source[src] = per_source.get(src, 0) + 1
            if written >= budget:
                break
            if written >= next_report:
                next_report += 64 * 1024**2
                fh.flush()
                save_state()
                pace = ((written - start_bytes) / 1024**2
                        / max(time.time() - started, 1))
                eta = (budget - written) / 1024**2 / max(pace, 0.01) / 60
                print(f"  {written / 1024**3:.2f} ГБ  ({pace:.1f} МБ/с, "
                      f"осталось ~{eta:.0f} мин)", flush=True)

    save_state()
    print(f"\nГотово: {written / 1024**3:.2f} ГБ за {time.time() - started:.0f} с")
    for src, n in sorted(per_source.items(), key=lambda kv: -kv[1]):
        print(f"  {src:<20} {n:>9,} документов")


def _iter_sample():
    path = SAMPLE_DIR / "sample.jsonl"
    if not path.exists():
        raise SystemExit(f"Нет сэмпла: {path}\nСначала: train_tokenizer.py sample")
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)["text"]


def cmd_train(args):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
            # add_prefix_space=False: иначе каждый документ начинается с пробела,
            # и первый токен ответа отличается от того же слова в середине.
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=config.VOCAB_SIZE,
        special_tokens=config.SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=args.min_frequency,
        show_progress=True,
    )

    print(f"Обучение BPE: словарь {config.VOCAB_SIZE:,}, "
          f"{len(config.SPECIAL_TOKENS)} служебных токенов")
    started = time.time()
    tok.train_from_iterator(sources.counted(_iter_sample(), every=200_000, label="док "),
                            trainer=trainer)
    print(f"Обучено за {time.time() - started:.0f} с")

    # Служебные токены обязаны стоять в начале и в том порядке, в каком объявлены:
    # PAD_ID и EOT_ID зашиты в config и в код обучения как числа.
    for expected_id, token in enumerate(config.SPECIAL_TOKENS[:config.N_PROTOCOL_TOKENS]):
        got = tok.token_to_id(token)
        if got != expected_id:
            raise SystemExit(
                f"Служебный токен {token} получил id {got}, а ожидался {expected_id}. "
                "Дальше нельзя: PAD_ID и EOT_ID зашиты в config числами."
            )

    tok.save(str(TOKENIZER_FILE))
    print(f"Сохранено: {TOKENIZER_FILE}  ({TOKENIZER_FILE.stat().st_size / 1024**2:.1f} МБ)")
    print(f"Реальный размер словаря: {tok.get_vocab_size():,}")


# Тексты для замера сжатия. Короткие и разные — задача не измерить корпус, а
# поймать грубую поломку: разъехавшиеся отступы в коде, развалившиеся адреса.
PROBES = {
    "en": "The transformer architecture processes sequences in parallel, which is "
          "why it replaced recurrent networks for language modelling tasks.",
    # Поисковая выдача — главный вход модели, и адреса в ней должны резаться
    # на куски, а не на отдельные буквы.
    "web": "[2] History of Canberra - https://en.wikipedia.org/wiki/History_of_Canberra "
           "- The city was officially named on 12 March 1913.",
    "code": "def entropy(logits):\n    p = softmax(logits, axis=-1)\n"
            "    return -(p * log(p + 1e-9)).sum(axis=-1)\n",
    "math": "Let f(x) = 3x^2 + 17x - 42. Then f'(x) = 6x + 17, so f'(0) = 17.",
    "digits": "In 1997 the population reached 12345678 people, up from 9876543.",
}


def cmd_eval(args):
    from tokenizers import Tokenizer

    if not TOKENIZER_FILE.exists():
        raise SystemExit(f"Нет токенизатора: {TOKENIZER_FILE}")
    tok = Tokenizer.from_file(str(TOKENIZER_FILE))

    print(f"{'проба':<8} {'симв':>6} {'токенов':>8} {'симв/токен':>11}")
    for name, text in PROBES.items():
        ids = tok.encode(text).ids
        ratio = len(text) / max(len(ids), 1)
        print(f"{name:<8} {len(text):>6} {len(ids):>8} {ratio:>11.2f}")

    # Порог из PLAN.md. Замер без числа результатом не считается.
    en = len(PROBES["en"]) / len(tok.encode(PROBES["en"]).ids)
    web = len(PROBES["web"]) / len(tok.encode(PROBES["web"]).ids)
    print()
    print(f"EN  {en:.2f} симв/ток (порог 4.20) — {'ок' if en >= 4.2 else 'НИЖЕ ПОРОГА'}")
    print(f"WEB {web:.2f} симв/ток (порог 3.00) — {'ок' if web >= 3.0 else 'НИЖЕ ПОРОГА'}")

    # Round-trip: декодированное должно совпадать с исходным байт в байт.
    for name, text in PROBES.items():
        back = tok.decode(tok.encode(text).ids)
        if back != text:
            print(f"\nROUND-TRIP СЛОМАН на {name!r}:")
            print(f"  было: {text!r}")
            print(f"  стало: {back!r}")
            raise SystemExit(1)
    print("Round-trip: совпадает на всех пробах.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample", help="выкачать сэмпл текста с Hub")
    p.add_argument("--gb", type=float, default=1.5,
                   help="сколько гигабайт текста собрать (по умолчанию 1.5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restart", action="store_true",
                   help="начать сбор заново, не подхватывая прошлую позицию")
    p.set_defaults(fn=cmd_sample)

    p = sub.add_parser("train", help="обучить BPE на собранном сэмпле")
    p.add_argument("--min-frequency", type=int, default=2)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval", help="замерить сжатие и round-trip")
    p.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
