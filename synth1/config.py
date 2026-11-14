"""Конфигурация модели.

Числа здесь — то, что описано в ARCHITECTURE.md. Всё, что помечено там как
гипотеза, живёт в этом файле как поле со значением по умолчанию: этап A меняет
значения, а не структуру.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    # --- Форма ---
    # Synth-1 — Synth-2 той же конструкции, уменьшенная вдвое: 579M всего и
    # 159M активных против 1228M и 330M. Сравнение — ARCHITECTURE.md, раздел 1.
    vocab_size: int = 32768
    d_model: int = 640
    n_layers: int = 20
    n_heads: int = 10
    n_kv_heads: int = 2          # GQA: впятеро меньше KV, впятеро меньше кэша
    head_dim: int = 64
    max_seq_len: int = 32768

    # --- Внимание ---
    window: int = 1024           # скользящее окно
    global_every: int = 5        # каждый n-й слой смотрит на весь контекст: 4 из 20
    rope_base: float = 10_000.0
    qk_norm: bool = True
    # Размер блока запросов во внимании. Полная матрица [B,H,T,T] на длине 2048
    # весит 1.5 ГБ на слой и в память чипа не помещается; блоками нужно 120 МБ.
    # 0 выключает блочность — только для коротких прогонов и тестов.
    attn_chunk: int = 256

    # Чем считать внимание.
    #
    #   "splash"   сплавленное ядро Pallas на TPU: матрица [C,S] живёт в VMEM и
    #              в память не выезжает вовсе. Окно достаётся бесплатно —
    #              блоки ключей вне окна ядро не считает.
    #   "blocked"  блоки на lax.scan (`_blocked_attention`). Работает везде,
    #              в том числе на CPU, и служит запасным путём.
    #
    # Умолчание — "blocked", и это результат замера, а не осторожность.
    #
    # Сессия 15.09.2026, батч 32 × 2048 на v5e-8, одна и та же команда:
    #     блочное   915.1 мс/шаг, 71,614 ток/с, MFU 4.3%
    #     splash   1063.9 мс/шаг, 61,598 ток/с, MFU 3.7%   (+16.3%)
    # За сессию это 25,432 шага вместо 29,598 — то есть ядро не ускорило, а
    # замедлило на седьмую часть.
    #
    # Гипотеза была: матрица [C,S] не выезжает в память, значит быстрее.
    # Память она действительно экономит, но выигрыш съеден. Два подозреваемых,
    # оба не проверены: `head_dim` 64 против 128 дорожек MXU (ядро считает
    # половину плитки впустую) и транспонирования [B,T,H,Dh] → [B,Hkv,G,T,Dh]
    # на входе и обратно на выходе, по три тензора на слой.
    #
    # Ядро оставлено рабочим и проверенным — переключается одним словом, когда
    # найдётся время мерить размер блока и укладку (`speed.py`, строки
    # «splash блоком 256/512»).
    attn_impl: str = "blocked"
    splash_block: int = 128
    # Ядро Pallas умеет считаться и вне TPU (interpret), но медленно: это режим
    # для проверки совпадения чисел, а не для работы. Проверки на CPU идут по
    # блочному пути, поэтому здесь False.
    splash_interpret: bool = False

    # --- MoE ---
    n_experts: int = 12
    n_shared_experts: int = 1
    d_ff: int = 1024             # на одного эксперта
    top_k: int = 2
    # Балансировка без вспомогательного лосса. Шаг знаковый, то есть скорость
    # исправления не зависит от величины перекоса — только от этого числа.
    # При 1e-3 смещение в слое 0 за 50 000 шагов уехало на 16.7 и всё равно не
    # догнало схлопнувшийся роутер (замер 02.09.2026). Поднято до 3e-3:
    # десять единиц отыгрываются за ~3300 шагов, а колебание у равновесия
    # равно самому шагу, то есть ±0.003 против логитов масштаба 2 — незаметно.
    router_bias_lr: float = 3e-3
    router_z_loss: float = 1e-4
    # Нормировать ли веса выбранных экспертов на их сумму.
    #
    # Сейчас вклад эксперта — его сырая вероятность из softmax по всем
    # двенадцати. Пока роутер не заострился, это около 1/12: разреженная ветвь
    # входит в выход слоя с коэффициентом ~0.17, тогда как общий эксперт — с
    # единицей. То есть двенадцать экспертов шепчут, а слышно одного общего.
    # Mixtral и DeepSeek нормируют top-k на сумму именно поэтому.
    #
    # Значение по умолчанию не меняется на ходу: это другая функция, и снимок,
    # снятый с одним значением, с другим считает не то, что учил. Проверять
    # надо абляцией с нуля (evals/ablations.md), а не переключением на живом
    # прогоне.
    router_renorm: bool = False

    # Укладка einsum'ов эксперта. См. moe.py: свёртка батча в ось слотов
    # убирает вторую батчевую ось и размножение весов по ней, но трогает ту
    # ось, по которой батч разрезан между чипами. Меряется в `speed.py`,
    # включается после замера, а не до.
    moe_fold_batch: bool = False

    # Как токены едут в слоты экспертов и обратно (moe.py, `permute_rows`).
    #
    #   "scatter"  .at[].set() туда, .at[].get() обратно — как было. Обратный
    #              проход gather'а на TPU — scatter-add, и он медленный.
    #   "gather"   оба направления — gather по номерам строк, и обратный
    #              проход тоже gather: отображение «назначение ↔ слот»
    #              взаимно однозначно, поэтому транспонирование перестановки —
    #              это перестановка обратная. Та же математика.
    #
    # Замер 06.10.2026 на v5e-8 под shard_map: 2048 × 32 — scatter 322 мс,
    # gather 304; 8192 × 8 — scatter 395, gather 363. Отсюда умолчание.
    moe_dispatch: str = "gather"

    # --- Неопределённость как управляющий сигнал (ARCHITECTURE.md, раздел 3) ---
    epistemic_head: bool = True
    epistemic_weight: float = 0.05     # λ_epi
    calib_weight: float = 1.0          # головка ответа на этапе E (data/calib.py)
    orpo_weight: float = 0.1           # λ ORPO на этапе F (data/prefs.py)
    pref_pairs: int = 16               # пар предпочтений на строку, не больше
    entropy_gate: bool = True          # ворота второго эксперта
    entropy_gate_threshold: float = 0.5   # τ
    entropy_gate_temp: float = 0.1        # T

    mod: bool = True                   # адаптивная глубина
    mod_every: int = 2                 # применять на каждом n-м слое
    mod_capacity: float = 0.5          # доля токенов, проходящих блок
    mod_entropy_alpha: float = 1.0     # вклад u в скор отбора
    mod_block: int = 128               # окно отбора; см. model.py, _moe_with_mod

    # --- Multi-token prediction ---
    mtp: bool = True
    mtp_weight: float = 0.3            # λ_mtp

    # --- Прочее ---
    # Пересчитывать активации слоя при обратном проходе вместо хранения.
    # Треть лишнего счёта в обмен на возможность держать большой батч —
    # см. model.py, _checkpointed_layer.
    remat: bool = True
    tie_embeddings: bool = True
    z_loss: float = 1e-4
    # Размер куска, которым лосс поднимает логиты. Полный тензор [B,T,V] при
    # батче 32 и длине 2048 весит 12.9 ГБ (1.5 на чип), и живёт он не в одном
    # экземпляре — на этом претрейн упирался в HBM. 0 выключает нарезку.
    loss_chunk: int = 256
    dtype: str = "bfloat16"

    def __post_init__(self):
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError(
                f"d_model={self.d_model} != n_heads*head_dim="
                f"{self.n_heads * self.head_dim}"
            )
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads должно делиться на n_kv_heads")
        if self.vocab_size % 128:
            raise ValueError(
                f"vocab_size={self.vocab_size} не кратен 128 — последняя плитка "
                "выходной матмули на TPU будет считаться вхолостую"
            )

    # --- Счётчики параметров ---
    def n_params(self) -> dict[str, int]:
        emb = self.vocab_size * self.d_model
        attn_per_layer = (
            self.d_model * self.n_heads * self.head_dim          # q
            + 2 * self.d_model * self.n_kv_heads * self.head_dim  # k, v
            + self.n_heads * self.head_dim * self.d_model         # o
        )
        expert = 3 * self.d_model * self.d_ff                     # SwiGLU
        moe_per_layer = expert * (self.n_experts + self.n_shared_experts)

        total = emb + self.n_layers * (attn_per_layer + moe_per_layer)
        if not self.tie_embeddings:
            total += emb
        if self.mtp:
            total += attn_per_layer + moe_per_layer + self.d_model * self.d_model

        active = (
            emb                                                   # выходные логиты
            + self.n_layers * attn_per_layer
            + self.n_layers * expert * (self.n_shared_experts + self.top_k)
        )
        return {"total": total, "active": active}

    def summary(self) -> str:
        n = self.n_params()
        return (
            f"Synth-1: всего {n['total'] / 1e6:.0f}M, "
            f"активно {n['active'] / 1e6:.0f}M на токен "
            f"(разрежённость {n['total'] / n['active']:.1f}x)"
        )

    def kv_cache_bytes(self, seq_len: int, bytes_per_elem: int = 2) -> int:
        """Сколько памяти съест кэш. Половина ответа на вопрос «почему быстрая»."""
        per_token_per_layer = 2 * self.n_kv_heads * self.head_dim * bytes_per_elem
        n_global = self.n_layers // self.global_every
        n_local = self.n_layers - n_global
        return (
            n_local * min(seq_len, self.window) * per_token_per_layer
            + n_global * seq_len * per_token_per_layer
        )

    def save(self, path: pathlib.Path) -> None:
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: pathlib.Path) -> "ModelConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


# Конфигурация для абляций: та же форма блока, но столько, сколько успевает
# обучиться за сорок минут. Меняется только размер, чтобы выводы переносились.
# Нарезка логитов здесь выключена намеренно. На модели в 43M активных
# параметров выходная проекция — половина всего счёта, и куски по 256 позиций
# замедлили шаг с 69 до 194 мс (замер 28.08.2026), то есть втрое. Памяти же
# нарезка тут не экономит ничего: логиты весят 0.8 ГБ на чип и влезают целиком.
# Целевой конфигурации нарезка нужна, абляциям — нет.
ABLATION = ModelConfig(
    loss_chunk=0,
    d_model=384,
    n_layers=8,
    n_heads=6,
    n_kv_heads=2,
    head_dim=64,
    n_experts=8,
    d_ff=768,
    max_seq_len=2048,
    window=512,
    global_every=4,
)

# Плотный контрольный конфиг: с ним сравнивается всё остальное на этапе A.
# Если MoE его не обгоняет при равных активных FLOPs — вся затея с разрежённостью
# не оправдана, и это надо узнать за сорок минут, а не за сто часов.
ABLATION_DENSE = dataclasses.replace(
    ABLATION,
    n_experts=1,
    n_shared_experts=0,
    top_k=1,
    d_ff=2048,
    entropy_gate=False,
    mod=False,
)

DEFAULT = ModelConfig()


if __name__ == "__main__":
    for name, cfg in [("целевая", DEFAULT), ("абляции", ABLATION),
                      ("абляции dense", ABLATION_DENSE)]:
        print(f"{name:<15} {cfg.summary()}")
    print()
    for n in (2048, 8192, 32768):
        print(f"KV-кэш на {n:>5} токенов: "
              f"{DEFAULT.kv_cache_bytes(n) / 1024**2:>6.1f} МБ")
