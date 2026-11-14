"""Цикл обучения. Один запуск — одна сессия Kaggle.

Устройство сессии:

    поднять модель          с Hub, если снимок есть; с нуля, если это старт
    считать шаги            пока не кончится время или заданное число шагов
    сохраниться и уйти      обязательно, даже при поломке

Последнее — не вежливость, а главное требование к этому файлу. Сессия
обрывается по лимиту в девять часов, и всё, что не легло в чекпоинт, потеряно
вместе с квотой. Поэтому выход по времени наступает заранее, с запасом на
выгрузку, а сохранение стоит в finally.

    py -3 train/pretrain.py --run ablation-dense --steps 200 --seq-len 512
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from synth1 import checkpoint as C  # noqa: E402
from synth1 import data as D  # noqa: E402
from data import sft as SFT  # noqa: E402
from data import search_sft as SS  # noqa: E402
from data import calib as CB  # noqa: E402
from synth1 import fsdp as F  # noqa: E402
from synth1 import holdout as HO  # noqa: E402
from synth1 import layers as L  # noqa: E402
from synth1 import model as M  # noqa: E402
from synth1 import optimizer as O  # noqa: E402
from synth1 import preflight as P  # noqa: E402
from synth1 import sharding as S  # noqa: E402
from synth1 import watch as W  # noqa: E402
from synth1.config import ABLATION, ABLATION_DENSE, DEFAULT, ModelConfig  # noqa: E402
from synth1.moe import update_router_bias  # noqa: E402

PRESETS = {"default": DEFAULT, "ablation": ABLATION, "ablation-dense": ABLATION_DENSE}


class FakeStream:
    """Случайные токены вместо корпуса.

    Нужен, чтобы отделить вопрос «считает ли железо» от вопроса «идут ли
    данные». На новой машине первым запускается именно он: если шаг
    компилируется и скорость разумная, значит остаётся только корпус.

    Лосс на таких данных не падает и падать не должен — предсказывать
    равномерный шум нечем, он держится около log(V).
    """

    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tokens_emitted = 0
        self.docs = 0
        self._key = jax.random.key(0)

    def describe(self) -> str:
        return "корпус: случайные токены (--fake-data), лосс держится у log(V)"

    def state(self) -> dict:
        return {"fake": True, "tokens_emitted": self.tokens_emitted}

    def restore(self, state: dict) -> None:
        self.tokens_emitted = state.get("tokens_emitted", 0)

    def batches(self, batch_size: int):
        while True:
            self._key, key = jax.random.split(self._key)
            self.tokens_emitted += batch_size * self.seq_len
            self.docs += batch_size
            yield {"tokens": jax.random.randint(
                key, (batch_size, self.seq_len + 1), 1, self.vocab_size,
                dtype=jnp.int32)}


def build_step(cfg: ModelConfig, opt: O.OptConfig, total_steps: int, mesh,
               fsdp: bool = True):
    """Скомпилированный шаг обучения.

    fsdp=True считает лосс под shard_map: каждый чип — свой кусок батча, веса
    собираются явно (synth1/fsdp.py). False — прежний путь, раскладку выводит
    XLA; на MoE он выводил её так, что каждый чип считал весь батч.

    Обновление router_bias живёт здесь же, но идёт мимо градиента: это
    регулировщик очереди к экспертам, а не параметр. Прятать его в оптимизатор
    нельзя — он двигается по загрузке, которую видно только после прохода.
    """

    def step(params, state, batch):
        if fsdp and mesh.shape[S.AXIS] > 1:
            loss_fn = F.sharded_loss(cfg, mesh, params)
        else:
            loss_fn = lambda p, b: M.loss_fn(p, b, cfg)  # noqa: E731
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch)
        params, state, opt_metrics = O.apply_updates(
            params, grads, state, total_steps, opt)

        if "load" in metrics:
            loads = metrics["load"]                      # [n_layers, n_experts]
            layers = []
            for i, layer in enumerate(params["layers"]):
                layer = dict(layer)
                moe = dict(layer["moe"])
                moe["router_bias"] = update_router_bias(
                    moe["router_bias"], loads[i], cfg.router_bias_lr)
                layer["moe"] = moe
                layers.append(layer)
            params = {**params, "layers": layers}

        return params, state, {**metrics, **opt_metrics}

    return jax.jit(step, donate_argnums=(0, 1))


def to_hub_now(step: int, push_every: int, checkpoint_every: int) -> bool:
    """Уезжает ли снимок этого шага на Hub. Чистая функция ради теста.

    push_every=0 — как раньше, каждый чекпоинт. Иначе заливка идёт по своей
    сетке, и она обязана быть кратна сетке чекпоинтов: снимок, которого нет на
    диске, залить нельзя, а «каждые 8000» при чекпоинте раз в 3000 дало бы ноль
    заливок за сессию — то есть ровно ту потерю, ради предотвращения которой
    заливка и существует.
    """
    if push_every <= 0:
        return True
    if push_every % checkpoint_every:
        raise SystemExit(
            f"--push-every {push_every} не кратен --checkpoint-every "
            f"{checkpoint_every}: заливать будет нечего"
        )
    return step % push_every == 0


def holdout_trend(marks: list[tuple[int, float]]):
    """Наклон отложенного лосса по всем замерам. Чистая функция ради теста.

    Возвращает (наклон на 1000 шагов, t, шум) или None, если замеров мало.

    Считать вердикт сессии по двум крайним замерам нельзя, и это стоило
    неверного вывода 15.09.2026: строка «2.1285 → 2.0854, −0.0431 — учится»
    сравнивала первую точку с последней, а обе случайно оказались из нижней
    части разброса. По всем 31 замеру наклон был +0.00002 на 1000 шагов при
    t = +0.04, то есть ровно ноль. При шуме 0.025 разность двух точек не
    отличима от нуля — нужен наклон и его значимость.
    """
    if len(marks) < 4:
        return None
    n = len(marks)
    mx = sum(s for s, _ in marks) / n
    my = sum(v for _, v in marks) / n
    sxx = sum((s - mx) ** 2 for s, _ in marks)
    if sxx <= 0:
        return None
    slope = sum((s - mx) * (v - my) for s, v in marks) / sxx
    intercept = my - slope * mx
    residuals = [v - (intercept + slope * s) for s, v in marks]
    noise = (sum(r * r for r in residuals) / (n - 2)) ** 0.5
    se = noise / sxx ** 0.5
    t = slope / se if se > 0 else 0.0
    return slope * 1000, t, noise


def apply_overrides(cfg: ModelConfig, pairs: list[str]) -> ModelConfig:
    """Правки конфигурации из командной строки: --set router_renorm=true.

    Тип берётся из самого поля, поэтому "true" ложится в bool, а "256" в int.
    Неизвестное имя — ошибка, а не тихое игнорирование: абляция, которая на
    самом деле ничего не изменила, но выглядит как изменившая, отравляет
    таблицу навсегда.
    """
    fields = {f.name: f.type for f in dataclasses.fields(cfg)}
    changes: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set ждёт ПОЛЕ=ЗНАЧЕНИЕ, а пришло {pair!r}")
        name, _, raw = pair.partition("=")
        name, raw = name.strip(), raw.strip()
        if name not in fields:
            near = [f for f in fields if name.lower() in f.lower()]
            raise SystemExit(
                f"в конфигурации нет поля {name!r}"
                + (f". Похожие: {', '.join(sorted(near))}" if near else ""))
        current = getattr(cfg, name)
        if isinstance(current, bool):
            if raw.lower() not in {"true", "false", "1", "0", "да", "нет"}:
                raise SystemExit(f"{name}: ждал true/false, а пришло {raw!r}")
            value = raw.lower() in {"true", "1", "да"}
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw
        changes[name] = value
    return dataclasses.replace(cfg, **changes)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="имя прогона; оно же имя репо на Hub")
    ap.add_argument("--preset", default="default", choices=list(PRESETS))
    ap.add_argument("--mixture", default="pretrain")
    ap.add_argument("--steps", type=int, default=10_000, help="сколько шагов за сессию")
    ap.add_argument("--total-steps", type=int, default=200_000,
                    help="длина всего обучения — от неё считается расписание LR")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--loss-chunk", type=int, default=None,
                    help="сколько позиций разом поднимать в логиты; меньше — "
                         "ниже пик памяти на выходной проекции")
    ap.add_argument("--hours", type=float, default=8.2,
                    help="через сколько часов сохраниться и выйти (лимит сессии 9)")
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--keep", type=int, default=1,
                    help="сколько снимков держать локально; история живёт на Hub, "
                         "а под /kaggle/working всего 20 ГБ")
    ap.add_argument("--no-checkpoint", dest="checkpoint", action="store_false",
                    help="не сохранять вообще — для прогонов, снимок которых "
                         "никому не нужен; --fake-data включает это сам")
    ap.add_argument("--push-every", type=int, default=0, metavar="ШАГОВ",
                    help="как часто снимок уезжает на Hub; 0 — на каждом "
                         "чекпоинте. Последний снимок сессии уходит всегда")
    ap.add_argument("--keep-remote", type=int, default=2,
                    help="сколько снимков держать на Hub. Приватная квота "
                         "кончается на седьмом снимке по 9.3 ГБ, а продолжение "
                         "читает только Hub — переполнение стоит целой сессии")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=1000,
                    help="каждые N шагов мерить лосс на отложенном наборе; 0 — "
                         "не мерить. Это единственное число, по которому видно "
                         "тренд: обучающий лосс шумит от смеси источников "
                         "сильнее, чем меняется за тысячи шагов")
    ap.add_argument("--eval-batches", type=int, default=4,
                    help="сколько батчей в отложенном наборе")
    ap.add_argument("--set", action="append", default=[], metavar="ПОЛЕ=ЗНАЧЕНИЕ",
                    dest="overrides",
                    help="поправить поле конфигурации модели, можно несколько "
                         "раз: --set router_renorm=true --set mod=false. "
                         "Нужно абляциям: одно изменение за прогон, и это "
                         "изменение должно быть видно в команде, а не в "
                         "правке файла между запусками")
    ap.add_argument("--summary", type=pathlib.Path, default=None,
                    help="куда положить итог прогона в JSON. Абляция без "
                         "машиночитаемого итога превращается в вычитывание "
                         "лога глазами, а таблицу надо собрать из десятка "
                         "прогонов подряд")
    ap.add_argument("--anneal", type=int, default=None, metavar="ШАГОВ",
                    help="отжечь LR до нуля за столько шагов, начиная с того "
                         "места, где стоит снимок. Расписание WSD держит "
                         "плато до 88% всей длины, поэтому обычный прогон на "
                         "середине пути идёт при неизменном LR и лосс на нём "
                         "стоит — это работа расписания, а не болезнь модели. "
                         "Отличить одно от другого можно только отжигом. "
                         "ВАЖНО: отожжённая модель — другая ветка обучения, "
                         "и продолжать с неё плато нельзя; запускать только с "
                         "--no-checkpoint или под другим --run")
    ap.add_argument("--warmup", type=int, default=None, metavar="ШАГОВ",
                    help="длина разогрева LR. По умолчанию 2000 — столько же, "
                         "сколько вся абляция, и тогда пикового LR модель не "
                         "видит ни разу, а сравниваются не механизмы, а "
                         "случайные точки на разгоне")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed инициализации весов и порядка корпуса. Нужен абляциям: без второго seed не с чем сравнивать разницу между строками — она может быть и шумом одного прогона")
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="множитель к обоим LR. Ручка для проверки гипотезы "
                         "«плато слишком высокое»: тот же снимок, тысяча шагов "
                         "с 1.0 и с 0.3, сравнить отложенный лосс")
    ap.add_argument("--show-every", type=int, default=0,
                    help="каждые N шагов печатать примеры генерации; 0 — не показывать. Это окно, а не проверка: смотреть глазами, что модель уже умеет")
    ap.add_argument("--show-tokens", type=int, default=40,
                    help="длина примеров генерации")
    ap.add_argument("--resume", action="store_true", help="продолжить с Hub")
    ap.add_argument("--init-from", default=None, metavar="ПРОГОН",
                    help="если у своего прогона на Hub ещё нет снимков, "
                         "стартовать со свежего снимка этого прогона: веса, "
                         "momentum, шаг и позиция в корпусе. Так настоящий "
                         "отжиг (этап C) идёт под своим --run и не трогает "
                         "снимки плато")
    ap.add_argument("--new-stage", action="store_true",
                    help="начать новый этап со снимка --init-from (обычно — "
                         "отожжённого): другая длина и смесь, LR заново "
                         "разогревается до пикового × --lr-scale и спадает до "
                         "пола за --anneal шагов. Позиция в каждом источнике "
                         "корпуса переносится, чтобы не повторять данные")
    ap.add_argument("--skip-batches", type=int, default=0,
                    help="выбросить N батчей после восстановления. Нужно для "
                         "перезапуска после nan: позиция в корпусе лежит в "
                         "снимке, и без сдвига прогон второй раз получит тот "
                         "же батч и тот же nan")
    ap.add_argument("--no-preflight", dest="preflight", action="store_false",
                    help="пропустить проверку первого шага; она стоит секунды "
                         "и говорит, где именно расходится, поэтому по умолчанию включена")
    ap.add_argument("--no-push", action="store_true", help="не отправлять на Hub")
    ap.add_argument("--auto-sharding", action="store_true",
                    help="лосс под обычным jit, раскладку выводит XLA — как "
                         "до 04.10.2026. По умолчанию лосс идёт под shard_map "
                         "(synth1/fsdp.py), каждый чип считает свой кусок батча")
    ap.add_argument("--sft-limit", type=int, default=0,
                    help="этап D1: брать не больше N вопросов каждого "
                         "источника. Только для проверки на CPU — вся база "
                         "D1 в памяти весит ~6 ГБ")
    ap.add_argument("--fake-data", action="store_true",
                    help="случайные токены вместо корпуса: проверить, что железо "
                         "считает и шаг компилируется, не трогая данные")
    args = ap.parse_args()

    # Снимок прогона на шуме — девять гигабайт весов, которые ничему не
    # научились. Ровно они и съели диск в прошлой сессии, а следом упал не
    # прогон, а сам ноутбук: papermill не смог дописать свой .ipynb.
    if args.fake_data and args.checkpoint:
        args.checkpoint = False
        print("--fake-data: чекпоинты выключены, снимок шума не нужен")

    cfg = dataclasses.replace(PRESETS[args.preset], max_seq_len=max(
        PRESETS[args.preset].max_seq_len, args.seq_len))
    if args.loss_chunk is not None:
        cfg = dataclasses.replace(cfg, loss_chunk=args.loss_chunk)
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)
        print("правки конфигурации: " + ", ".join(args.overrides))
    opt = O.OptConfig()
    if args.warmup is not None:
        opt = dataclasses.replace(opt, warmup_steps=args.warmup)
        print(f"разогрев LR: {args.warmup:,} шагов")
    if args.lr_scale != 1.0:
        opt = dataclasses.replace(opt, lr_muon=opt.lr_muon * args.lr_scale,
                                  lr_adam=opt.lr_adam * args.lr_scale)

    print(f"прогон {args.run!r}: {cfg.summary()}")
    if args.lr_scale != 1.0:
        print(f"LR умножен на {args.lr_scale}: muon {opt.lr_muon:.5f}, "
              f"adam {opt.lr_adam:.6f}")
    print(f"бюджет сессии: {D.token_budget(args.seq_len, args.batch_size, args.steps)}")

    mesh = S.make_mesh()
    print(f"устройств: {len(jax.devices())} × {jax.devices()[0].device_kind}")

    # Сплавленное внимание — до всего остального: если ядро Pallas на этом
    # железе не собирается, узнать об этом надо за две секунды, а не на первом
    # шаге через восемь часов. Отказ не фатален, блочный путь никуда не делся.
    if cfg.attn_impl == "splash":
        if L.probe_splash(cfg, args.seq_len, S.batch_sharding(mesh),
                          args.batch_size):
            print("внимание: сплавленное ядро (splash), блок "
                  f"{cfg.splash_block}")
        else:
            cfg = dataclasses.replace(cfg, attn_impl="blocked")
            print(f"внимание: блочное, блок {cfg.attn_chunk}")

    # Дешёвые проверки — раньше дорогих. Инициализация модели, компиляция шага и
    # предполётный прогон стоят минуты; несходящееся число видно сразу.
    S.check_batch(args.batch_size, mesh)

    # --- Состояние ---------------------------------------------------------
    # Раскладка по чипам идёт сразу, до всего остального. Нешардированная
    # модель целиком лежит на нулевом чипе: 4.9 ГБ веса плюс столько же
    # momentum, и это ещё до того, как с Hub приедет снимок с такой же парой.
    # В шестнадцать гигабайт HBM две пары не помещаются — на этом сорвалась
    # сессия 28.08.2026, причём на всех батчах одинаково, потому что батч тут
    # ни при чём.
    params = S.distribute(M.init_model(jax.random.key(args.seed), cfg), mesh)
    # m и v рождаются из params и раскладку наследуют, но полагаться на это
    # нельзя: гарантий у поэлементных операций нет, а цена ошибки — тот же
    # переполненный нулевой чип. Раскладываем явно, для уже разложенного это
    # ничего не стоит.
    state = O.init_state(params, opt)
    state = {**state, "m": S.distribute(state["m"], mesh),
             "v": S.distribute(state["v"], mesh)}
    print(S.report(params, mesh))
    print(S.report(state["m"], mesh, "momentum"))

    snapshot_gb = C.size_gb(params, state["m"], state["v"])
    if args.checkpoint:
        disk = C.free_gb(config.ROOT)
        print(f"диск: снимок {snapshot_gb:.2f} ГБ, свободно {disk:.1f} ГБ, "
              f"держим локально {args.keep}")
        if disk < snapshot_gb * (args.keep + 1):
            print("  места впритык: снимки будут вытеснять друг друга "
                  "(история всё равно уходит на Hub)")
    else:
        print(f"диск: чекпоинты выключены (снимок был бы {snapshot_gb:.2f} ГБ)")

    if args.fake_data:
        stream = FakeStream(args.seq_len, cfg.vocab_size)
    elif args.mixture == "d1":
        # Этап D1: вопросы с документами, лосс только по ответу (data/sft.py).
        stream = SFT.SFTStream(args.seq_len, seed=args.seed,
                               limit=args.sft_limit or None)
    elif args.mixture == "d2":
        # Этап D2: траектории поиска + D1 + претрейн (data/search_sft.py).
        stream = SS.MixStream(args.seq_len, seed=args.seed,
                              limit=args.sft_limit or None)
    elif args.mixture == "d2w":
        # D2 третьей версии: поиск по Википедии в среде замера + почищенный
        # OpenResearcher + D1 + претрейн (data/wiki_search.py).
        from data import wiki_search as WS
        stream = WS.SearchMixStream(args.seq_len, seed=args.seed,
                                  limit=args.sft_limit or None)
    elif args.mixture == "e":
        # Этап E: калибровка уверенности + D2-смесь (data/calib.py).
        stream = CB.CalibStream(args.seq_len, seed=args.seed,
                                limit=args.sft_limit or None)
    elif args.mixture == "f":
        # Этап F: пары предпочтений (ORPO) из ответов этапа E (data/prefs.py).
        from data import prefs as PF
        stream = PF.PrefStream(args.seq_len, seed=args.seed,
                               limit=args.sft_limit or None)
    elif args.mixture == "chat":
        # Этап G: живой чат (OASST2) и личность из системного промпта
        # (data/chat.py) поверх всего прошлого.
        from data import chat as CH
        stream = CH.ChatStream(args.seq_len, seed=args.seed,
                               limit=args.sft_limit or None)
    elif args.mixture == "g":
        # Этап G: личность поверх всего прошлого (data/persona.py).
        from data import persona as PE
        stream = PE.PersonaStream(args.seq_len, seed=args.seed,
                                  limit=args.sft_limit or None)
    else:
        stream = D.TokenStream(args.mixture, seq_len=args.seq_len, seed=args.seed)
    start_step = 0
    # Шаг, с которого начался отжиг. Лежит в снимке: сессия, продолжившая
    # отжиг после падения или по лимиту времени, должна спускать LR по той же
    # кривой, а не начинать спуск заново от своего стартового шага.
    anneal_from = None

    if args.resume:
        # missing_ok: первая сессия запускается с тем же --resume, что и все
        # последующие, и отсутствие репозитория для неё — норма, а не ошибка.
        local = C.pull(args.run, missing_ok=True)
        if local is None and args.init_from:
            print(f"у '{args.run}' снимков нет — стартуем со снимка "
                  f"'{args.init_from}'", flush=True)
            local = C.pull(args.init_from)
        if local is None and args.new_stage:
            raise SystemExit("--new-stage: нет снимка ни у прогона, ни у "
                             "--init-from — начинать этап не с чего")
        if local is not None:
            params, state, meta = C.load(local, params, state)
            start_step = int(state["step"])
            anneal_from = meta.get("anneal_from")
            fresh_stage = args.new_stage and meta.get("run") != args.run
            if fresh_stage:
                # Снимок чужого этапа: его спад LR уже пройден, и шаг, с
                # которого он начался, к новому этапу отношения не имеет.
                anneal_from = None
                if meta.get("corpus") and not args.fake_data:
                    moved = stream.carry_over(meta["corpus"])
                    print(f"корпус: новый этап, длина {args.seq_len}, смесь "
                          f"{args.mixture!r}; позиция перенесена для "
                          f"{', '.join(moved)}", flush=True)
            elif meta.get("corpus"):
                stream.restore(meta["corpus"])
                print(f"корпус: продолжаем с {meta['corpus']['tokens_emitted']:,} токенов")

    if args.new_stage and not args.anneal:
        raise SystemExit("--new-stage требует --anneal: длину этапа")
    if args.anneal:
        # total и decay_frac подбираются так, чтобы спад начинался ровно здесь
        # и упирался в пол ровно через `anneal` шагов:
        #     decay_start = total * (1 - decay_frac) = start_step
        if anneal_from is None:
            anneal_from = start_step
        total = anneal_from + args.anneal
        opt = dataclasses.replace(opt, decay_frac=args.anneal / total)
        if args.new_stage:
            # Новый этап: разогрев от первого шага этапа, затем спад до пола
            # к концу этапа. Шаг начала этапа лежит в снимке (anneal_from),
            # так что сессия, продолжившая этап, пойдёт по той же кривой.
            warm = min(args.warmup or 500, args.anneal // 2)
            opt = dataclasses.replace(
                opt, warmup_from=anneal_from, warmup_steps=warm,
                decay_frac=(args.anneal - warm) / total)
            print(f"новый этап: разогрев {warm:,} шагов до LR ×{args.lr_scale}, "
                  f"затем спад до пола к шагу {total:,}", flush=True)
        args.total_steps = total
        # Отжиг кончается на полу LR: считать дальше незачем.
        args.steps = min(args.steps, max(total - start_step, 0))
        if args.checkpoint and args.run == "pretrain":
            raise SystemExit(
                "Отжиг без --no-checkpoint запрещён. Отожжённая модель — "
                "другая ветка: её веса больше не годятся для продолжения "
                "плато, а снимок ляжет на Hub последним и следующая сессия "
                "поднимет именно его. Если отжиг задуман как настоящий "
                "(этап C), запускайте его под своим --run, а не под "
                "'pretrain'.")
        print(f"отжиг: {args.anneal:,} шагов от {anneal_from:,} до {total:,}, "
              f"LR идёт от пикового до пола"
              + (f"; продолжаем с {start_step:,}" if start_step > anneal_from else ""),
              flush=True)

    print(stream.describe(), flush=True)

    batches = D.Prefetcher(stream.batches(args.batch_size), depth=4)

    if args.skip_batches:
        skipper = iter(batches)
        for _ in range(args.skip_batches):
            next(skipper)
        print(f"пропущено {args.skip_batches} батчей — обходим то, на чём "
              f"сорвались", flush=True)
    tokenizer = W.load_tokenizer() if args.show_every else None

    if args.preflight:
        # Один шаг под микроскопом до начала прогона. Стоит секунды и отвечает
        # на вопрос «где именно nan», за который иначе платят целой сессией.
        first = next(iter(batches))
        P.run(params, state, first, cfg, opt, args.total_steps,
              apply_updates=O.apply_updates)

    if cfg.attn_impl == "splash" and not args.auto_sharding:
        # Ядро Pallas само обёрнуто в shard_map и внутрь второго не ляжет.
        print("сплавленное внимание не совместимо с shard_map лосса — "
              "идём блочным путём", flush=True)
        cfg = dataclasses.replace(cfg, attn_impl="blocked")
    step_fn = build_step(cfg, opt, args.total_steps, mesh,
                         fsdp=not args.auto_sharding)

    # Отложенный набор. Собирается один раз и держится в памяти хоста: те же
    # токены на каждом замере — единственный способ увидеть изменение меньше
    # разброса смеси. Не собрался — прогон всё равно идёт, просто вслепую.
    holdout = evaluate = None
    if args.eval_every and not args.fake_data:
        try:
            begun = time.time()
            if args.mixture == "d1":
                holdout = SFT.holdout(args.seq_len, args.batch_size,
                                      args.eval_batches,
                                      limit=args.sft_limit or None)
                size = holdout[0].size
            elif args.mixture in ("d2w", "e", "f", "g", "chat"):
                # Поиск без учителя (data/wiki_search.py): отложенные — с
                # валидации HotpotQA, MuSiQue, SQuAD. Смотрим, не теряется ли.
                from data import wiki_search as WS
                holdout = WS.holdout(args.seq_len, args.batch_size,
                                     args.eval_batches,
                                     limit=args.sft_limit or None)
                size = holdout[0].size
            elif args.mixture == "d2":
                holdout = SS.holdout(args.seq_len, args.batch_size,
                                     args.eval_batches,
                                     limit=args.sft_limit or None)
                size = holdout[0].size
            else:
                holdout = HO.take(args.mixture, args.seq_len, args.batch_size,
                                  args.eval_batches)
                size = holdout.size
            evaluate = HO.evaluator(cfg)
            print(f"отложенный набор: {args.eval_batches}×{args.batch_size}×"
                  f"{args.seq_len} = {size / 1e6:.1f}M токенов "
                  f"за {time.time() - begun:.0f}с", flush=True)
        except Exception as exc:                          # noqa: BLE001
            print(f"отложенный набор не собрался ({type(exc).__name__}: {exc}) — "
                  f"идём без него", flush=True)
            holdout = None

    snapshot_dir = config.ROOT / "checkpoints" / args.run
    deadline = time.time() + args.hours * 3600
    step = start_step
    history: list[float] = []
    ema: float | None = None
    marks: list[tuple[int, float]] = []
    shares_trace: list[float] = []          # вклад экспертов на каждом логе
    layer_shares: list[float] = []          # последний послойный разрез
    started = time.time()
    reason = "заданное число шагов"

    saved_at_step = -1
    pushed_at_step = -1
    push_failures = 0
    skips = jnp.zeros((), jnp.float32)

    def measure():
        """Замер на отложенном наборе. Печатает сдвиг от прошлого раза."""
        if holdout is None:
            return
        value, spent = evaluate(
            params, holdout,
            lambda x: jax.device_put(x, S.batch_sharding(mesh)))
        move = ""
        if marks:
            was_step, was = marks[-1]
            move = (f", {value - was:+.4f} за {step - was_step:,} шагов"
                    if step > was_step else "")
        marks.append((step, value))
        print(f"  отложенный лосс {value:.4f} (ppl {float(jnp.exp(value)):.1f}"
              f"{move}), {spent:.0f}с", flush=True)

    def snapshot(tag: str, to_hub: bool = True):
        """Снимок на диск и, если `to_hub`, на Hub.

        Разделено потому, что на диск снимок стоит десять секунд, а на Hub —
        свои 9.3 ГБ квоты, и квота эта не возвращается: удаление папок и
        схлопывание истории оставляют блобы висеть (замер 08.09.2026 в
        checkpoint.prune_remote). Сессия 07.09 залила семь снимков, съела 65 ГБ
        и встала на восьмом. С тех пор частота заливки задаётся отдельно —
        `--push-every`, — а на диске ротация как была: локальный снимок
        переживает падение прогона и вместе с output ноутбука переживает
        контейнер.
        """
        nonlocal saved_at_step
        if not args.checkpoint:
            return
        if step == saved_at_step:
            # Снимок этого шага уже на диске. Заливка всё же может быть
            # должна: последний снимок сессии уходит на Hub всегда, а
            # периодический на том же шаге мог её пропустить по --push-every.
            if to_hub:
                send(snapshot_dir / f"step-{step}")
            return
        if not all(bool(jnp.all(jnp.isfinite(w))) for w in jax.tree.leaves(params)):
            # Снимок с nan хуже отсутствия снимка: он ложится на Hub последним
            # и следующая сессия поднимет с него мёртвую модель, потратив
            # квоту на прогон, который уже не мог сойтись.
            print("снимок не пишем: в весах nan, последний рабочий остаётся "
                  "на Hub", flush=True)
            return
        directory = snapshot_dir / f"step-{step}"
        # Последний снимок этапа SFT — только веса: продолжать его некому,
        # следующий этап начнёт с разогрева, а место на Hub кончается.
        final_stage_snapshot = bool(args.new_stage and step >= args.total_steps)
        # Ротация до записи, а не после: иначе на диске на миг оказывается на
        # один снимок больше, чем он способен вместить, и это тот самый миг.
        C.rotate(snapshot_dir, keep=max(args.keep - 1, 0))

        def write():
            C.save(directory, params, state, {
                "run": args.run, "preset": args.preset, "mixture": args.mixture,
                "seed": args.seed,
                "seq_len": args.seq_len, "batch_size": args.batch_size,
                "config": dataclasses.asdict(cfg), "corpus": stream.state(),
                "reason": tag, "anneal_from": anneal_from if args.anneal else None,
            }, weights_only=final_stage_snapshot)

        try:
            write()
        except OSError as exc:                            # noqa: BLE001
            # Место кончилось. Сносим всё локальное и пробуем ещё раз: этот
            # снимок нужнее любого из предыдущих, те уже на Hub.
            print(f"снимок не пишется ({exc}); чистим локальные и пробуем снова",
                  flush=True)
            C.rotate(snapshot_dir, keep=0)
            try:
                write()
            except OSError as exc2:                       # noqa: BLE001
                print(f"снимок потерян: {exc2}", flush=True)
                return
        saved_at_step = step
        if to_hub:
            send(directory)
        C.rotate(snapshot_dir, keep=args.keep)

    def send(directory: pathlib.Path):
        """Заливка снимка на Hub. Отдельно от записи — см. snapshot."""
        nonlocal push_failures, pushed_at_step
        if args.no_push or pushed_at_step == step or push_failures >= 2:
            return
        if not directory.exists():
            return
        pushed_at_step = step
        try:
            C.push(directory, args.run, keep=args.keep_remote)
            push_failures = 0
        except C.StorageFull as exc:
            # Место на Hub, в отличие от сети, само не появится. Локальный
            # снимок переживёт только контейнер, а продолжение читает
            # только Hub — значит всё, что считается дальше, считается в
            # никуда. Один раз прощаем (вдруг ротация освободит место со
            # следующей попытки), на втором отказе выходим.
            push_failures += 1
            print(f"на Hub НЕ ХВАТИЛО МЕСТА ({exc}).\n"
                  f"  Снимок лежит в {directory} и переживёт контейнер как\n"
                  "  output ноутбука. Квоту освобождает не ротация, а снос\n"
                  "  репозитория — ячейка 6б со входом-output его и делает.",
                  flush=True)
        except Exception as exc:                      # noqa: BLE001
            # Сеть моргнула — локальный снимок уже есть, следующий уедет.
            print(f"на Hub не ушло ({exc}); снимок лежит в {directory}",
                  flush=True)

    measure()

    try:
        for batch in batches:
            batch = {k: jax.device_put(jnp.asarray(v), S.batch_sharding(mesh))
                     for k, v in batch.items()}
            params, state, metrics = step_fn(params, state, batch)
            step += 1
            # Копится на устройстве, читается раз в лог: пропущенный шаг — это
            # шаг, на котором веса не двинулись, и знать их число надо точно,
            # а не по одному замеру из пятидесяти.
            skips = skips + metrics["skipped"]

            if step % args.log_every == 0:
                loss = float(metrics["loss_lm"])
                history.append(loss)
                # Скользящее среднее: одиночный замер шумит на ±0.25 от того,
                # какие источники попали в батч, и читать по нему тренд нельзя.
                ema = loss if ema is None else 0.95 * ema + 0.05 * loss
                elapsed = time.time() - started
                per_step = elapsed / max(step - start_step, 1)
                seen = stream.tokens_emitted / 1e9
                n_skips = float(skips)
                print(
                    f"шаг {step:>7,} | loss {loss:6.3f} (сред {ema:5.3f}) | "
                    f"ppl {float(metrics['perplexity']):8.1f} | "
                    f"lr {float(metrics['lr_scale']):.3f} | "
                    f"|g| {float(metrics['grad_norm']):6.2f} | "
                    f"u {float(metrics.get('u_mean', 0)):.3f} | "
                    f"ворота {float(metrics.get('gate2_open', 0)):.2f} | "
                    f"сброс {float(metrics.get('moe_dropped', 0)):.2f} | "
                    f"эксперты {float(metrics.get('expert_share', 0)):.2f} | "
                    + (f"калибровка {float(metrics['loss_calib_0']):.3f}/"
                       f"{float(metrics['loss_calib_1']):.3f} | "
                       if "loss_calib" in metrics else "")
                    + (f"ORPO {float(metrics['loss_or']):.3f} (пар верно "
                       f"{float(metrics['pref_acc']):.0%}) | "
                       if "loss_or" in metrics else "")
                    + f"{per_step * 1000:.0f} мс/шаг | {seen:.2f}B токенов"
                    + (f" | ПРОПУЩЕНО ШАГОВ {n_skips:.0f}" if n_skips else ""),
                    flush=True,
                )
                if "expert_share" in metrics:
                    shares_trace.append(float(metrics["expert_share"]))

                if not jnp.isfinite(metrics["loss"]):
                    # Лосс уже nan — значит либо веса отравлены, либо их
                    # отравил именно этот батч. Разница решает, чинить
                    # оптимизатор или данные, а второго шанса посмотреть не
                    # будет: снимок таких весов мы не пишем.
                    bad = [jax.tree_util.keystr(p)
                           for p, w in jax.tree_util.tree_flatten_with_path(params)[0]
                           if not bool(jnp.all(jnp.isfinite(w)))]
                    if bad:
                        print(f"  веса отравлены: {len(bad)} тензоров, первые "
                              f"{bad[:5]}", flush=True)
                    else:
                        print("  веса конечны — nan родился в этом проходе, "
                              "не в весах", flush=True)
                    reason = "лосс перестал быть конечным"
                    break

            if args.eval_every and step % args.eval_every == 0:
                measure()

            if args.show_every and step % args.show_every == 0:
                # Послойный разрез вклада экспертов. Среднее в строке шага
                # прячет главное: замер 02.09.2026 дал 0.3% в слое 0 и 47% в
                # слое 5 при среднем 22%. Лечить надо слои, а не среднее.
                shares = metrics.get("expert_share_layers")
                if shares is not None:
                    layer_shares = [float(v) for v in shares]
                    row = " ".join(f"{float(v):.2f}" for v in shares)
                    print(f"  вклад экспертов по слоям: {row}", flush=True)
            if args.show_every and step % args.show_every == 0 and tokenizer:
                W.show(params, cfg, tokenizer, step, max_tokens=args.show_tokens)

            if step % args.checkpoint_every == 0:
                snapshot("периодический",
                         to_hub=to_hub_now(step, args.push_every,
                                           args.checkpoint_every))
                if push_failures >= 2:
                    reason = "снимок некуда девать: на Hub нет места"
                    break

            if step - start_step >= args.steps:
                break
            if time.time() > deadline:
                reason = "лимит времени сессии"
                break
        else:
            reason = "корпус кончился"
    except KeyboardInterrupt:
        reason = "прервано вручную"
    finally:
        batches.stop()
        # Сохранение в finally — единственное, что отделяет оборванную сессию
        # от потерянной.
        snapshot(reason)

    if marks and step > marks[-1][0]:
        measure()

    elapsed = time.time() - started
    print(f"\nостановка: {reason}")
    print(f"шагов за сессию: {step - start_step:,} за {elapsed / 3600:.2f} ч")
    print(f"токенов: {stream.tokens_emitted / 1e9:.2f}B, документов: {stream.docs:,}")

    # Первый и последний замер — это два случайных батча, и их разность
    # говорит о составе смеси больше, чем об обучении. Сравниваются средние по
    # краям, а вердикт выносит отложенный набор.
    if history:
        edge = max(1, min(20, len(history) // 4))
        head = sum(history[:edge]) / edge
        tail = sum(history[-edge:]) / edge
        print(f"loss на обучении: {head:.3f} → {tail:.3f} "
              f"(средние по {edge} замерам с краёв)")
    fit = holdout_trend(marks)
    if marks:
        (first_step, first), (last_step, last) = marks[0], marks[-1]
        print(f"отложенный лосс: {first:.4f} (шаг {first_step:,}) → "
              f"{last:.4f} (шаг {last_step:,}), {last - first:+.4f} за сессию")
        if fit is None:
            print(f"  замеров {len(marks)} — на тренд не хватает, "
                  f"разность двух точек вердиктом не считается")
        else:
            slope, t, noise = fit
            # Порог по значимости, а не по величине: при шуме 0.025 сдвиг на
            # 0.04 между двумя точками — это ноль, а наклон −0.003 на тысяче
            # замеров — уже спуск.
            verdict = ("учится" if t <= -2 else
                       "РАСТЁТ" if t >= 2 else "СТОИТ НА МЕСТЕ")
            print(f"  наклон {slope:+.5f} на 1000 шагов, t {t:+.2f}, "
                  f"шум {noise:.4f} по {len(marks)} замерам — {verdict}")

            # Ровный лосс на плато расписания — это норма, а не диагноз, и
            # три сессии подряд он читался как диагноз. WSD держит LR на
            # плато до 88% длины, и отложенный лосс в этом режиме стоит по
            # построению: модель копит, а не улучшает предсказание. Замер
            # 22.09.2026 со снимка 84,425 — плато 55k…84k давало наклон
            # +0.00002 при t=+0.04, а отжиг на 2000 шагов с той же точки
            # сразу дал −0.0929 при t=−19.9 и лосс 2.0854 → 1.9170.
            if abs(t) < 2 and not args.anneal and float(
                    O.lr_schedule(jnp.asarray(step), args.total_steps, opt)) > 0.99:
                print("  это плато расписания, а не остановка: LR всё ещё "
                      "пиковый, отжиг не начинался.\n"
                      "  Прогресс на плато отложенным лоссом не меряется — "
                      "мерить коротким отжигом:\n"
                      "  --resume --anneal 2000 --no-checkpoint "
                      "(эталон для сравнения: 84,425 → 1.9170)")

    if args.summary:
        # Хвост, а не последнее число: вклад экспертов шумит от батча к батчу
        # так же, как лосс, и одиночный замер из него ничего не значит.
        tail_n = max(1, len(shares_trace) // 5)
        done = step - start_step
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps({
            "run": args.run,
            "preset": args.preset,
            "overrides": args.overrides,
            "steps": done,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "tokens": stream.tokens_emitted,
            "seconds": elapsed,
            "ms_per_step": elapsed / max(done, 1) * 1000,
            "tokens_per_second": stream.tokens_emitted / max(elapsed, 1e-9),
            "holdout_first": marks[0][1] if marks else None,
            "holdout_last": marks[-1][1] if marks else None,
            "holdout_all": [[st, float(v)] for st, v in marks],
            "holdout_slope_1000": fit[0] if fit else None,
            "holdout_t": fit[1] if fit else None,
            "expert_share": (sum(shares_trace[-tail_n:]) / tail_n
                             if shares_trace else None),
            "expert_share_layers": layer_shares,
            "reason": reason,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"итог прогона: {args.summary}")


if __name__ == "__main__":
    main()
