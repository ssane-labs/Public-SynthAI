"""Чекпоинты: сохранение, загрузка, отправка на Hugging Face Hub.

Правило проекта: сессия, из которой нельзя продолжить, считается потерянной.
Поэтому в снимок кладётся всё, что нужно для точного продолжения, а не только
веса:

    веса                    что выучено
    состояние оптимизатора  momentum, без него первые шаги после перезапуска
                            идут вразнос
    позиция в корпусе       (шард, группа строк) по каждому источнику плюс
                            состояние жребия смеси — иначе после перезапуска
                            модель второй раз увидит те же данные, а пропорции
                            источников поедут
    шаг и конфигурация      чтобы расписание LR продолжилось, а не началось

Веса и буферы лежат в fp32 — в них модель и считается между шагами, bf16
появляется только внутри прямого прохода. Тензоры идут в safetensors, всё
остальное в JSON рядом.

Про «снимок живёт только на Hub»: это верно про диск контейнера, но не про
output ноутбука. Всё, что лежало в `/kaggle/working` к концу прогона, Kaggle
сохраняет как output версии, и оттуда снимок достаётся — см. `recover`.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402

TENSORS = "tensors.safetensors"
META = "meta.json"


def free_gb(path: pathlib.Path | None = None) -> float:
    """Сколько гигабайт свободно там, где лежат снимки."""
    path = pathlib.Path(path or config.ROOT)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1024**3


def size_gb(*trees) -> float:
    """Во сколько гигабайт обойдётся снимок этих деревьев.

    Считать надо до записи, а не после: под /kaggle/working ровно 20 ГБ, а
    полный снимок Synth-1 весит девять с лишним. Два таких — и диск кончился
    вместе с сессией.
    """
    total = 0
    for tree in trees:
        for leaf in jax.tree.leaves(tree):
            total += leaf.size * leaf.dtype.itemsize
    return total / 1024**3


def _flatten(tree, prefix: str) -> dict[str, np.ndarray]:
    """Дерево параметров в плоский словарь. Ключ — путь, как его печатает JAX."""
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        key = prefix + jax.tree_util.keystr(path)
        out[key] = np.asarray(jax.device_get(leaf))
    return out


def _unflatten(flat: dict[str, np.ndarray], reference, prefix: str):
    """Обратно в дерево — по образцу.

    Образец задаёт не только структуру, но и раскладку по чипам: каждый тензор
    едет сразу туда, где ему жить. Наивное `jnp.asarray` кладёт всё на нулевой
    чип, и модель на 1.2B параметров занимает там 4.9 ГБ — вместе с momentum и
    с тем, что уже было в памяти, это выходит за шестнадцать гигабайт HBM.
    Сессия 28.08.2026 умерла именно так, на аллокации в 54 МБ.
    """
    pairs = jax.tree_util.tree_flatten_with_path(reference)[0]
    paths = [prefix + jax.tree_util.keystr(p) for p, _ in pairs]
    missing = [p for p in paths if p not in flat]
    if missing:
        raise SystemExit(
            f"В чекпоинте нет {len(missing)} тензоров, например {missing[:3]}.\n"
            "Скорее всего, снимок снят с другой конфигурацией модели."
        )
    leaves = []
    for path, (_, ref) in zip(paths, pairs):
        sharding = getattr(ref, "sharding", None)
        leaves.append(jax.device_put(flat[path], sharding) if sharding is not None
                      else jnp.asarray(flat[path]))
    return jax.tree.unflatten(jax.tree.structure(reference), leaves)


def save(directory: pathlib.Path, params, opt_state, meta: dict,
         margin_gb: float = 0.5, weights_only: bool = False) -> pathlib.Path:
    """Снимок на диск. Не начинает писать, если места заведомо не хватит.

    Половинный снимок хуже отсутствующего: он выглядит как снимок, а
    разваливается на загрузке — в следующей сессии, когда чинить уже поздно.
    Поэтому место проверяется заранее, а недописанная папка удаляется.

    weights_only — без состояния оптимизатора, вдвое легче. Для последнего
    снимка этапа SFT: следующий этап всё равно начинает с разогрева, а место
    на Hub кончилось (STATUS.md, 28.10.2026).
    """
    from safetensors.numpy import save_file

    directory = pathlib.Path(directory)
    need = (size_gb(params) if weights_only
            else size_gb(params, opt_state["m"], opt_state["v"]))
    have = free_gb(directory.parent)
    if have < need + margin_gb:
        raise OSError(
            f"снимок весит {need:.2f} ГБ, свободно {have:.2f} ГБ — не пишем, "
            f"чтобы не оставить обрубок в {directory}"
        )

    directory.mkdir(parents=True, exist_ok=True)

    tensors = {}
    tensors.update(_flatten(params, "p"))
    if not weights_only:
        tensors.update(_flatten(opt_state["m"], "m"))
        tensors.update(_flatten(opt_state["v"], "v"))

    started = time.time()
    try:
        save_file(tensors, str(directory / TENSORS))
    except OSError:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    full_meta = dict(meta)
    full_meta["step"] = int(opt_state["step"])
    full_meta["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    full_meta["weights_only"] = weights_only
    (directory / META).write_text(
        json.dumps(full_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    size = (directory / TENSORS).stat().st_size / 1024**3
    print(f"чекпоинт: шаг {full_meta['step']:,}, {size:.2f} ГБ, "
          f"{time.time() - started:.1f}с → {directory} "
          f"(свободно {free_gb(directory.parent):.1f} ГБ)", flush=True)
    return directory


def load(directory: pathlib.Path, params_ref, opt_state_ref):
    """Загрузка по образцу структуры. Возвращает (params, opt_state, meta)."""
    from safetensors.numpy import load_file

    directory = pathlib.Path(directory)
    flat = load_file(str(directory / TENSORS))
    meta = json.loads((directory / META).read_text(encoding="utf-8"))

    params = _unflatten(flat, params_ref, "p")
    if any(k.startswith("m") for k in flat):
        m = _unflatten(flat, opt_state_ref["m"], "m")
        v = _unflatten(flat, opt_state_ref["v"], "v")
    else:
        # Снимок без оптимизатора (конец этапа SFT): состояние — с нуля.
        m, v = opt_state_ref["m"], opt_state_ref["v"]
        print("в снимке только веса — состояние оптимизатора с нуля", flush=True)
    opt_state = {"step": jnp.asarray(meta["step"], dtype=jnp.int32), "m": m, "v": v}
    print(f"продолжаем с шага {meta['step']:,} "
          f"(снят {meta.get('saved_at', '?')})", flush=True)
    return params, opt_state, meta


# --- Hugging Face Hub ------------------------------------------------------

class StorageFull(RuntimeError):
    """Hub отказал по месту, а не по сети.

    Разница решает, что делать дальше. Сеть моргнула — следующий чекпоинт через
    две тысячи шагов уйдёт, терять сессию не из-за чего. Место кончилось — само
    оно не появится, и все последующие снимки тоже никуда не уедут: сессия
    досчитает восемь часов и умрёт вместе с контейнером, потому что продолжение
    читает только Hub. Сессия 29.08.2026 потеряла так 19,477 шагов.
    """


def _is_storage_full(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(mark in text for mark in
               ("storage limit", "quota", "exceeded your storage", "over the limit"))


def repo_name(run: str) -> str:
    return f"{config.HF_NAMESPACE}/synth-1-{run}"


def remote_steps(run: str, api=None) -> list[int]:
    """Какие снимки лежат на Hub. Пусто, если репозитория ещё нет."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    api = api or HfApi(token=config.key("HF_TOKEN"))
    try:
        files = api.list_repo_files(repo_name(run), repo_type="model")
    except RepositoryNotFoundError:
        return []
    return sorted(int(f.split("/")[0].removeprefix("step-"))
                  for f in files
                  if f.startswith("step-") and f.endswith(META))


def steps_to_prune(steps: list[int], keep: int) -> list[int]:
    """Какие снимки снести, чтобы осталось keep свежих. Чистая функция ради теста."""
    if keep <= 0:
        return sorted(steps)
    return sorted(steps)[:-keep]


def prune_remote(run: str, keep: int = 1, api=None, squash: bool = True) -> list[int]:
    """Снести на Hub всё, кроме последних keep снимков.

    Снимок весит 9.3 ГБ, и приватная квота Hub кончается на седьмом. Локально
    ротация была с самого начала, на Hub её не было: там снимки копились вечно,
    «история живёт на Hub» — и история же место и съела.

    squash: удаление файла не освобождает место само по себе — блоб остаётся в
    истории репозитория. super_squash_history схлопывает историю в один коммит.
    Долго считалось, что после этого старое перестаёт считаться; замер 08.09.2026
    говорит иначе. В `synth-1-pretrain` после ротации и схлопывания в HEAD лежал
    ровно один снимок и ровно один коммит, а Hub насчитывал 65.08 ГБ — это
    семь на 9.3, то есть все семь заливок сессии 07.09 до единой. Сутки спустя
    число не сдвинулось.

    Значит, схлопывание квоту освобождает не сразу, а когда до блобов дойдёт
    сборщик мусора Hub, и на горизонте сессии этого ждать нельзя. Планировать
    надо так, будто каждая заливка занимает свои 9.3 ГБ навсегда: держать число
    заливок за сессию маленьким (`--push-every` в pretrain.py), а место
    освобождать `wipe_remote`, который сносит репозиторий целиком.
    """
    from huggingface_hub import HfApi

    api = api or HfApi(token=config.key("HF_TOKEN"))
    repo = repo_name(run)
    doomed = steps_to_prune(remote_steps(run, api), keep)
    for step in doomed:
        api.delete_folder(path_in_repo=f"step-{step}", repo_id=repo,
                          repo_type="model",
                          commit_message=f"ротация: снят step-{step}")
        print(f"  Hub: снесён step-{step}", flush=True)
    if doomed and squash:
        try:
            api.super_squash_history(repo_id=repo, repo_type="model")
        except Exception as exc:                          # noqa: BLE001
            # Не у всех аккаунтов есть право схлопывать историю. Место тогда
            # освободится позже, сборщиком мусора Hub, — сказать об этом надо,
            # но останавливаться не из-за чего.
            print(f"  Hub: историю схлопнуть не вышло ({type(exc).__name__}); "
                  f"место освободится не сразу", flush=True)
    return doomed


def wipe_remote(run: str, api=None, private: bool = True) -> None:
    """Снести репозиторий целиком и создать пустой заново.

    Единственное, что освобождает квоту сразу. Удаление папок и схлопывание
    истории оставляют блобы висеть до сборщика мусора Hub — замер 08.09.2026
    в `prune_remote`. Удаление репозитория уносит блобы вместе с ним.

    Вызывать можно только тогда, когда снимок лежит где-то ещё: в output
    ноутбука Kaggle или на диске. Отсюда и единственное место вызова —
    `recover(wipe=True)`, где источник заведомо не на Hub.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    api = api or HfApi(token=config.key("HF_TOKEN"))
    repo = repo_name(run)
    try:
        api.delete_repo(repo_id=repo, repo_type="model")
        print(f"  Hub: {repo} снесён целиком — квота освобождена", flush=True)
    except RepositoryNotFoundError:
        print(f"  Hub: {repo} и так нет", flush=True)
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)


def storage_report(namespace: str | None = None, api=None) -> list[tuple]:
    """Сколько Hub считает занятым — по репозиториям, сверху вниз.

    По списку файлов этого не видно: там HEAD, а квоту ест история. Разрыв
    между двумя числами и есть то, что не даёт залить следующий снимок, — а
    заметить его иначе можно только по отказу посреди восьмичасовой сессии.
    """
    from huggingface_hub import HfApi

    api = api or HfApi(token=config.key("HF_TOKEN"))
    namespace = namespace or config.HF_NAMESPACE
    rows = []
    for kind, lister in (("model", api.list_models), ("dataset", api.list_datasets),
                         ("space", api.list_spaces)):
        for item in lister(author=namespace):
            info = api.repo_info(item.id, repo_type=kind,
                                 expand=["usedStorage", "private"])
            rows.append(((info.used_storage or 0) / 1024**3,
                         kind, bool(info.private), item.id))
    return sorted(rows, reverse=True)


def push(directory: pathlib.Path, run: str, private: bool = True,
         keep: int = 2) -> str:
    """Отправить снимок на Hub. Репозиторий создаётся, если его ещё нет.

    keep — сколько снимков остаётся на Hub вместе с этим. Чистка идёт до
    заливки, как и локальная ротация: место нужно освободить раньше, чем
    занять. Держать на время заливки хотя бы один прежний снимок обязательно —
    иначе сорвавшаяся отправка оставит репозиторий пустым.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=config.key("HF_TOKEN"))
    repo = repo_name(run)
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)

    directory = pathlib.Path(directory)
    step = json.loads((directory / META).read_text(encoding="utf-8"))["step"]

    try:
        prune_remote(run, keep=max(keep - 1, 1), api=api)
    except Exception as exc:                              # noqa: BLE001
        print(f"  Hub: ротация не прошла ({exc})", flush=True)

    started = time.time()
    try:
        api.upload_folder(
            folder_path=str(directory),
            repo_id=repo,
            repo_type="model",
            # Одна папка на шаг: так старые снимки остаются доступными, а откат
            # к предыдущему — это смена пути, а не поиск по истории коммитов.
            path_in_repo=f"step-{step}",
            commit_message=f"чекпоинт {run}, шаг {step}",
        )
    except Exception as exc:                              # noqa: BLE001
        raise (StorageFull(exc) if _is_storage_full(exc) else exc) from exc
    print(f"на Hub: {repo} за {time.time() - started:.0f}с", flush=True)
    return repo


def bloat(run: str, api=None) -> tuple[float, float]:
    """(сколько Hub считает занятым, сколько лежит в HEAD) — в гигабайтах.

    Разрыв между числами и есть мусор в истории: после ротации со схлопыванием
    в HEAD остаётся один снимок, а считается всё, что когда-либо заливалось.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    api = api or HfApi(token=config.key("HF_TOKEN"))
    try:
        info = api.repo_info(repo_name(run), repo_type="model",
                             files_metadata=True, expand=None)
    except RepositoryNotFoundError:
        return 0.0, 0.0
    head = sum((f.size or 0) for f in info.siblings) / 1024**3
    used = api.repo_info(repo_name(run), repo_type="model",
                         expand=["usedStorage"]).used_storage or 0
    return used / 1024**3, head


def needs_compacting(used_gb: float, head_gb: float, slack: float = 1.5) -> bool:
    """Стоит ли перекладывать репозиторий ради квоты. Чистая функция ради теста.

    Считается занятым заметно больше, чем лежит, — значит в истории висят
    блобы, и следующая заливка упрётся в квоту раньше, чем сборщик мусора Hub
    до них доберётся. Счётчик отстаёт на часы и в обе стороны, поэтому порог с
    запасом: ошибка вверх стоит десяти минут сессии, ошибка вниз — всей сессии.
    """
    return head_gb > 0 and used_gb > slack * head_gb


def compact(run: str, into: pathlib.Path | None = None, api=None,
            slack: float = 1.5) -> bool:
    """Переложить репозиторий: скачать свежий снимок, снести репо, залить назад.

    Нужно, когда снимок восстанавливать неоткуда (вход-output не подключён), а
    квота уже съедена историей. Ротация тут не помогает — блобы висят до
    сборщика мусора Hub (замер 08.09.2026 в prune_remote), а снос репозитория
    без второго экземпляра снимка делать нельзя.

    Порядок именно такой: сначала скачали и проверили, потом снесли. Между
    сносом и концом заливки единственный экземпляр лежит на диске контейнера —
    но он же попадёт в output версии, то есть переживёт и контейнер тоже.

    Возвращает True, если перекладывали.
    """
    used, head = bloat(run, api)
    if not needs_compacting(used, head, slack):
        print(f"перекладывать не нужно: Hub считает {used:.1f} ГБ при "
              f"{head:.1f} ГБ в снимках", flush=True)
        return False

    print(f"Hub считает {used:.1f} ГБ, а в снимках {head:.1f} ГБ — "
          f"перекладываем репозиторий", flush=True)
    directory = pull(run, into=into, missing_ok=True)
    if directory is None or not (directory / TENSORS).exists():
        print("  скачать нечего — оставляем как есть", flush=True)
        return False
    print(f"  скачан {directory.name}, "
          f"{(directory / TENSORS).stat().st_size / 1024**3:.2f} ГБ", flush=True)
    wipe_remote(run, api=api)
    push(directory, run, keep=1)
    return True


def find_snapshot(root: pathlib.Path) -> pathlib.Path | None:
    """Самый свежий целый снимок под `root`. Нужен, чтобы поднять то, что
    осталось от сорвавшейся сессии.

    Output ноутбука Kaggle переживает контейнер: всё, что лежало в
    `/kaggle/working` на момент конца прогона, сохраняется как output версии.
    Сессия 29.08.2026 оставила там `step-33477` — 19,477 шагов, которые считали
    потерянными, потому что на Hub они не уехали. Этот путь их возвращает.

    Целый — значит с обоими файлами: половинный снимок выглядит как снимок и
    разваливается на загрузке, в следующей сессии, когда чинить уже поздно.
    """
    root = pathlib.Path(root)
    if not root.exists():
        return None
    best: tuple[int, pathlib.Path] | None = None
    for meta in root.rglob(f"step-*/{META}"):
        directory = meta.parent
        if not (directory / TENSORS).exists():
            continue
        try:
            step = int(directory.name.removeprefix("step-"))
        except ValueError:
            continue
        if best is None or step > best[0]:
            best = (step, directory)
    return best[1] if best else None


def recover(run: str, root: pathlib.Path, api=None,
            free_first: bool = False, wipe: bool = False) -> pathlib.Path | None:
    """Найти снимок под `root` и отправить его на Hub, если он новее тамошнего.

    Идемпотентна: если на Hub уже лежит шаг не меньше, ничего не делает. Это
    важно, потому что ячейка ноутбука выполняется при каждом запуске, а
    заливать девять гигабайт второй раз незачем.
    """
    directory = find_snapshot(root)
    if directory is None:
        print(f"восстанавливать нечего: под {root} снимков нет", flush=True)
        return None

    step = int(directory.name.removeprefix("step-"))
    remote = remote_steps(run, api)
    if remote and max(remote) >= step:
        print(f"на Hub уже есть шаг {max(remote):,} — {directory.name} "
              f"восстанавливать не нужно", flush=True)
        return None

    print(f"восстанавливаем {directory.name} из {directory.parent} "
          f"(на Hub сейчас {max(remote) if remote else 'пусто'})", flush=True)

    if wipe:
        # Сносим репозиторий целиком. Ротация с схлопыванием так не умеет:
        # блобы висят до сборщика мусора Hub, и сессия 07.09 упёрлась в квоту,
        # имея в HEAD один снимок и один коммит (замер в prune_remote).
        # Здесь это безопасно и только здесь: источник лежит в output ноутбука,
        # заливку всегда можно повторить с того же входа.
        print("  сносим репозиторий целиком: источник в output, "
              "потерять его нельзя", flush=True)
        wipe_remote(run, api=api)
    elif free_first and remote:
        # Обычно на время заливки один прежний снимок держится обязательно:
        # сорвавшаяся отправка иначе оставит репозиторий пустым. Здесь можно
        # иначе, и это важно при нехватке места: источник лежит в output
        # ноутбука и никуда не денется, поэтому неудачную заливку всегда можно
        # повторить с того же входа. Пик занятого места падает вдвое — на
        # тесной квоте ровно это и решает, пройдёт отправка или нет.
        #
        # Освобождает место оно только на бумаге — см. prune_remote. Оставлено
        # ради случая, когда сносить репозиторий не хочется.
        print("  освобождаем место до заливки: источник в output, "
              "потерять его нельзя", flush=True)
        prune_remote(run, keep=0, api=api)

    try:
        push(directory, run)
    except StorageFull as exc:
        raise SystemExit(
            chr(10).join([
                "на Hub НЕ ХВАТИЛО МЕСТА при восстановлении:",
                f"  {exc}",
                "",
                "Снимок остался в output и никуда не делся — заливку можно",
                "повторить, когда место будет. Считать дальше нельзя: снимки",
                "этой сессии уедут в ту же переполненную квоту.",
            ])
        ) from exc
    return directory


def pull(run: str, step: int | None = None,
         into: pathlib.Path | None = None,
         missing_ok: bool = False) -> pathlib.Path | None:
    """Скачать снимок. Без step берётся самый свежий.

    missing_ok — вернуть None вместо падения, если репозитория ещё нет или он
    пуст. Это обычное состояние первой сессии: `--resume` включён всегда, чтобы
    вторая и последующие продолжали сами, и первая не должна на этом спотыкаться.
    """
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    token = config.key("HF_TOKEN")
    repo = repo_name(run)
    api = HfApi(token=token)

    if step is None:
        try:
            files = api.list_repo_files(repo, repo_type="model")
        except RepositoryNotFoundError:
            if missing_ok:
                print(f"на Hub нет {repo} — начинаем с нуля")
                return None
            raise SystemExit(f"Нет репозитория {repo}")
        steps = sorted(
            int(f.split("/")[0].removeprefix("step-"))
            for f in files
            if f.startswith("step-") and f.endswith(META)
        )
        if not steps:
            if missing_ok:
                print(f"в {repo} пока нет снимков — начинаем с нуля")
                return None
            raise SystemExit(f"В {repo} нет ни одного чекпоинта")
        step = steps[-1]

    into = pathlib.Path(into or (config.ROOT / "checkpoints" / run))
    path = snapshot_download(repo, repo_type="model", token=token,
                             allow_patterns=[f"step-{step}/*"],
                             local_dir=str(into))
    return pathlib.Path(path) / f"step-{step}"


def rotate(directory: pathlib.Path, keep: int = 1) -> None:
    """Локально держим только последние снимки: на Kaggle диск не резиновый.

    keep=0 — снести все: так освобождают место перед записью нового снимка.
    Настоящая история живёт на Hub, локальная папка — это буфер на одну запись.
    """
    directory = pathlib.Path(directory)
    if not directory.exists():
        return
    snaps = sorted((d for d in directory.iterdir() if d.name.startswith("step-")),
                   key=lambda d: int(d.name.removeprefix("step-")))
    for old in (snaps[:-keep] if keep else snaps):
        shutil.rmtree(old, ignore_errors=True)
        print(f"удалён локальный снимок {old.name}", flush=True)


if __name__ == "__main__":
    # Разбор завалов на Hub руками — с ноутбука, не дожидаясь сессии.
    #
    #     py -3 synth1/checkpoint.py list  pretrain
    #     py -3 synth1/checkpoint.py usage
    #     py -3 synth1/checkpoint.py compact pretrain
    #     py -3 synth1/checkpoint.py prune pretrain 1
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    run = sys.argv[2] if len(sys.argv) > 2 else "pretrain"

    if cmd == "list":
        steps = remote_steps(run)
        print(f"{repo_name(run)}: снимков {len(steps)}")
        for s in steps:
            print(f"  step-{s:,}")
    elif cmd == "usage":
        # Что Hub считает занятым. Отдельная команда, потому что по списку
        # снимков этого не видно: там HEAD, а квоту ест история.
        rows = storage_report()
        total = private = 0.0
        for gb, kind, is_private, name in rows:
            total += gb
            private += gb if is_private else 0.0
            mark = "приватный" if is_private else "открытый "
            print(f"  {gb:7.2f} ГБ  {mark}  {kind:7s}  {name}")
        print(f"  {private:7.2f} ГБ  приватного (в квоту идёт оно)")
        print(f"  {total:7.2f} ГБ  всего")
    elif cmd == "compact":
        # Переложить репозиторий ради квоты: скачать свежий снимок, снести
        # репо, залить назад. Нужно, когда вход-output не подключён и
        # восстанавливать нечем.
        compact(run)
    elif cmd == "prune":
        keep = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        gone = prune_remote(run, keep=keep)
        print(f"снесено {len(gone)}, осталось {len(remote_steps(run))}")
    elif cmd == "recover":
        # --wipe: снести репозиторий целиком до заливки. Единственное, что
        # освобождает квоту сразу (--free-first только схлопывает историю, а
        # блобы после этого висят сутками — замер в prune_remote). Безопасно
        # только когда источник лежит не на Hub — как раз случай output.
        # Поднять снимок, оставшийся от сорвавшейся сессии: в output ноутбука
        # Kaggle (/kaggle/input, если output подключён входом) или просто на
        # диске. Безопасно запускать всегда — на Hub новее не перезапишет.
        rest = sys.argv[3:]
        free_first = "--free-first" in rest
        wipe = "--wipe" in rest
        rest = [a for a in rest if not a.startswith("--")]
        root = pathlib.Path(rest[0] if rest else "/kaggle/input")
        recover(run, root, free_first=free_first, wipe=wipe)
    else:
        raise SystemExit("Команды: list | usage | compact | prune | recover")
