"""Конвейер ретроспективной бинарной классификации оттока ``churned``.

Лабораторная работа 6.1: рефакторинг ноутбука в CLI-конвейер.

Важно про постановку: здесь оценивается учебная классификация уже
известной метки ``churned`` на случайном стратифицированном разбиении
60/20/20. Это НЕ прогноз будущего оттока на следующем временном окне;
временная утечка исходной постановки не маскируется высоким ROC-AUC.

Публичный контракт:

* ``load_data(config) -> DataFrame`` — признаки + метка, только whitelist;
* ``preprocess(frame, config) -> dict`` — сплиты train/val/test 60/20/20;
* ``train_model(X, y, config) -> sklearn.pipeline.Pipeline`` — полный пайплайн;
* ``evaluate(model, splits, config) -> dict`` — метрики val/test;
* ``run_pipeline(config_path) -> dict`` — полный запуск и артефакты.

CLI: ``python pipeline.py --config config.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import tempfile
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401  # pandas читает parquet через pyarrow; версия идёт в manifest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_NAME = "config.json"
SPLIT_NAMES = ("train", "val", "test")
_REQUIRED_SECTIONS = ("data", "preprocessing", "model", "evaluation", "artifacts", "logging")
_CONFIG_SECTION_KEYS = {
    "data": (
        "features_path",
        "churn_path",
        "key_column",
        "target_column",
        "excluded_columns",
        "feature_columns",
    ),
    "preprocessing": ("test_size", "val_size", "random_state", "fill_value"),
    "model": (
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "class_weight",
        "random_state",
        "n_jobs",
    ),
    "evaluation": ("threshold",),
    "artifacts": (
        "dir",
        "model_name",
        "features_name",
        "metrics_name",
        "manifest_name",
        "splits_name",
    ),
    "logging": ("level", "format", "log_name"),
}
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------


def _require_mapping(value, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Секция конфигурации '{name}' должна быть JSON-объектом")


def _require_exact_keys(section: dict, name: str) -> None:
    required = _CONFIG_SECTION_KEYS[name]
    missing = [key for key in required if key not in section]
    if missing:
        raise ValueError(f"В секции '{name}' нет обязательных ключей: {missing}")
    unknown = sorted(set(section) - set(required))
    if unknown:
        raise ValueError(f"Неизвестные ключи в секции '{name}': {unknown}")


def _require_str(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Параметр '{name}' должен быть непустой строкой")
    return value


def _require_basename(value, name: str) -> str:
    """Имя выходного файла должно быть простым basename без каталогов."""
    text = _require_str(value, name)
    if text in (".", "..") or "/" in text or "\\" in text or Path(text).name != text:
        raise ValueError(f"'{name}' должен быть именем файла без каталогов: {text!r}")
    return text


def _require_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Параметр '{name}' должен быть числом")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Параметр '{name}' должен быть конечным числом")
    return number


def _require_int(value, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Параметр '{name}' должен быть целым числом")
    if minimum is not None and value < minimum:
        raise ValueError(f"Параметр '{name}' должен быть >= {minimum}")
    return value


def _feature_columns(config: dict) -> list[str]:
    """Получить упорядоченный список признаков из конфигурации и проверить его."""
    data = config.get("data") if isinstance(config, dict) else None
    if not isinstance(data, dict):
        raise ValueError("В конфигурации отсутствует секция 'data'")
    feature_columns = data.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns or not all(
        isinstance(column, str) and column.strip() for column in feature_columns
    ):
        raise ValueError("'data.feature_columns' должен быть непустым списком непустых строк")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("'data.feature_columns' содержит дубликаты")
    if data.get("key_column") in feature_columns:
        raise ValueError("Ключ 'data.key_column' не может быть признаком модели")
    if data.get("target_column") in feature_columns:
        raise ValueError("Целевая переменная 'data.target_column' не может быть признаком модели")
    excluded = data.get("excluded_columns")
    if not isinstance(excluded, list) or not all(
        isinstance(column, str) and column.strip() for column in excluded
    ):
        raise ValueError("'data.excluded_columns' должен быть списком непустых строк")
    overlap = sorted(set(excluded) & set(feature_columns))
    if overlap:
        raise ValueError(f"'data.excluded_columns' пересекается с признаками: {overlap}")
    return list(feature_columns)


def _validate_config(config: dict) -> None:
    """Проверить структуру, типы и границы всех параметров конфигурации."""
    _require_mapping(config, "config")
    missing = [name for name in _REQUIRED_SECTIONS if name not in config]
    if missing:
        raise ValueError(f"В конфигурации нет обязательных секций: {missing}")
    unknown = sorted(
        name for name in config if not str(name).startswith("_") and name not in _REQUIRED_SECTIONS
    )
    if unknown:
        raise ValueError(f"Неизвестные секции конфигурации: {unknown}")
    for name in _REQUIRED_SECTIONS:
        _require_mapping(config[name], name)
        _require_exact_keys(config[name], name)

    data = config["data"]
    _require_str(data["features_path"], "data.features_path")
    _require_str(data["churn_path"], "data.churn_path")
    key_column = _require_str(data["key_column"], "data.key_column")
    target_column = _require_str(data["target_column"], "data.target_column")
    _feature_columns(config)
    if target_column == key_column:
        raise ValueError("Ключ и целевая переменная должны различаться")
    excluded = data["excluded_columns"]
    if key_column in excluded:
        raise ValueError("Ключ нельзя исключать из признаков: он нужен для splits.csv")

    preprocessing = config["preprocessing"]
    test_size = _require_number(preprocessing["test_size"], "preprocessing.test_size")
    val_size = _require_number(preprocessing["val_size"], "preprocessing.val_size")
    if not 0.0 < test_size < 1.0:
        raise ValueError("'preprocessing.test_size' должен быть в интервале (0, 1)")
    if not 0.0 < val_size < 1.0:
        raise ValueError("'preprocessing.val_size' должен быть в интервале (0, 1)")
    _require_int(preprocessing["random_state"], "preprocessing.random_state", minimum=0)
    _require_number(preprocessing["fill_value"], "preprocessing.fill_value")

    model = config["model"]
    _require_int(model["n_estimators"], "model.n_estimators", minimum=1)
    if model["max_depth"] is not None:
        _require_int(model["max_depth"], "model.max_depth", minimum=1)
    _require_int(model["min_samples_split"], "model.min_samples_split", minimum=2)
    _require_int(model["min_samples_leaf"], "model.min_samples_leaf", minimum=1)
    class_weight = model["class_weight"]
    if class_weight is not None and class_weight not in ("balanced", "balanced_subsample"):
        raise ValueError(
            "'model.class_weight' должен быть null, 'balanced' или 'balanced_subsample'"
        )
    _require_int(model["random_state"], "model.random_state", minimum=0)
    n_jobs = _require_int(model["n_jobs"], "model.n_jobs")
    if n_jobs == 0:
        raise ValueError("'model.n_jobs' не может быть 0")

    threshold = _require_number(config["evaluation"]["threshold"], "evaluation.threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("'evaluation.threshold' должен быть в интервале [0, 1]")

    artifacts = config["artifacts"]
    _require_str(artifacts["dir"], "artifacts.dir")
    names = [
        _require_basename(artifacts[key], f"artifacts.{key}")
        for key in ("model_name", "features_name", "metrics_name", "manifest_name", "splits_name")
    ]
    if len(set(names)) != len(names):
        raise ValueError(f"Имена артефактов должны быть уникальны: {names}")

    log = config["logging"]
    if log["level"] not in _LOG_LEVELS:
        raise ValueError(f"'logging.level' должен быть одним из {list(_LOG_LEVELS)}")
    _require_str(log["format"], "logging.format")
    _require_basename(log["log_name"], "logging.log_name")


def load_config(config_path) -> dict:
    """Прочитать и провалидировать JSON-конфиг.

    Пути внутри конфига трактуются относительно каталога самого конфига;
    для этого в возвращаемый словарь добавляются служебные ключи ``_base_dir``
    и ``_config_path``.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Некорректный JSON в конфигурации {path}: {error}") from error
    _validate_config(config)
    resolved = Path(path).resolve()
    config["_config_path"] = str(resolved)
    config["_base_dir"] = str(resolved.parent)
    return config


def _resolve_path(config: dict, value: str) -> Path:
    """Разрешить путь относительно каталога конфига (а не текущего каталога)."""
    path = Path(value)
    if path.is_absolute():
        return path
    base_dir = config.get("_base_dir")
    return Path(base_dir) / path if base_dir else Path.cwd() / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Безопасность записи
# ---------------------------------------------------------------------------


def _validate_output_paths(config: dict) -> dict:
    """Запретить артефактам затирать входные данные, конфиг или код конвейера.

    Конфиг может быть кастомным (Docker, CI), поэтому имена проверяются после
    разрешения путей: никакой выходной файл не должен совпасть ни с другим
    выходным, ни с features/churn/config/pipeline.py.
    """
    artifacts_config = config["artifacts"]
    artifacts_dir = _resolve_path(config, artifacts_config["dir"])
    artifacts = {
        "model": artifacts_dir / artifacts_config["model_name"],
        "features": artifacts_dir / artifacts_config["features_name"],
        "metrics": artifacts_dir / artifacts_config["metrics_name"],
        "manifest": artifacts_dir / artifacts_config["manifest_name"],
        "splits": artifacts_dir / artifacts_config["splits_name"],
    }
    outputs = {**artifacts, "log": artifacts_dir / config["logging"]["log_name"]}
    resolved_outputs = [path.resolve() for path in outputs.values()]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError(f"Выходные файлы конвейера совпадают: {[str(p) for p in outputs.values()]}")
    for index, first in enumerate(resolved_outputs):
        for second in resolved_outputs[index + 1:]:
            if first.exists() and second.exists() and first.samefile(second):
                raise ValueError(f"Выходные файлы совпадают (hardlink): {first}, {second}")
    inputs = {
        "features": _resolve_path(config, config["data"]["features_path"]),
        "churn": _resolve_path(config, config["data"]["churn_path"]),
        "config": Path(config["_config_path"]),
        "pipeline": Path(__file__).resolve(),
    }
    for output_name, output_path in outputs.items():
        for input_name, input_path in inputs.items():
            if (output_path.resolve() == input_path.resolve()
                    or (output_path.exists() and input_path.exists()
                        and output_path.samefile(input_path))):
                raise ValueError(
                    f"Выходной файл '{output_name}' ({output_path}) совпадает "
                    f"с входным '{input_name}' — запись уничтожила бы данные"
                )
    return {"artifacts_dir": artifacts_dir, "artifacts": artifacts, "log": outputs["log"]}


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------


def _validate_keys(frame: pd.DataFrame, key_column: str, label: str) -> None:
    if key_column not in frame.columns:
        raise ValueError(f"В {label} нет колонки-ключа '{key_column}'")
    keys = frame[key_column]
    if keys.isna().any():
        raise ValueError(f"В {label} есть пропуски в ключе '{key_column}'")
    if keys.duplicated().any():
        duplicate = keys[keys.duplicated()].iloc[0]
        raise ValueError(f"В {label} дубликаты ключей, например: {duplicate!r}")
    if keys.astype("string").str.strip().eq("").fillna(False).any():
        raise ValueError(f"В {label} есть пустые значения ключа '{key_column}'")


def _load_inputs(config: dict) -> tuple[pd.DataFrame, dict]:
    """Прочитать признаки и эталон, проверить контракт и вернуть данные + метаданные."""
    feature_columns = _feature_columns(config)
    data = config["data"]
    key_column = data["key_column"]
    target_column = data["target_column"]
    features_path = _resolve_path(config, data["features_path"])
    churn_path = _resolve_path(config, data["churn_path"])
    for path, label in ((features_path, "признаков"), (churn_path, "эталона")):
        if not path.is_file():
            raise FileNotFoundError(f"Файл {label} не найден: {path}")

    features_sha256 = _sha256_file(features_path)
    churn_sha256 = _sha256_file(churn_path)
    features = pd.read_parquet(features_path)
    truth = pd.read_csv(churn_path)

    _validate_keys(features, key_column, "признаках")
    _validate_keys(truth, key_column, "эталоне")
    if target_column not in truth.columns:
        raise ValueError(f"В эталоне нет целевой колонки '{target_column}'")
    missing = [column for column in feature_columns if column not in features.columns]
    if missing:
        raise ValueError(f"В признаках нет колонок из 'data.feature_columns': {missing}")
    for column in feature_columns:
        series = features[column]
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"Признак '{column}' должен быть числовым, dtype={series.dtype}")

    raw_features_rows = int(len(features))
    raw_features_columns = int(features.shape[1])
    dropped = [
        column
        for column in features.columns
        if column != key_column and column not in feature_columns
    ]
    if dropped:
        logger.warning(
            "Из признаков исключены служебные/лишние колонки: %s", dropped
        )
    features = features[[key_column, *feature_columns]]

    target_values = truth[target_column]
    if target_values.isna().any():
        raise ValueError(f"В эталоне есть пропуски в целевой колонке '{target_column}'")
    if not set(target_values.unique().tolist()) <= {True, False, 0, 1}:
        raise ValueError(
            f"Целевая колонка '{target_column}' должна содержать только 0/1 (bool или int)"
        )
    truth = truth[[key_column, target_column]].copy()
    truth[target_column] = truth[target_column].astype(int)

    # left join сохраняет исходный порядок строк признаков; indicator проверяет
    # полное покрытие каждого ключа признаков в эталоне.
    merged = features.merge(
        truth, on=key_column, how="left", validate="one_to_one", indicator=True
    )
    left_only = int((merged["_merge"] == "left_only").sum())
    if left_only:
        raise ValueError(
            f"{left_only} ключей из признаков отсутствуют в эталоне: полное покрытие нарушено"
        )
    matched = int((merged["_merge"] == "both").sum())
    unmatched_truth = int(len(truth) - matched)
    merged = merged.drop(columns=["_merge"])
    if merged.empty:
        raise ValueError("После объединения признаков и эталона не осталось строк")
    if merged[target_column].isna().any():
        raise ValueError(f"После объединения есть пропуски в '{target_column}'")
    merged[target_column] = merged[target_column].astype(int)
    classes = sorted(merged[target_column].unique().tolist())
    if classes != [0, 1]:
        raise ValueError(f"Метка '{target_column}' должна содержать оба класса 0/1, найдено: {classes}")

    if unmatched_truth:
        logger.info(
            "В эталоне %d строк без признаков: они не участвуют в обучении", unmatched_truth
        )
    logger.info(
        "Загружено %d строк; признаков: %d; доля оттока %.4f",
        len(merged),
        len(feature_columns),
        float(merged[target_column].mean()),
    )
    metadata = {
        "features_path": data["features_path"],
        "features_sha256": features_sha256,
        "features_rows": raw_features_rows,
        "features_columns": raw_features_columns,
        "churn_path": data["churn_path"],
        "churn_sha256": churn_sha256,
        "churn_rows": int(len(truth)),
        "joined_rows": int(len(merged)),
        "unmatched_truth_rows": unmatched_truth,
    }
    return merged, metadata


def load_data(config: dict) -> pd.DataFrame:
    """Объединить признаки с меткой ``churned`` без служебных полей."""
    frame, _ = _load_inputs(config)
    return frame


# ---------------------------------------------------------------------------
# Предобработка
# ---------------------------------------------------------------------------


def preprocess(frame: pd.DataFrame, config: dict) -> dict:
    """Стратифицированно разбить данные 60/20/20 и вернуть сплиты.

    Признаки остаются «сырыми» (с NaN/inf): их обрабатывает сам sklearn
    Pipeline при обучении, поэтому заполнение и scaler не видят val/test.
    """
    feature_columns = _feature_columns(config)
    data = config["data"]
    key_column = data["key_column"]
    target_column = data["target_column"]
    for column in (key_column, target_column, *feature_columns):
        if column not in frame.columns:
            raise ValueError(f"В данных нет колонки '{column}'")

    y = frame[target_column]
    classes = sorted(y.unique().tolist())
    if classes != [0, 1]:
        raise ValueError(
            f"Метка '{target_column}' должна содержать оба класса 0/1, найдено: {classes}"
        )
    y = y.astype(int)
    X = frame[feature_columns]
    preparation = config["preprocessing"]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=preparation["test_size"],
        stratify=y,
        random_state=preparation["random_state"],
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=1.0 - preparation["val_size"],
        stratify=y_holdout,
        random_state=preparation["random_state"],
    )

    labels = pd.Series("train", index=frame.index, dtype=object)
    labels.loc[X_val.index] = "val"
    labels.loc[X_test.index] = "test"
    split_table = pd.DataFrame(
        {key_column: frame[key_column].to_numpy(), "split": labels.to_numpy()}
    )

    sizes = {
        "train": int(len(X_train)),
        "val": int(len(X_val)),
        "test": int(len(X_test)),
    }
    logger.info(
        "Разбиение: train=%d, val=%d, test=%d; признаков=%d",
        sizes["train"],
        sizes["val"],
        sizes["test"],
        len(feature_columns),
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_index": X_train.index,
        "val_index": X_val.index,
        "test_index": X_test.index,
        "split_table": split_table[[key_column, "split"]],
        "feature_columns": feature_columns,
        "random_state": int(preparation["random_state"]),
        "test_size": float(preparation["test_size"]),
        "val_size": float(preparation["val_size"]),
        "sizes": sizes,
    }


# ---------------------------------------------------------------------------
# Модель и метрики
# ---------------------------------------------------------------------------


def _build_pipeline(config: dict) -> Pipeline:
    """Полный sklearn Pipeline.

    Шаги: выбор whitelist по именам (лишние колонки отбрасываются, порядок
    колонок не важен), inf -> NaN, NaN -> fill_value, StandardScaler, RandomForest.
    Все шаги — стандартные sklearn-трансформеры, поэтому Pipeline сериализуется
    и загружается без самодельных классов.
    """
    fill_value = float(config["preprocessing"]["fill_value"])
    return Pipeline(
        steps=[
            (
                "select_features",
                ColumnTransformer(
                    transformers=[(
                        "features",
                        FunctionTransformer(pd.DataFrame.astype, kw_args={"dtype": "float64"}),
                        _feature_columns(config),
                    )],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
            (
                "fill_inf_nan",
                FunctionTransformer(
                    np.nan_to_num,
                    kw_args={"nan": np.nan, "posinf": np.nan, "neginf": np.nan},
                    validate=False,
                ),
            ),
            (
                "impute",
                SimpleImputer(
                    strategy="constant",
                    fill_value=fill_value,
                    keep_empty_features=True,
                ),
            ),
            ("scale", StandardScaler()),
            ("model", RandomForestClassifier(**dict(config["model"]))),
        ]
    )


def _predict_proba(model, X) -> np.ndarray:
    """Детерминированные вероятности положительного класса.

    RandomForest с n_jobs > 1 агрегирует голоса деревьев параллельно и даёт
    расхождения ~1e-16 между вызовами. Для воспроизводимых метрик и проверки
    reload предсказание считается в один поток; параметр модели восстанавливается.
    """
    forest = model.named_steps.get("model") if isinstance(model, Pipeline) else None
    n_jobs = getattr(forest, "n_jobs", None)
    if forest is None or n_jobs in (None, 1):
        return model.predict_proba(X)
    forest.n_jobs = 1
    try:
        return model.predict_proba(X)
    finally:
        forest.n_jobs = n_jobs


def train_model(X, y, config: dict) -> Pipeline:
    """Обучить полный Pipeline; scaler и заполнение фитятся только на train."""
    model = _build_pipeline(config)
    started = time.perf_counter()
    model.fit(X, y)
    logger.info("RandomForest обучен за %.2f сек", time.perf_counter() - started)
    return model


def evaluate(model, splits: dict, config: dict) -> dict:
    """Посчитать ROC-AUC/F1/precision/recall на val и test при фиксированном пороге."""
    threshold = float(config["evaluation"]["threshold"])
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Модель не содержит положительный класс 1")
    positive_index = classes.index(1)
    metrics: dict = {"threshold": threshold}
    for split in ("val", "test"):
        X = splits[f"X_{split}"]
        y = splits[f"y_{split}"]
        if len(set(y.unique().tolist())) != 2:
            raise ValueError(f"В split '{split}' должен быть оба класса для ROC-AUC")
        probabilities = _predict_proba(model, X)[:, positive_index]
        # Порог фиксирован и берётся из конфига; тест не участвует в его выборе.
        predictions = (probabilities >= threshold).astype(int)
        metrics[f"{split}_auc"] = float(roc_auc_score(y, probabilities))
        metrics[f"{split}_f1"] = float(f1_score(y, predictions, zero_division=0))
        metrics[f"{split}_precision"] = float(
            precision_score(y, predictions, zero_division=0)
        )
        metrics[f"{split}_recall"] = float(recall_score(y, predictions, zero_division=0))
    logger.info("Метрики (порог %.2f): %s", threshold, json.dumps(metrics, ensure_ascii=False))
    return metrics


# ---------------------------------------------------------------------------
# Логирование и безопасная запись артефактов
# ---------------------------------------------------------------------------


def _configure_logging(config: dict) -> Path:
    """Настроить логирование в artifacts/<log_name> (перезапись) и stderr.

    Лог лежит внутри каталога артефактов, чтобы попадать в Docker bind mount
    и в перечень выходных файлов конвейера. Время фиксируется только в логах.
    """
    artifacts_dir = _resolve_path(config, config["artifacts"]["dir"])
    log_path = artifacts_dir / config["logging"]["log_name"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"],
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


def _atomic_write(path: Path, writer) -> None:
    """Записать файл через временный файл в том же каталоге и os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload) -> None:
    def writer(temporary: Path) -> None:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    _atomic_write(path, writer)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write(path, lambda temporary: frame.to_csv(temporary, index=False))


def _atomic_write_joblib(path: Path, value) -> None:
    _atomic_write(path, lambda temporary: joblib.dump(value, temporary, compress=3))


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def _environment_versions() -> dict:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "joblib"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover
            versions[package] = "not-installed"
    return versions


def _build_manifest(
    config: dict,
    inputs: dict,
    splits: dict,
    metrics: dict,
    reload_check: dict,
    artifact_names: list[str],
) -> dict:
    """Манифест воспроизводимости: SHA данных/кода/конфига, версии, параметры.

    В манифесте нет времени и абсолютных путей, поэтому два запуска с одинаковыми
    входами дают байт-в-байт одинаковый manifest.json.
    """
    target_column = config["data"]["target_column"]
    return {
        "schema_version": 1,
        "problem": "binary_classification",
        "scope": (
            "ретроспективная классификация метки churned на случайном "
            "стратифицированном разбиении; не прогноз будущего оттока"
        ),
        "target": target_column,
        "inputs": dict(inputs),
        "code": {
            "pipeline_sha256": _sha256_file(Path(__file__).resolve()),
            "config_sha256": _sha256_file(Path(config["_config_path"])),
        },
        "environment": _environment_versions(),
        "features": _feature_columns(config),
        "model": {
            "estimator": "RandomForestClassifier",
            "params": dict(config["model"]),
            "threshold": metrics["threshold"],
            "preprocessing": [
                "ColumnTransformer: whitelist по именам, float64 cast, remainder=drop",
                "FunctionTransformer(nan_to_num): +-inf -> NaN",
                "SimpleImputer(constant fill_value)",
                "StandardScaler",
            ],
        },
        "splits": {
            "train": splits["sizes"]["train"],
            "val": splits["sizes"]["val"],
            "test": splits["sizes"]["test"],
            "test_size": splits["test_size"],
            "val_size": splits["val_size"],
            "random_state": splits["random_state"],
            "stratify": target_column,
        },
        "artifacts": list(artifact_names),
        "reload_check": reload_check,
    }


# ---------------------------------------------------------------------------
# Полный запуск
# ---------------------------------------------------------------------------


def run_pipeline(config_path) -> dict:
    """Полный идемпотентный запуск: данные -> сплиты -> модель -> артефакты."""
    config = load_config(config_path)
    paths = _validate_output_paths(config)
    artifact_paths = paths["artifacts"]
    artifacts_dir = paths["artifacts_dir"]
    artifact_names = [path.name for path in artifact_paths.values()]
    artifact_names.append(config["logging"]["log_name"])

    log_path = _configure_logging(config)
    logger.info("Запуск конвейера; конфиг: %s", config["_config_path"])

    logger.info("Шаг 1/7: загрузка и проверка данных")
    frame, inputs = _load_inputs(config)
    logger.info("Шаг 2/7: стратифицированное разбиение 60/20/20")
    splits = preprocess(frame, config)
    logger.info("Шаг 3/7: обучение полного sklearn Pipeline")
    model = train_model(splits["X_train"], splits["y_train"], config)
    logger.info("Шаг 4/7: оценка качества на val/test")
    metrics = evaluate(model, splits, config)

    logger.info("Шаг 5/7: сохранение модели, признаков и сплитов")
    _atomic_write_joblib(artifact_paths["model"], model)
    _atomic_write_json(artifact_paths["features"], splits["feature_columns"])
    split_table = splits["split_table"]
    _atomic_write_csv(artifact_paths["splits"], split_table)

    logger.info("Шаг 6/7: проверка загрузки модели и вероятностей")
    reloaded = joblib.load(artifact_paths["model"])
    probabilities_before = _predict_proba(model, splits["X_test"])
    probabilities_after = _predict_proba(reloaded, splits["X_test"])
    max_difference = (
        float(np.max(np.abs(probabilities_before - probabilities_after)))
        if probabilities_before.size
        else 0.0
    )
    if not np.allclose(
        probabilities_before, probabilities_after, rtol=0.0, atol=1e-9
    ) or reloaded.predict(splits["X_test"]).shape != (len(splits["X_test"]),):
        raise RuntimeError(
            "Проверка reload не пройдена: вероятности после joblib.load различаются"
        )
    reload_check = {
        "passed": True,
        "max_abs_probability_diff": round(max_difference, 12),
        "checked_split": "test",
        "checked_rows": int(len(splits["X_test"])),
    }
    logger.info("Reload проверен: max |delta p| = %.3e", max_difference)

    logger.info("Шаг 7/7: сохранение метрик и manifest")
    _atomic_write_json(artifact_paths["metrics"], metrics)
    manifest = _build_manifest(
        config, inputs, splits, metrics, reload_check, artifact_names
    )
    _atomic_write_json(artifact_paths["manifest"], manifest)
    logger.info("Артефакты записаны в %s", artifacts_dir)
    logger.info("Конвейер успешно завершён")

    return {
        "metrics": metrics,
        "artifacts": {
            **{name: str(path) for name, path in artifact_paths.items()},
            "log": str(log_path),
        },
        "manifest": manifest,
        "reload_check": reload_check,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    """Точка входа CLI: 0 при успехе, 1 при любой ошибке (с логированием)."""
    parser = argparse.ArgumentParser(
        description=(
            "Конвейер ретроспективной бинарной классификации churned "
            "(случайное разбиение 60/20/20, не прогноз будущего оттока)"
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_NAME,
        help="путь к JSON-конфигу (по умолчанию config.json)",
    )
    arguments = parser.parse_args(argv)
    try:
        result = run_pipeline(arguments.config)
    except Exception:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                stream=sys.stderr,
            )
        logger.exception("Конвейер завершился с ошибкой (конфиг: %s)", arguments.config)
        return 1
    metrics = result["metrics"]
    print(f"Модель сохранена: {result['artifacts']['model']}")
    print(f"Manifest: {result['artifacts']['manifest']}")
    print(
        "test_auc={test_auc:.4f} test_f1={test_f1:.4f} "
        "test_precision={test_precision:.4f} test_recall={test_recall:.4f}".format(**metrics)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
