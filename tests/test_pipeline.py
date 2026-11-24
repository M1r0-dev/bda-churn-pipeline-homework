"""Контрактные тесты конвейера лабораторной работы 6.1.

Тесты проверяют публичный контракт pipeline.py и не зависят от сырых данных:
для проверок используется синтетическая выборка, построенная в этом же файле.
Реальные данные проверяются отдельным прогоном (см. TestRealDataset).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline  # noqa: E402

KEY = "subscriber_key"
TARGET = "churned"
LEAKY = "churn_day_index"
FEATURE_COLUMNS = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))[
    "data"
]["feature_columns"]
ARTIFACT_NAMES = [
    "model.joblib",
    "features.json",
    "metrics.json",
    "manifest.json",
    "splits.csv",
    "pipeline.log",
]


def make_config() -> dict:
    """Ожидаемое содержимое lab2/config.json: 60/20/20, seed 42, threshold 0.5."""
    return {
        "data": {
            "features_path": "data/churn_features.parquet",
            "churn_path": "data/churn_truth.csv",
            "key_column": KEY,
            "target_column": TARGET,
            "excluded_columns": [LEAKY],
            "feature_columns": list(FEATURE_COLUMNS),
        },
        "preprocessing": {
            "test_size": 0.4,
            "val_size": 0.5,
            "random_state": 42,
            "fill_value": 0.0,
        },
        "model": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": 4,
        },
        "evaluation": {"threshold": 0.5},
        "artifacts": {
            "dir": "artifacts",
            "model_name": "model.joblib",
            "features_name": "features.json",
            "metrics_name": "metrics.json",
            "manifest_name": "manifest.json",
            "splits_name": "splits.csv",
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            "log_name": "pipeline.log",
        },
    }


def build_fixture(n: int = 240, seed: int = 7, extra_truth: int = 3):
    """Синтетические признаки (10 колонок, NaN/inf) и эталон с меткой churned."""
    rng = np.random.default_rng(seed)
    keys = np.array([f"SUB{i:06d}" for i in range(n)])
    base = {}
    for index, name in enumerate(FEATURE_COLUMNS):
        base[name] = np.round(rng.normal(loc=index * 0.05, scale=1.0, size=n), 6)
    # Метка с реальным сигналом: должна отделяться лучше случайного угадывания.
    score = (
        base["events_30d"]
        + base["tickets_30d"]
        + base["recency_days"]
        + rng.normal(scale=0.75, size=n)
    )
    churned = score > np.quantile(score, 0.7)
    features = pd.DataFrame(base)
    features.insert(0, KEY, keys)
    # Пропуски и бесконечности внутри признаков — как в реальных данных HW5.
    features.loc[features.index[:20], "traffic_change_ratio"] = np.nan
    features.loc[features.index[20:30], "network_error_rate_30d"] = np.nan
    features.loc[features.index[30:35], "traffic_gb_30d"] = np.inf
    features.loc[features.index[35:40], "recency_days"] = -np.inf
    extra_keys = np.array([f"SUBX{i:05d}" for i in range(extra_truth)])
    truth = pd.DataFrame(
        {
            KEY: np.concatenate([keys, extra_keys]),
            LEAKY: rng.integers(-1, 120, size=n + extra_truth),
            TARGET: np.concatenate([churned, rng.random(extra_truth) > 0.5]),
        }
    )
    return features, truth


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_TIMING_TOKENS = ("time", "elapsed", "duration", "seconds", "runtime", "timestamp")


def assert_no_timing_keys(case: unittest.TestCase, payload) -> None:
    """В артефактах не должно быть времени: оно живёт только в логах."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            for token in _TIMING_TOKENS:
                case.assertNotIn(token, lowered, f"в артефакте есть время: {key}")
            assert_no_timing_keys(case, value)
    elif isinstance(payload, list):
        for item in payload:
            assert_no_timing_keys(case, item)


class PipelineFixture(unittest.TestCase):
    """Общая синтетическая песочница: project/data + project/config.json."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.project = Path(self._tmp.name) / "project"
        (self.project / "data").mkdir(parents=True)
        self.features, self.truth = build_fixture()
        self.features_path = self.project / "data" / "churn_features.parquet"
        self.truth_path = self.project / "data" / "churn_truth.csv"
        self.write_inputs()
        self.config_path = self.write_config()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_inputs(self) -> None:
        self.features.to_parquet(self.features_path, index=False)
        self.truth.to_csv(self.truth_path, index=False)

    def write_config(self, mutate=None) -> Path:
        config = make_config()
        if mutate is not None:
            mutate(config)
        path = self.project / "config.json"
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def load_config(self):
        return pipeline.load_config(str(self.config_path))

    def splits(self, config=None, frame=None):
        config = config or self.load_config()
        frame = pipeline.load_data(config) if frame is None else frame
        return pipeline.preprocess(frame, config)


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------


class TestRepoConfig(unittest.TestCase):
    def test_config_file_matches_contract(self) -> None:
        loaded = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(loaded, make_config())

    def test_config_paths_are_relative_to_config_file(self) -> None:
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        for value in (config["data"]["features_path"], config["data"]["churn_path"]):
            self.assertFalse(Path(value).is_absolute(), value)
        self.assertFalse(Path(config["artifacts"]["dir"]).is_absolute())
        self.assertFalse(Path(config["logging"]["log_name"]).is_absolute())

    def test_original_seed42_random_forest_parameters(self) -> None:
        model = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["model"]
        self.assertEqual(
            model,
            {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": 4,
            },
        )

    def test_threshold_and_split_ratio(self) -> None:
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["evaluation"]["threshold"], 0.5)
        self.assertEqual(config["preprocessing"]["test_size"], 0.4)
        self.assertEqual(config["preprocessing"]["val_size"], 0.5)
        self.assertEqual(config["preprocessing"]["random_state"], 42)
        self.assertEqual(config["preprocessing"]["fill_value"], 0.0)


class TestConfigValidation(PipelineFixture):
    def test_unknown_or_missing_sections_rejected(self) -> None:
        self.write_config(lambda c: c.update({"unexpected": 1}))
        with self.assertRaises(ValueError):
            pipeline.load_config(str(self.config_path))
        self.write_config(lambda c: c.pop("evaluation"))
        with self.assertRaises(ValueError):
            pipeline.load_config(str(self.config_path))

    def test_split_and_threshold_bounds_rejected(self) -> None:
        mutations = (
            lambda c: c["preprocessing"].update({"test_size": 0.0}),
            lambda c: c["preprocessing"].update({"test_size": 1.5}),
            lambda c: c["preprocessing"].update({"val_size": 0.0}),
            lambda c: c["preprocessing"].update({"val_size": 2.0}),
            lambda c: c["evaluation"].update({"threshold": -0.1}),
            lambda c: c["evaluation"].update({"threshold": 1.1}),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.write_config(mutate)
                with self.assertRaises(ValueError):
                    pipeline.load_config(str(self.config_path))

    def test_invalid_feature_whitelist_rejected(self) -> None:
        mutations = (
            lambda c: c["data"].update({"feature_columns": []}),
            lambda c: c["data"].update({"feature_columns": None}),
            lambda c: c["data"].update({"feature_columns": "events_30d"}),
            lambda c: c["data"]["feature_columns"].append(" "),
            lambda c: c["data"]["feature_columns"].append(1),
            lambda c: c["data"]["feature_columns"].append(FEATURE_COLUMNS[0]),
            lambda c: c["data"]["feature_columns"].append(KEY),
            lambda c: c["data"]["feature_columns"].append(TARGET),
            lambda c: c["data"]["feature_columns"].append(LEAKY),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.write_config(mutate)
                with self.assertRaises(ValueError):
                    pipeline.load_config(str(self.config_path))

    def test_artifact_and_logging_section_rejected(self) -> None:
        self.write_config(lambda c: c["artifacts"].update({"features_name": "model.joblib"}))
        with self.assertRaises(ValueError):
            pipeline.load_config(str(self.config_path))
        self.write_config(lambda c: c["logging"].update({"level": "LOUD"}))
        with self.assertRaises(ValueError):
            pipeline.load_config(str(self.config_path))

    def test_output_names_must_be_basenames(self) -> None:
        self.write_config(lambda c: c["artifacts"].update({"model_name": "data/model.joblib"}))
        with self.assertRaises(ValueError):
            pipeline.load_config(str(self.config_path))
        self.write_config(lambda c: c["logging"].update({"log_name": "../pipeline.log"}))
        with self.assertRaises(ValueError):
            pipeline.load_config(str(self.config_path))

    def test_missing_or_broken_config_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            pipeline.load_config(str(self.project / "missing.json"))
        broken = self.project / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            pipeline.load_config(str(broken))


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------


class TestLoadData(PipelineFixture):
    def test_returns_key_target_and_fixed_whitelist(self) -> None:
        frame = pipeline.load_data(self.load_config())
        self.assertEqual(list(frame.columns), [KEY, *FEATURE_COLUMNS, TARGET])
        self.assertEqual(len(frame), len(self.features))
        self.assertEqual(frame[TARGET].dtype.kind, "i")

    def test_preserves_source_row_order(self) -> None:
        frame = pipeline.load_data(self.load_config())
        self.assertEqual(frame[KEY].tolist(), self.features[KEY].tolist())

    def test_extra_truth_rows_are_not_joined(self) -> None:
        frame = pipeline.load_data(self.load_config())
        self.assertEqual(len(frame), len(self.features))
        self.assertEqual(set(frame[KEY]), set(self.features[KEY]))
        self.assertTrue(set(self.truth[KEY]) - set(self.features[KEY]))

    def test_drops_target_and_leaky_columns_from_features(self) -> None:
        # В файле признаков лежат поддельные churned/churn_day_index: источник
        # метки — только эталон, служебные колонки не должны попасть в X.
        features = self.features.copy()
        truth_target = self.truth.set_index(KEY).loc[features[KEY], TARGET].astype(int)
        features[TARGET] = 1 - truth_target.to_numpy()
        features[LEAKY] = np.arange(len(features))
        features["future_leak"] = 1.0
        features.to_parquet(self.features_path, index=False)
        frame = pipeline.load_data(self.load_config())
        self.assertNotIn(LEAKY, frame.columns)
        self.assertNotIn("future_leak", frame.columns)
        self.assertEqual(list(frame.columns), [KEY, *FEATURE_COLUMNS, TARGET])
        np.testing.assert_array_equal(frame[TARGET].to_numpy(), truth_target.to_numpy())

    def test_rejects_feature_key_missing_from_truth(self) -> None:
        truth = self.truth.iloc[1:]
        truth.to_csv(self.truth_path, index=False)
        with self.assertRaises(ValueError):
            pipeline.load_data(self.load_config())

    def test_rejects_duplicate_feature_keys(self) -> None:
        features = pd.concat([self.features, self.features.iloc[[0]]], ignore_index=True)
        features.to_parquet(self.features_path, index=False)
        with self.assertRaises(ValueError):
            pipeline.load_data(self.load_config())

    def test_rejects_duplicate_truth_keys(self) -> None:
        truth = pd.concat([self.truth, self.truth.iloc[[0]]], ignore_index=True)
        truth.to_csv(self.truth_path, index=False)
        with self.assertRaises(ValueError):
            pipeline.load_data(self.load_config())

    def test_rejects_null_or_empty_keys(self) -> None:
        features = self.features.copy()
        features.loc[0, KEY] = None
        features.to_parquet(self.features_path, index=False)
        with self.assertRaises(ValueError):
            pipeline.load_data(self.load_config())

        features = self.features.copy()
        features.loc[0, KEY] = "   "
        features.to_parquet(self.features_path, index=False)
        with self.assertRaises(ValueError):
            pipeline.load_data(self.load_config())

    def test_rejects_missing_key_column(self) -> None:
        self.features.drop(columns=[KEY]).to_parquet(self.features_path, index=False)
        with self.assertRaises(ValueError):
            pipeline.load_data(self.load_config())

    def test_rejects_missing_target_column(self) -> None:
        self.truth.drop(columns=[TARGET]).to_csv(self.truth_path, index=False)
        with self.assertRaises(ValueError):
            pipeline.load_data(self.load_config())

    def test_rejects_missing_feature_column(self) -> None:
        self.features.drop(columns=[FEATURE_COLUMNS[3]]).to_parquet(
            self.features_path, index=False
        )
        with self.assertRaises(ValueError):
            pipeline.load_data(self.load_config())

    def test_rejects_non_numeric_feature(self) -> None:
        features = self.features.copy()
        features["tickets_30d"] = "many"
        features.to_parquet(self.features_path, index=False)
        with self.assertRaises(ValueError):
            pipeline.load_data(self.load_config())

    def test_rejects_single_class_target(self) -> None:
        truth = self.truth.copy()
        truth[TARGET] = False
        truth.to_csv(self.truth_path, index=False)
        with self.assertRaises(ValueError):
            pipeline.load_data(self.load_config())

    def test_rejects_missing_input_files(self) -> None:
        self.features_path.unlink()
        with self.assertRaises(FileNotFoundError):
            pipeline.load_data(self.load_config())


# ---------------------------------------------------------------------------
# Предобработка
# ---------------------------------------------------------------------------


class TestPreprocess(PipelineFixture):
    def test_split_sizes_60_20_20(self) -> None:
        splits = self.splits()
        self.assertEqual(splits["sizes"], {"train": 144, "val": 48, "test": 48})
        self.assertEqual(len(splits["X_train"]), 144)
        self.assertEqual(len(splits["y_val"]), 48)

    def test_splits_are_disjoint_and_cover_every_key(self) -> None:
        frame = pipeline.load_data(self.load_config())
        splits = self.splits(frame=frame)
        train = set(splits["train_index"])
        val = set(splits["val_index"])
        test = set(splits["test_index"])
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(train | val | test, set(frame.index))

    def test_each_split_is_stratified(self) -> None:
        splits = self.splits()
        overall = float(pd.concat([splits["y_train"], splits["y_val"], splits["y_test"]]).mean())
        for name in ("train", "val", "test"):
            y = splits[f"y_{name}"]
            self.assertEqual(set(y.unique()), {0, 1})
            self.assertAlmostEqual(float(y.mean()), overall, delta=0.05)

    def test_keeps_raw_nan_and_inf_for_the_pipeline(self) -> None:
        frame = pipeline.load_data(self.load_config())
        splits = self.splits(frame=frame)
        raw = frame[FEATURE_COLUMNS]
        self.assertGreater(raw.isna().to_numpy().sum(), 0)
        self.assertGreater(np.isinf(raw.to_numpy(dtype=float)).sum(), 0)
        combined = pd.concat([splits["X_train"], splits["X_val"], splits["X_test"]])
        self.assertEqual(
            int(combined.isna().to_numpy().sum()), int(raw.isna().to_numpy().sum())
        )
        self.assertEqual(
            int(np.isinf(combined.to_numpy(dtype=float)).sum()),
            int(np.isinf(raw.to_numpy(dtype=float)).sum()),
        )

    def test_split_table_has_key_and_split_columns(self) -> None:
        frame = pipeline.load_data(self.load_config())
        splits = self.splits(frame=frame)
        table = splits["split_table"]
        self.assertEqual(list(table.columns), [KEY, "split"])
        self.assertEqual(table[KEY].tolist(), frame[KEY].tolist())
        self.assertEqual(set(table["split"]), {"train", "val", "test"})
        self.assertEqual(len(table), len(frame))

    def test_split_is_deterministic_for_same_seed(self) -> None:
        first = self.splits()
        second = self.splits()
        self.assertEqual(first["train_index"].tolist(), second["train_index"].tolist())
        self.assertEqual(first["test_index"].tolist(), second["test_index"].tolist())

    def test_feature_columns_order_preserved(self) -> None:
        splits = self.splits()
        self.assertEqual(splits["feature_columns"], FEATURE_COLUMNS)
        self.assertEqual(list(splits["X_train"].columns), FEATURE_COLUMNS)

    def test_rejects_frame_without_target(self) -> None:
        frame = pipeline.load_data(self.load_config()).drop(columns=[TARGET])
        with self.assertRaises(ValueError):
            pipeline.preprocess(frame, self.load_config())

    def test_rejects_single_class_frame(self) -> None:
        frame = pipeline.load_data(self.load_config())
        frame[TARGET] = 0
        with self.assertRaises(ValueError):
            pipeline.preprocess(frame, self.load_config())


# ---------------------------------------------------------------------------
# Модель: полный sklearn Pipeline
# ---------------------------------------------------------------------------


class TestTrainModel(PipelineFixture):
    def _trained(self):
        config = self.load_config()
        frame = pipeline.load_data(config)
        splits = pipeline.preprocess(frame, config)
        model = pipeline.train_model(splits["X_train"], splits["y_train"], config)
        return config, frame, splits, model

    def test_pipeline_uses_only_standard_sklearn_steps(self) -> None:
        _, _, _, model = self._trained()
        self.assertIsInstance(model, Pipeline)
        steps = dict(model.steps)
        self.assertIsInstance(steps["select_features"], ColumnTransformer)
        self.assertEqual(steps["select_features"].remainder, "drop")
        self.assertIsInstance(steps["fill_inf_nan"], FunctionTransformer)
        self.assertIs(steps["fill_inf_nan"].func, np.nan_to_num)
        self.assertIsInstance(steps["impute"], SimpleImputer)
        self.assertIsInstance(steps["scale"], StandardScaler)
        self.assertIsInstance(steps["model"], RandomForestClassifier)
        for name, step in model.steps:
            self.assertTrue(
                type(step).__module__.startswith("sklearn"),
                f"шаг {name} не из sklearn: {type(step).__module__}",
            )

    def test_original_random_forest_parameters_used(self) -> None:
        config, _, _, model = self._trained()
        forest = model.named_steps["model"]
        for name, value in config["model"].items():
            self.assertEqual(getattr(forest, name), value)

    def test_accepts_raw_dataframe_with_nan_and_inf(self) -> None:
        _, _, splits, model = self._trained()
        probabilities = model.predict_proba(splits["X_val"])
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
        self.assertEqual(probabilities.shape[0], len(splits["X_val"]))
        predictions = model.predict(splits["X_test"])
        self.assertEqual(set(np.unique(predictions)), {0, 1})

    def test_nullable_numeric_features_with_missing_values(self) -> None:
        config = self.load_config()
        splits = self.splits(config=config)
        train = splits['X_train'].replace([np.inf, -np.inf], np.nan).round()
        validation = splits['X_val'].replace([np.inf, -np.inf], np.nan).round()
        reference = pipeline.train_model(train, splits['y_train'], config)
        expected = pipeline._predict_proba(reference, validation)
        for dtype in ('Float64', 'Int64'):
            with self.subTest(dtype=dtype):
                model = pipeline.train_model(train.astype(dtype), splits['y_train'], config)
                path = self.project / 'nullable.joblib'
                joblib.dump(model, path)
                loaded = joblib.load(path)
                actual = pipeline._predict_proba(loaded, validation.astype(dtype))
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_scaler_fitted_on_train_rows_only(self) -> None:
        _, frame, splits, model = self._trained()
        scaler = model.named_steps["scale"]
        raw_train = splits["X_train"].to_numpy(dtype=float)
        expected_train = np.nan_to_num(
            np.where(np.isfinite(raw_train), raw_train, np.nan), nan=0.0
        )
        np.testing.assert_allclose(
            scaler.mean_, expected_train.mean(axis=0), rtol=0, atol=1e-12
        )
        raw_all = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        expected_all = np.nan_to_num(
            np.where(np.isfinite(raw_all), raw_all, np.nan), nan=0.0
        ).mean(axis=0)
        self.assertGreater(
            float(np.max(np.abs(scaler.mean_ - expected_all))),
            1e-9,
            "scaler не должен обучаться на val/test",
        )

    def test_seed_makes_predictions_reproducible(self) -> None:
        config, _, splits, first = self._trained()
        second = pipeline.train_model(splits["X_train"], splits["y_train"], config)
        # Деревья воспроизводимы; у predict_proba с n_jobs>1 возможны
        # расхождения ~1e-16 из-за параллельного усреднения по деревьям.
        np.testing.assert_allclose(
            first.predict_proba(splits["X_test"]),
            second.predict_proba(splits["X_test"]),
            rtol=0,
            atol=1e-12,
        )

    def test_selector_ignores_reordered_and_extra_columns(self) -> None:
        _, _, splits, model = self._trained()
        baseline = model.predict_proba(splits["X_test"])
        reordered = splits["X_test"][list(reversed(FEATURE_COLUMNS))].copy()
        reordered[LEAKY] = 123
        reordered["unexpected_text"] = "not a feature"
        np.testing.assert_allclose(
            model.predict_proba(reordered), baseline, rtol=0, atol=1e-12
        )

    def test_predict_rejects_wrong_feature_set(self) -> None:
        _, _, splits, model = self._trained()
        incomplete = splits["X_test"].drop(columns=[FEATURE_COLUMNS[0]])
        with self.assertRaises(ValueError):
            model.predict_proba(incomplete)


# ---------------------------------------------------------------------------
# Оценка качества
# ---------------------------------------------------------------------------


class _FixedProbabilityModel:
    """Тестовая заглушка с заранее известными вероятностями."""

    classes_ = np.array([0, 1])

    def __init__(self, probabilities):
        self._probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, X):
        positive = self._probabilities[: len(X)]
        return np.column_stack([1.0 - positive, positive])


class TestEvaluate(PipelineFixture):
    def _fixed_split(self, probabilities=(0.49, 0.5, 0.51)):
        X = pd.DataFrame({name: np.zeros(3) for name in FEATURE_COLUMNS})
        y = pd.Series([0, 1, 1])
        return {
            "X_val": X,
            "y_val": y,
            "X_test": X.copy(),
            "y_test": y.copy(),
        }, _FixedProbabilityModel(probabilities)

    def test_threshold_is_inclusive_at_0_5(self) -> None:
        config = self.load_config()
        splits, model = self._fixed_split()
        metrics = pipeline.evaluate(model, splits, config)
        self.assertEqual(metrics["threshold"], 0.5)
        # y=[0,1,1], p=[0.49,0.5,0.51] и порог 0.5: предсказания [0,1,1].
        self.assertAlmostEqual(metrics["val_f1"], 1.0)
        self.assertAlmostEqual(metrics["val_precision"], 1.0)
        self.assertAlmostEqual(metrics["val_recall"], 1.0)
        self.assertAlmostEqual(metrics["test_auc"], 1.0)

    def test_threshold_from_config_is_used(self) -> None:
        self.write_config(lambda c: c["evaluation"].update({"threshold": 0.505}))
        config = self.load_config()
        splits, model = self._fixed_split()
        metrics = pipeline.evaluate(model, splits, config)
        # p=[0.49,0.5,0.51] >= 0.505 превращается в [0,0,1]: tp=1, fp=0, fn=1.
        self.assertAlmostEqual(metrics["test_precision"], 1.0)
        self.assertAlmostEqual(metrics["test_recall"], 0.5)
        self.assertAlmostEqual(metrics["test_f1"], 2.0 / 3.0)

    def test_metrics_keys_and_bounds(self) -> None:
        config = self.load_config()
        frame = pipeline.load_data(config)
        splits = pipeline.preprocess(frame, config)
        model = pipeline.train_model(splits["X_train"], splits["y_train"], config)
        metrics = pipeline.evaluate(model, splits, config)
        expected = {"threshold"} | {
            f"{split}_{name}"
            for split in ("val", "test")
            for name in ("auc", "f1", "precision", "recall")
        }
        self.assertEqual(set(metrics), expected)
        for key, value in metrics.items():
            if key == "threshold":
                continue
            self.assertTrue(0.0 <= float(value) <= 1.0, f"{key}={value}")
        self.assertGreater(metrics["test_auc"], 0.6, "фикстура должна иметь сигнал")
        self.assertNotIn("train_time_sec", metrics)


class TestRunPipeline(PipelineFixture):
    def _run(self) -> dict:
        return pipeline.run_pipeline(str(self.config_path))

    def test_configured_features_used_throughout_pipeline(self) -> None:
        self.features["custom_numeric"] = np.arange(len(self.features), dtype=float)
        self.write_inputs()
        selections = (
            list(reversed(FEATURE_COLUMNS)),
            [FEATURE_COLUMNS[5], FEATURE_COLUMNS[0]],
            ["custom_numeric", FEATURE_COLUMNS[3]],
            ["custom_numeric"],
        )
        for selected in selections:
            with self.subTest(selected=selected):
                self.write_config(lambda c: c["data"].update({"feature_columns": selected}))
                config = self.load_config()
                frame = pipeline.load_data(config)
                self.assertEqual(list(frame.columns), [KEY, *selected, TARGET])
                splits = pipeline.preprocess(frame, config)
                self.assertEqual(splits["feature_columns"], selected)
                for split in ("train", "val", "test"):
                    self.assertEqual(list(splits[f"X_{split}"].columns), selected)
                result = self._run()
                features = json.loads(Path(result["artifacts"]["features"]).read_text())
                manifest = json.loads(Path(result["artifacts"]["manifest"]).read_text())
                self.assertEqual(features, selected)
                self.assertEqual(manifest["features"], selected)
                model = joblib.load(result["artifacts"]["model"])
                self.assertEqual(
                    model.named_steps["select_features"].transformers_[0][2], selected
                )
                self.assertEqual(model.named_steps["scale"].n_features_in_, len(selected))
                # Для предсказания достаточно выбранных колонок, в любом порядке.
                raw = self.features[list(reversed(selected))]
                probabilities = model.predict_proba(raw)
                self.assertEqual(probabilities.shape, (len(self.features), 2))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(result["reload_check"]["passed"])

    def test_missing_configured_feature_rejected(self) -> None:
        self.write_config(lambda c: c["data"]["feature_columns"].append("missing_numeric"))
        config = self.load_config()
        with self.assertRaisesRegex(ValueError, "missing_numeric"):
            pipeline.load_data(config)

    def test_artifacts_written_next_to_config(self) -> None:
        result = self._run()
        artifacts_dir = self.project / "artifacts"
        self.assertEqual(
            sorted(path.name for path in artifacts_dir.iterdir()), sorted(ARTIFACT_NAMES)
        )
        for name in ("model", "features", "metrics", "manifest", "splits", "log"):
            self.assertTrue(Path(result["artifacts"][name]).is_file(), name)
        self.assertEqual(Path(result["artifacts"]["log"]), artifacts_dir / "pipeline.log")

    def test_splits_csv_has_key_and_split(self) -> None:
        result = self._run()
        table = pd.read_csv(self.project / "artifacts" / "splits.csv")
        self.assertEqual(list(table.columns), [KEY, "split"])
        self.assertEqual(len(table), len(self.features))
        counts = {name: int(value) for name, value in table["split"].value_counts().items()}
        self.assertEqual(counts, {"train": 144, "val": 48, "test": 48})
        self.assertEqual(result["manifest"]["splits"]["test"], 48)

    def test_features_and_metrics_json(self) -> None:
        result = self._run()
        artifacts_dir = self.project / "artifacts"
        features = json.loads((artifacts_dir / "features.json").read_text(encoding="utf-8"))
        metrics = json.loads((artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(features, FEATURE_COLUMNS)
        expected = {"threshold"} | {
            f"{split}_{name}"
            for split in ("val", "test")
            for name in ("auc", "f1", "precision", "recall")
        }
        self.assertEqual(set(metrics), expected)
        self.assertAlmostEqual(metrics["threshold"], 0.5)
        self.assertEqual(metrics, result["metrics"])
        assert_no_timing_keys(self, metrics)

    def test_manifest_provenance(self) -> None:
        result = self._run()
        manifest = json.loads(
            (self.project / "artifacts" / "manifest.json").read_text(encoding="utf-8")
        )
        inputs = manifest["inputs"]
        self.assertEqual(inputs["features_sha256"], sha256_file(self.features_path))
        self.assertEqual(inputs["churn_sha256"], sha256_file(self.truth_path))
        self.assertEqual(
            (inputs["features_rows"], inputs["churn_rows"], inputs["joined_rows"]), (240, 243, 240)
        )
        self.assertEqual(inputs["unmatched_truth_rows"], 3)
        self.assertEqual(manifest["code"]["pipeline_sha256"], sha256_file(ROOT / "pipeline.py"))
        self.assertEqual(manifest["code"]["config_sha256"], sha256_file(self.config_path))
        self.assertEqual(manifest["features"], FEATURE_COLUMNS)
        self.assertIn("pipeline.log", manifest["artifacts"])
        self.assertEqual(manifest["model"]["params"], make_config()["model"])
        self.assertEqual(manifest["model"]["threshold"], 0.5)
        for package in ("python", "numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "joblib"):
            self.assertIn(package, manifest["environment"])
        self.assertEqual(result["manifest"], manifest)
        assert_no_timing_keys(self, manifest)

    def test_model_reload_check_and_raw_prediction(self) -> None:
        result = self._run()
        self.assertTrue(result["reload_check"]["passed"])
        self.assertLessEqual(result["reload_check"]["max_abs_probability_diff"], 1e-9)
        self.assertEqual(result["reload_check"]["checked_rows"], 48)
        model = joblib.load(result["artifacts"]["model"])
        raw = self.features[FEATURE_COLUMNS]
        probabilities = model.predict_proba(raw)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertEqual(len(model.predict(raw)), len(raw))

    def test_second_run_overwrites_identically(self) -> None:
        first = self._run()
        artifacts_dir = self.project / "artifacts"
        stable_names = ("features.json", "metrics.json", "splits.csv", "manifest.json")
        first_bytes = {name: (artifacts_dir / name).read_bytes() for name in stable_names}
        second = self._run()
        for name, payload in first_bytes.items():
            self.assertEqual((artifacts_dir / name).read_bytes(), payload, name)
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(
            sorted(path.name for path in artifacts_dir.iterdir()), sorted(ARTIFACT_NAMES)
        )
        log_text = (artifacts_dir / "pipeline.log").read_text(encoding="utf-8")
        self.assertEqual(log_text.count("Конвейер успешно завершён"), 1)
        self.assertIn("INFO", log_text)

    def test_outputs_cannot_overwrite_inputs(self) -> None:
        # artifacts.dir=data + model_name=churn_features.parquet заменил бы вход.
        self.write_config(
            lambda c: c["artifacts"].update(
                {"dir": "data", "model_name": "churn_features.parquet"}
            )
        )
        with self.assertRaises(ValueError):
            pipeline.run_pipeline(str(self.config_path))
        self.assertTrue(self.features_path.is_file())
        # log_name не должен совпасть с именем артефакта в том же каталоге.
        self.write_config(
            lambda c: (
                c["artifacts"].update({"dir": "."}),
                c["logging"].update({"log_name": "model.joblib"}),
            )
        )
        with self.assertRaises(ValueError):
            pipeline.run_pipeline(str(self.config_path))

    def test_log_hardlink_cannot_truncate_an_input(self) -> None:
        before = self.truth_path.read_bytes()
        directory = self.project / 'artifacts'
        directory.mkdir()
        (directory / 'pipeline.log').hardlink_to(self.truth_path)
        with self.assertRaisesRegex(ValueError, 'совпадает'):
            self._run()
        self.assertEqual(self.truth_path.read_bytes(), before)

    def test_missing_data_file_fails(self) -> None:
        self.features_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self._run()


class TestCLI(PipelineFixture):
    def _cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(ROOT / "pipeline.py"), *arguments],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )

    def test_cli_success_exit_code_zero(self) -> None:
        completed = self._cli("--config", str(self.config_path))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("test_auc", completed.stdout)
        self.assertTrue((self.project / "artifacts" / "model.joblib").is_file())

    def test_cli_errors_exit_nonzero_with_logging(self) -> None:
        missing = self.project / "missing_config.json"
        completed = self._cli("--config", str(missing))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("ERROR", completed.stderr)
        broken = self.project / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        completed = self._cli("--config", str(broken))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("ERROR", completed.stderr)


DATA_FEATURES = ROOT / "data" / "churn_features.parquet"
DATA_TRUTH = ROOT / "data" / "churn_truth.csv"
DATA_READY = DATA_FEATURES.is_file() and DATA_TRUTH.is_file()
BASELINE_PATH = ROOT / "provenance" / "baseline_metrics.json"
BASELINE_PROBABILITIES = ROOT / ".scratch" / "baseline_test_probabilities.npy"
BASELINE_INDICES = ROOT / ".scratch" / "baseline_test_indices.npy"


@unittest.skipUnless(DATA_READY, "реальные данные lab2/data не подготовлены")
class TestRealDataset(unittest.TestCase):
    """Интеграционная проверка на реальных данных без записи в ROOT/artifacts."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.project = Path(self._tmp.name)
        self.artifacts_dir = self.project / "artifacts"
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        # Изоляция: копия конфига во временном каталоге, абсолютные пути к данным
        # и временный artifacts, чтобы unittest не затирал рабочую модель CLI.
        config["data"]["features_path"] = str(DATA_FEATURES)
        config["data"]["churn_path"] = str(DATA_TRUTH)
        config["artifacts"]["dir"] = str(self.artifacts_dir)
        self.config_path = self.project / "config.json"
        self.config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_reproduces_notebook_baseline(self) -> None:
        result = pipeline.run_pipeline(str(self.config_path))
        for name, path in result["artifacts"].items():
            self.assertTrue(Path(path).is_relative_to(self.project), f"{name}: {path}")
        manifest = result["manifest"]
        inputs = manifest["inputs"]
        self.assertEqual(
            (inputs["features_rows"], inputs["churn_rows"], inputs["joined_rows"]),
            (198578, 200000, 198578),
        )
        self.assertEqual(inputs["unmatched_truth_rows"], 1422)
        self.assertEqual(
            [manifest["splits"][name] for name in ("train", "val", "test")],
            [119146, 39716, 39716],
        )
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        for name, value in baseline["metrics"].items():
            self.assertAlmostEqual(result["metrics"][name], value, places=12, msg=name)
        self.assertEqual(
            [manifest["splits"][name] for name in ("train", "val", "test")],
            baseline["split_rows"],
        )
        if BASELINE_PROBABILITIES.is_file() and BASELINE_INDICES.is_file():
            # Все reload-проверки — только на артефактах из временного каталога.
            config = pipeline.load_config(str(self.config_path))
            frame = pipeline.load_data(config)
            splits = pipeline.preprocess(frame, config)
            np.testing.assert_array_equal(
                splits["test_index"].to_numpy(), np.load(BASELINE_INDICES)
            )
            model = joblib.load(result["artifacts"]["model"])
            positive = list(model.classes_).index(1)
            probabilities = pipeline._predict_proba(model, splits["X_test"])[:, positive]
            np.testing.assert_allclose(
                probabilities, np.load(BASELINE_PROBABILITIES), rtol=0, atol=1e-9
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
