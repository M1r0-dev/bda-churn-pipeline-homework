# 001 — Эквивалентность рефакторинга и повторный запуск

## 2026-11-20 — start
- commit: 53bdc9c1cdea87d8a9336c312ed6b9177a67863f (dirty: параллельно готовятся новые файлы конвейера и Docker).
- config: unknown (финальный config.json ещё пишет ML-агент; перед первым запуском полный JSON и SHA будут добавлены отдельной записью).
- data: TDG S, generator1.0.0, seed20260827; 198578 строк признаков; SHA-256 features=769b2b7627315f62fd85cfc0f05bcc40f4543b7adab8dd5ba01248a392b1b222, truth=20e8d860b8a21dcb71de11c5ddff8eb771f196749d6785fbc5c2db095b246058.
- sampling: отсутствует, все исходные строки признаков; метка churned из one-to-one join. Split: stratified60/20/20, исходный порядок, seed42.
- seed(s): 42; два одинаковых запуска не являются двумя seed.
- hardware: CPU linux/amd64, четыре потока RF; точная модель CPU/память пока unknown (будет добавлена перед запуском).
- hypothesis: полный сохраняемый Pipeline воспроизводит исходный notebook, исключая необходимость отдельно загружать scaler; детерминированные выходы перезаписываются без накопления.
- metric: положительный F1 при threshold0.5; приёмка по совпадению baseline с допуском1e-8 и предсказаний после загрузки; улучшение качества не заявляется.
- logs/tracker: ожидается artifacts/pipeline.log и verification.json; результатов пока нет.
- decision: change-one-thing (инженерный рефакторинг, без изменения задачи и подбора по test).
- next: закончить ML-код и зафиксировать фактический полный config перед запуском.

## 2026-11-20 — baseline start
- commit: 53bdc9c1cdea87d8a9336c312ed6b9177a67863f (dirty).
- hardware: Intel Core Ultra7 155H, x86_64, 22 logical CPU, 30GiB RAM; RF n_jobs4.
- environment: CPython3.11.18, зависимости из requirements.txt; pip check без ошибок. Первое создание venv через uv не включало pip (exit1 на проверке pip); исправлено python -m ensurepip, pip install -r requirements.txt и pip check (exit0). Данные и обучение от этого не менялись.
- baseline: фактический запуск архивных оригинальных функций, не копирование notebook outputs. Полная эффективная конфигурация baseline:
```json
{
  "data": {"features_path": "data/churn_features.parquet", "churn_path": "data/churn_truth.csv"},
  "preprocessing": {"test_size": 0.4, "val_size": 0.5, "random_state": 42},
  "model": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 5, "class_weight": "balanced", "random_state": 42, "n_jobs": 4},
  "evaluation": {"threshold": 0.5}
}
```
- code: provenance/original_functions.py (committed bytes 53bdc9c); seed42; полные входы с SHA из начальной записи; метрики val/test без подбора параметров.
- outputs: .scratch/baseline.json и .scratch/baseline_test_probabilities.npy (локальные проверочные данные, не публичные артефакты).
- decision: keep (проверяем эквивалентность; baseline не использован для выбора модели).
- next: выполнить baseline и затем сравнить с рефакторингом.

## 2026-11-20 — baseline result
- commit: 53bdc9c1cdea87d8a9336c312ed6b9177a67863f (dirty), тот же baseline config и SHA данных.
- command: .venv/bin/python .scratch/baseline.py; exit0; baseline — исходные функции, извлечённые AST без модификации тел функций.
- metrics (источник .scratch/baseline.json): val_auc=0.9689088149805355, val_f1=0.6778992651012726, val_precision=0.5368346344925479, val_recall=0.9195234621930465; test_auc=0.9695817244936411, test_f1=0.6883493386483058, test_precision=0.5485128501299451, test_recall=0.9238813229571985.
- split_rows: train119146, val39716, test39716; train_time_sec=4.92279406900343; seeds1.
- baseline Δ: 0 (сам baseline); опубликованные значения notebook фактически воспроизведены, это не выбор модели по test.
- decision: keep; next: сравнить рефакторинг с baseline и дважды запустить единый CLI.

## 2026-11-20 — sanity result и неудачный probe
- commit: 53bdc9c1cdea87d8a9336c312ed6b9177a67863f (dirty); конфиг и данные baseline выше, модель не перенастраивалась.
- Первый probe: перемешан только train y (default_rng42), val y исходные. validation_AUC=0.6365931677095619; assert ожидания0.45–0.55 завершился exit1. Это **не** замалчивается как успешная проверка. Один такой probe не даёт калиброванного null для сильно зависимых от признаков реальных меток; полноценное распределение перестановок не оценивалось. Отсутствие временного leakage не доказано и не заявляется.
- Диагностика: реально изменены22126 train labels, ключи уникальны, пересечение split=0. Повторён тот же train shuffle, только val y независимо перемешан seed43: AUC=0.49850400661005273 (exit0). Проверяется случайный null, не качество/выбор модели.
- majority_negative_validation_f1=0.0; constant_score_validation_auc=0.5.
- source: .scratch/sanity.json; debug protocol .scratch/sanity-investigation.md. Эти sanity не меняют принятый config.
- decision: keep инженерный рефакторинг, не делать исследовательских выводов о качестве будущего прогноза.
- next: проверить рефакторинг на двух последовательных запусках.

## 2026-11-20 — refactored pipeline start
- commit: 53bdc9c1cdea87d8a9336c312ed6b9177a67863f (dirty: новый ML-код ещё не закоммичен); pipeline_sha256=83bf438a7159c20ec93241241181595aa55aedd6beeb9d0ec22db96995cd9fdb; config_sha256=b89009d9af08704e10ee6a305a65a2d8741741c66182b562d5f776d09e2a0f6b.
- environment/hardware/data: те же CPython3.11.18, Intel Core Ultra7 155H, CPU4 потока, полные данные с SHA выше; без sampling. Split seed42; модель seed42.
- full config (effective user JSON; приватные служебные _base_dir/_config_path добавляются в памяти, не влияют на обучение):
```json
{
  "data": {
    "features_path": "data/churn_features.parquet", "churn_path": "data/churn_truth.csv",
    "key_column": "subscriber_key", "target_column": "churned", "excluded_columns": ["churn_day_index"],
    "feature_columns": ["traffic_gb_30d", "traffic_change_ratio", "voice_minutes_30d", "events_30d", "active_days_30d", "recency_days", "roaming_share_30d", "tickets_30d", "unresolved_share_30d", "network_error_rate_30d"]
  },
  "preprocessing": {"test_size": 0.4, "val_size": 0.5, "random_state": 42, "fill_value": 0.0},
  "model": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 5, "class_weight": "balanced", "random_state": 42, "n_jobs": 4},
  "evaluation": {"threshold": 0.5},
  "artifacts": {"dir": "artifacts", "model_name": "model.joblib", "features_name": "features.json", "metrics_name": "metrics.json", "manifest_name": "manifest.json", "splits_name": "splits.csv"},
  "logging": {"level": "INFO", "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s", "log_name": "pipeline.log"}
}
```
- criterion: F1 baseline delta≤1e-8; вероятности loaded Pipeline vs baseline atol1e-12; одинаковые SHA пяти выходных артефактов между двумя CLI запусками; log mode=w, без накопления файлов.
- decision: keep fixed config, без подбора по test; next: Lead запускает .scratch/verify_local.py и полный unittest.

## 2026-11-20 — review fixes / final acceptance start
- commit: 305811fafc035ffde14bcfabbf327ac8c0764f79 (dirty fixes); pipeline_sha256=707aeec7b0f7a8d4cea1c35dce511f39f71dda138eb15522a2505e0e7e465aea. Config/data/seed/hardware не менялись; полный config выше, SHA=b89009d9af08704e10ee6a305a65a2d8741741c66182b562d5f776d09e2a0f6b.
- Первый Lead suite59 и два CLI прошли, но real-dataset test писал в рабочую artifacts. Исправлена изоляция теста: только TemporaryDirectory, и последовательный CLI повторён после окончания suite. Результаты того proof исторические; финальный proof будет заново после review fixes.
- Независимый Standards review нашёл два edge-case дефекта. Оба воспроизведены regression tests (оба red exit1): nullable Float64/Int64 с pd.NA падали в passthrough selector; log hardlink к входному CSV позволял truncate input. Исправлены кастом НЕ классом, а стандартным FunctionTransformer(DataFrame.astype,float64) внутри ColumnTransformer и проверкой Path.samefile до открытия логов. Оба targeted tests после исправления green exit0.
- Public baseline_metrics.json содержит реально воспроизведённые значения исходных функций; проверка метрик теперь обязательна и на fresh clone, не зависит от локального .scratch.
- Docker historical build был успешен, но запуск uid1000 выявил COPY permissions0600; verifier исправил COPY --chmod=644, historical два запуска exit0. Финальная сборка/запуск будут повторены с исправленным ML-кодом; старый образ не выдаётся за финальную проверку.
- decision: keep fixed original model config; next: full suite и два последовательных CLI на исправленном коде, затем final Docker rebuild и повторный review.
