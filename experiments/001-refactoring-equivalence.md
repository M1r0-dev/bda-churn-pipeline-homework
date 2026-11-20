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

## 2026-11-20 — final result
- commit: 49314d8068005fb70fddf411741f77e72bb81651 (dirty: дописываются только документация и receipts); pipeline_sha256=707aeec7b0f7a8d4cea1c35dce511f39f71dda138eb15522a2505e0e7e465aea, config/data/seed/hardware неизменны.
- Lead command: python -m unittest discover -s tests -v →61tests OK, exit0; compileall и pip check exit0. Затем два **последовательных** CLI run на полных входах exit0,5/5 содержательных SHA одинаковы, лог содержит только текущий запуск, файлов ровно6. После исправления изоляции suite не пишет в рабочую artifacts.
- metrics (verification.json, обе runs): val_auc=0.9689088149805355, val_f1=0.6778992651012726, val_precision=0.5368346344925479, val_recall=0.9195234621930465; test_auc=0.9695817244936411, test_f1=0.6883493386483058, test_precision=0.5485128501299451, test_recall=0.9238813229571985. F1 delta от baseline=0, seeds1. Все39716testprobabilities после загрузки совпали с исходным baseline при atol1e-12; reload max|delta|=0.
- Docker verifier: final build exit0,2nonroot bind runs exit0,5/5hashes stable/log overwrite;61tests в final image OK, pipcheck0. Lead независимо выполнил docker build и nonroot run в отдельный hostmount: exit0; все5SHA совпали с local и обоими verifier outputs;100probabilities Docker-модели после загрузки на host совпали atol1e-12. Docker/host CPython3.11.18 и package pins совпали. Подробный durable receipt и SHA — verification.json.
- Model SHA323c15d9832ee7652685f4bff9830f29bfaaeef06b0fd20e92ab94023d5c8471; metricsSHAd0f2998ac776dfdd7498387cbdd1ce959f75cac0af1c75576d25d159800a765a; manifestSHA7546b5e3afb3dbf4fe297448f346ce10a3516e0c5826c4e6f1b7855a45170cdf; featuresSHAee0c371a601c8b5741d15c182164899eb175a29f468448c38aea02f7b2c87914; splitsSHA9b07d8aff0412156a3f4f13d1aaa00cced3f88644daaf2ad8ff808fbc80539bd.
- Independent review: Standards two Critical initially, both confirmed/fixed/regression-tested; fresh Standards re-review PASS and Spec core/tests PASS, без открытых findings в этих scopes. Документация будет проверена отдельно.
- decision: keep refactoring (эквивалентность и инженерная воспроизводимость подтверждены, улучшение качества не заявляется).
- next: закончить документацию, пройти публикационный аудит и опубликовать репозиторий.

## 2026-11-20 — publication result
- commit: 1a938395c8adc6470ccaf54a5e7b1aa577fb65de (clean functional snapshot published to main); source/config/data SHA неизменны. После этого добавлена только запись публикации, не модель/настройки/данные.
- final review: свежие независимые Standards и Spec по документации PASS, без открытых замечаний. Lead independently проверил60doclinks, точный HTML SHA, strict provenance check и browser-check (exit0); визуальная perceptual оценка не заявляется.
- prepublication suite:61tests/22.396s OK, exit0; pip check0. Audit:25trackedfiles,12,293,495bytes на этом снимке; credential signatures0, generated/private paths0, файлов>20MB нет; авторский email уже присутствует в существующих публичных GitHub-коммитах (значение не публикуется в отчёте).
- publication: https://github.com/M1r0-dev/bda-churn-pipeline-homework ; gh repo create --public exit0; ff merge и git push -u origin main exit0, безforce. GitHub REST подтвердил public/defaultmain, тот же HEAD и SHA всех25remote blobs. Анонимный web GET репозитория вернул HTTP200.
- clean local Git clone:61tests/21.063s OK; actual clone с публичного GitHub HTTPS URL:61tests/21.310s OK (exit0), commit1a938395c8adc6470ccaf54a5e7b1aa577fb65de. Оба запуска без исходного локального notebook и без локальных scratch baseline arrays. Использовано то же pinned Python окружение; чистая установка независимо проверена Docker-сборкой.
- environment failures: отдельные GitHub REST/GraphQL GET однажды завершались TLS/i-o timeout; повторные ограниченные чтения прошли. Никакой недоступный GET не выдавался за успешную проверку; push и remote tree проверены фактически.
- decision: keep, инженерная цель выполнена для сохранённой ретроспективной постановки; next: при необходимости будущего прогноза сначала определить новый целевой временной интервал, не использовать эти метрики как future-churn оценку.
