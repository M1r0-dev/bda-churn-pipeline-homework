# Archived original functions extracted without modification from the local notebook.
# Historical context only; run ../pipeline.py instead.

def load_data(data_path, churn_path):
    """Объединить признаки с меткой без служебных полей эталона."""
    features = pd.read_parquet(data_path)
    truth = pd.read_csv(churn_path)
    if 'subscriber_key' not in features.columns:
        raise ValueError('В признаках нет subscriber_key')
    if not {'subscriber_key', 'churned'}.issubset(truth.columns):
        raise ValueError('В эталоне нужны subscriber_key и churned')
    if features['subscriber_key'].isna().any() or truth['subscriber_key'].isna().any():
        raise ValueError('subscriber_key не должен содержать пропуски')
    # churn_day_index раскрывает ответ; его нельзя включать в признаки.
    features = features.drop(columns=['churned', 'churn_day_index'], errors='ignore')
    df_model = features.merge(
        truth[['subscriber_key', 'churned']], on='subscriber_key',
        how='inner', validate='one_to_one'
    )
    if df_model.empty:
        raise ValueError('Нет общих абонентов в признаках и эталоне')
    if df_model['churned'].isna().any() or set(df_model['churned'].unique()) != {0, 1}:
        raise ValueError('churned должна содержать оба класса: 0/1 или False/True')
    df_model['churned'] = df_model['churned'].astype(int)
    logger.info('Загружено %d строк; доля оттока %.4f',
                len(df_model), df_model['churned'].mean())
    return df_model

def preprocess(df_model, seed=42, test_size=0.4, val_size=0.5):
    """test_size — доля отложенных данных, val_size — доля val внутри них."""
    X = df_model.drop(columns=['subscriber_key', 'churned', 'churn_day_index'],
                      errors='ignore').select_dtypes(include=[np.number])
    feature_cols = X.columns.tolist()
    if not feature_cols:
        raise ValueError('Нет числовых признаков для обучения')
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_model['churned']
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=1 - val_size,
        stratify=y_holdout, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    logger.info('Разбиение: train=%d, val=%d, test=%d; признаков=%d',
                len(y_train), len(y_val), len(y_test), len(feature_cols))
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler

def train_model(X_train, y_train, params):
    model_params = dict(n_estimators=100, max_depth=10,
                        class_weight='balanced', random_state=SEED, n_jobs=4)
    model_params.update(params)
    model = RandomForestClassifier(**model_params)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - started
    logger.info('RandomForest обучен за %.2f сек', train_time)
    return model, train_time

def evaluate(model, X_val, y_val, X_test, y_test):
    metrics = {}
    positive_class = list(model.classes_).index(1)
    # Фиксированный порог: тестовая выборка не используется для его выбора.
    for name, X, y in [('val', X_val, y_val), ('test', X_test, y_test)]:
        probabilities = model.predict_proba(X)[:, positive_class]
        predictions = (probabilities >= 0.5).astype(int)
        metrics[f'{name}_auc'] = float(roc_auc_score(y, probabilities))
        metrics[f'{name}_f1'] = float(f1_score(y, predictions, zero_division=0))
        metrics[f'{name}_precision'] = float(precision_score(y, predictions, zero_division=0))
        metrics[f'{name}_recall'] = float(recall_score(y, predictions, zero_division=0))
    logger.info('Метрики: %s', metrics)
    return metrics

def save_artifacts(model, scaler, feature_cols, metrics, config):
    artifacts = config['artifacts']
    model_dir = artifacts['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    for name, value, is_binary in [
        ('model_name', model, True), ('scaler_name', scaler, True),
        ('features_name', feature_cols, False), ('metrics_name', metrics, False),
    ]:
        path = os.path.join(model_dir, artifacts[name])
        if os.path.exists(path):
            os.remove(path)
            logger.info('Удалён старый артефакт: %s', path)
        if is_binary:
            joblib.dump(value, path)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        logger.info('Сохранён артефакт: %s', path)
    print(f'Артефакты сохранены идемпотентно в {model_dir}')

def configure_logging(config, project_dir):
    # force=True применяет настройки и при повторном запуске ячейки.
    logging.basicConfig(
        level=getattr(logging, config['logging']['level'].upper()),
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(os.path.join(project_dir, 'pipeline.log'),
                                mode='w', encoding='utf-8'),
            logging.StreamHandler(),
        ],
        force=True,
    )

def load_data_safe(data_path, churn_path):
    try:
        for path in (data_path, churn_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Файл данных не найден: {path}')
        return load_data(data_path, churn_path)
    except Exception:
        logger.exception('Ошибка загрузки данных')
        raise

def run_pipeline(config_path):
    with open(config_path, encoding='utf-8') as f:
        pipeline_config = json.load(f)
    configure_logging(pipeline_config, os.path.dirname(os.path.abspath(config_path)))
    try:
        logger.info('Шаг 1/5: загрузка данных')
        df_model = load_data_safe(pipeline_config['data']['features_path'],
                                  pipeline_config['data']['churn_path'])
        logger.info('Шаг 2/5: предобработка')
        prep = pipeline_config['preprocessing']
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler = preprocess(
            df_model, seed=prep['random_state'],
            test_size=prep['test_size'], val_size=prep['val_size']
        )
        logger.info('Шаг 3/5: обучение')
        model, train_time = train_model(X_train, y_train, pipeline_config['model'])
        logger.info('Шаг 4/5: оценка качества')
        metrics = evaluate(model, X_val, y_val, X_test, y_test)
        metrics['train_time_sec'] = float(train_time)
        logger.info('Шаг 5/5: сохранение артефактов')
        save_artifacts(model, scaler, feature_cols, metrics, pipeline_config)
        logger.info('Конвейер успешно завершён')
        return model, metrics
    except Exception:
        logger.exception('Конвейер завершился с ошибкой')
        raise
