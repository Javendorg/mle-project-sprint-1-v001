# Оценка стоимости квартир — проект MLE, спринт 1

MVP-пайплайн собирает и очищает данные о квартирах, обучает CatBoost-регрессор для предсказания `price` и сохраняет модель в S3 через DVC.

## Данные и результаты Airflow

Персональная база данных: `mle_20261110_b5e9872894d9`, схема `public`. Исходные таблицы `flats` и `buildings` доступны через Airflow connection `destination_db`;

| DAG ID | Файл | Функция DAG | Результат |
|---|---|---|---|
| `build_flats_features` | `part1_airflow/dags/build_flats_features.py` | `build_flats_features_dag` | `public.flats_features` |
| `clean_flats_features` | `part1_airflow/dags/clean_flats_features.py` | `clean_flats_features_dag` | `public.flats_cleaned` |

Оба DAG содержат задачи `create_table`, `extract`, `transform` и `load`; промежуточные данные передаются через staging-таблицы в БД, а не через XCom. Общие функции работы с БД находятся в `part1_airflow/plugins/db_utils.py`. Проверки качества и очистка реализованы в `part1_airflow/plugins/cleaning.py`: `diagnose_data` и `clean_flats`.

Ноутбук анализа и очистки: `part1_airflow/notebooks/data_analysis.ipynb`. Он ожидает предварительно выгруженный CSV таблицы `public.flats_features` по пути `part2_dvc/data/raw/flats_features.csv`; сам этот исходный CSV в Git не хранится.

Последняя проверка БД: `flats_features` — 141 362 строки; `flats_cleaned` — 138 534 строки, все `flat_id` уникальны.

## DVC-пайплайн

Рабочая директория: `part2_dvc/`.

| Этап | Файл |
|---|---|
| Разделение train/test и исключение ID/утечки | `part2_dvc/scripts/prepare.py` |
| Подготовка признаков и общие функции | `part2_dvc/scripts/common.py` |
| Обучение CatBoost | `part2_dvc/scripts/train.py` |
| Оценка на отложенной выборке | `part2_dvc/scripts/evaluate.py` |
| Экспорт таблицы из PostgreSQL (утилита) | `part2_dvc/scripts/export_data.py` |
| Конфигурация стадий | `part2_dvc/dvc.yaml` |
| Параметры разбиения и модели | `part2_dvc/params.yaml` |
| Зафиксированные входы/выходы стадий | `part2_dvc/dvc.lock` |

Стадии `dvc.yaml`: `prepare`, `train`, `evaluate`. Входной датасет — `part2_dvc/data/raw/flats_cleaned.csv`, его DVC-указатель хранится в `part2_dvc/data/raw/flats_cleaned.csv.dvc`. Не добавляйте сам CSV или модель в Git: их версии хранятся в DVC remote.

Ноутбук предварительного анализа модели: `part2_dvc/notebooks/model_experiments.ipynb`. Результаты оценки: `part2_dvc/cv_results/metrics.json`; MLflow-метаданные экспериментов: `part2_dvc/mlruns/`.

## Модель и метрики

Финальная модель: `models/catboost.joblib`, сохранена в DVC remote. На тестовой выборке из 27 707 объектов получены:

| Метрика | Значение |
|---|---:|
| MAE | 3 005 796 |
| RMSE | 5 989 618 |
| MAPE | 17,91% |
| R² | 0,8985 |

## S3 и воспроизведение

Бакет: `mle-20261110-b5e9872894d9`. DVC remote `storage` использует endpoint `https://storage.yandexcloud.net`. Ключи доступа не хранятся в Git: локальные настройки находятся в игнорируемом `.dvc/config.local`.

После настройки локальных S3 credentials и установки зависимостей выполните из `part2_dvc`:

```bash
python -m pip install -r requirements.txt
dvc pull
dvc repro
dvc metrics show
dvc push -r storage
```

Для Airflow требуется настроенный connection `destination_db`; имена connections и секреты в репозитории не хранятся.
