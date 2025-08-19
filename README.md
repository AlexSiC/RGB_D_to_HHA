# Rgb_D_to_HHA

Короткое описание: Проект для обработки RGB/Depth данных и преобразования глубины в представление HHA, с поддержкой разметки, inpainting и аугментаций.

## Установка

Рекомендуется использовать виртуальное окружение.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Использование

Основной конвейер запускается через `main.py` с указанием пути к конфигурации.

```bash
python main.py --config ./configs/config_example.yaml
```

### Переопределение параметров через CLI

- Удобные флаги:
  - `--augmentation-seed 43` — заменить `augmentation.seed` на 43
  - `--augmentation-enabled true|false` — включить/выключить аугментации
  - `--inpainting-method rbf|linear_nearest|none` — метод восстановления глубины
  - `--train-export true|false` — включить/выключить экспорт в `data/train`
  - `--processed-export true|false` — включить/выключить экспорт в `processed`

- Произвольные поля конфига можно менять через повторяемый аргумент `--set`:

```bash
python main.py --config ./configs/config_example.yaml \
  --set augmentation.seed=41 \
  --set augmentation.rotate_limit=20 \
  --set outputs.save_hha_u8_in_train=true
```

### Структура результатов

- Все артефакты сохраняются в подкаталог `paths.processed_dir/run_YYYYMMDD_HHMMSS`
- Базовый запуск (без аугментаций) сохраняется в корень `run_*`
- Для нескольких `augmentation.seeds` создаются подпапки `seed_<value>`
- При включённом экспорте `outputs.enable_train_export` данные пишутся в `outputs.train_dir/{images,masks}`

### Отдельные утилиты

Скрипты для изолированного запуска стадий находятся в `scripts/`:

- `run_inpainting.py`, `run_annotations.py`, `run_hha.py`, `run_augmentation.py`, `run_ordered.py`

## Конфигурация

Пример находится в `configs/config_example.yaml`. Ниже описание ключевых полей:

```yaml
paths:
  raw_dir: ./data/raw            # Входные данные: rgb/, depth/, annotations/
  processed_dir: ./data/processed # Выходные артефакты по запускам

inpainting:
  method: linear_nearest         # linear_nearest | rbf | none

augmentation:
  enabled: true                  # Глобальный переключатель аугментаций
  seed: 42                       # Базовый сид для детерминизма
  seeds: [41, 42, 43, 44, 45]    # Список сидов для вариативных запусков
  horizontal_flip_prob: 0.5
  random_scale_limit: 0.1
  crop_size: [800, 600]          # [width, height]
  rotate_limit: 15
  pad_if_needed: true

cameras:
  rgb_camera_matrix:             # Матрица внутренних параметров RGB-камеры
    fx: 692.683
    fy: 692.738
    cx: 408.508
    cy: 295.696
  depth_camera_matrix:           # Матрица внутренних параметров depth-камеры
    fx: 712.929
    fy: 712.808
    cx: 396.903
    cy: 295.821

hha:
  bottom_band_frac: 0.25         # Доля нижней части кадра для оценки пола
  side_band_frac: 0.05           # Боковые полосы (доля ширины)
  center_exclude_width_frac: 0.40 # Исключение центральной зоны
  ransac_thresh_cm: 1.5
  min_inlier_ratio: 0.12
  ransac_seed: 42
  gravity_init: -z               # Начальная ориентация гравитации

outputs:
  save_hha_channels_jet: true    # Доп. визуализации каналов HHA
  enable_train_export: true      # Экспорт пары (image, mask) в каталог обучающих данных
  train_dir: data/train
  save_hha_u8_in_train: false    # Использовать uint8 визуализации вместо uint16
  enable_processed_export: false # Полный набор артефактов в run_* (depth, hha, rgb, masks)
```

## Формат входных данных

- `paths.raw_dir/rgb/*.jpg` — RGB изображения вида `rgb_frame_<id>_png.rf.<hash>.jpg`
- `paths.raw_dir/depth/depth_data_<id>.txt` или `depth_data_frame_<id>.txt` — глубина в мм
- `paths.raw_dir/annotations/rgb_frame_<id>_png.rf.*.txt` — полигоны в нормированных координатах

## Логи и отчётность

- Логи пишутся в `logs/pipeline.log`
- Список упавших файлов — `logs/failed_files.txt`
- В консоль выводится прогресс-бар и финальный отчёт


