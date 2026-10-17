# CashGuard — выявление нарушений кассовой дисциплины по видео

AI-сервис для хакатона ForteBank. Принимает видео с камеры кассы (.mov), находит нарушения
кассовой дисциплины и возвращает отчёт: вердикт, таймкоды, кадры-доказательства и ссылки на
нормативные акты НБРК на adilet.zan.kz.

- Полностью локальный: YOLOv8 и ffmpeg работают на вашей машине, внешних AI-сервисов и API нет.
- Распознавание лиц не используется: модель видит только «человек» как силуэт, без личности.
- Укладывается в 5 минут на видео: проверено на 20-минутном ролике 720p на 2 ядрах CPU — 224 с.

## Архитектура

```
 Жюри / React UI
      │  POST /api/v1/analyze (multipart .mov, X-API-Key)
      ▼
 FastAPI ── сохраняет файл потоково ── job_id сразу (202) ──► клиент
      │
      ▼  asyncio.Queue (персистентные job-файлы JSON)
 Worker (asyncio.to_thread, не блокирует event loop)
      │
      ├─ ffmpeg: декод + fps-фильтр + масштаб ──► кадры BGR (N к/с)
      ├─ YOLOv8 (yolov8n.pt COCO + ваша кастомная модель) ──► детекции
      ├─ Rule engine (rules.yaml + zones.yaml) ──► эпизоды нарушений
      └─ Evidence: кадры с рамками и зонами ──► JPEG ──► base64
      │
      ▼
 GET /api/v1/jobs/{job_id} ──► processing | done (verdict, violations) | failed
```

**Почему asyncio-очередь, а не Celery/Redis.** Одна машина, одна модель в памяти. Очередь на
`asyncio.Queue` даёт тот же неблокирующий POST без лишних сервисов, которые могут упасть на
защите. Тяжёлый цикл YOLO выполняется в потоке через `asyncio.to_thread`, поэтому сервер
продолжает отвечать на `GET /jobs/{id}` во время обработки. Задачи пишутся в `data/jobs/*.json`:
после перезапуска готовые отчёты доступны, незавершённые задачи ставятся в очередь заново.
Число воркеров задаёт `WORKERS` (для нескольких GPU или мощного CPU).

**Как соблюдается лимит 5 минут.**
1. Дедлайн задачи фиксируется в момент загрузки (`JOB_DEADLINE_SEC=270`), ожидание в очереди
   входит в бюджет.
2. При старте сервер замеряет скорость детектора на этой машине. Частота выборки кадров
   выбирается заранее: длинное видео анализируется реже (от `TARGET_FPS=4` до `MIN_FPS=0.5`).
3. ffmpeg сам снижает частоту и разрешение кадров в отдельном процессе, Python получает только
   нужные кадры.
4. Во время работы скорость пересчитывается каждые 10 кадров: если не успеваем — пропускаем
   часть кадров, если успеваем с запасом — возвращаемся к полной частоте.
5. Если дедлайн всё же наступает, анализ останавливается и отчёт возвращается по
   обработанной части (`meta.coverage`, `meta.stopped_early`).

## Какие нарушения выявляются

Маппинг «класс YOLO → нарушение → норма» лежит в `backend/config/rules.yaml`. Код при добавлении
нового нарушения менять не нужно.

| type | Что детектируется | Норма РК | Модель |
|---|---|---|---|
| `mobile_phone_in_cash_zone` | телефон в кассовой зоне ≥ 2 с | Правила № 231, п. 12 | стоковая COCO |
| `personal_items_in_cash_zone` | сумка, рюкзак, чемодан в кассовой зоне ≥ 5 с | Правила № 231, п. 12 | стоковая COCO |
| `unauthorized_person_in_cash_zone` | людей в кассовой зоне больше 2 (кассир + контролёр) ≥ 3 с | Правила № 14, п. 8; Правила № 231, п. 75 | стоковая COCO |
| `cashier_left_during_service` | клиент у окна, в зоне кассира никого ≥ 10 с | Правила № 231, п. 22 | стоковая COCO, нужна `client_zone` |
| `camera_obstructed` | кадр резко потемнел или потерял резкость ≥ 3 с | Правила № 14, п. 16 пп. 6 | без модели (OpenCV) |
| `cash_left_unattended` | наличные / открытый ящик без кассира ≥ 5 с | Правила № 231, пп. 20, 22 | кастомная (`cash`, `cash_drawer_open`) |
| `cash_zone_door_open` | дверь кассы открыта ≥ 10 с | Правила № 14, п. 8 | кастомная (`door_open`) |

Нормативные акты:
- Правила осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей в
  БВУ… — постановление Правления НБРК от 29.11.2019 № 231:
  https://adilet.zan.kz/rus/docs/V1900019680
- Правила организации охраны и устройства помещений БВУ… — постановление Правления НБРК от
  24.02.2020 № 14: https://adilet.zan.kz/rus/docs/V2000020075

> Перед защитой откройте обе ссылки и сверьте пункты с актуальной редакцией. Часть пунктов
> Правил № 14 (в т. ч. п. 16) имеет отметку о приостановлении до 01.01.2027 с временной редакцией.

Правила, для которых нет зоны или нет модели с нужным классом, пропускаются автоматически.
Список пропущенных правил приходит в `meta.skipped_rules`.

## API для жюри

Все маршруты, кроме `/api/v1/health`, требуют заголовок `X-API-Key` (значение — переменная
окружения `API_KEY`).

```bash
# 1. Отправить видео
curl -s -H "X-API-Key: $API_KEY" -F "file=@video.mov" http://localhost:8000/api/v1/analyze
# {"job_id":"a296db09812043fd981ffd70556a1386","status":"processing"}

# 2. Опрашивать статус
curl -s -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/jobs/a296db09812043fd981ffd70556a1386
```

`status`: `processing` (в очереди или в работе, плюс `progress` от 0 до 1), `done`, `failed`
(плюс `error`). При `done`:

```json
{
  "job_id": "a296db09812043fd981ffd70556a1386",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "unauthorized_person_in_cash_zone",
      "timecode_sec": 8.0,
      "end_sec": 16.0,
      "duration_sec": 8.0,
      "confidence": 0.882,
      "severity": "high",
      "title": "Постороннее лицо в кассовой зоне",
      "description": "В кассовой зоне одновременно находится больше людей, чем допускается… Нормативное основание: Правила охраны и устройства помещений (пост. Правления НБРК от 24.02.2020 № 14), п. 8: … (https://adilet.zan.kz/rus/docs/V2000020075); Правила кассовых операций (пост. Правления НБРК от 29.11.2019 № 231), п. 75: … (https://adilet.zan.kz/rus/docs/V1900019680).",
      "legal_basis": [
        {"short": "Правила охраны и устройства помещений (пост. Правления НБРК от 24.02.2020 № 14)",
         "clause": "п. 8", "summary": "…", "url": "https://adilet.zan.kz/rus/docs/V2000020075", "act": "…"}
      ],
      "evidence_frames": [
        {"timecode_sec": 8.0, "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD..."}
      ]
    }
  ],
  "meta": {"duration_sec": 30.0, "sample_fps": 4.0, "frames_analyzed": 120, "coverage": 1.0,
           "processing_time_sec": 9.34, "models": ["yolov8n.pt"], "zones": ["cash_zone"]}
}
```

Обязательные поля из ТЗ: `verdict` (`no_violation` / `violation`), `violations[].type`,
`timecode_sec`, `description` (с актом РК), `evidence_frames[].image_base64` (JPEG в base64 без
префикса `data:`). Остальные поля — дополнительные.

Необязательные поля формы `POST /analyze`:
- `profile` — имя профиля зон из `backend/config/zones.yaml` (по умолчанию `default`: весь кадр —
  кассовая зона).
- `zones` — JSON с полигонами в долях кадра:
  `{"cash_zone": [[0,0],[0.55,0],[0.55,1],[0,1]], "client_zone": [[0.6,0],[1,0],[1,1],[0.6,1]]}`.

Ответы об ошибках: 401 — нет или неверный ключ, 404 — нет такой задачи, 413 — файл больше
`MAX_UPLOAD_MB`, 415 — не видеоформат, 422 — неверные зоны.

## Запуск локально

Нужны Python 3.10+ и Node.js 20.19+ (для сборки фронтенда, Vite 8). ffmpeg ставится как
pip-пакет `imageio-ffmpeg`, отдельно устанавливать не нужно.

### 1. Бэкенд

```bash
cd backend
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Только CPU: лёгкий torch без CUDA (экономит ~2 ГБ). С NVIDIA GPU пропустите эту строку.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/download_weights.py   # один раз скачивает models/yolov8n.pt, дальше всё офлайн

export API_KEY=my-secret-key         # Windows: set API_KEY=my-secret-key
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Проверка: `curl http://localhost:8000/api/v1/health`. Документация Swagger:
http://localhost:8000/docs.

### 2. Фронтенд (режим разработки)

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, запросы /api проксируются на :8000
```

Введите `X-API-Key` в шапке страницы, перетащите видео, при желании разметьте зоны на кадре и
нажмите «Запустить анализ».

### 3. Один порт для защиты

```bash
cd frontend && npm run build      # собирает frontend/dist
cd ../backend && uvicorn app.main:app --host 0.0.0.0 --port 8000
```

FastAPI сам раздаёт `frontend/dist`: интерфейс на http://localhost:8000, API там же.

### Docker

```bash
API_KEY=my-secret-key docker compose up --build
```

Образ содержит собранный фронтенд и веса модели. После сборки контейнеру сеть не нужна.

### Проверка «как жюри»

```bash
cd backend
python scripts/make_demo_video.py data/demo.mov      # синтетический ролик 30 с из картинок Ultralytics
python scripts/smoke_test.py data/demo.mov --key my-secret-key
# status=done total=10.5s OK
# verdict: violation
#       8.00s  unauthorized_person_in_cash_zone   frames=2
#      16.00s  camera_obstructed                  frames=3

python -m pytest -q                                  # 9 тестов: правила и контракт API
```

## Зоны кадра

Правила работают в зонах, а не по всему кадру: телефон у клиента в зале не нарушение, телефон у
кассира — нарушение. Зоны — полигоны в долях кадра (0..1), поэтому не зависят от разрешения.

- `cash_zone` — рабочее место кассира, кассовый узел.
- `client_zone` — место клиента у кассового окна.

Для каждой камеры заведите профиль в `backend/config/zones.yaml` или нарисуйте зоны в
веб-интерфейсе (режим «Разметить вручную»: клики по кадру видео). Профиль `default` (весь кадр —
кассовая зона) используется для запросов без зон, в том числе от жюри. Если камеры жюри известны
заранее, поменяйте профиль по умолчанию через `ZONE_PROFILE`.

## Обучение своей модели (банковские классы)

Стоковая YOLOv8 знает людей, телефоны и сумки, но не знает наличные, открытый денежный ящик
или дверь кассы. Для правил `cash_left_unattended` и `cash_zone_door_open` нужна дообученная
модель.

1. Нарежьте кадры из обучающих роликов (почти одинаковые кадры отбрасываются):
   ```bash
   python ml/extract_frames.py videos/ dataset_raw/ --every 1.0
   ```
2. Загрузите `dataset_raw/` в Roboflow, разметьте классы из `ml/data.yaml`: `cash`,
   `cash_drawer_open`, `door_open`. Ориентир: 300–500 кадров на класс, кадры с разных камер и
   при разном освещении, плюс кадры без объектов (негативы). Лица не размечаются.
3. Экспортируйте датасет в формате «YOLOv8» и обучите:
   ```bash
   python ml/train.py --data path/to/export/data.yaml --epochs 80 --device 0
   ```
   Веса копируются в `backend/models/cash_custom.pt`.
4. Запустите сервис с двумя моделями, детекции объединяются по имени класса:
   ```bash
   MODEL_PATHS=models/yolov8n.pt,models/cash_custom.pt uvicorn app.main:app --port 8000
   ```
   Две модели — два прохода на кадр. Бюджет времени это учитывает: частота кадров снизится
   автоматически. На GPU запас огромный.

### Как добавить нарушение

Пример: кастомный класс `cash_handover` (передача наличных) вне кассового окна.

1. Добавьте класс в разметку и переобучите модель.
2. Добавьте зону, например `hall_zone` (зал вне окна), в профиль `zones.yaml`.
3. Добавьте правило в `rules.yaml`:
   ```yaml
   - type: cash_handover_outside_window
     title: "Передача наличных вне кассы"
     kind: object_in_zone
     classes: ["cash_handover"]
     zone: hall_zone
     min_conf: 0.5
     min_duration_sec: 1
     severity: high
     description: "Зафиксирована передача наличных вне кассового окна."
     norms:
       - ref: rules_231
         clause: "п. …"          # сверьте пункт на adilet.zan.kz
         summary: "…"
   ```

Типы правил (`kind`): `object_in_zone`, `person_count_exceeds`, `zone_unattended`,
`camera_obstructed`. Параметры описаны в начале `backend/app/pipeline/rules.py`.

## Настройки (переменные окружения)

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `API_KEY` | `change-me` | значение заголовка `X-API-Key` |
| `MODEL_PATHS` | `models/yolov8n.pt` | веса через запятую |
| `DEVICE` | авто | `cpu`, `0` (GPU), `mps` (Apple) |
| `IMGSZ` / `CONF` | `640` / `0.35` | размер входа и порог уверенности YOLO |
| `WORKERS` | `1` | число параллельных обработчиков (у каждого своя модель) |
| `JOB_DEADLINE_SEC` | `270` | бюджет на задачу от момента загрузки |
| `TARGET_FPS` / `MIN_FPS` | `4` / `0.5` | частота анализа кадров |
| `ZONE_PROFILE` | `default` | профиль зон по умолчанию |
| `MAX_UPLOAD_MB` | `2048` | лимит размера файла |
| `KEEP_VIDEOS` | `false` | хранить видео после обработки |
| `DATA_DIR` | `backend/data` | загрузки и JSON задач |

## Ограничения и что улучшить

- «Постороннее лицо» определяется по количеству людей в зоне, а не по личности: распознавание лиц
  запрещено ТЗ. Точнее можно сделать по форменной одежде или бейджу как отдельному классу
  кастомной модели.
- Без кастомной модели правила про наличные, ящик и дверь пропускаются (это видно в
  `meta.skipped_rules`).
- `yolov8n` выбрана ради скорости на CPU. На GPU можно поставить `yolov8s.pt`/`yolov8m.pt`:
  `python scripts/download_weights.py yolov8s.pt` и `MODEL_PATHS=models/yolov8s.pt`.
- Пороги длительности и уверенности в `rules.yaml` подобраны как стартовые. Подстройте их на
  размеченных роликах, чтобы сбалансировать ложные срабатывания и пропуски.

## Структура проекта

```
backend/
  app/
    main.py            маршруты FastAPI, загрузка файла, раздача фронтенда
    auth.py            проверка X-API-Key
    jobs.py            очередь asyncio, воркеры, сохранение задач
    config.py          настройки из переменных окружения
    pipeline/
      video.py         ffmpeg: probe, кадры с нужной частотой
      detector.py      YOLOv8, несколько моделей, офлайн-режим
      rules.py         зоны, временные эпизоды, маппинг на нормы
      evidence.py      отрисовка кадров-доказательств, JPEG → base64
      analyzer.py      оркестрация и бюджет времени
  config/rules.yaml    нарушения и нормативные акты РК
  config/zones.yaml    профили зон камер
  scripts/             загрузка весов, демо-видео, smoke-тест
  tests/               pytest
frontend/src/
  App.jsx              шаги: видео → зоны → анализ → отчёт, опрос статуса
  components/          UploadDropzone, ZoneEditor, Report
  api.js               клиент API (XHR с прогрессом загрузки)
ml/                    нарезка кадров, шаблон датасета, обучение
Dockerfile, docker-compose.yml
```
