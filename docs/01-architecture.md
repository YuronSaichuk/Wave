# 01. Загальна архітектура

## Компоненти

```
┌─────────────────────────────────────────────────────────────┐
│  wave-ui (Angular)                                          │
│  бібліотека · конструктор сету · плеєр · візуалізація       │
└───────────────┬─────────────────────────┬───────────────────┘
                │ REST                    │ SignalR (WS)
                ▼                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Wave.Api (ASP.NET Core)                                    │
│  · CRUD бібліотеки, сетів, світлових пресетів               │
│  · оркестрація завдань аналізу/рендеру                      │
│  · роздача аудіо (Range requests)                           │
│  · транспорт відтворення (play/pause/seek) як єдиний clock   │
└──────┬────────────────────────┬──────────────────┬──────────┘
       │ HTTP                   │ SQL              │ in-proc
       ▼                        ▼                  ▼
┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐
│ wave-core        │  │ PostgreSQL       │  │ Wave.Lighting   │
│ (Python/FastAPI) │  │ + pgvector       │  │ cue engine      │
│                  │  │                  │  │ + драйвери      │
│ · analyze        │  │ tracks           │  └────────┬────────┘
│ · similar        │  │ track_features   │           │ UDP
│ · plan_mix       │  │ sets / set_items │           ▼
│ · render_mix     │  │ light_presets    │   WLED / Art-Net /
│ · render_lights  │  └──────────────────┘   віртуальний емулятор
└────────┬─────────┘
         │ read/write
         ▼
   data/library/*.mp3
   data/cache/<track_id>.npz     покадрові фічі
   data/mixes/<set_id>.wav       відрендерений мікс
   data/mixes/<set_id>.lights    світлова доріжка
```

## Чому такий поділ

Кордон між Python і .NET проходить рівно по кордону "важка математика"
vs "мережа й стан". Python нічого не знає про HTTP-клієнтів, сесії,
SignalR — він приймає шляхи до файлів і повертає JSON. .NET нічого
не знає про librosa — він оркеструє й віддає.

**Python↔.NET — просто HTTP.** RabbitMQ виглядає спокусливо (він у тебе
в руках щодня), але для пет-проекту він додає інфраструктуру без
виграшу. Черга стане потрібною, коли захочеш аналізувати бібліотеку
в кілька тисяч треків паралельними воркерами — тоді M1 переїде на
RabbitMQ за півдня, бо контракт уже буде "одне завдання = один трек".

Аналіз одного треку — 5–30 секунд. Тому:

- `POST /analyze` у Python приймає завдання і **повертає `202` з job_id**,
  а не тримає з'єднання. Прогрес пише в БД, .NET опитує або слухає.
- Рендер міксу на 60 хвилин — це хвилини роботи. Той самий патерн.

## Лейаут репозиторію

```
Wave/
├─ docs/
│  ├─ 01-architecture.md
│  ├─ 02-audio-analysis.md
│  ├─ 03-similarity-and-mixing.md
│  ├─ 04-lighting.md
│  ├─ 05-roadmap.md
│  └─ tasks/
│     ├─ M0-skeleton.md
│     └─ M1-analysis.md
├─ src/
│  ├─ wave-core/                 Python
│  │  ├─ wave_core/
│  │  │  ├─ analysis/            STFT, фічі, сегментація
│  │  │  ├─ similarity/          скоринг, Camelot, побудова сету
│  │  │  ├─ mixing/              точки переходу, crossfade, рендер
│  │  │  ├─ lighting/            генерація світлової доріжки
│  │  │  ├─ storage/             доступ до Postgres і кешу
│  │  │  ├─ api.py               FastAPI
│  │  │  └─ cli.py               те саме без HTTP — для розробки
│  │  ├─ tests/
│  │  └─ pyproject.toml
│  ├─ Wave.Api/                  ASP.NET Core
│  ├─ Wave.Lighting/             class lib: драйвери + cue engine
│  ├─ Wave.Domain/               моделі, спільні з Api та Lighting
│  └─ wave-ui/                   Angular
├─ data/                         (в .gitignore)
│  ├─ library/
│  ├─ cache/
│  └─ mixes/
├─ docker-compose.yml
└─ .gitignore
```

`cli.py` — не другорядна деталь. Уся розробка ядра йде через нього:
`wave analyze ./data/library`, `wave similar <id>`, `wave mix <id> <id>`.
FastAPI — тонка обгортка над тими самими функціями. Це дозволяє
розвивати ядро, поки .NET-частини ще не існує.

## Схема БД

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Трек: файл + метадані з тегів
CREATE TABLE tracks (
    id              uuid PRIMARY KEY,
    file_path       text NOT NULL UNIQUE,
    file_hash       text NOT NULL,          -- sha256, щоб не аналізувати двічі
    title           text,
    artist          text,
    album           text,
    duration_sec    real,
    sample_rate     int,
    added_at        timestamptz NOT NULL DEFAULT now(),
    analyzed_at     timestamptz,
    analysis_version int                    -- бампимо при зміні алгоритму
);

-- Результат аналізу: одне до одного з треком
CREATE TABLE track_features (
    track_id        uuid PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,

    -- ритм
    bpm             real,
    bpm_confidence  real,
    beat_grid_path  text,        -- у .npz, бо це тисячі значень
    first_beat_sec  real,

    -- гармонія
    key_pitch       smallint,    -- 0=C .. 11=B
    key_mode        smallint,    -- 0=minor, 1=major
    key_confidence  real,
    camelot         text,        -- '8A', '8B' — денормалізовано для зручності

    -- гучність
    lufs_integrated real,
    lufs_range      real,
    true_peak_db    real,

    -- агреговані спектральні (mean+std по треку)
    centroid_mean   real, centroid_std real,
    rolloff_mean    real, rolloff_std  real,
    flatness_mean   real, flatness_std real,
    bandwidth_mean  real, bandwidth_std real,
    zcr_mean        real, zcr_std      real,

    -- вектори для пошуку
    timbre_vec      vector(40),   -- MFCC 20 × (mean, std), нормалізовано
    chroma_vec      vector(12),   -- усереднена хрома, для гармонійних сусідів
    band_vec        vector(8),    -- середня енергія по 8 лог-смугах

    frames_path     text NOT NULL, -- data/cache/<track_id>.npz
    sections        jsonb          -- [{start, end, label, energy}, ...]
);

CREATE INDEX ON track_features USING hnsw (timbre_vec vector_cosine_ops);
CREATE INDEX ON track_features (bpm);
CREATE INDEX ON track_features (camelot);

-- Сет = впорядкований список треків із точками переходу
CREATE TABLE sets (
    id           uuid PRIMARY KEY,
    name         text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    rendered_at  timestamptz,
    audio_path   text,
    lights_path  text,
    total_sec    real
);

CREATE TABLE set_items (
    set_id           uuid REFERENCES sets(id) ON DELETE CASCADE,
    position         int NOT NULL,
    track_id         uuid REFERENCES tracks(id),
    -- параметри переходу З цього треку в наступний
    transition       jsonb,   -- {type, out_at, in_at, bars, bpm_target, eq}
    PRIMARY KEY (set_id, position)
);

-- Світловий пресет: як фічі мапляться в кольори
CREATE TABLE light_presets (
    id       uuid PRIMARY KEY,
    name     text NOT NULL,
    fixtures jsonb NOT NULL,  -- розкладка приладів
    rules    jsonb NOT NULL   -- правила band→color, beat→flash тощо
);
```

`analysis_version` — не бюрократія. Ти точно кілька разів переробиш
набір фіч, і без цього поля не зрозумієш, які треки треба
переаналізувати, а які ні.

## Ключові контракти API

### Python (wave-core, внутрішній)

```
POST /analyze          {file_path}         → 202 {job_id}
GET  /jobs/{id}                            → {status, progress, error}
POST /similar          {track_id, filters, limit}
                                           → [{track_id, score, breakdown}]
POST /plan-set         {seed_track_id, length_min, curve}
                                           → {items: [...], transitions: [...]}
POST /render-mix       {set_id}            → 202 {job_id}
POST /render-lights    {set_id, preset_id} → 202 {job_id}
```

### .NET (Wave.Api, публічний)

```
GET    /api/tracks?q=&bpm_min=&bpm_max=&key=
POST   /api/tracks/scan            запустити скан папки
GET    /api/tracks/{id}/similar
GET    /api/tracks/{id}/waveform   даунсемплений пік-масив для UI

GET    /api/sets/{id}
POST   /api/sets                   створити вручну
POST   /api/sets/auto              згенерувати з seed-треку
POST   /api/sets/{id}/render

GET    /api/lighting/drivers
POST   /api/lighting/fixtures
POST   /api/lighting/test          статичний колір, перевірка заліза
POST   /api/shows/{setId}/start
POST   /api/shows/{setId}/stop

hub    /hub/transport              позиція відтворення, стан
hub    /hub/lights                 кадри для віртуального емулятора
```

## Важливі архітектурні рішення

**Годинник один.** Позиція відтворення живе у Wave.Api, і аудіо, і світло
підписані на неї. Не буде ситуації, коли світло рахує свій час, а плеєр
свій, і за 20 хвилин вони розійшлися на секунду.

**Аудіо грає браузер, не сервер.** Мікс уже відрендерено у файл — браузер
просто програє його через `<audio>`. Сервер знає позицію, бо UI шле її
у SignalR ~10 разів на секунду. Мінус — світло залежить від мережевої
затримки до сервера; це компенсується полем `latency_offset_ms` у пресеті.
Альтернатива (сервер грає у свою звукову карту) простіша по таймінгу,
але тоді "концерт" прив'язаний до машини, де крутиться сервер.

**Файли — джерело правди для важких даних.** У БД лежать вектори і
метадані; покадрові матриці (STFT-похідні, beat grid, світлові кадри)
лежать файлами. Postgres не для 40 МБ float32 на трек.

## docker-compose

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: wave
      POSTGRES_DB: wave
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]

  core:
    build: ./src/wave-core
    volumes: ["./data:/data"]
    environment:
      WAVE_DB: "postgresql://postgres:wave@db:5432/wave"
    ports: ["8100:8000"]
    depends_on: [db]

  api:
    build: ./src/Wave.Api
    volumes: ["./data:/data"]
    network_mode: host   # потрібен для UDP-броадкасту в світлове залізо
    depends_on: [db, core]

volumes:
  pgdata:
```

`network_mode: host` для api — не забудь. Art-Net і DDP шлються по UDP
у локальну мережу, і NAT докера їх з'їсть. На Windows це працює тільки
через WSL2 або запуск Wave.Api поза докером — для пет-проекту простіше
тримати `db` і `core` у докері, а `api` запускати з IDE.
