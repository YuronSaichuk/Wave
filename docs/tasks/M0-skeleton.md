# M0 — Каркас

**Мета:** з чистого клону проект піднімається однією командою, і
є куди писати код.

Оцінка: 1 вечір. Якщо розтягнулось на три — щось пішло не так,
зупинись і спрости.

---

## T0.1 — Git і структура папок

- [x] `git init` у `Wave/`
- [x] створити дерево з [01-architecture.md](../01-architecture.md#лейаут-репозиторію)
- [x] `.gitignore`:

```gitignore
data/
*.npz
*.lights
*.wav
*.mp3

# python
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
*.egg-info/

# dotnet
bin/
obj/
*.user

# node
node_modules/
dist/

# ide
.vs/
.idea/
.vscode/
```

- [x] `data/library/.gitkeep`, `data/cache/.gitkeep`, `data/mixes/.gitkeep`
- [x] перший коміт

---

## T0.2 — PostgreSQL + pgvector (Docker Compose)

- [x] налаштовано підтримку `.env` для `wave-core` та `appsettings.Development.json` для `Wave.Api` (порт 5435)
- [x] створено `.env.example` та `.env` зі зразком конфігурації підключення
- [x] `docker-compose.yml` запущено: контейнер `wave-db` (`pgvector/pgvector:pg16`) на зовнішньому порту `5435`
- [x] підключення та pgvector перевірено через `wave db-check`

**Примітка:** зовнішній порт призначено як `5435:5432`, оскільки 5432 і 5434 зайняті локальними службами Windows.

---

## T0.3 — Схема БД

- [x] `src/wave-core/migrations/001_init.sql` — увесь DDL із
      [01-architecture.md](../01-architecture.md#схема-бд)
- [x] додати `CREATE EXTENSION IF NOT EXISTS vector;` першим рядком
- [x] додати таблицю для нормалізації:

```sql
CREATE TABLE feature_stats (
    id            int PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    track_count   int NOT NULL,
    stats         jsonb NOT NULL   -- {"mfcc_1": {"mean": .., "std": ..}, ...}
);
```

- [x] додати таблицю джобів:

```sql
CREATE TABLE jobs (
    id          uuid PRIMARY KEY,
    kind        text NOT NULL,     -- analyze | render_mix | render_lights
    payload     jsonb NOT NULL,
    status      text NOT NULL,     -- queued | running | done | failed
    progress    real DEFAULT 0,
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);
CREATE INDEX ON jobs (status, created_at);
```

- [x] інструмент міграцій: `alembic` або простий раннер, що виконує
      `.sql` файли по порядку і пише в `schema_migrations`. Для пет-проекту
      другий варіант чесніший — менше магії
- [x] перевірити: таблиці створено та підтверджено через `wave db-check`

**Рішення, яке варто зафіксувати:** схему створює і володіє нею
Python-частина. .NET читає ту саму базу через Dapper або EF з
`DbContext` у режимі "database first", **без** власних міграцій.
Дві системи міграцій на одну базу — гарантований біль.

---

## T0.4 — Каркас wave-core (Python)

- [x] Python 3.11+, `python -m venv .venv`
- [x] `pyproject.toml`:

```toml
[project]
name = "wave-core"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "librosa>=0.10",
    "numpy",
    "scipy",
    "soundfile",
    "mutagen",
    "pyloudnorm",
    "psycopg[binary]>=3.1",
    "pgvector",
    "typer",
    "rich",
    "pydantic",
    "pydantic-settings",
    "fastapi",
    "uvicorn",
]

[project.optional-dependencies]
dev = ["pytest", "matplotlib", "ruff"]

[project.scripts]
wave = "wave_core.cli:app"
```

- [x] `pip install -e ".[dev]"`
- [x] пакети-заглушки: `analysis/`, `similarity/`, `mixing/`,
      `lighting/`, `storage/`
- [x] `config.py` через `pydantic-settings`: `WAVE_DB`, `WAVE_LIBRARY`,
      `WAVE_CACHE`, `WAVE_MIXES`
- [x] `cli.py` на `typer` з командою `version`
- [x] `storage/db.py` — пул з'єднань psycopg3, реєстрація pgvector
- [x] перевірити: `wave version` і `wave db-check` (селект з `tracks`)

**Пастка з librosa на Windows:** їй потрібен `ffmpeg` або `audioread`
для MP3. Постав ffmpeg і додай у PATH — інакше отримаєш
`NoBackendError` на першому ж файлі.

---

## T0.5 — Каркас Wave.Api (.NET)

- [x] `dotnet new sln -n Wave`
- [x] `dotnet new webapi -n Wave.Api -o src/Wave.Api`
- [x] `dotnet new classlib -n Wave.Lighting -o src/Wave.Lighting`
- [x] `dotnet new classlib -n Wave.Domain -o src/Wave.Domain`
- [x] додати проекти в sln, посилання Api → Lighting, Api → Domain
- [x] `/health` → 200
- [x] `appsettings.Development.json`: рядок підключення до БД,
      базовий URL wave-core (`http://localhost:8100`)
- [x] `HttpClient` до wave-core через `IHttpClientFactory`, з
      таймаутом і `AddStandardResilienceHandler`
- [x] CORS для Angular dev-сервера (`http://localhost:4200`)
- [x] перевірити: `dotnet run`, `/health` відповідає

---

## T0.6 — FastAPI-обгортка

- [x] `api.py`: `/health`, `/jobs/{id}`
- [x] `uvicorn wave_core.api:app --port 8100 --reload`
- [x] перевірити ланцюжок: Wave.Api викликає `/health` wave-core
      і віддає його статус у своєму `/health`

Це найкорисніша перевірка M0 — вона доводить, що обидві половини
проекту бачать одна одну.

---

## T0.7 — README запуску

- [x] як підняти БД, як поставити Python-залежності, як запустити
      обидва сервіси, які змінні середовища потрібні
- [x] окремо: як поставити ffmpeg і rubberband-cli на Windows

---

## Definition of Done

```powershell
git clone <repo> && cd Wave
docker compose up -d db
cd src/wave-core; python -m venv .venv; .venv\Scripts\activate
pip install -e ".[dev]"
wave db-check                       # OK
uvicorn wave_core.api:app --port 8100   # у окремому вікні
cd ../Wave.Api; dotnet run              # у окремому вікні
curl http://localhost:5000/health       # {"api":"ok","core":"ok","db":"ok"}
```

Усе. Ніякої логіки — тільки те, що воно живе.
