CREATE EXTENSION IF NOT EXISTS vector;

-- Трек: файл + метадані з тегів
CREATE TABLE IF NOT EXISTS tracks (
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
CREATE TABLE IF NOT EXISTS track_features (
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

CREATE INDEX IF NOT EXISTS idx_track_features_timbre_hnsw ON track_features USING hnsw (timbre_vec vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_track_features_bpm ON track_features (bpm);
CREATE INDEX IF NOT EXISTS idx_track_features_camelot ON track_features (camelot);

-- Сет = впорядкований список треків із точками переходу
CREATE TABLE IF NOT EXISTS sets (
    id           uuid PRIMARY KEY,
    name         text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    rendered_at  timestamptz,
    audio_path   text,
    lights_path  text,
    total_sec    real
);

CREATE TABLE IF NOT EXISTS set_items (
    set_id           uuid REFERENCES sets(id) ON DELETE CASCADE,
    position         int NOT NULL,
    track_id         uuid REFERENCES tracks(id),
    -- параметри переходу З цього треку в наступний
    transition       jsonb,   -- {type, out_at, in_at, bars, bpm_target, eq}
    PRIMARY KEY (set_id, position)
);

-- Світловий пресет: як фічі мапляться в кольори
CREATE TABLE IF NOT EXISTS light_presets (
    id       uuid PRIMARY KEY,
    name     text NOT NULL,
    fixtures jsonb NOT NULL,  -- розкладка приладів
    rules    jsonb NOT NULL   -- правила band→color, beat→flash тощо
);

-- Нормалізація фіч по бібліотеці
CREATE TABLE IF NOT EXISTS feature_stats (
    id            int PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    track_count   int NOT NULL,
    stats         jsonb NOT NULL   -- {"mfcc_1": {"mean": .., "std": ..}, ...}
);

-- Черга завдань
CREATE TABLE IF NOT EXISTS jobs (
    id          uuid PRIMARY KEY,
    kind        text NOT NULL,     -- analyze | render_mix | render_lights
    payload     jsonb NOT NULL,
    status      text NOT NULL,     -- queued | running | done | failed
    progress    real DEFAULT 0,
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created_at ON jobs (status, created_at);
