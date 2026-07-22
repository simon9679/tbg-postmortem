-- Canonical PostgreSQL bootstrap for the current TBG runtime.
-- This schema matches the storage model used by api.py, tbg_engine.py, and fact_engine.py.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS user_tbg (
    user_id TEXT PRIMARY KEY,
    nodes_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    edges_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    message_count INTEGER NOT NULL DEFAULT 0,
    last_sync TIMESTAMPTZ DEFAULT NOW(),
    last_decay TIMESTAMPTZ DEFAULT NOW(),
    turning_points_data JSONB NOT NULL DEFAULT '[]'::jsonb,
    archive_data JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Migration for existing databases (idempotent):
ALTER TABLE user_tbg ADD COLUMN IF NOT EXISTS turning_points_data JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE user_tbg ADD COLUMN IF NOT EXISTS archive_data JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS tbg_history (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    message_count INTEGER NOT NULL,
    snapshot_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, message_count)
);

CREATE INDEX IF NOT EXISTS idx_tbg_history_user_id
    ON tbg_history (user_id, message_count DESC);

CREATE TABLE IF NOT EXISTS user_facts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    fact TEXT NOT NULL,
    embedding vector(384),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', fact)) STORED,
    source TEXT DEFAULT 'inferred',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_facts_user_id
    ON user_facts (user_id);

CREATE INDEX IF NOT EXISTS idx_user_facts_tsv
    ON user_facts USING GIN (tsv);

CREATE INDEX IF NOT EXISTS idx_user_facts_embedding
    ON user_facts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

CREATE TABLE IF NOT EXISTS fact_history (
    id BIGSERIAL PRIMARY KEY,
    fact_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    old_fact TEXT,
    new_fact TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fact_history_user_id
    ON fact_history (user_id);

CREATE INDEX IF NOT EXISTS idx_fact_history_fact_id
    ON fact_history (fact_id);

-- ============================================================
-- TBG TBG engine tables (v1.0)
-- ============================================================

-- Current psychological mode per user
CREATE TABLE IF NOT EXISTS user_mode (
    user_id    TEXT PRIMARY KEY,
    mode       TEXT NOT NULL DEFAULT 'exploration',
    confidence FLOAT NOT NULL DEFAULT 0.5,
    stability  FLOAT NOT NULL DEFAULT 0.5,
    triggers   JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Internal dissonance state per user
CREATE TABLE IF NOT EXISTS user_dissonance (
    user_id           TEXT PRIMARY KEY,
    total_score       FLOAT NOT NULL DEFAULT 0.0,
    drive_conflict    FLOAT NOT NULL DEFAULT 0.0,
    role_strain       FLOAT NOT NULL DEFAULT 0.0,
    want_should_gap   FLOAT NOT NULL DEFAULT 0.0,
    decision_friction FLOAT NOT NULL DEFAULT 0.0,
    hotspots          JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Per-user intervention sensitivity profile
CREATE TABLE IF NOT EXISTS user_sensitivity (
    user_id              TEXT PRIMARY KEY,
    challenge_tolerance  FLOAT NOT NULL DEFAULT 0.5,
    validation_need      FLOAT NOT NULL DEFAULT 0.5,
    shame_sensitivity    FLOAT NOT NULL DEFAULT 0.5,
    autonomy_sensitivity FLOAT NOT NULL DEFAULT 0.5,
    structure_preference FLOAT NOT NULL DEFAULT 0.5,
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
