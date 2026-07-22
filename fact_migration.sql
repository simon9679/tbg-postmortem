-- FactEngine v2.0 migration

CREATE TABLE IF NOT EXISTS user_facts (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    fact        TEXT NOT NULL,
    embedding   vector(384),
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', fact)) STORED,
    source      TEXT DEFAULT 'inferred',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_facts_user ON user_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_facts_tsv  ON user_facts USING GIN(tsv);
CREATE INDEX IF NOT EXISTS idx_facts_emb  ON user_facts
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- NEW in v2.0: full audit history of all ADD/UPDATE/DELETE operations
CREATE TABLE IF NOT EXISTS fact_history (
    id          BIGSERIAL PRIMARY KEY,
    fact_id     TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    action      TEXT NOT NULL,   -- ADD | UPDATE | DELETE | NOOP
    old_fact    TEXT,
    new_fact    TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fact_history_user ON fact_history(user_id);
CREATE INDEX IF NOT EXISTS idx_fact_history_fact ON fact_history(fact_id);
