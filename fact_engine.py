"""
FactEngine v2.0
Full competitor to Mem0. Atomic fact memory with LLM-driven lifecycle.

Architecture (mirrors Mem0 two-phase pipeline):
  add(user_text, assistant_text) ->
    Phase 1: Extraction  — LLM extracts candidate facts
    Phase 2: Update      — for each fact, find top-K similar, LLM decides:
                           ADD / UPDATE / DELETE / NOOP
    -> pgvector + BM25 storage + history log

  search(query) ->
    hybrid RRF (semantic + BM25) -> top-K facts

Advantages over Mem0:
  - Full audit history (fact_history table)
  - Hybrid BM25+semantic (Mem0 OSS is semantic-only)
  - Pairs with TBGEngine for belief layer on top of facts
  - Any LLM via llm_call_fn, any PostgreSQL

SQL (run once):
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

    CREATE TABLE IF NOT EXISTS fact_history (
        id          BIGSERIAL PRIMARY KEY,
        fact_id     TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        action      TEXT NOT NULL,
        old_fact    TEXT,
        new_fact    TEXT,
        created_at  TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_fact_history_user ON fact_history(user_id);
    CREATE INDEX IF NOT EXISTS idx_fact_history_fact ON fact_history(fact_id);
"""

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:  # asyncpg is needed only for the live DB path, not for offline replay
    from asyncpg import Pool

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = float(os.getenv("FACT_SIMILARITY_THRESHOLD", "0.75"))  # raise to 0.80 to reduce LLM calls
REINFORCE_THRESHOLD = 0.82    # above this = semantic duplicate, auto-reinforce
DEDUP_THRESHOLD = 0.96        # above this = near-identical, auto-NOOP
UPDATE_CANDIDATES = 5         # top-K similar facts shown to LLM
MAX_FACTS = 200
EMBEDDING_DIM = 384
EMBED_MODEL = "all-MiniLM-L6-v2"
W_SEMANTIC = 0.6
W_BM25 = 0.4

_embed_model = None

def get_embed_model():
    """Shared embedding model — imported by tbg_extractor and tbg_bot as single instance."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
        logger.info(f"FactEngine: loaded {EMBED_MODEL}")
    return _embed_model

def embed(text: str) -> np.ndarray:
    return get_embed_model().encode(text, normalize_embeddings=True, show_progress_bar=False)

def embed_batch(texts: List[str]) -> List[np.ndarray]:
    return get_embed_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)


EXTRACTION_PROMPT = """Extract atomic facts about the user from this conversation.

Rules:
- ONE fact = ONE event or detail. Never combine multiple details into one sentence.
- Be SPECIFIC: include exact dates, numbers, names, amounts, locations when mentioned.
- Bad: "User is worried about money" — too abstract.
- Good: "User has a mortgage and two kids and says wife salary won't cover everything."
- Bad: "User visited a place in May" — too vague.
- Good: "On 7 May 2023: User went to LGBTQ support group for the first time."
- If a date is mentioned, prefix the fact with it: "On [date]: ..."
- Date context for this session: {date_context}
- Only explicit information. Not assistant opinions or interpretations.
- Each fact must be a single standalone sentence in English.

Return ONLY valid JSON, no markdown:
{{"facts": ["fact 1", "fact 2", "fact 3"]}}

If nothing factual: {{"facts": []}}

User: "{user_text}"
Assistant: "{assistant_text}"
Date context: {date_context}
"""

UPDATE_PROMPT = """You manage a personal memory store. A new fact was extracted.
Compare it to existing memories and decide what to do.

NEW FACT:
{new_fact}

EXISTING RELATED MEMORIES:
{existing_memories}

Choose one action:
- ADD: new fact is distinct from all existing memories
- UPDATE: new fact updates or enriches an existing memory (provide its ID and the merged text)
- DELETE: new fact directly contradicts an existing memory (provide its ID)
- NOOP: new fact is already covered by existing memory

Return ONLY valid JSON:
{{
  "action": "ADD" | "UPDATE" | "DELETE" | "NOOP",
  "memory_id": "ID of existing memory or null",
  "updated_fact": "merged fact text if UPDATE, else null",
  "reasoning": "brief explanation"
}}

Rules:
- UPDATE: existing says "likes coffee", new says "likes black coffee no sugar" -> merge
- DELETE: existing says "lives in Moscow", new says "moved to Berlin" -> delete old, add new
- NOOP: essentially the same information
- ADD: genuinely new
"""


def _truncate(text: str, max_chars: int = 600) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    return text[:head] + "\n...\n" + text[-(max_chars - head):]


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1:
        return raw[s:e + 1]
    return raw


class FactEngine:
    """
    Full-lifecycle fact memory: extract -> decide (ADD/UPDATE/DELETE/NOOP) -> store.
    Audit history included. Hybrid BM25+semantic retrieval.
    """

    def __init__(self, db_pool: "Pool"):
        self.db = db_pool

    @staticmethod
    async def _init_conn(conn):
        from pgvector.asyncpg import register_vector
        await register_vector(conn)

    @staticmethod
    async def register(pool: "Pool"):
        from pgvector.asyncpg import register_vector
        async with pool.acquire() as conn:
            await register_vector(conn)

    async def _extract_facts(self, user_text: str, assistant_text: str, llm_call_fn, date_context: str = "") -> List[str]:
        prompt = EXTRACTION_PROMPT.format(
            user_text=_truncate(user_text, 600),
            assistant_text=_truncate(assistant_text, 300),
            date_context=date_context or "not specified",
        )
        try:
            raw = await llm_call_fn(prompt)
            data = json.loads(_clean_json(raw))
            return [f.strip() for f in data.get("facts", []) if isinstance(f, str) and f.strip()]
        except Exception as e:
            logger.warning(f"FactEngine extraction: {e}")
            return []

    async def _find_similar(self, user_id: str, emb: np.ndarray, top_k: int = 5) -> List[Dict]:
        rows = await self.db.fetch(
            """
            SELECT id, fact, 1 - (embedding <=> $1) AS similarity
            FROM user_facts WHERE user_id=$2
            ORDER BY embedding <=> $1 LIMIT $3
            """,
            emb, user_id, top_k
        )
        return [{"id": r["id"], "fact": r["fact"], "similarity": float(r["similarity"])} for r in rows]

    async def _decide_action(self, new_fact: str, new_emb: np.ndarray, user_id: str, llm_call_fn) -> Dict:
        similar = await self._find_similar(user_id, new_emb, top_k=UPDATE_CANDIDATES)

        if not similar:
            return {"action": "ADD", "memory_id": None, "updated_fact": None}

        if similar[0]["similarity"] >= DEDUP_THRESHOLD:
            return {"action": "NOOP", "memory_id": similar[0]["id"], "updated_fact": None}

        if similar[0]["similarity"] >= REINFORCE_THRESHOLD:
            return {"action": "REINFORCE", "memory_id": similar[0]["id"], "updated_fact": None}

        relevant = [s for s in similar if s["similarity"] >= SIMILARITY_THRESHOLD]
        if not relevant:
            return {"action": "ADD", "memory_id": None, "updated_fact": None}

        existing_str = "\n".join(
            f"  ID={s['id']}: {s['fact']} (sim={s['similarity']:.2f})"
            for s in relevant
        )
        prompt = UPDATE_PROMPT.format(new_fact=new_fact, existing_memories=existing_str)

        try:
            raw = await llm_call_fn(prompt)
            result = json.loads(_clean_json(raw))
            action = result.get("action", "ADD").upper()
            if action not in ("ADD", "UPDATE", "DELETE", "NOOP"):
                action = "ADD"
            return {
                "action": action,
                "memory_id": result.get("memory_id"),
                "updated_fact": result.get("updated_fact"),
                "reasoning": str(result.get("reasoning", ""))[:200]
            }
        except Exception as e:
            logger.warning(f"FactEngine decision: {e}")
            return {"action": "ADD", "memory_id": None, "updated_fact": None}

    async def _apply_action(self, action: Dict, new_fact: str, new_emb: np.ndarray, user_id: str) -> str:
        now = datetime.now(timezone.utc)
        op = action["action"]

        if op == "ADD":
            fid = str(uuid.uuid4())[:16]
            await self.db.execute(
                "INSERT INTO user_facts (id, user_id, fact, embedding, source, updated_at) VALUES ($1,$2,$3,$4,'explicit',$5)",
                fid, user_id, new_fact, new_emb, now
            )
            await self._log(fid, user_id, "ADD", None, new_fact)
            return "ADD"

        elif op == "REINFORCE" and action.get("memory_id"):
            mid = action["memory_id"]
            await self.db.execute(
                "UPDATE user_facts SET updated_at=$1 WHERE id=$2 AND user_id=$3",
                now, mid, user_id
            )
            await self._log(mid, user_id, "REINFORCE", None, None)
            return "REINFORCE"

        elif op == "UPDATE" and action.get("memory_id"):
            mid = action["memory_id"]
            updated = action.get("updated_fact") or new_fact
            loop = asyncio.get_running_loop()
            updated_emb = await loop.run_in_executor(None, lambda: embed(updated)) if updated != new_fact else new_emb
            old = await self.db.fetchrow("SELECT fact FROM user_facts WHERE id=$1 AND user_id=$2", mid, user_id)
            if not old:
                return await self._apply_action({"action": "ADD", "memory_id": None, "updated_fact": None}, new_fact, new_emb, user_id)
            await self.db.execute(
                "UPDATE user_facts SET fact=$1, embedding=$2, updated_at=$3 WHERE id=$4 AND user_id=$5",
                updated, updated_emb, now, mid, user_id
            )
            await self._log(mid, user_id, "UPDATE", old["fact"], updated)
            return "UPDATE"

        elif op == "DELETE" and action.get("memory_id"):
            mid = action["memory_id"]
            old = await self.db.fetchrow("SELECT fact FROM user_facts WHERE id=$1 AND user_id=$2", mid, user_id)
            if old:
                await self.db.execute("DELETE FROM user_facts WHERE id=$1 AND user_id=$2", mid, user_id)
                await self._log(mid, user_id, "DELETE", old["fact"], None)
            # Insert the new contradicting fact without counting it as a separate ADD.
            fid = str(uuid.uuid4())[:16]
            await self.db.execute(
                "INSERT INTO user_facts (id, user_id, fact, embedding, source, updated_at) VALUES ($1,$2,$3,$4,'explicit',$5)",
                fid, user_id, new_fact, new_emb, now
            )
            await self._log(fid, user_id, "ADD", None, new_fact)
            return "DELETE"

        else:
            return "NOOP"

    async def _log(self, fact_id: str, user_id: str, action: str, old: Optional[str], new: Optional[str]):
        try:
            await self.db.execute(
                "INSERT INTO fact_history (fact_id, user_id, action, old_fact, new_fact) VALUES ($1,$2,$3,$4,$5)",
                fact_id, user_id, action, old, new
            )
        except Exception as e:
            logger.warning(f"FactEngine history log: {e}")

    async def add(
        self,
        user_id: str,
        user_text: str,
        assistant_text: str,
        llm_call_fn,
        date_context: str = ""
    ) -> Dict[str, int]:
        """Extract facts and apply ADD/UPDATE/DELETE/NOOP. Returns action counts."""
        facts = await self._extract_facts(user_text, assistant_text, llm_call_fn, date_context)
        if not facts:
            return {"add": 0, "update": 0, "delete": 0, "noop": 0}

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, lambda: embed_batch(facts))
        counts: Dict[str, int] = {"add": 0, "update": 0, "delete": 0, "noop": 0, "reinforce": 0}

        # Phase 1: parallel LLM decisions — semaphore caps concurrent calls per
        # user to 3, preventing LLM rate limit spikes on messages with many facts.
        sem = asyncio.Semaphore(3)

        async def _decide(fact, emb):
            async with sem:
                return await self._decide_action(fact, emb, user_id, llm_call_fn)

        decisions = await asyncio.gather(
            *[_decide(f, e) for f, e in zip(facts, embeddings)],
            return_exceptions=True
        )

        # Phase 2: sequential DB writes — order matters, prevents concurrent
        # writes to the same memory_id.
        for fact, emb, decision in zip(facts, embeddings, decisions):
            if isinstance(decision, Exception):
                logger.warning(f"FactEngine decide failed for fact '{fact[:40]}': {decision}")
                decision = {"action": "ADD", "memory_id": None, "updated_fact": None}
            result = await self._apply_action(decision, fact, emb, user_id)
            counts[result.lower()] = counts.get(result.lower(), 0) + 1

        await self._prune(user_id)

        logger.info(
            f"FactEngine user={user_id[:8]} "
            f"ADD={counts['add']} UPD={counts['update']} DEL={counts['delete']} "
            f"NOOP={counts['noop']} REINFORCE={counts['reinforce']}"
        )
        return counts

    async def search(self, user_id: str, query: str, limit: int = 8) -> List[str]:
        """Hybrid RRF: semantic + BM25. Returns facts with temporal prefix."""
        loop = asyncio.get_running_loop()
        emb = await loop.run_in_executor(None, lambda: embed(query))

        sem_rows = await self.db.fetch(
            "SELECT id, fact, updated_at FROM user_facts WHERE user_id=$2 ORDER BY embedding <=> $1 LIMIT $3",
            emb, user_id, limit * 2
        )
        bm25 = await self.db.fetch(
            """SELECT id, fact, updated_at FROM user_facts
               WHERE user_id=$2 AND tsv @@ plainto_tsquery('english', $1)
               ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $1)) DESC LIMIT $3""",
            query, user_id, limit * 2
        )

        scores: Dict[str, float] = {}
        facts_by_id: Dict[str, str] = {}
        timestamps_by_id: Dict[str, object] = {}

        for rank, row in enumerate(sem_rows, 1):
            scores[row["id"]] = scores.get(row["id"], 0) + W_SEMANTIC / (rank + 60)
            facts_by_id[row["id"]] = row["fact"]
            timestamps_by_id[row["id"]] = row.get("updated_at")

        for rank, row in enumerate(bm25, 1):
            scores[row["id"]] = scores.get(row["id"], 0) + W_BM25 / (rank + 60)
            facts_by_id[row["id"]] = row["fact"]
            timestamps_by_id[row["id"]] = row.get("updated_at")

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for fid, _ in ranked[:limit]:
            fact = facts_by_id[fid]
            ts = timestamps_by_id.get(fid)
            if ts:
                try:
                    date_str = ts.strftime("%d %b %Y") if hasattr(ts, "strftime") else str(ts)[:10]
                    fact = f"{date_str}: {fact}"
                except Exception:
                    pass
            result.append(fact)
        return result

    async def get_context(self, user_id: str, query: str, limit: int = 6) -> str:
        facts = await self.search(user_id, query, limit=limit)
        if not facts:
            return ""
        return "Known facts about user:\n" + "\n".join(f"- {f}" for f in facts)

    async def history(self, user_id: str, fact_id: str, limit: int = 50) -> List[Dict]:
        rows = await self.db.fetch(
            "SELECT action, old_fact, new_fact, created_at FROM fact_history "
            "WHERE user_id=$1 AND fact_id=$2 ORDER BY created_at DESC LIMIT $3",
            user_id, fact_id, limit
        )
        return [dict(r) for r in rows]

    async def get_all(self, user_id: str) -> List[Dict]:
        rows = await self.db.fetch(
            "SELECT id, fact, source, updated_at FROM user_facts WHERE user_id=$1 ORDER BY updated_at DESC",
            user_id
        )
        return [dict(r) for r in rows]

    async def delete_fact(self, user_id: str, fact_id: str) -> bool:
        row = await self.db.fetchrow("SELECT fact FROM user_facts WHERE id=$1 AND user_id=$2", fact_id, user_id)
        if not row:
            return False
        await self.db.execute("DELETE FROM user_facts WHERE id=$1 AND user_id=$2", fact_id, user_id)
        await self._log(fact_id, user_id, "DELETE", row["fact"], None)
        return True

    async def delete_all(self, user_id: str):
        await self.db.execute("DELETE FROM user_facts WHERE user_id=$1", user_id)
        await self.db.execute("DELETE FROM fact_history WHERE user_id=$1", user_id)

    async def _prune(self, user_id: str):
        # Single query: DELETE is a no-op when count ≤ MAX_FACTS, no COUNT(*) needed
        await self.db.execute(
            """DELETE FROM user_facts WHERE user_id=$1 AND id NOT IN (
                SELECT id FROM user_facts WHERE user_id=$1 ORDER BY updated_at DESC LIMIT $2)""",
            user_id, MAX_FACTS
        )


async def update_facts_background(
    user_id: str,
    user_text: str,
    assistant_text: str,
    engine: FactEngine,
    llm_call_fn
):
    """Background task — call via asyncio.create_task()."""
    try:
        counts = await engine.add(user_id, user_text, assistant_text, llm_call_fn)
        meaningful = counts.get("add", 0) + counts.get("update", 0) + counts.get("delete", 0)
        if meaningful:
            logger.info(f"FactEngine bg {user_id[:8]}: {counts}")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"FactEngine bg failed {user_id[:8]}: {e}", exc_info=True)
