#!/usr/bin/env python3
"""
In-memory storage-shim для FactEngine (вариант (b) из ТЗ LoCoMo).
Реализует интерфейс asyncpg.Pool (fetch/fetchrow/execute), эмулируя ИМЕННО те
SQL-запросы, что шлёт FactEngine. Логика FactEngine (add/_extract_facts/
_decide_action/_apply_action/search/_prune) вызывается КАК ЕСТЬ — подменяется
только storage primitive.

Данные: facts[id] = {user_id, fact, embedding(np, normalized), source, updated_at}.
Векторный порядок: cosine distance = 1 - dot (эмбеддинги нормированы).
BM25: аппроксимация (overlap токенов) — точный ts_rank_cd не нужен, RRF сглаживает.
Помечено: bm25 ≈ approx.
"""
import re
import numpy as np


def _norm_sql(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip().lower()


def _toks(s: str):
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 1]


class InMemoryPool:
    """Утиный asyncpg.Pool. Только fetch/fetchrow/execute (FactEngine больше ничего не зовёт)."""

    def __init__(self):
        self.facts: dict = {}      # id -> row dict
        self.history: list = []    # fact_history rows

    # ---- helpers ----
    def _user_rows(self, uid):
        return [r for r in self.facts.values() if r["user_id"] == uid]

    # ---- asyncpg.Pool surface ----
    async def fetch(self, query, *args):
        q = _norm_sql(query)

        # _find_similar: SELECT id, fact, 1-(embedding <=> $1) AS similarity ... ORDER BY embedding <=> $1
        if "as similarity" in q:
            emb, uid, k = args[0], args[1], args[2]
            rows = self._user_rows(uid)
            scored = [(r, 1.0 - float(np.dot(emb, r["embedding"]))) for r in rows]  # cosine distance
            scored.sort(key=lambda x: x[1])  # ascending distance
            return [{"id": r["id"], "fact": r["fact"], "similarity": 1.0 - d}
                    for r, d in scored[:k]]

        # search semantic: SELECT id, fact, updated_at ... ORDER BY embedding <=> $1 LIMIT $3
        if "order by embedding <=>" in q and "as similarity" not in q:
            emb, uid, k = args[0], args[1], args[2]
            rows = self._user_rows(uid)
            scored = [(r, 1.0 - float(np.dot(emb, r["embedding"]))) for r in rows]
            scored.sort(key=lambda x: x[1])
            return [{"id": r["id"], "fact": r["fact"], "updated_at": r["updated_at"]}
                    for r, _ in scored[:k]]

        # search bm25: WHERE tsv @@ plainto_tsquery(...) ORDER BY ts_rank_cd(...)
        if "plainto_tsquery" in q:
            query_text, uid, k = args[0], args[1], args[2]
            qt = set(_toks(query_text))
            if not qt:
                return []
            scored = []
            for r in self._user_rows(uid):
                ft = _toks(r["fact"])
                overlap = sum(1 for t in ft if t in qt)
                if overlap > 0:
                    scored.append((r, overlap))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [{"id": r["id"], "fact": r["fact"], "updated_at": r["updated_at"]}
                    for r, _ in scored[:k]]

        # get_all: SELECT id, fact, source, updated_at ... ORDER BY updated_at DESC
        if "select id, fact, source, updated_at" in q:
            uid = args[0]
            rows = sorted(self._user_rows(uid), key=lambda r: r["updated_at"], reverse=True)
            return [{"id": r["id"], "fact": r["fact"], "source": r["source"],
                     "updated_at": r["updated_at"]} for r in rows]

        # fact_history select
        if "from fact_history" in q:
            return []

        return []

    async def fetchrow(self, query, *args):
        q = _norm_sql(query)
        # SELECT fact FROM user_facts WHERE id=$1 AND user_id=$2
        if "select fact from user_facts where id=" in q:
            fid, uid = args[0], args[1]
            r = self.facts.get(fid)
            return {"fact": r["fact"]} if r and r["user_id"] == uid else None
        return None

    async def execute(self, query, *args):
        q = _norm_sql(query)

        # INSERT INTO user_facts (id,user_id,fact,embedding,source,updated_at) VALUES ...'explicit'...
        if q.startswith("insert into user_facts"):
            fid, uid, fact, emb, now = args[0], args[1], args[2], args[3], args[4]
            self.facts[fid] = {"id": fid, "user_id": uid, "fact": fact,
                               "embedding": np.asarray(emb, dtype=float),
                               "source": "explicit", "updated_at": now}
            return

        # INSERT INTO fact_history (...)
        if q.startswith("insert into fact_history"):
            self.history.append(args)
            return

        # UPDATE user_facts SET updated_at=$1 WHERE id=$2 AND user_id=$3  (REINFORCE)
        if q.startswith("update user_facts set updated_at=$1 where id="):
            now, mid, uid = args[0], args[1], args[2]
            r = self.facts.get(mid)
            if r and r["user_id"] == uid:
                r["updated_at"] = now
            return

        # UPDATE user_facts SET fact=$1, embedding=$2, updated_at=$3 WHERE id=$4 AND user_id=$5
        if q.startswith("update user_facts set fact="):
            fact, emb, now, mid, uid = args[0], args[1], args[2], args[3], args[4]
            r = self.facts.get(mid)
            if r and r["user_id"] == uid:
                r.update({"fact": fact, "embedding": np.asarray(emb, dtype=float), "updated_at": now})
            return

        # _prune: DELETE ... WHERE user_id=$1 AND id NOT IN (SELECT ... ORDER BY updated_at DESC LIMIT $2)
        if "not in" in q and "delete from user_facts" in q:
            uid, keep = args[0], args[1]
            rows = sorted(self._user_rows(uid), key=lambda r: r["updated_at"], reverse=True)
            keep_ids = {r["id"] for r in rows[:keep]}
            for r in list(rows):
                if r["id"] not in keep_ids:
                    self.facts.pop(r["id"], None)
            return

        # DELETE FROM user_facts WHERE id=$1 AND user_id=$2
        if q.startswith("delete from user_facts where id="):
            mid, uid = args[0], args[1]
            r = self.facts.get(mid)
            if r and r["user_id"] == uid:
                self.facts.pop(mid, None)
            return

        # delete_all
        if q.startswith("delete from user_facts where user_id="):
            uid = args[0]
            for fid in [k for k, r in self.facts.items() if r["user_id"] == uid]:
                self.facts.pop(fid, None)
            return
        if q.startswith("delete from fact_history where user_id="):
            return

        # CREATE/register schema — no-op for in-memory
        return


# ── selftest: гоняем РЕАЛЬНЫЙ FactEngine на shim с фейковым LLM (без сети) ──
if __name__ == "__main__":
    import sys, os, asyncio, json
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from fact_engine import FactEngine

    async def fake_llm(prompt: str) -> str:
        # extraction prompt -> вернуть факты; update prompt -> ADD
        if "Return ONLY valid JSON" in prompt or "facts" in prompt.lower() and "extract" in prompt.lower():
            if "Alice" in prompt and "Berlin" in prompt:
                return json.dumps({"facts": ["Alice lives in Berlin", "Alice has a dog named Rex"]})
            if "vegan" in prompt:
                return json.dumps({"facts": ["Alice is vegan"]})
            return json.dumps({"facts": []})
        return json.dumps({"action": "ADD", "memory_id": None, "updated_fact": None})

    async def main():
        pool = InMemoryPool()
        fe = FactEngine(pool)
        c1 = await fe.add("u1", "Hi, I'm Alice and I just moved to Berlin with my dog Rex.", "", fake_llm)
        c2 = await fe.add("u1", "By the way I'm vegan now.", "", fake_llm)
        print("add#1:", c1)
        print("add#2:", c2)
        print("stored facts:", [r["fact"] for r in pool.facts.values()])
        res = await fe.search("u1", "where does Alice live?", limit=5)
        print("search 'where does Alice live':", res)
        res2 = await fe.search("u1", "what does Alice eat?", limit=5)
        print("search 'what does Alice eat':", res2)
        assert any("Berlin" in f for f in res), "semantic retrieval failed"
        print("\nSELFTEST OK — FactEngine runs unmodified on the in-memory shim.")

    asyncio.run(main())
