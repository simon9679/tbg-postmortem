import os
import asyncio
import logging
import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from tbg_engine import TBGEngine, update_tbg_background
from fact_engine import FactEngine, update_facts_background
from mode_engine import ModeEngine
from dissonance_engine import DissonanceEngine
from intervention_engine import InterventionSimulator

load_dotenv()

RATE_LIMIT_PER_SEC = int(os.getenv("RATE_LIMIT_PER_SEC", "30"))

# ---------------------------------------------------------------------------
# LLM Providers
# LLM_PROVIDER=gemini|openai|anthropic|groq  (default: gemini)
# LLM_MODEL=<optional model override>
# ---------------------------------------------------------------------------

async def __active_llm_fn(prompt: str) -> str:
    from llm_client import gemini_call
    return await gemini_call(prompt)


async def _openai_llm_call(prompt: str) -> str:
    import httpx
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.1, "max_tokens": 1024, "response_format": {"type": "json_object"}}
        )
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.text[:200]}")
        return response.json()["choices"][0]["message"]["content"].strip()


async def _anthropic_llm_call(prompt: str) -> str:
    import httpx
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"},
            json={"model": model, "max_tokens": 1024,
                  "messages": [{"role": "user", "content": prompt}]}
        )
        if response.status_code != 200:
            raise Exception(f"Anthropic API error: {response.text[:200]}")
        return response.json()["content"][0]["text"].strip()


async def _groq_llm_call(prompt: str) -> str:
    import httpx
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.1, "max_tokens": 1024}
        )
        if response.status_code != 200:
            raise Exception(f"Groq API error: {response.text[:200]}")
        return response.json()["choices"][0]["message"]["content"].strip()


_PROVIDERS = {
    "gemini":    __active_llm_fn,
    "openai":    _openai_llm_call,
    "anthropic": _anthropic_llm_call,
    "groq":      _groq_llm_call,
}

# Set at startup via _init_llm_fn()
_active_llm_fn = None

def _init_llm_fn():
    global _active_llm_fn
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(f"Unknown LLM_PROVIDER={provider!r}. Choose: {list(_PROVIDERS)}")
    _active_llm_fn = fn
    logger.info(f"LLM provider: {provider} | model: {os.getenv('LLM_MODEL', 'default')}")

app = FastAPI(title="TBG Memory API", version="2.0.0", docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_pool = None
redis_client = None
engine = None
fact_engine = None
mode_engine = None
dissonance_engine = None
_updating_users: set = set()  # guards against concurrent writes for the same user_id
intervention_simulator = None

class MemoryAddRequest(BaseModel):
    user_id: str
    text: str
    assistant_response: str
    sync: bool = False  # True: wait for writes to complete; False: background (default)

class MemorySearchRequest(BaseModel):
    user_id: str
    query: str   # user's latest message — drives hybrid semantic+BM25 search
    limit: int = 6

class MemoryContextRequest(BaseModel):
    """Simplified endpoint — pass user_id + latest message, get back a ready
    context string to inject directly into your LLM system prompt."""
    user_id: str
    message: str      # user's latest message (used as search query)
    limit: int = 5

class MemoryInsightRequest(BaseModel):
    user_id: str

class CognitiveDirectiveRequest(BaseModel):
    user_id: str
    message: str

class CognitiveUpdateRequest(BaseModel):
    user_id: str
    message: str
    assistant_response: Optional[str] = None
    used_strategy: str

class TimelineResponse(BaseModel):
    user_id: str
    message_count: int
    turning_points: List[dict]
    belief_trajectories: dict
    summary: str

class APIResponse(BaseModel):
    status: str
    data: Optional[dict] = None
    error: Optional[str] = None

async def verify_api_key(api_key: str = Header(..., alias="X-API-Key")):
    if api_key != os.getenv("API_SECRET_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


async def _init_db_conn(conn):
    import json
    await conn.set_type_codec(
        'jsonb',
        encoder=json.dumps,
        decoder=json.loads,
        schema='pg_catalog'
    )
    await conn.set_type_codec(
        'json',
        encoder=json.dumps,
        decoder=json.loads,
        schema='pg_catalog'
    )
    from pgvector.asyncpg import register_vector
    await register_vector(conn)

@app.on_event("startup")
async def startup():
    global db_pool, redis_client, engine, fact_engine, mode_engine, dissonance_engine, intervention_simulator

    # asyncpg pool with pgvector type registration
    db_pool = await asyncpg.create_pool(
        os.getenv("DATABASE_URL"),
        init=_init_db_conn
    )

    redis_client = redis.from_url(os.getenv("REDIS_URL"))
    engine = TBGEngine(db_pool=db_pool)
    fact_engine = FactEngine(db_pool=db_pool)
    mode_engine = ModeEngine(db_pool=db_pool)
    dissonance_engine = DissonanceEngine(db_pool=db_pool)
    intervention_simulator = InterventionSimulator(db_pool=db_pool)

    # Init LLM provider from LLM_PROVIDER env var
    _init_llm_fn()

    # Pre-load embedding model so the first request doesn't block (~100MB load)
    loop = asyncio.get_running_loop()
    from fact_engine import get_embed_model
    await loop.run_in_executor(None, get_embed_model)
    logger.info("Embedding model pre-loaded")

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.close()

async def _run_memory_updates(user_id: str, tbg_fn, facts_fn):
    """
    Run TBG and Facts updates in parallel.
    Skips if a previous update is still running for this user.
    On any failure: logs, then retries the failed task(s) once.

    Accepts callables (not coroutines) so failed tasks can be retried
    by calling them again without re-binding arguments.
    """
    if user_id in _updating_users:
        logger.warning(f"Concurrent update skipped for {user_id[:8]} — previous still running")
        return
    _updating_users.add(user_id)
    try:
        names = ["TBG", "Facts"]
        fns = [tbg_fn, facts_fn]

        results = await asyncio.gather(tbg_fn(), facts_fn(), return_exceptions=True)

        failed_indices = [i for i, r in enumerate(results) if isinstance(r, Exception)]
        if not failed_indices:
            return

        failed_names = [names[i] for i in failed_indices]
        ok_names = [names[i] for i in range(len(results)) if i not in set(failed_indices)]

        if ok_names:
            logger.warning(
                f"Partial memory failure for {user_id[:8]}: "
                f"{failed_names} failed, {ok_names} ok — retrying"
            )
        else:
            logger.error(f"All memory updates failed for {user_id[:8]}: {failed_names} — retrying")

        # One retry for failed tasks — prevents permanent state divergence on transient errors
        retry_results = await asyncio.gather(
            *[fns[i]() for i in failed_indices],
            return_exceptions=True
        )
        still_failed = [
            names[failed_indices[i]] for i, r in enumerate(retry_results)
            if isinstance(r, Exception)
        ]
        if still_failed:
            logger.error(
                f"Memory retry failed for {user_id[:8]}: {still_failed} — state may diverge"
            )
    finally:
        _updating_users.discard(user_id)


@app.post("/memory/add", response_model=APIResponse)
async def add_memory(req: MemoryAddRequest, api_key: str = Depends(verify_api_key)):
    """Store a conversation turn into the TBG graph and fact store."""
    try:
        # Rate limiting — graceful fallback if Redis is unavailable
        try:
            rate_key = f"rate:{req.user_id}"
            current = await redis_client.incr(rate_key)
            if current == 1:
                await redis_client.expire(rate_key, 1)
            if current > RATE_LIMIT_PER_SEC:
                raise HTTPException(429, "Too many requests")
        except HTTPException:
            raise
        except Exception as redis_err:
            logger.warning(f"Rate limit check skipped (Redis unavailable): {redis_err}")

        # Pass callables (not coroutines) so _run_memory_updates can retry on failure
        def _tbg_fn():
            return update_tbg_background(
                user_id=req.user_id,
                user_text=req.text,
                assistant_text=req.assistant_response,
                engine=engine,
                llm_call_fn=_active_llm_fn
            )

        def _facts_fn():
            return update_facts_background(
                user_id=req.user_id,
                user_text=req.text,
                assistant_text=req.assistant_response,
                engine=fact_engine,
                llm_call_fn=_active_llm_fn
            )

        if req.sync:
            await _run_memory_updates(req.user_id, _tbg_fn, _facts_fn)
            return APIResponse(status="ok", data={"queued": False, "written": True})
        else:
            asyncio.create_task(_run_memory_updates(req.user_id, _tbg_fn, _facts_fn))
            return APIResponse(status="ok", data={"queued": True, "written": False})

    except HTTPException:
        raise
    except Exception as e:
        return APIResponse(status="error", error=str(e))

@app.post("/memory/search", response_model=APIResponse)
async def search_memory(req: MemorySearchRequest, api_key: str = Depends(verify_api_key)):
    """Hybrid search: semantic + BM25 over facts, plus TBG belief state."""
    try:
        # Parallel: facts + TBG graph
        facts_task = asyncio.create_task(
            fact_engine.search(req.user_id, req.query, limit=req.limit)
        )
        tbg_task = asyncio.create_task(
            engine.load(req.user_id)
        )

        facts, tbg = await asyncio.gather(facts_task, tbg_task)
        tbg_insight = engine.get_insight(tbg)

        return APIResponse(status="ok", data={
            "facts": facts,
            "tbg_insight": tbg_insight,
            "context": _build_context(facts, tbg_insight)
        })

    except Exception as e:
        return APIResponse(status="error", error=str(e))


@app.post("/memory/context", response_model=APIResponse)
async def get_context(req: MemoryContextRequest, api_key: str = Depends(verify_api_key)):
    """One-call endpoint: returns a context string ready to inject into a system prompt."""
    try:
        facts_task = asyncio.create_task(
            fact_engine.search(req.user_id, req.message, limit=req.limit)
        )
        tbg_task = asyncio.create_task(
            engine.load(req.user_id)
        )

        facts, tbg = await asyncio.gather(facts_task, tbg_task)
        tbg_insight = engine.get_insight(tbg)
        context = _build_context(facts, tbg_insight)

        return APIResponse(status="ok", data={
            "context": context,
            "has_memory": bool(context)
        })

    except Exception as e:
        return APIResponse(status="error", error=str(e))

@app.post("/memory/insight", response_model=APIResponse)
async def get_insight(req: MemoryInsightRequest, api_key: str = Depends(verify_api_key)):
    """TBG belief insight + fact count for a user."""
    try:
        tbg_task = asyncio.create_task(engine.load(req.user_id))
        snapshots_task = asyncio.create_task(engine.load_last_snapshots(req.user_id, limit=2))
        fact_count_task = asyncio.create_task(
            db_pool.fetchval("SELECT COUNT(*) FROM user_facts WHERE user_id = $1", req.user_id)
        )

        tbg, snapshots, fact_count = await asyncio.gather(tbg_task, snapshots_task, fact_count_task)
        insight = engine.get_insight(tbg, snapshots=snapshots)

        return APIResponse(status="ok", data={
            "insight": insight,
            "tbg_nodes": len(tbg.nodes),
            "tbg_edges": len(tbg.edges),
            "fact_count": fact_count or 0
        })

    except Exception as e:
        return APIResponse(status="error", error=str(e))

@app.delete("/memory/{user_id}", response_model=APIResponse)
async def delete_memory(user_id: str, api_key: str = Depends(verify_api_key)):
    """Delete all user memory (TBG + facts)."""
    try:
        await asyncio.gather(
            db_pool.execute("DELETE FROM user_tbg WHERE user_id = $1", user_id),
            db_pool.execute("DELETE FROM tbg_history WHERE user_id = $1", user_id),
            db_pool.execute("DELETE FROM user_mode WHERE user_id = $1", user_id),
            db_pool.execute("DELETE FROM user_dissonance WHERE user_id = $1", user_id),
            db_pool.execute("DELETE FROM user_sensitivity WHERE user_id = $1", user_id),
            fact_engine.delete_all(user_id)
        )
        return APIResponse(status="ok", data={"deleted": user_id})
    except Exception as e:
        return APIResponse(status="error", error=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/memory/timeline/{user_id}", response_model=APIResponse)
async def get_timeline(user_id: str, api_key: str = Depends(verify_api_key)):
    """Returns turning points + belief confidence trajectories for a user."""
    try:
        tbg = await engine.load(user_id)

        turning_points = [tp.model_dump() for tp in tbg.turning_points]

        # Collect trajectories: only nodes with >= 3 history points
        belief_trajectories = {}
        for node in tbg.nodes.values():
            if len(node.confidence_history) >= 3:
                belief_trajectories[node.label] = node.confidence_history

        if turning_points:
            last_tp = turning_points[-1]
            summary = (
                f"{len(turning_points)} turning point(s) over {tbg.message_count} messages. "
                f"Latest at msg {last_tp['message_count']}: "
                f"{', '.join(last_tp['top_nodes'][:2])}"
            )
        else:
            summary = f"No turning points yet. {tbg.message_count} messages."

        return APIResponse(status="ok", data=TimelineResponse(
            user_id=user_id,
            message_count=tbg.message_count,
            turning_points=turning_points,
            belief_trajectories=belief_trajectories,
            summary=summary,
        ).model_dump())
    except Exception as e:
        return APIResponse(status="error", error=str(e))

@app.post("/cognitive/directive", response_model=APIResponse)
async def get_cognitive_directive(req: CognitiveDirectiveRequest, api_key: str = Depends(verify_api_key)):
    """
    The heart of the TBG engine.
    Runs the full pipeline: Beliefs -> Mode -> Dissonance -> Intervention Simulation.
    Returns a ready-to-use directive for the LLM.

    Pipeline order:
      Phase 1 (parallel, DB only): TBG load + facts search + sensitivity load
      Phase 2 (parallel, LLM):     mode inference (with TBG insight) + dissonance compute
      Phase 3 (sync, CPU):         intervention selection (with TBG conflict count)

    TBG is loaded in Phase 1 so its insight can ground the mode inference prompt
    in Phase 2, and its conflict edges can adjust intervention scoring in Phase 3.
    """
    try:
        # Phase 1: parallel DB reads — no LLM yet
        tbg_task = asyncio.create_task(engine.load(req.user_id))
        facts_task = asyncio.create_task(fact_engine.search(req.user_id, req.message, limit=5))
        sensitivity_task = asyncio.create_task(intervention_simulator.load_sensitivity(req.user_id))
        snapshots_task = asyncio.create_task(engine.load_last_snapshots(req.user_id, limit=2))

        tbg, facts, sensitivity, snapshots = await asyncio.gather(
            tbg_task, facts_task, sensitivity_task, snapshots_task
        )

        # Build TBG-derived signals (pure CPU, instant)
        # snapshots passed to get_insight() to include recent belief drift ("Shifting: X↑, Y↓")
        tbg_insight = engine.get_insight(tbg, snapshots=snapshots)
        tbg_conflict_count = sum(
            1 for e in tbg.edges.values()
            if e.relation in ("blocks", "contradicts", "conflicts_with")
            and e.confidence >= 0.55
        )

        # Phase 2: parallel LLM calls — mode now sees TBG belief context
        mode_task = asyncio.create_task(mode_engine.infer(
            req.user_id, req.message, [], _active_llm_fn,
            tbg_insight=tbg_insight,
        ))
        dissonance_task = asyncio.create_task(dissonance_engine.compute(
            req.user_id, req.message, tbg, _active_llm_fn
        ))

        mode_state, dissonance_state = await asyncio.wait_for(
            asyncio.gather(mode_task, dissonance_task),
            timeout=25.0
        )

        # Compute recent turning point signal
        recent_tp = (
            bool(tbg.turning_points)
            and (tbg.message_count - tbg.turning_points[-1].message_count) <= 10
        )

        cold_start = tbg.message_count < 5

        # Compute AMF aggregate signals from per-node AMF states.
        # Strategy: confidence-weighted mean over top-5 nodes by confidence.
        # Falls back to neutral defaults on cold start or missing state.
        amf_state_map = getattr(tbg, "_amf_state", None) or {}
        amf_conf_agg  = 0.5
        amf_ambiv_agg = 0.0
        if amf_state_map and tbg.nodes:
            top_nodes = sorted(tbg.nodes.values(), key=lambda n: n.confidence, reverse=True)[:5]
            total_w = sum(n.confidence for n in top_nodes)
            if total_w > 0:
                node_states = [amf_state_map.get(n.id) for n in top_nodes]
                amf_conf_agg  = sum(
                    (s.amf_conf  if s else 0.5) * n.confidence
                    for s, n in zip(node_states, top_nodes)
                ) / total_w
                amf_ambiv_agg = sum(
                    (s.amf_ambiv if s else 0.0) * n.confidence
                    for s, n in zip(node_states, top_nodes)
                ) / total_w

        # Phase 3: select best intervention — scoring adjusted by TBG conflicts + turning point + AMF
        directive = intervention_simulator.select(
            mode_state, dissonance_state, sensitivity,
            tbg_conflict_count=tbg_conflict_count,
            recent_turning_point=recent_tp,
            cold_start=cold_start,
            amf_conf=round(amf_conf_agg, 3),
            amf_ambiv=round(amf_ambiv_agg, 3),
        )

        context = _build_context(facts, tbg_insight)

        # Enrich context with turning point narrative
        if tbg.turning_points:
            last_tp = tbg.turning_points[-1]
            msgs_ago = tbg.message_count - last_tp.message_count
            narrative = (
                f"\n[COGNITIVE SHIFT: {msgs_ago} messages ago, "
                f"beliefs shifted significantly around: {', '.join(last_tp.top_nodes)}. "
                f"Magnitude: {last_tp.cascade_magnitude:.2f}]"
            )
            context += narrative

        return APIResponse(status="ok", data={
            "directive": directive.to_dict(),
            "prompt_block": directive.to_prompt_block(),
            "context": context,
            "cognitive_state": {
                "mode": mode_state.to_dict(),
                "dissonance": dissonance_state.to_dict(),
                "amf_state": {
                    "conf": round(amf_conf_agg, 3),
                    "ambiv": round(amf_ambiv_agg, 3),
                    "regime": (
                        "grounding" if amf_conf_agg < 0.3
                        else "calibration" if amf_ambiv_agg > 0.4
                        else "continuation"
                    ),
                }
            }
        })
    except Exception as e:
        logger.error(f"Cognitive directive failed: {e}")
        return APIResponse(status="error", error=str(e))

@app.post("/cognitive/update", response_model=APIResponse)
async def update_cognitive_state(req: CognitiveUpdateRequest, api_key: str = Depends(verify_api_key)):
    """
    Update cognitive sensitivity after a response.
    Call this after the assistant responds to the user.
    """
    try:
        # We infer the mode based on the user's new message (their reaction)
        mode_state = await mode_engine.infer(
            req.user_id, req.message, [], _active_llm_fn
        )
        
        # Update sensitivity if a previous strategy was specified
        if req.used_strategy:
            await intervention_simulator.update_sensitivity(
                req.user_id,
                req.used_strategy,
                mode_state.mode
            )
        
        return APIResponse(status="ok", data={"updated": True, "resulting_mode": mode_state.mode})
    except Exception as e:
        return APIResponse(status="error", error=str(e))

@app.post("/cognitive/state", response_model=APIResponse)
async def get_cognitive_state(req: MemoryInsightRequest, api_key: str = Depends(verify_api_key)):
    """Get the full current cognitive state for debugging/UI."""
    try:
        mode_task = asyncio.create_task(mode_engine.load(req.user_id))
        dissonance_task = asyncio.create_task(dissonance_engine.load(req.user_id))
        sensitivity_task = asyncio.create_task(intervention_simulator.load_sensitivity(req.user_id))
        
        mode, diss, sens = await asyncio.gather(mode_task, dissonance_task, sensitivity_task)
        
        return APIResponse(status="ok", data={
            "mode": mode.to_dict(),
            "dissonance": diss.to_dict(),
            "sensitivity": vars(sens)
        })
    except Exception as e:
        return APIResponse(status="error", error=str(e))

def _build_context(facts: list, tbg_insight: str) -> str:
    """Build context block for system prompt."""
    parts = []
    if facts:
        facts_str = "\n".join(f"- {f}" for f in facts)
        parts.append(f"Known facts about user:\n{facts_str}")
    if tbg_insight:
        parts.append(f"Belief state: {tbg_insight}")
    return "\n\n".join(parts)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
