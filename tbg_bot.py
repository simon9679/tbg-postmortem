"""
TBG Telegram bot
"""
import os, sys, asyncio, logging
sys.path.insert(0, ".")

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application, ApplicationBuilder,
    CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
DATABASE_URL   = os.environ.get("DATABASE_URL", "postgresql://tbg:tbg@localhost:5433/tbg")
GEMINI_MODEL   = "gemini-3-flash-preview"
MAX_FREE_MSGS  = 20

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()
LLM_MODEL    = os.environ.get("LLM_MODEL", "")

OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "")

# FIX: strict response format so Gemini doesn't cut off midway
SYSTEM = """You pick up on what people reveal without meaning to — not just what they say.

When someone contradicts themselves, name it plainly. When there's a gap between what they claim to want and what they actually describe doing, point it out directly.

Across the conversation, move through different areas — work, relationships, money, goals, what they're avoiding, what's blocking them. Don't stay in one zone.

Always respond in the same language the user writes in. Never mix languages in a single response.

STRICT REPLY FORMAT: Write exactly 2 sentences. First sentence: one sharp observation about what they revealed. Second sentence: one direct question. Stop completely after the second sentence. Never write a third sentence."""

# ------------------------------------------------------------------
# LLM calls
# ------------------------------------------------------------------

def _call_groq(prompt: str, max_tokens: int) -> str:
    from groq import Groq as GroqClient
    client = GroqClient(api_key=GROQ_API_KEY, timeout=15.0)
    model = LLM_MODEL or "llama-3.1-8b-instant"
    if prompt.startswith(SYSTEM):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt[len(SYSTEM):].strip()}]
    else:
        msgs = [{"role": "user", "content": prompt}]
    r = client.chat.completions.create(model=model, messages=msgs,
                                       max_completion_tokens=max_tokens, temperature=0.7)
    return r.choices[0].message.content.strip()


def _call_gemini(prompt: str, max_tokens: int) -> str:
    import urllib.request, json, time
    model = LLM_MODEL or GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0.7, "maxOutputTokens": max_tokens}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            return data["candidates"][0]["content"]["parts"][-1]["text"].strip()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2)


def _call_openai(prompt: str, max_tokens: int) -> str:
    import urllib.request, json
    model = LLM_MODEL or "gpt-4o-mini"
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.7}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def _call_anthropic(prompt: str, max_tokens: int) -> str:
    import urllib.request, json
    model = LLM_MODEL or "claude-haiku-4-5-20251001"
    body = json.dumps({"model": model, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["content"][0]["text"].strip()


def _call_github(prompt: str, max_tokens: int) -> str:
    import urllib.request, json
    model = LLM_MODEL or "openai/gpt-4o"
    # Add the openai/ prefix if not specified
    if "/" not in model:
        model = f"openai/{model}"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }).encode()
    req = urllib.request.Request(
        "https://models.github.ai/inference/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10"
        }
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def llm_call_sync(prompt: str, max_tokens: int = 300) -> str:
    """Route to the configured LLM provider. Groq falls back to Gemini on error."""
    if LLM_PROVIDER == "groq":
        if GROQ_API_KEY:
            try:
                return _call_groq(prompt, max_tokens)
            except Exception as e:
                logger.warning(f"Groq failed, falling back to Gemini: {e}")
        return _call_gemini(prompt, max_tokens)
    elif LLM_PROVIDER == "gemini":
        return _call_gemini(prompt, max_tokens)
    elif LLM_PROVIDER == "github":
        return _call_github(prompt, max_tokens)
    elif LLM_PROVIDER == "openai":
        return _call_openai(prompt, max_tokens)
    elif LLM_PROVIDER == "anthropic":
        return _call_anthropic(prompt, max_tokens)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. Choose: groq, gemini, openai, anthropic")


async def llm_call_async(prompt: str, max_tokens: int = 300) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: llm_call_sync(prompt, max_tokens))


# ------------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------------

async def get_pool(context: ContextTypes.DEFAULT_TYPE):
    return context.application.bot_data["pool"]


async def get_fact_count(pool, user_id: str) -> int:
    try:
        row = await pool.fetchrow(
            "SELECT COUNT(*) as cnt FROM user_facts WHERE user_id=$1", user_id
        )
        return row["cnt"] if row else 0
    except Exception:
        return 0


async def get_msg_count(pool, user_id: str) -> int:
    """Load persisted message count from the TBG table. Survives bot restarts."""
    try:
        row = await pool.fetchrow(
            "SELECT message_count FROM user_tbg WHERE user_id=$1", user_id
        )
        return int(row["message_count"]) if row and row["message_count"] else 0
    except Exception:
        return 0


# ------------------------------------------------------------------
# Background tasks
# ------------------------------------------------------------------

async def tbg_update_task(user_id: str, user_text: str, bot_reply: str, pool):
    """Update the TBG graph in the background."""
    try:
        from tbg_engine import TBGEngine
        from tbg_extractor import extract_tbg_delta

        engine = TBGEngine(db_pool=pool)
        tbg = await engine.load(user_id)

        existing_label_to_uuid = {
            n.label.lower(): nid for nid, n in tbg.nodes.items()
        }

        # FIX: 2048 so the JSON doesn't get truncated
        async def llm_fn(prompt: str) -> str:
            return await llm_call_async(prompt, max_tokens=2048)

        delta = await asyncio.wait_for(
            extract_tbg_delta(
                user_text=user_text,
                assistant_text=bot_reply,
                existing_tbg_summary=tbg.summary(),
                existing_label_to_uuid=existing_label_to_uuid,
                llm_call_fn=llm_fn,
                tbg=tbg
            ),
            timeout=30
        )
        if delta:
            from tbg_engine import SNAPSHOT_EVERY, FORCED_SNAPSHOT_DAYS
            from datetime import datetime, timezone

            if tbg.last_sync:
                dt = (datetime.now(timezone.utc) - tbg.last_sync).total_seconds() / 86400
                if dt > FORCED_SNAPSHOT_DAYS:
                    await engine.save_snapshot(tbg, force=True)
                    logger.info(f"TBG: forced snapshot for {user_id[:8]} (idle {dt:.1f}d)")

            tbg = engine.apply_delta(tbg, delta)
            await engine.save(tbg)
            logger.info(f"TBG updated for {user_id[:8]}: +{len(delta.add_nodes)} nodes")

            if tbg.message_count % SNAPSHOT_EVERY == 0:
                await engine.save_snapshot(tbg)
    except Exception as e:
        logger.error(f"TBG bg update failed for {user_id[:8]}: {e}", exc_info=True)


async def facts_update_task(user_id: str, user_text: str, bot_reply: str, pool):
    """Extract and persist facts using the full FactEngine pipeline."""
    try:
        from fact_engine import FactEngine, update_facts_background

        # FIX: 2048 so JSON extraction doesn't get truncated
        async def llm_fn(prompt: str) -> str:
            return await llm_call_async(prompt, max_tokens=2048)

        engine = FactEngine(db_pool=pool)
        await update_facts_background(
            user_id=user_id,
            user_text=user_text,
            assistant_text=bot_reply,
            engine=engine,
            llm_call_fn=llm_fn,
        )
    except Exception as e:
        logger.warning(f"Facts update failed for {user_id[:8]}: {e}")


# ------------------------------------------------------------------
# Graph visualization
# ------------------------------------------------------------------

async def get_graph_state(pool, user_id: str) -> dict:
    try:
        from tbg_engine import TBGEngine
        
        engine = TBGEngine(db_pool=pool)
        tbg = await engine.load(user_id)
        
        if not tbg or not tbg.nodes:
            return {"nodes": 0, "edges": 0, "text": None, "insight": None}

        nodes = []
        for nid, n in tbg.nodes.items():
            if n.confidence > 0.25:
                nodes.append({
                    "id": nid,
                    "label": n.label,
                    "category": n.category,
                    "confidence": n.confidence
                })

        if not nodes:
            return {"nodes": 0, "edges": 0, "text": None, "insight": None}

        cat_emoji = {
            "goals": "🎯", "fears": "😰", "career": "💼", "mood": "🌡",
            "values": "💎", "identity": "🪞", "relationships": "👥", "finances": "💰"
        }

        lines = []
        by_cat = {}
        for n in nodes:
            by_cat.setdefault(n["category"], []).append(n)

        for cat in ["identity", "goals", "fears", "career", "mood", "values", "relationships", "finances"]:
            cat_nodes = sorted(by_cat.get(cat, []), key=lambda x: -x["confidence"])[:3]
            if cat_nodes:
                lines.append(f"\n{cat_emoji.get(cat, '•')} *{cat.upper()}*")
                for n in cat_nodes:
                    filled = int(n["confidence"] * 10)
                    bar = "█" * filled + "░" * (10 - filled)
                    lines.append(f"  `{bar}` {n['confidence']:.0%}  {n['label']}")

        edges = []
        for k, e in tbg.edges.items():
            if e.relation in ("blocks", "contradicts", "conflicts_with") and e.confidence > 0.55:
                edges.append(e)
        
        node_map = {n["id"]: n["label"] for n in nodes}
        if edges:
            lines.append("\n⚡ *CONFLICTS*")
            for e in edges[:3]:
                src = node_map.get(e.source_id, "?")[:25]
                tgt = node_map.get(e.target_id, "?")[:25]
                lines.append(f"  {src} ⛔ {tgt}")

        cognitive = _compute_insight(tbg)

        return {
            "nodes": len(nodes),
            "edges": len(tbg.edges),
            "text": "\n".join(lines),
            "insight": cognitive
        }
    except Exception as e:
        logger.error(f"get_graph_state error: {e}")
        return {"nodes": 0, "edges": 0, "text": None, "insight": None}




def _compute_insight(tbg) -> str:
    """Pure math from TBG graph. No hardcoded content strings."""
    if not tbg or not tbg.nodes:
        return ""

    nodes = tbg.nodes
    edges = tbg.edges

    # --- Core conflict ---
    conflict_edges = [
        e for e in edges.values()
        if e.relation in ("blocks", "contradicts", "conflicts_with") and e.confidence >= 0.5
    ]
    core_conflict = ""
    if conflict_edges:
        top = sorted(conflict_edges, key=lambda e: e.confidence, reverse=True)[0]
        src = nodes.get(top.source_id)
        tgt = nodes.get(top.target_id)
        if src and tgt:
            core_conflict = f"{src.label} vs {tgt.label} ({top.confidence:.0%})"

    # --- Node categories ---
    fear_nodes     = [n for n in nodes.values() if n.category == "fears" and n.confidence >= 0.4]
    goal_nodes     = [n for n in nodes.values() if n.category in ("goals", "career") and n.confidence >= 0.4]
    identity_nodes = [n for n in nodes.values() if n.category == "identity" and n.confidence >= 0.5]
    mood_nodes     = [n for n in nodes.values() if n.category == "mood" and n.confidence >= 0.4]

    avg_fear  = sum(n.confidence for n in fear_nodes)  / max(1, len(fear_nodes))
    avg_goal  = sum(n.confidence for n in goal_nodes)  / max(1, len(goal_nodes))
    avg_ident = sum(n.confidence for n in identity_nodes) / max(1, len(identity_nodes)) if identity_nodes else 0.0

    # Action proxy: nodes that signal actual movement (not just desire)
    action_nodes = [
        n for n in nodes.values()
        if any(kw in n.label.lower() for kw in ("start", "quit", "launch", "build", "resign", "left", "founded", "started"))
        and n.confidence >= 0.5
    ]
    avg_action = sum(n.confidence for n in action_nodes) / max(1, len(action_nodes)) if action_nodes else 0.0

    conflict_density = len(conflict_edges) / max(1, len(nodes))

    # --- Pattern detection (1. avoidance loop, 2. committed, 3. blocked, 4. exploration) ---
    # Avoidance: high fear + high desire + low action = stuck
    avoidance = avg_fear >= 0.6 and avg_goal >= 0.5 and avg_action < 0.35
    committed = avg_ident >= 0.75 and avg_goal > avg_fear and avg_action >= 0.4
    blocked   = conflict_density >= 0.25 and avg_fear >= 0.5 and avg_action < 0.3

    if committed:
        pattern = "commitment"
    elif avoidance:
        pattern = "avoidance loop"
    elif blocked:
        pattern = "blocked by conflict"
    else:
        pattern = "exploration"

    # --- Drift: only nodes with real previous value, threshold >= 0.05 ---
    drift_items = []
    for n in nodes.values():
        if n.confidence_prev is None:
            continue
        d = n.confidence - n.confidence_prev
        if abs(d) < 0.05:
            continue
        arrow = chr(8593) if d > 0 else chr(8595)
        drift_items.append((abs(d), f"{n.label[:28]} {arrow} {abs(d):.0%}"))

    drift_items.sort(key=lambda x: x[0], reverse=True)
    drift_parts = [label for _, label in drift_items[:3]]
    drift_str = ", ".join(drift_parts) if drift_parts else "insufficient data"

    # --- Decision Pressure ---
    dp = int(min(1.0, conflict_density * 2) * 35 + max(0.0, 0.8 - avg_fear) * 30 + avg_ident * 35)
    dp = max(0, min(100, dp))

    # --- Risk label (semantic, not numeric) ---
    if avoidance and avg_fear >= 0.7:
        risk_label = "chronic stagnation"
    elif len(conflict_edges) >= 3 and avg_fear >= 0.6:
        risk_label = "active conflict"
    elif avg_fear < 0.4 and avg_action >= 0.4:
        risk_label = "approaching decision"
    elif committed:
        risk_label = "stable"
    else:
        risk_label = "unresolved tension"

    # --- Intervention window ---
    if avg_ident >= 0.85 and dp >= 70:
        window = "closed — decision made"
    elif dp >= 50 and conflict_edges:
        window = "open"
    else:
        window = "partial"

    lines = []
    if core_conflict:
        lines.append(f"*Core conflict:* {core_conflict}")
    lines.append(f"*Pattern:* {pattern}")
    lines.append(f"*Drift:* {drift_str}")
    lines.append(f"*Pressure:* {dp}/100  |  *Risk:* {risk_label}")
    lines.append(f"*Window:* {window}")
    return "\n".join(lines)

# ------------------------------------------------------------------
# Handlers
# ------------------------------------------------------------------

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pool = await get_pool(ctx)
    count = await get_msg_count(pool, str(update.effective_user.id))
    remaining = max(0, MAX_FREE_MSGS - count)

    welcome = (
        "This maps what's actually going on with you — beliefs, conflicts, patterns.\n\n"
        "Builds a map of your beliefs, patterns, and conflicts from what you share.\n\n"
        "Talk about work, relationships, money, goals, what you're stuck on. "
        "The more specific, the more accurate.\n\n"
        f"You have *{remaining}* free messages.\n\n"
        "Commands: /graph · /insight · /reset\n\n"
        "To start: what's something in your life you keep telling yourself is fine — but probably isn't?"
    )
    await update.message.reply_text(welcome, parse_mode="Markdown")


async def cmd_graph(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    pool = await get_pool(ctx)
    msg = await update.message.reply_text("⏳ Building your map...")

    state = await get_graph_state(pool, user_id)

    if state["nodes"] == 0:
        await msg.edit_text(
            "🧠 Not enough yet.\n\n"
            "Talk about something that actually matters — work, relationships, money, "
            "what you want, what's blocking you."
        )
        return

    text = (
        f"🧠 *YOUR BELIEF MAP*\n"
        f"_{state['nodes']} beliefs · {state['edges']} connections_"
        f"{state['text']}"
    )
    if state["insight"]:
        text += f"\n\n💡 *Pattern:*\n_{state['insight']}_"

    await msg.edit_text(text, parse_mode="Markdown")


async def cmd_shift(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Use /graph — it shows your full belief map with changes over time."
    )


async def cmd_insight(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    pool = await get_pool(ctx)
    state = await get_graph_state(pool, user_id)

    if not state["insight"]:
        await update.message.reply_text(
            "Not enough to go on yet. Talk about something real — work, relationships, money, what you want."
        )
        return

    await update.message.reply_text(
        f"💡 *What I'm seeing:*\n\n_{state['insight']}_",
        parse_mode="Markdown"
    )


async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    pool = await get_pool(ctx)
    try:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM user_tbg WHERE user_id=$1", user_id)
            await conn.execute("DELETE FROM tbg_history WHERE user_id=$1", user_id)
            await conn.execute("DELETE FROM user_facts WHERE user_id=$1", user_id)
            await conn.execute("DELETE FROM fact_history WHERE user_id=$1", user_id)
            await conn.execute("DELETE FROM user_mode WHERE user_id=$1", user_id)
            await conn.execute("DELETE FROM user_dissonance WHERE user_id=$1", user_id)
            await conn.execute("DELETE FROM user_sensitivity WHERE user_id=$1", user_id)
        ctx.user_data.clear()
        await update.message.reply_text("Done. Memory cleared. Starting fresh.")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    user_text = update.message.text
    pool = await get_pool(ctx)

    if "msg_count" not in ctx.user_data:
        ctx.user_data["msg_count"] = await get_msg_count(pool, user_id)
    msg_count = ctx.user_data["msg_count"]
    if msg_count >= MAX_FREE_MSGS:
        await update.message.reply_text(
            f"You've used all {MAX_FREE_MSGS} free messages.\n\nTo continue: @alexey_tbg"
        )
        return

    ctx.user_data["msg_count"] = msg_count + 1

    history = ctx.user_data.get("history", [])
    history_str = "\n".join(f"{role}: {text}" for role, text in history[-6:])

    prompt = f"{SYSTEM}\n\nConversation so far:\n{history_str}\n\nUser: {user_text}\nYou:"
    try:
        # FIX: 250 tokens — enough for 2 sentences, doesn't truncate
        reply = await asyncio.wait_for(llm_call_async(prompt, max_tokens=250), timeout=45.0)
    except asyncio.TimeoutError:
        logger.error(f"Reply timed out for user {user_id[:8]}")
        reply = "Go on."
    except Exception as e:
        logger.error(f"Reply generation failed: {e}")
        reply = "Go on."

    history.append(("You", user_text))
    history.append(("Me", reply))
    ctx.user_data["history"] = history[-12:]

    await update.message.reply_text(reply)

    ctx.application.create_task(tbg_update_task(user_id, user_text, reply, pool))
    ctx.application.create_task(facts_update_task(user_id, user_text, reply, pool))

    new_count = msg_count + 1
    if new_count % 6 == 0 and new_count < MAX_FREE_MSGS:
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🧠 Belief map", callback_data="graph")],
            [InlineKeyboardButton("💡 Pattern", callback_data="insight")]
        ])
        remaining = MAX_FREE_MSGS - new_count
        await update.message.reply_text(
            f"_{new_count} messages in. The map is building._\n{remaining} left.",
            parse_mode="Markdown",
            reply_markup=keyboard
        )
    elif new_count % 3 == 0 and new_count < MAX_FREE_MSGS:
        await update.message.reply_text(
            "_Still mapping. Keep going — contradictions are what make it interesting._",
            parse_mode="Markdown"
        )


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "graph":
        user_id = str(query.from_user.id)
        pool = await get_pool(ctx)
        state = await get_graph_state(pool, user_id)

        if state["nodes"] == 0:
            await query.message.reply_text("Not enough data yet.")
            return

        text = f"🧠 *YOUR MAP*\n_{state['nodes']} beliefs · {state['edges']} connections_{state['text']}"
        if state["insight"]:
            text += f"\n\n💡 *Pattern:* {state['insight']}"

        await query.message.reply_text(text, parse_mode="Markdown")

    elif query.data == "insight":
        user_id = str(query.from_user.id)
        pool = await get_pool(ctx)
        state = await get_graph_state(pool, user_id)
        if state["insight"]:
            await query.message.reply_text(f"💡 {state['insight']}", parse_mode="Markdown")
        else:
            await query.message.reply_text("Not enough data yet.")


# ------------------------------------------------------------------
# Startup
# ------------------------------------------------------------------

async def post_init(app: Application):
    import asyncpg
    import json
    from pgvector.asyncpg import register_vector

    logger.info("Connecting to database...")
    
    async def init_connection(conn):
        await register_vector(conn)
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog"
        )

    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=2,
        max_size=10,
        init=init_connection
    )

    app.bot_data["pool"] = pool
    logger.info("Database pool ready.")

    from fact_engine import get_embed_model as _get_embed_model
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _get_embed_model)


async def post_shutdown(app: Application):
    pool = app.bot_data.get("pool")
    if pool:
        await pool.close()
        logger.info("Database pool closed.")


def main():
    if not TELEGRAM_TOKEN:
        print("ERROR: Set TELEGRAM_TOKEN environment variable")
        return

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("graph", cmd_graph))
    app.add_handler(CommandHandler("shift", cmd_shift))
    app.add_handler(CommandHandler("insight", cmd_insight))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_callback))

    logger.info("TBG Demo Bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
