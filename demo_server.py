"""
TBG Demo Server - FastAPI backend with SQLite storage and mock LLM
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from tbg_schema import UserTBG, BeliefNode, BeliefEdge, TBGDelta, ConfidenceSnapshot
from tbg_engine import TBGEngine, update_tbg_background

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import aiosqlite

DB_PATH = Path("demo_tbg.db")

class SQLiteTBGAdapter:
    """Drop-in replacement for asyncpg.Pool in demo mode."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
    
    async def _init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_tbg (
                    user_id TEXT PRIMARY KEY,
                    nodes_data TEXT,
                    edges_data TEXT,
                    message_count INTEGER DEFAULT 0,
                    last_sync TEXT,
                    last_decay TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tbg_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    message_count INTEGER,
                    snapshot_data TEXT,
                    created_at TEXT,
                    UNIQUE(user_id, message_count)
                )
            """)
            await db.commit()
    
    def _parse_datetime(self, value):
        """Convert SQLite string to datetime."""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                # Try ISO format
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except:
                return None
        return value
    
    async def fetchrow(self, query: str, *args):
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            processed_args = tuple(json.dumps(arg) if isinstance(arg, dict) else arg for arg in args)
            async with db.execute(query, processed_args) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                result = dict(row)
                # Parse JSON strings back to dicts
                for key in ['nodes_data', 'edges_data', 'snapshot_data']:
                    if key in result and result[key]:
                        try:
                            result[key] = json.loads(result[key])
                        except (json.JSONDecodeError, TypeError):
                            pass
                
                # Convert datetime fields
                if 'last_sync' in result:
                    result['last_sync'] = self._parse_datetime(result['last_sync'])
                if 'last_decay' in result:
                    result['last_decay'] = self._parse_datetime(result['last_decay'])
                
                return result
    
    async def fetch(self, query: str, *args):
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, args) as cursor:
                rows = await cursor.fetchall()
                results = []
                for row in rows:
                    result = dict(row)
                    for key in ['nodes_data', 'edges_data', 'snapshot_data']:
                        if key in result and result[key]:
                            try:
                                result[key] = json.loads(result[key])
                            except (json.JSONDecodeError, TypeError):
                                pass
                    
                    # Convert datetime fields
                    if 'last_sync' in result:
                        result['last_sync'] = self._parse_datetime(result['last_sync'])
                    if 'last_decay' in result:
                        result['last_decay'] = self._parse_datetime(result['last_decay'])
                    
                    results.append(result)
                return results
    
    async def execute(self, query: str, *args):
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            processed_args = []
            for arg in args:
                if isinstance(arg, (dict, list)):
                    processed_args.append(json.dumps(arg, ensure_ascii=False))
                elif isinstance(arg, datetime):
                    # datetime -> ISO string for SQLite
                    processed_args.append(arg.isoformat())
                else:
                    processed_args.append(arg)
            await db.execute(query, tuple(processed_args))
            await db.commit()

MOCK_LLM_RESPONSES = [
    {
        "reasoning": "User expressed career anxiety",
        "add_nodes": [
            {"category": "career", "label": "IT job", "confidence": 0.7, "source": "explicit"},
            {"category": "mood", "label": "anxiety", "confidence": 0.6, "source": "inferred"},
        ],
        "add_edges": [
            {"source_id": "IT job", "target_id": "anxiety", "relation": "causes", "confidence": 0.8}
        ],
        "reinforce_ids": [],
        "contradict_ids": []
    },
    {
        "reasoning": "User mentioned learning goals",
        "add_nodes": [
            {"category": "goals", "label": "learn Python", "confidence": 0.8, "source": "explicit"},
            {"category": "career", "label": "skill growth", "confidence": 0.6, "source": "inferred"},
        ],
        "add_edges": [
            {"source_id": "learn Python", "target_id": "skill growth", "relation": "motivates", "confidence": 0.7}
        ],
        "reinforce_ids": [],
        "contradict_ids": []
    },
    {
        "reasoning": "User talked about relationships",
        "add_nodes": [
            {"category": "relationships", "label": "family conflict", "confidence": 0.5, "source": "explicit"},
            {"category": "mood", "label": "sadness", "confidence": 0.5, "source": "inferred"},
        ],
        "add_edges": [
            {"source_id": "family conflict", "target_id": "sadness", "relation": "causes", "confidence": 0.6}
        ],
        "reinforce_ids": [],
        "contradict_ids": []
    },
    {
        "reasoning": "User is optimistic about future",
        "add_nodes": [
            {"category": "values", "label": "optimism", "confidence": 0.7, "source": "explicit"},
            {"category": "goals", "label": "long-term planning", "confidence": 0.6, "source": "inferred"},
        ],
        "add_edges": [
            {"source_id": "optimism", "target_id": "long-term planning", "relation": "motivates", "confidence": 0.7}
        ],
        "reinforce_ids": [],
        "contradict_ids": []
    },
]

_llm_counter = 0

async def mock_llm_call(prompt: str) -> str:
    """Simulate LLM extraction with rotating responses."""
    global _llm_counter
    await asyncio.sleep(0.3)
    response = MOCK_LLM_RESPONSES[_llm_counter % len(MOCK_LLM_RESPONSES)]
    _llm_counter += 1
    logger.info(f"MOCK LLM: {response.get('reasoning')}")
    return json.dumps(response, ensure_ascii=False)

app = FastAPI(title="TBG Demo")

# Serve static files
STATIC_DIR = Path(__file__).parent / "demo_static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global engine instance
sqlite_db = SQLiteTBGAdapter()
engine = TBGEngine(sqlite_db)

class ChatMessage(BaseModel):
    user_id: str
    message: str

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the landing page."""
    landing_path = Path(__file__).parent / "landing.html"
    if landing_path.exists():
        return landing_path.read_text(encoding="utf-8")
    return "<h1>TBG Demo</h1><p>Create landing.html</p>"

@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Serve the demo HTML page."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>TBG Demo</h1><p>Create demo_static/index.html</p>"

@app.get("/api/graph/{user_id}")
async def get_graph(user_id: str):
    """Get current belief graph for visualization."""
    tbg = await engine.load(user_id)
    
    nodes = []
    for node_id, node in tbg.nodes.items():
        nodes.append({
            "id": node_id,
            "label": node.label,
            "category": node.category,
            "confidence": node.confidence,
            "confidence_prev": node.confidence_prev,
            "source": node.source,
            "updated_at": node.updated_at.isoformat() if node.updated_at else None
        })
    
    edges = []
    for edge_key, edge in tbg.edges.items():
        edges.append({
            "source": edge.source_id,
            "target": edge.target_id,
            "relation": edge.relation,
            "confidence": edge.confidence
        })
    
    return {
        "user_id": user_id,
        "message_count": tbg.message_count,
        "nodes": nodes,
        "edges": edges,
        "summary": tbg.summary()
    }

@app.post("/api/chat")
async def chat(msg: ChatMessage):
    """Process chat message and update TBG."""
    user_text = msg.message
    
    assistant_responses = [
        "Got it. Tell me more.",
        "Interesting. How does that affect you?",
        "I see. What else is on your mind?",
        "Thanks for sharing. Let's continue.",
        "Understood. That's an important topic."
    ]
    import random
    assistant_text = random.choice(assistant_responses)
    
    async def _update_with_logging():
        try:
            logger.info(f"🚀 TBG update for {msg.user_id[:8]}...")
            await update_tbg_background(
                user_id=msg.user_id,
                user_text=user_text,
                assistant_text=assistant_text,
                engine=engine,
                llm_call_fn=mock_llm_call
            )
            logger.info(f"✅ TBG update completed for {msg.user_id[:8]}")
        except Exception as e:
            logger.error(f"❌ TBG update failed: {e}", exc_info=True)
    
    asyncio.create_task(_update_with_logging())
    
    return {
        "user_id": msg.user_id,
        "user_message": user_text,
        "assistant_message": assistant_text
    }

@app.post("/api/reset/{user_id}")
async def reset_user(user_id: str):
    """Reset user data for demo."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM user_tbg WHERE user_id = ?", (user_id,))
        await db.execute("DELETE FROM tbg_history WHERE user_id = ?", (user_id,))
        await db.commit()
    return {"status": "ok", "message": f"User {user_id} reset"}

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket for real-time graph updates."""
    await websocket.accept()
    logger.info(f"🔌 WebSocket connected for {user_id[:8]}")
    
    last_message_count = -1
    
    try:
        while True:
            tbg = await engine.load(user_id)
            
            if tbg.message_count != last_message_count:
                last_message_count = tbg.message_count
                
                nodes = []
                for node_id, node in tbg.nodes.items():
                    nodes.append({
                        "id": node_id,
                        "label": node.label,
                        "category": node.category,
                        "confidence": round(node.confidence, 2),
                        "source": node.source
                    })
                
                edges = []
                for edge in tbg.edges.values():
                    edges.append({
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "relation": edge.relation,
                        "confidence": round(edge.confidence, 2)
                    })
                
                await websocket.send_json({
                    "message_count": tbg.message_count,
                    "nodes": nodes,
                    "edges": edges,
                    "summary": tbg.summary()
                })
                logger.info(f"📤 Sent update: {len(nodes)} nodes, {len(edges)} edges")
            
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected for {user_id[:8]}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    import sys
    
    port = 8000
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])
    
    print(f"\n🚀 TBG Demo Server starting on http://localhost:{port}")
    print(f"   Landing page: http://localhost:{port}")
    print(f"   Demo: http://localhost:{port}/demo")
    print(f"   API docs: http://localhost:{port}/docs\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
