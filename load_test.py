#!/usr/bin/env python3
"""
TBG Local Load Test
Run: py load_test.py

Tests the local API against parallel requests.
Shows where the bottleneck is.
"""
import asyncio
import time
import json
import sys
from datetime import datetime

try:
    import httpx
except ImportError:
    print("Установи: pip install httpx")
    sys.exit(1)

API_URL    = "http://localhost:8001"
API_KEY = "your-super-secret-key-change-it"  # from your .env API_SECRET_KEY
USERS      = [f"loadtest_user_{i}" for i in range(10)]

MESSAGES = [
    "Устал от работы. Думаю о своём деле.",
    "Боюсь потерять стабильность — ипотека.",
    "Нашёл конкурента, но это значит рынок есть.",
    "Решил строить по вечерам, не уходить с работы.",
    "Договорился с другом — он войдёт как партнёр.",
]

RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
YEL   = "\033[93m"
CYAN  = "\033[96m"
DIM   = "\033[2m"

# ------------------------------------------------------------------
# Single request
# ------------------------------------------------------------------

async def single_request(client: httpx.AsyncClient, user_id: str, message: str) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.post(
            f"{API_URL}/memory/context",
            headers={"X-API-Key": API_KEY},
            json={"user_id": user_id, "message": message},
            timeout=30.0
        )
        elapsed = time.perf_counter() - t0
        return {
            "ok":      r.status_code == 200,
            "status":  r.status_code,
            "elapsed": elapsed,
            "user":    user_id,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "ok":      False,
            "status":  0,
            "elapsed": elapsed,
            "error":   str(e)[:60],
            "user":    user_id,
        }

# ------------------------------------------------------------------
# Test 1 — sequential (baseline)
# ------------------------------------------------------------------

async def test_sequential(client: httpx.AsyncClient):
    print(f"\n{BOLD}{CYAN}TEST 1 — Последовательно (baseline){RESET}")
    results = []
    for i, (user, msg) in enumerate(zip(USERS[:5], MESSAGES)):
        r = await single_request(client, user, msg)
        col = GREEN if r["ok"] else RED
        print(f"  {i+1}. {col}{'OK' if r['ok'] else 'ERR'}{RESET} "
              f"{r['elapsed']:.2f}s  {user[:20]}")
        results.append(r)

    ok      = sum(1 for r in results if r["ok"])
    avg     = sum(r["elapsed"] for r in results) / len(results)
    print(f"\n  Итог: {ok}/{len(results)} OK  avg={avg:.2f}s\n")
    return results

# ------------------------------------------------------------------
# Test 2 — N users in parallel
# ------------------------------------------------------------------

async def test_parallel(client: httpx.AsyncClient, concurrency: int):
    print(f"{BOLD}{CYAN}TEST 2 — Параллельно {concurrency} пользователей{RESET}")
    tasks = [
        single_request(client, f"loadtest_parallel_{i}", MESSAGES[i % len(MESSAGES)])
        for i in range(concurrency)
    ]
    t0      = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total   = time.perf_counter() - t0

    ok      = sum(1 for r in results if r["ok"])
    avg     = sum(r["elapsed"] for r in results) / len(results)
    mx      = max(r["elapsed"] for r in results)
    mn      = min(r["elapsed"] for r in results)

    col = GREEN if ok == concurrency else YEL if ok > concurrency // 2 else RED
    print(f"  {col}{ok}/{concurrency} OK{RESET}  "
          f"total={total:.2f}s  avg={avg:.2f}s  "
          f"min={mn:.2f}s  max={mx:.2f}s")

    errors = [r for r in results if not r["ok"]]
    if errors:
        print(f"  {RED}Ошибки:{RESET}")
        for e in errors[:3]:
            print(f"    status={e['status']}  {e.get('error', '')}")
    print()
    return results

# ------------------------------------------------------------------
# Test 3 — memory add + search
# ------------------------------------------------------------------

async def test_memory_cycle(client: httpx.AsyncClient):
    print(f"{BOLD}{CYAN}TEST 3 — Цикл add → search{RESET}")
    user_id = "loadtest_cycle_user"
    results = []

    for i, msg in enumerate(MESSAGES):
        # add
        t0 = time.perf_counter()
        try:
            r = await client.post(
                f"{API_URL}/memory/add",
                headers={"X-API-Key": API_KEY},
                json={
                    "user_id": user_id,
                    "text": msg,
                    "assistant_response": "Понял, продолжай.",
                    "sync": True
                },
                timeout=60.0
            )
            add_time = time.perf_counter() - t0
            add_ok   = r.status_code == 200
        except Exception as e:
            add_time = time.perf_counter() - t0
            add_ok   = False

        # search
        t0 = time.perf_counter()
        try:
            r = await client.post(
                f"{API_URL}/memory/search",
                headers={"X-API-Key": API_KEY},
                json={"user_id": user_id, "query": msg},
                timeout=15.0
            )
            search_time = time.perf_counter() - t0
            search_ok   = r.status_code == 200
            facts_count = len(r.json().get("data", {}).get("facts", [])) if search_ok else 0
        except Exception as e:
            search_time = time.perf_counter() - t0
            search_ok   = False
            facts_count = 0

        add_col    = GREEN if add_ok    else RED
        search_col = GREEN if search_ok else RED
        print(f"  msg{i+1}  "
              f"add={add_col}{'OK' if add_ok else 'ERR'}{RESET} {add_time:.2f}s  "
              f"search={search_col}{'OK' if search_ok else 'ERR'}{RESET} {search_time:.2f}s  "
              f"facts={facts_count}")
        results.append({"add_ok": add_ok, "search_ok": search_ok,
                        "add_time": add_time, "search_time": search_time})

    print()
    return results

# ------------------------------------------------------------------
# Test 4 — cognitive directive (the heaviest)
# ------------------------------------------------------------------

async def test_cognitive(client: httpx.AsyncClient, concurrency: int = 3):
    print(f"{BOLD}{CYAN}TEST 4 — Cognitive Directive x{concurrency} параллельно{RESET}")
    tasks = [
        client.post(
            f"{API_URL}/cognitive/directive",
            headers={"X-API-Key": API_KEY},
            json={
                "user_id": f"loadtest_cog_{i}",
                "message": MESSAGES[i % len(MESSAGES)]
            },
            timeout=60.0
        )
        for i in range(concurrency)
    ]

    t0      = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total   = time.perf_counter() - t0

    ok = sum(1 for r in results if not isinstance(r, Exception) and r.status_code == 200)
    col = GREEN if ok == concurrency else RED
    print(f"  {col}{ok}/{concurrency} OK{RESET}  total={total:.2f}s\n")

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  {RED}req{i}: {str(r)[:60]}{RESET}")
        elif r.status_code != 200:
            print(f"  {RED}req{i}: status={r.status_code}{RESET}")

# ------------------------------------------------------------------
# Final summary
# ------------------------------------------------------------------

def print_summary(seq_results, par5, par10):
    print(f"\n{BOLD}{CYAN}{'='*50}{RESET}")
    print(f"{BOLD}{CYAN}  ИТОГ{RESET}")
    print(f"{BOLD}{CYAN}{'='*50}{RESET}\n")

    avg_seq = sum(r["elapsed"] for r in seq_results) / len(seq_results)

    ok5  = sum(1 for r in par5  if r["ok"])
    ok10 = sum(1 for r in par10 if r["ok"])

    print(f"  Baseline (1 запрос):     {avg_seq:.2f}s avg")
    print(f"  5 параллельных:          {ok5}/5 OK")
    print(f"  10 параллельных:         {ok10}/10 OK")

    if ok10 == 10:
        verdict = f"{GREEN}Держит 10 параллельных — норм для MVP{RESET}"
    elif ok5 >= 4:
        verdict = f"{YEL}Держит 5, проблемы на 10 — нужна оптимизация{RESET}"
    else:
        verdict = f"{RED}Падает уже на 5 — критично починить до продажи{RESET}"

    print(f"\n  Вердикт: {verdict}\n")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def run():
    print(f"\n{BOLD}TBG LOCAL LOAD TEST — {datetime.now().strftime('%H:%M:%S')}{RESET}")
    print(f"{DIM}Target: {API_URL}{RESET}\n")

    # Check that the API is alive
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{API_URL}/health", timeout=5.0)
            if r.status_code == 200:
                print(f"{GREEN}API alive{RESET}\n")
            else:
                print(f"{RED}API вернул {r.status_code}{RESET}")
                return
        except Exception as e:
            print(f"{RED}API недоступен: {e}{RESET}")
            print(f"{DIM}Убедись что запущен: docker compose up -d{RESET}")
            return

    async with httpx.AsyncClient() as client:
        seq     = await test_sequential(client)
        par5    = await test_parallel(client, 5)
        par10   = await test_parallel(client, 10)
        await test_memory_cycle(client)
        await test_cognitive(client, 3)
        print_summary(seq, par5, par10)

if __name__ == "__main__":
    asyncio.run(run())
