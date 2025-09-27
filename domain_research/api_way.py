#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import asyncio
import aiohttp
import aiofiles
import csv
import logging
from pathlib import Path
from typing import Any, Optional, Tuple
from datetime import datetime

# ========= НАЛАШТУВАННЯ =========
INPUT         = Path("domains.txt")
OUTPUT        = Path("results.csv")
API_URL       = "https://api.apilayer.com/whois/query"
API_KEY       = os.getenv("WHOIS_API_KEY") or "D9u6u1a3PxrMjKtTg3LGZjFV8ZKZboAA"

WORKERS       = 300
RETRIES_500   = 10000
COOLDOWN_500  = 1.0

HUMAN_DATES   = True
LOGLEVEL      = logging.INFO
LOG_EVERY     = 100
# =================================

ALLOWED_TLDS = {
    ".com",".me",".net",".org",".sh",".io",".co",".club",".biz",".mobi",".info",".us",
    ".domains",".cloud",".fr",".au",".ru",".uk",".nl",".fi",".br",".hr",".ee",".ca",
    ".sk",".se",".no",".cz",".it",".in",".icu",".top",".xyz",".cn",".cf",".hk",".sg",
    ".pt",".site",".kz",".si",".ae",".do",".yoga",".xxx",".ws",".work",".wiki",".watch",
    ".wtf",".world",".website",".vip",".ly",".dev",".network",".company",".page",".rs",
    ".run",".science",".sex",".shop",".solutions",".so",".studio",".style",".tech",
    ".travel",".vc",".pub",".pro",".app",".press",".ooo",".de"
}

CSV_HEADER = ["domain", "creation", "expiration", "last_update"]

logging.basicConfig(level=LOGLEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("whois-apilayer")

def to_human(d: Optional[str]) -> str:
    if not d:
        return ""
    s = d.strip()
    if not s:
        return ""
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        if "T" in s:
            s = s.split("T", 1)[0]
        dt = datetime.fromisoformat(s)
        return dt.strftime("%d.%m.%Y")
    except Exception:
        try:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
            return dt.strftime("%d.%m.%Y")
        except Exception:
            return d

def parse_single(obj: Any) -> Tuple[str, str, str]:
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return "", "", ""
    src = obj.get("domain") if isinstance(obj.get("domain"), dict) else obj

    def pick(dct: dict, *names: str) -> str:
        for k in names:
            v = dct.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    created = pick(src, "created_date", "creation_date", "created", "registered")
    expires = pick(src, "expiration_date", "expiry_date", "expires", "paid_till")
    updated = pick(src, "updated_date", "update_date", "updated", "last_update")
    return created, expires, updated

async def ensure_csv_header(csv_path: Path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        async with aiofiles.open(csv_path, "w", encoding="utf-8", newline="") as f:
            await f.write(",".join(CSV_HEADER) + "\n")

def read_processed(csv_path: Path) -> set[str]:
    done: set[str] = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return done
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if not header or [h.lower() for h in header] != [h.lower() for h in CSV_HEADER]:
            f.seek(0); rdr = csv.reader(f)
        for row in rdr:
            if row:
                done.add(row[0].strip().lower())
    log.info(f"Resume: у CSV вже {len(done)} доменів")
    return done

def build_url(domain: str) -> str:
    return f"{API_URL}?domain={domain}"

async def fetch_json(session: aiohttp.ClientSession, url: str) -> Optional[Any]:
    attempt = 0
    while True:
        try:
            async with session.get(url, allow_redirects=True) as r:
                if r.status == 200:
                    return await r.json(content_type=None)
                if r.status == 404:
                    return {}
                if r.status == 500:
                    if attempt < RETRIES_500:
                        attempt += 1
                        await asyncio.sleep(COOLDOWN_500)
                        continue
                    log.warning(f"Give up after {RETRIES_500} retries: HTTP 500 -> {url}")
                    return {}
                if r.status in (401, 403):
                    log.error("Auth error (401/403). Перевір API-ключ WHOIS_API_KEY.")
                    return {}
                if r.status == 429:
                    log.warning("Rate limited (429) — пропускаю без ретраю.")
                    return {}
                log.warning(f"HTTP {r.status} (no retry) -> {url}")
                return {}
        except Exception as e:
            log.error(f"Request failed (no retry): {e} -> {url}")
            return {}

async def worker(name: str,
                 queue: "asyncio.Queue[str]",
                 session: aiohttp.ClientSession,
                 fout,
                 write_lock: asyncio.Lock,
                 progress: dict):
    while True:
        dom = await queue.get()
        if dom is None:
            queue.task_done()
            break

        # TLD check
        tld_ok = any(dom.endswith(tld) for tld in ALLOWED_TLDS)
        if not tld_ok:
            async with write_lock:
                await fout.write(f"{dom},,,\n")
                progress["n"] += 1
                if progress["n"] % LOG_EVERY == 0:
                    log.info(f"[{name}] done {progress['n']}")
                    await fout.flush()
            queue.task_done()
            continue

        obj = await fetch_json(session, build_url(dom))

        c = e = u = ""
        if obj:
            payload = obj.get("result") if isinstance(obj, dict) and "result" in obj else obj
            pc, pe, pu = parse_single(payload)
            c, e, u = (to_human(pc), to_human(pe), to_human(pu)) if HUMAN_DATES else (pc, pe, pu)

        async with write_lock:
            await fout.write(f"{dom},{c},{e},{u}\n")

        progress["n"] += 1
        if progress["n"] % LOG_EVERY == 0:
            log.info(f"[{name}] done {progress['n']}")
            await fout.flush()

        queue.task_done()

async def run():
    if not API_KEY or API_KEY.strip() == "":
        log.error("Немає API-ключа. Задай WHOIS_API_KEY або пропиши в API_KEY.")
        return
    if not INPUT.exists():
        log.error(f"Не знайдено {INPUT}")
        return

    await ensure_csv_header(OUTPUT)
    processed = read_processed(OUTPUT)

    domains: list[str] = []
    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            d = line.strip().lower()
            if not d or d.startswith("#"):
                continue
            if d in processed:
                continue
            domains.append(d)

    if not domains:
        log.info("Немає нових доменів — все вже в CSV.")
        return

    log.info(f"Старт: {len(domains)} доменів | WORKERS={WORKERS} | API={API_URL}")

    q: asyncio.Queue[str] = asyncio.Queue()
    for d in domains:
        q.put_nowait(d)
    for _ in range(WORKERS):
        q.put_nowait(None)

    timeout_obj = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=WORKERS, limit_per_host=WORKERS, ssl=True, ttl_dns_cache=300)

    session_headers = {
        "User-Agent": "whois-apilayer/1.0",
        "apikey": API_KEY,
    }

    write_lock = asyncio.Lock()
    progress = {"n": 0}

    async with aiohttp.ClientSession(timeout=timeout_obj, headers=session_headers, connector=connector) as session, \
               aiofiles.open(OUTPUT, "a", encoding="utf-8", newline="") as fout:

        tasks = [asyncio.create_task(worker(f"W{i:03d}", q, session, fout, write_lock, progress))
                 for i in range(WORKERS)]
        await q.join()
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(run())
