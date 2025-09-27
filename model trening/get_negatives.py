#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xlsx_to_text_json.py
- Читає XLSX, колонка "hre" (також підтримує "href"), бере URL-и
- Відкриває сторінки через Playwright (5 паралельних браузерів)
- Браузери НЕ закриваються між навігаціями; кожні 100 відкриттів — перезапуск конкретного браузера
- Тягне HTML, чистить до plain text (без скриптів/стилів), записує у JSON як { "domain": "text", ... }
- Повторно домени не пише (перевіряє існуючий JSON)

Залежності:
    pip install playwright pandas beautifulsoup4 lxml
    playwright install chromium
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ───────── CONFIG ─────────
INPUT_XLSX      = Path("Ukraine 1970 IT Product companies.xlsx")
OUTPUT_JSON     = Path("output.json")        # мапа {domain: text}
CONCURRENCY     = 3                         # кількість паралельних браузерів
NAV_TIMEOUT_MS  = 25000                      # таймаут завантаження сторінки
RETRIES         = 1                          # додаткові спроби на фейл
RESTART_EVERY   = 100                        # перезапуск браузера після N навігацій
HEADLESS        = True                       # запуск без GUI
WAIT_UNTIL      = "networkidle"              # умова очікування контенту
# ─────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("xlsx2json")


def load_urls_from_xlsx(xlsx_path: Path) -> List[str]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Немає файлу: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    col = None
    for name in df.columns:
        if str(name).strip().lower() in ("hre", "href"):
            col = name
            break
    if col is None:
        raise ValueError("У XLSX не знайдено колонку 'hre' або 'href'")

    urls = []
    for v in df[col].dropna().astype(str):
        s = v.strip()
        if not s:
            continue
        # Додай схему якщо відсутня
        if not s.lower().startswith(("http://", "https://")):
            s = "http://" + s
        urls.append(s)
    return urls


def extract_domain(url: str) -> Optional[str]:
    try:
        p = urlparse(url)
        if p.netloc:
            return p.netloc.lower()
    except Exception:
        return None
    return None


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # вирізаємо скрипти/стилі/носкрипти
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # компактний пробіл
    text = " ".join(text.split())
    return text


def load_existing(output_path: Path) -> Dict[str, str]:
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception as e:
            log.warning(f"Не вдалось прочитати існуючий JSON ({e}). Стартуємо з порожнього.")
    return {}


async def robust_goto(page, url: str) -> None:
    await page.goto(url, wait_until=WAIT_UNTIL, timeout=NAV_TIMEOUT_MS)


class BrowserWorker:
    def __init__(
        self,
        id_: int,
        url_queue: "asyncio.Queue[Tuple[str, str]]",
        results: Dict[str, str],
        results_lock: asyncio.Lock,
        existing_domains: Set[str],
    ):
        self.id = id_
        self.url_queue = url_queue
        self.results = results
        self.results_lock = results_lock
        self.existing_domains = existing_domains

        self.browser = None
        self.context = None
        self.page = None
        self.nav_count = 0

    async def start(self, pw):
        await self._start_browser(pw)

    async def _start_browser(self, pw):
        if self.browser:
            try:
                await self.browser.close()
            except Exception:
                pass

        self.browser = await pw.chromium.launch(headless=HEADLESS)
        self.context = await self.browser.new_context()
        # одна вкладка на воркер (перевідкривати сторінки у ній)
        self.page = await self.context.new_page()
        self.nav_count = 0
        log.info(f"[W{self.id}] Браузер запущено")

    async def _maybe_restart(self, pw):
        if self.nav_count >= RESTART_EVERY:
            log.info(f"[W{self.id}] Перезапуск браузера після {self.nav_count} навігацій")
            await self._start_browser(pw)

    async def run(self, pw):
        await self.start(pw)
        try:
            while True:
                try:
                    url, domain = await asyncio.wait_for(self.url_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    # черга спорожніла
                    break

                # якщо домен вже є — пропустити
                if domain in self.existing_domains:
                    self.url_queue.task_done()
                    continue

                ok = await self.process_one(pw, url, domain)
                self.url_queue.task_done()
                if ok:
                    async with self.results_lock:
                        self.results[domain] = ok  # ok — це text
                        self.existing_domains.add(domain)
        finally:
            # НЕ закриваємо глобально всі браузери тут — тільки свій
            try:
                if self.browser:
                    await self.browser.close()
            except Exception:
                pass
            log.info(f"[W{self.id}] Завершено")

    async def process_one(self, pw, url: str, domain: str) -> Optional[str]:
        tries = 1 + max(0, int(RETRIES))
        for attempt in range(1, tries + 1):
            try:
                await self._maybe_restart(pw)
                await robust_goto(self.page, url)
                # невелике очікування рендеру
                await self.page.wait_for_timeout(300)
                html = await self.page.content()
                text = html_to_text(html)
                self.nav_count += 1
                if text:
                    log.info(f"[W{self.id}] OK {domain} ({len(text)} симв.)")
                    return text
                else:
                    log.warning(f"[W{self.id}] Порожній текст: {domain}")
                    return ""
            except PWTimeout:
                log.warning(f"[W{self.id}] Таймаут: {url} (спроба {attempt}/{tries})")
            except Exception as e:
                log.warning(f"[W{self.id}] Помилка {domain}: {e} (спроба {attempt}/{tries})")
            # невеликий backoff між спробами
            await asyncio.sleep(0.5 * attempt)
        return None


async def save_periodically(path: Path, data_ref: Dict[str, str], stop_evt: asyncio.Event):
    """Періодично флешить JSON на диск, щоб не втратити прогрес."""
    while not stop_evt.is_set():
        await asyncio.sleep(3.0)
        try:
            tmp = path.with_suffix(".tmp.json")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data_ref, f, ensure_ascii=False, indent=2)
            tmp.replace(path)
        except Exception as e:
            log.error(f"Помилка збереження JSON: {e}")
    # фінальний запис
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data_ref, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Фінальне збереження провалено: {e}")


async def main():
    urls = load_urls_from_xlsx(INPUT_XLSX)

    # нормалізуємо до (url, domain) і фільтруємо валідні
    pairs: List[Tuple[str, str]] = []
    for u in urls:
        d = extract_domain(u)
        if d:
            pairs.append((u, d))
        else:
            log.warning(f"Пропуск (немає домену): {u}")

    # завантажуємо існуючі результати
    results: Dict[str, str] = load_existing(OUTPUT_JSON)
    existing_domains: Set[str] = set(results.keys())
    log.info(f"Вже у файлі: {len(existing_domains)} доменів")

    # готуємо чергу тільки з нових
    q: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
    new_count = 0
    for u, d in pairs:
        if d not in existing_domains:
            q.put_nowait((u, d))
            new_count += 1
    log.info(f"До збору: {new_count} з {len(pairs)}")

    if new_count == 0:
        log.info("Немає нових доменів. Вихід.")
        return

    results_lock = asyncio.Lock()
    stop_evt = asyncio.Event()

    saver_task = asyncio.create_task(save_periodically(OUTPUT_JSON, results, stop_evt))

    async with async_playwright() as pw:
        workers = [
            BrowserWorker(i + 1, q, results, results_lock, existing_domains)
            for i in range(CONCURRENCY)
        ]
        tasks = [asyncio.create_task(w.run(pw)) for w in workers]
        # очікуємо виконання усіх воркерів
        await asyncio.gather(*tasks, return_exceptions=False)

    # зупиняємо saver
    stop_evt.set()
    await saver_task

    log.info(f"Готово. Всього у JSON: {len(results)} доменів")
    log.info(f"Файл: {OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Зупинено користувачем")
