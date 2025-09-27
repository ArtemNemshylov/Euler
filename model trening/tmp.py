import asyncio
import json
import re
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

SITEMAP_URL = "https://jobs.dou.ua/sitemap-vacancies.xml"
OUTPUT_FILE = Path("vacancies.jsonl")

BROWSERS = 5
TIMEOUT = 3000
RETRIES = 3


async def fetch_sitemap(page):
    await page.goto(SITEMAP_URL, wait_until="domcontentloaded")
    content = await page.content()
    urls = re.findall(r"<loc>(.*?)</loc>", content)
    print(f"Знайдено {len(urls)} вакансій у sitemap")
    return urls


async def scrape_vacancy(page, url):
    for attempt in range(1, RETRIES + 2):  # перша спроба + ретраї
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT)

            date = await page.locator("div.date").first.inner_text(timeout=TIMEOUT)
            title = await page.locator("h1.g-h2").first.inner_text(timeout=TIMEOUT)
            location = await page.locator("div.sh-info span.place").first.inner_text(timeout=TIMEOUT)
            desc = await page.locator("div.b-typo.vacancy-section").first.inner_html(timeout=TIMEOUT)

            return {
                "url": url,
                "title": title.strip(),
                "date": date.strip(),
                "location": location.strip(),
                "description_html": desc.strip()
            }
        except PlaywrightTimeoutError:
            print(f"[WARN] Timeout on {url}, attempt {attempt}")
            if attempt > RETRIES:
                return {"url": url, "error": "Timeout after retries"}
        except Exception as e:
            print(f"[ERR] {url} -> {e}")
            return {"url": url, "error": str(e)}


async def browser_worker(playwright, queue, out_file, worker_id):
    browser = await playwright.chromium.launch(headless=False)
    page = await browser.new_page()
    try:
        while not queue.empty():
            url = await queue.get()
            data = await scrape_vacancy(page, url)
            out_file.write(json.dumps(data, ensure_ascii=False) + "\n")
            out_file.flush()
            print(f"[Browser {worker_id}] scraped: {url}")
            queue.task_done()
    finally:
        await browser.close()


async def main():
    async with async_playwright() as p:
        # отримуємо список вакансій
        tmp_browser = await p.chromium.launch(headless=True)
        page = await tmp_browser.new_page()
        urls = await fetch_sitemap(page)
        await tmp_browser.close()

        # черга всіх вакансій
        queue = asyncio.Queue()
        for u in urls:
            await queue.put(u)

        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            tasks = [asyncio.create_task(browser_worker(p, queue, f, idx+1)) for idx in range(BROWSERS)]
            await queue.join()
            for t in tasks:
                t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
