#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from bs4 import BeautifulSoup

INPUT = "vacancies.jsonl"
OUTPUT = "vacancies_clean.jsonl"

def clean_html(raw_html: str) -> str:
    return BeautifulSoup(raw_html, "html.parser").get_text(" ", strip=True)

with open(INPUT, "r", encoding="utf-8") as fin, open(OUTPUT, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        url = obj.get("url")
        raw_html = obj.get("description_html")
        if url and raw_html:
            text = clean_html(raw_html)
            fout.write(json.dumps({"url": url, "text": text}, ensure_ascii=False) + "\n")

print(f"Готово. Записано у {OUTPUT}")
