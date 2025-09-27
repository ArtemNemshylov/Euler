#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "vacancy_clf"
MAX_LENGTH = 512

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def load_best_threshold(model_dir: str, default_thr: float = 0.5) -> float:
    path = os.path.join(model_dir, "best_threshold.json")
    if os.path.exists(path):
        try:
            obj = json.load(open(path, "r", encoding="utf-8"))
            return float(obj.get("best_threshold", default_thr))
        except Exception:
            return default_thr
    return default_thr

def predict_one(text: str, tok, mdl, threshold: float):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        out = mdl(**inputs)
        logit = out.logits.squeeze(-1).cpu().numpy()
    prob = float(sigmoid(logit))
    return {
        "is_vacancy": bool(prob >= threshold),
        "score_vacancy": prob,
        "score_not_vacancy": 1.0 - prob,
    }

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    mdl.eval()
    thr = load_best_threshold(MODEL_DIR, 0.5)

    tests = [
        # 1) Пряме заперечення найму (схоже на сторінку кар’єри, але НЕ вакансія)
        "Ми зараз не наймаємо та вакансій немає. Слідкуйте за оновленнями на сторінці кар’єри.",
        # 2) Вакансія закрита
        "Вакансія «Python розробник» ЗАКРИТА. Нові заявки не приймаємо.",
        # 3) Список вимог без слова «вакансія»
        "Обов'язки: підтримка мікросервісів, CI/CD, моніторинг. Вимоги: Docker, Kubernetes, Python 3.10.",
        # 4) Кар’єрна сторінка без конкретних позицій
        "Ми цінуємо людей та створюємо комфортні умови, але зараз відкритих позицій немає.",
        # 5) Стаття про співбесіди (контент про найм, але не пропозиція роботи)
        "Як скласти технічну співбесіду: поради рекрутера, частина 2.",
        # 6) Лише «надсилайте резюме» без опису
        "Надсилайте резюме на hr@company.com — розглянемо кандидатів. Деталі згодом.",
        # 7) Реклама курсів зі словами «зарплата», «працевлаштування»
        "Онлайн-курс QA: зарплата до $2000, гарантія працевлаштування, відгуки студентів.",
        # 8) Новина з ключовими словами «вакансії»
        "Мінцифра опублікувала статистику: кількість IT-вакансій зросла на 12% у вересні.",
        # 9) HTML-сміття як у сирих сторінках
        "<div><h1>Senior Golang Engineer</h1><p>Обов'язки: розробка API.</p><p>Віддалено</p></div>",
        # 10) Дуже короткий спам-сигнал
        "Робота!!! Терміново потрібні люди!!!",
        # 11) Лише посилання (без контенту)
        "https://example.com/jobs/python-developer",
        # 12) Мішанина мов та напів-опис
        "We’re hiring React dev у Києві: TypeScript, hooks, SSR. Full-time, релокейт/remote.",
        # 13) Оголошення, схоже на вакансію, але це підряд/аутстаф
        "Шукаємо партнерів для підряду на проєкт: React/Node, оплата погодинна, короткостроково.",
        # 14) Політика компанії щодо рівних можливостей (кар’єрна сторінка)
        "Наша компанія дотримується політики рівних можливостей працевлаштування. Поточні позиції відсутні.",
        # 15) Повноцінний опис вакансії
        "Вакансія: Data Engineer. Обов'язки: побудова ETL на Airflow, Spark. Вимоги: SQL, Python, AWS. Віддалено, $3–5k.",
    ]

    for i, t in enumerate(tests, 1):
        r = predict_one(t, tok, mdl, thr)
        print(json.dumps({"id": i, "text": t[:160], **r}, ensure_ascii=False))

if __name__ == "__main__":
    main()
