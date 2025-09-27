#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repair_vacancy_model.py
Ремонт збереженої моделі з кастомної обгортки:
- підтримка pytorch_model.bin та model.safetensors
- прибирає префікс 'base.' у ключах
- зберігає валідну HF-модель у vacancy_clf/
"""

import os
import json
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

OUTPUT_DIR = "vacancy_clf"
BASE_MODEL_ID = "xlm-roberta-base"   # така ж, як під час тренування

def load_state_any(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file as st_load
        return st_load(path, device="cpu")
    else:
        # pytorch_model.bin або ін.
        return torch.load(path, map_location="cpu", weights_only=False)

def main():
    if not os.path.isdir(OUTPUT_DIR):
        raise FileNotFoundError(f"Немає каталогу: {OUTPUT_DIR}")

    # знайти файл ваг
    candidates = [
        os.path.join(OUTPUT_DIR, "model.safetensors"),
        os.path.join(OUTPUT_DIR, "pytorch_model.bin"),
        # іноді Trainer кладе у підчекпойнти
    ]

    # якщо в корені немає — пошукай останній checkpoint-*/
    if not any(os.path.exists(p) for p in candidates):
        ckpts = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        ckpts = [d for d in ckpts if os.path.isdir(d)]
        if not ckpts:
            raise FileNotFoundError("Не знайдено ані ваг у корені, ані підкаталогів checkpoint-*")
        ckpts.sort(key=lambda p: int(p.split("-")[-1]))
        latest = ckpts[-1]
        candidates = [
            os.path.join(latest, "model.safetensors"),
            os.path.join(latest, "pytorch_model.bin"),
        ]
        print(f"[INFO] Використовую чекпойнт: {latest}")

    ckpt_path = None
    for p in candidates:
        if os.path.exists(p):
            ckpt_path = p
            break
    if ckpt_path is None:
        raise FileNotFoundError("Не знайдено ваг: model.safetensors або pytorch_model.bin")

    print(f"[INFO] Ваги: {ckpt_path}")

    # підняти чисту базову модель з правильним config
    cfg = AutoConfig.from_pretrained(BASE_MODEL_ID)
    cfg.num_labels = 1
    cfg.problem_type = "multi_label_classification"
    base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_ID, config=cfg)

    # завантажити state_dict з підтримкою safetensors/pickle
    state = load_state_any(ckpt_path)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # зняти можливі префікси
    new_state = {}
    for k, v in state.items():
        if k.startswith("base."):
            new_state[k[len("base."):]] = v
        elif k.startswith("model."):  # на всяк
            new_state[k[len("model."):]] = v
        else:
            new_state[k] = v

    missing, unexpected = base.load_state_dict(new_state, strict=False)
    print("[INFO] missing keys:", missing)
    print("[INFO] unexpected keys:", unexpected)

    # зберегти валідну модель у корінь OUTPUT_DIR
    base.save_pretrained(OUTPUT_DIR)
    # залиш токенайзер як є (він уже там має бути після тренування)

    print("[OK] Готово. Тепер працює:")
    print("from transformers import AutoModelForSequenceClassification;")
    print("mdl = AutoModelForSequenceClassification.from_pretrained('vacancy_clf')")

if __name__ == "__main__":
    main()
