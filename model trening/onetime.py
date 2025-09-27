#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repair_vacancy_model.py
Відновлює валідну HF-модель у папці OUTPUT_DIR з ваг, збережених Trainer-ом для обгортки.
"""
import os, torch, json
from transformers import AutoConfig, AutoModelForSequenceClassification

OUTPUT_DIR = "vacancy_clf"
BASE_MODEL_ID = "xlm-roberta-base"   # та ж базова, що й під час тренування

def main():
    # шукаємо ваги
    candidates = [
        os.path.join(OUTPUT_DIR, "pytorch_model.bin"),
        os.path.join(OUTPUT_DIR, "adapter_model.bin"),
        os.path.join(OUTPUT_DIR, "model.safetensors"),
    ]
    ckpt_path = None
    for p in candidates:
        if os.path.exists(p):
            ckpt_path = p
            break
    if ckpt_path is None:
        raise FileNotFoundError("Не знайдено ваги у vacancy_clf (pytorch_model.bin / model.safetensors).")

    # готуємо базову модель з правильним config
    cfg = AutoConfig.from_pretrained(BASE_MODEL_ID)
    cfg.num_labels = 1
    cfg.problem_type = "multi_label_classification"
    base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_ID, config=cfg)

    # вантажимо стейт
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # якщо ключі мають префікс "base.", знімемо його
    new_state = {}
    for k, v in state.items():
        if k.startswith("base."):
            new_state[k[len("base."):]] = v
        else:
            new_state[k] = v

    # пробуємо завантажити у базову модель
    missing, unexpected = base.load_state_dict(new_state, strict=False)
    print("[INFO] missing keys:", missing)
    print("[INFO] unexpected keys:", unexpected)

    # зберігаємо валідну модель
    base.save_pretrained(OUTPUT_DIR)
    print("[OK] Модель перепаковано. Тепер можна: AutoModelForSequenceClassification.from_pretrained('vacancy_clf')")

if __name__ == "__main__":
    main()
