#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_vacancy_classifier.py (full, compat with old transformers)
Вхід:
  positives: vacancies_clean.jsonl -> {"url": "...", "text": "..."}
  negatives: output.jsonl          -> довільні ключі; текст витягується автоматично

Налаштовано під 12GB VRAM:
  xlm-roberta-base, MAX_LENGTH=512, batch=8, grad_acc=2, epochs=6, fp16=True
BCEWithLogitsLoss (+ pos_weight), метрики: acc/prec/recall/F1/ROC-AUC/LogLoss.
Cosine LR, warmup_ratio=0.1, early stopping (якщо бібліотека підтримує callbacks).
Зворотна сумісність з давніми версіями transformers (без evaluation_strategy).
"""

import os
import json
import inspect
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, log_loss, roc_curve
)
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed
)

# EarlyStopping може бути відсутній у старих версіях — врахуємо це нижче.
try:
    from transformers import EarlyStoppingCallback
except Exception:
    EarlyStoppingCallback = None

from bs4 import BeautifulSoup

# ───────── CONFIG ─────────
POSITIVES = "vacancies_clean.jsonl"   # {"url": "...", "text": "..."}
NEGATIVES = "output.jsonl"            # будь-що; текст дістаємо самі
TEXT_FIELD: Optional[str] = None

MODEL_ID = "xlm-roberta-base"
OUTPUT_DIR = "vacancy_clf"

SEED = 42
EPOCHS = 6
BATCH_SIZE = 8
GRAD_ACC = 2              # ефективний батч ≈ 16
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
LR_SCHEDULER = "cosine"   # у старих версіях може бути не підтримано
MAX_LENGTH = 512
VAL_SIZE = 0.1
PATIENCE = 3
GRADIENT_CHECKPOINTING = False
FP16 = True               # на CUDA True; на CPU/без CUDA ігнорується
BF16 = False

FREEZE_EMBEDDINGS = False
USE_CLASS_WEIGHT = True

LOGGING_STEPS = 50
EVAL_STEPS = None
SAVE_TOTAL_LIMIT = 3

TEXT_KEYS_CANDIDATES = (
    "text", "content", "clean_text", "body", "html", "description_html",
    "article", "data", "raw"
)

# ───────── сумісність аргументів ─────────
def supports_arg(cls, name: str) -> bool:
    try:
        return name in inspect.signature(cls.__init__).parameters
    except Exception:
        return False

def filter_kwargs(cls, kwargs: dict) -> dict:
    sig = {}
    try:
        sig = inspect.signature(cls.__init__).parameters
    except Exception:
        return {}
    return {k: v for k, v in kwargs.items() if k in sig}

# ───────── IO helpers ─────────
def file_info(path: str) -> Tuple[bool, int]:
    if not os.path.exists(path):
        return False, 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return True, sum(1 for _ in f)
    except Exception:
        return True, 0

def sniff_jsonl_keys(path: str, sample_lines: int = 50) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    seen = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if seen >= sample_lines:
                break
            line = line.strip()
            if not line or (not line.startswith("{")):
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, (str, list)) or v is None:
                            freq[k] = freq.get(k, 0) + 1
                seen += 1
            except Exception:
                continue
    return dict(sorted(freq.items(), key=lambda x: (-x[1], x[0])))

def clean_html(raw_html: str) -> str:
    return BeautifulSoup(raw_html, "html.parser").get_text(" ", strip=True)

def _extract_text_from_obj(row: Dict[str, Any], text_field: Optional[str] = None) -> Optional[str]:
    if text_field and text_field in row and isinstance(row[text_field], str):
        t = row[text_field].strip();  return t if t else None
    for html_k in ("description_html", "html"):
        if html_k in row and isinstance(row[html_k], str) and row[html_k].strip():
            cleaned = clean_html(row[html_k])
            if cleaned: return cleaned
    for k in TEXT_KEYS_CANDIDATES:
        if k in row and isinstance(row[k], str) and row[k].strip():
            return row[k].strip()
    for k, v in row.items():
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            joined = "\n".join(v).strip()
            if joined: return joined
    return None

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("{") and s.endswith("}"):
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
            else:
                rows.append({"text": s})
    return rows

def build_dataset(positives_path: str, negatives_path: str, text_field: Optional[str] = None) -> Dataset:
    pos_raw = _read_jsonl(positives_path)  # {"url","text"}
    neg_raw = _read_jsonl(negatives_path)  # довільно

    rows = []
    bad_pos, bad_neg = 0, 0

    for r in pos_raw:
        t = r.get("text")
        if isinstance(t, str) and t.strip():
            rows.append({"text": t.strip(), "label": 1})
        else:
            bad_pos += 1

    for r in neg_raw:
        t = _extract_text_from_obj(r, text_field)
        if t and t.strip():
            rows.append({"text": t, "label": 0})
        else:
            bad_neg += 1

    print(f"[INFO] Positives: total={len(pos_raw)}, ok={len(pos_raw)-bad_pos}, skipped={bad_pos}")
    print(f"[INFO] Negatives: total={len(neg_raw)}, ok={len(neg_raw)-bad_neg}, skipped={bad_neg}")

    if not rows:
        print("[ERROR] Порожній датасет.")
        if os.path.exists(positives_path):
            print(f"[DIAG] keys in {positives_path}:", sniff_jsonl_keys(positives_path))
        if os.path.exists(negatives_path):
            print(f"[DIAG] keys in {negatives_path}:", sniff_jsonl_keys(negatives_path))
        raise ValueError("Порожній датасет. Перевір файли/поля/кодування.")
    return Dataset.from_list(rows)

# ───────── Model wrapper ─────────
class ModelWrap(nn.Module):
    def __init__(self, base: nn.Module, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.base = base
        self.pos_weight = pos_weight
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        logits = out.logits  # (bs,1)
        loss = None
        if labels is not None:
            labels = labels.float().unsqueeze(-1)
            bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight) if self.pos_weight is not None else nn.BCEWithLogitsLoss()
            loss = bce(logits, labels)
        return {"loss": loss, "logits": logits}

# ───────── Metrics ─────────
def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def compute_metrics_builder(threshold: float = 0.5):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = sigmoid_np(logits.reshape(-1))
        preds = (probs >= threshold).astype(int)
        labels = labels.astype(int)
        acc = accuracy_score(labels, preds)
        pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = float("nan")
        try:
            ll = log_loss(labels, probs, eps=1e-7)
        except Exception:
            ll = float("nan")
        return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1, "roc_auc": auc, "log_loss": ll}
    return compute_metrics

def compute_class_pos_weight_from_dataset(dataset: Dataset) -> Optional[float]:
    col = "label" if "label" in dataset.column_names else ("labels" if "labels" in dataset.column_names else None)
    if col is None:
        return None
    if isinstance(dataset.features.get(col), ClassLabel):
        y = [int(dataset.features[col].int2str(v) != "not") for v in dataset[col]]
    else:
        y = [int(v) for v in dataset[col]]
    pos, neg = sum(y), len(y) - sum(y)
    if pos == 0 or neg == 0:
        return None
    return float(neg) / float(pos)

def find_best_threshold(logits: np.ndarray, labels: np.ndarray):
    probs = sigmoid_np(logits.reshape(-1))
    best_thr, best_f1 = 0.5, -1.0
    best_scores = {}
    try:
        fpr, tpr, thresholds = roc_curve(labels.astype(int), probs)
        candidates = np.unique(np.concatenate([thresholds, np.linspace(0.05, 0.95, 19)]))
    except Exception:
        candidates = np.linspace(0.05, 0.95, 19)
    for thr in candidates:
        preds = (probs >= thr).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(labels.astype(int), preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_scores = {"precision": float(pr), "recall": float(rc), "f1": float(f1)}
    return best_thr, best_scores

# ───────── Train ─────────
def main():
    ok_p, ln_p = file_info(POSITIVES)
    ok_n, ln_n = file_info(NEGATIVES)
    if not ok_p:
        raise FileNotFoundError(f"Не знайдено файл: {POSITIVES}")
    if not ok_n:
        raise FileNotFoundError(f"Не знайдено файл: {NEGATIVES}")
    print(f"[INFO] {POSITIVES}: {ln_p} lines")
    print(f"[INFO] {NEGATIVES}: {ln_n} lines")

    set_seed(SEED)

    # Дані
    ds = build_dataset(POSITIVES, NEGATIVES, text_field=TEXT_FIELD)

    # Стратифікація
    try:
        ds = ds.cast_column("label", ClassLabel(num_classes=2, names=["not", "vacancy"]))
        dsd = ds.train_test_split(test_size=VAL_SIZE, seed=SEED, stratify_by_column="label")
        print("[INFO] Stratify via ClassLabel: OK")
    except Exception as e:
        print(f"[WARN] ClassLabel stratify failed: {e}. Fallback to sklearn.train_test_split()")
        idx = list(range(len(ds)))
        y = [int(v) for v in ds["label"]]
        tr_idx, te_idx = train_test_split(idx, test_size=VAL_SIZE, random_state=SEED, stratify=y)
        dsd = DatasetDict({"train": ds.select(tr_idx), "test": ds.select(te_idx)})

    print(f"[INFO] Train size: {len(dsd['train'])} | Val size: {len(dsd['test'])}")

    # pos_weight ДО мапінгу
    pos_weight_value = compute_class_pos_weight_from_dataset(dsd["train"]) if USE_CLASS_WEIGHT else None
    if pos_weight_value is not None:
        print(f"[INFO] pos_weight = {pos_weight_value:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding=False, truncation=True, max_length=MAX_LENGTH)

    dsd = dsd.map(tokenize_fn, batched=True, remove_columns=[c for c in ds.column_names if c != "label"])
    dsd = dsd.rename_column("label", "labels")

    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_labels = 1
    config.problem_type = "multi_label_classification"
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, config=config)

    if FREEZE_EMBEDDINGS:
        if hasattr(base_model, "roberta"):
            for p in base_model.roberta.embeddings.parameters(): p.requires_grad = False
        elif hasattr(base_model, "xlm_roberta"):
            for p in base_model.xlm_roberta.embeddings.parameters(): p.requires_grad = False

    pos_weight_tensor = torch.tensor([pos_weight_value]) if (USE_CLASS_WEIGHT and pos_weight_value is not None) else None
    model = ModelWrap(base=base_model, pos_weight=pos_weight_tensor)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # динамічні eval_steps
    train_len = len(dsd["train"])
    steps_per_epoch = max(1, train_len // (BATCH_SIZE))
    dyn_eval_steps = max(100, steps_per_epoch // 2)
    eval_steps_used = dyn_eval_steps if EVAL_STEPS is None else EVAL_STEPS
    print(f"[INFO] steps_per_epoch={steps_per_epoch} | eval_steps={eval_steps_used}")

    # Пакуємо аргументи з фільтрацією під старі версії
    ta_kwargs = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=max(4, BATCH_SIZE),
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=FP16,
        bf16=BF16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        dataloader_num_workers=2,
        report_to=["none"],
    )
    # Новіші параметри — додамо лише якщо підтримуються
    if supports_arg(TrainingArguments, "lr_scheduler_type"):
        ta_kwargs["lr_scheduler_type"] = LR_SCHEDULER
    if supports_arg(TrainingArguments, "load_best_model_at_end"):
        ta_kwargs["load_best_model_at_end"] = True
    if supports_arg(TrainingArguments, "metric_for_best_model"):
        ta_kwargs["metric_for_best_model"] = "eval_roc_auc"
    if supports_arg(TrainingArguments, "greater_is_better"):
        ta_kwargs["greater_is_better"] = True
    if supports_arg(TrainingArguments, "evaluation_strategy"):
        ta_kwargs["evaluation_strategy"] = "steps"
    if supports_arg(TrainingArguments, "eval_steps"):
        ta_kwargs["eval_steps"] = eval_steps_used
    if supports_arg(TrainingArguments, "save_strategy"):
        ta_kwargs["save_strategy"] = "steps"
    if supports_arg(TrainingArguments, "save_steps"):
        ta_kwargs["save_steps"] = eval_steps_used

    # Для дуже старих версій (до evaluation_strategy):
    if not supports_arg(TrainingArguments, "evaluation_strategy") and supports_arg(TrainingArguments, "evaluate_during_training"):
        ta_kwargs["evaluate_during_training"] = True  # еквівалент "steps" у старих версіях

    train_args = TrainingArguments(**filter_kwargs(TrainingArguments, ta_kwargs))

    # Trainer kwargs з фільтрацією
    tr_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_builder(threshold=0.5),
    )
    # callbacks (EarlyStopping) тільки якщо підтримується
    if EarlyStoppingCallback is not None:
        # Деякі старі версії не вміють callbacks у __init__ — перевіримо:
        if supports_arg(Trainer, "callbacks"):
            tr_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)]

    trainer = Trainer(**filter_kwargs(Trainer, tr_kwargs))

    trainer.train()

    # оцінка при thr=0.5
    metrics = trainer.evaluate()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, ensure_ascii=False, indent=2)

    # пошук кращого порогу
    preds = trainer.predict(dsd["test"])
    logits = preds.predictions
    labels = preds.label_ids
    best_thr, best_scores = find_best_threshold(logits, labels)
    with open(os.path.join(OUTPUT_DIR, "best_threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"best_threshold": best_thr, **best_scores}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Best threshold by F1: {best_thr:.3f} | scores={best_scores}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # інференс-скрипт
    infer_path = os.path.join(OUTPUT_DIR, "predict_is_vacancy.py")
    with open(infer_path, "w", encoding="utf-8") as f:
        f.write(f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, sys, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def load_best_threshold(model_dir: str, default_thr: float = 0.5) -> float:
    path = os.path.join(model_dir, "best_threshold.json")
    if os.path.exists(path):
        try:
            obj = json.load(open(path, "r", encoding="utf-8"))
            return float(obj.get("best_threshold", default_thr))
        except Exception:
            return default_thr
    return default_thr
def predict(text: str, model_dir: str, max_length: int = 512, threshold: float = None):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.eval()
    if threshold is None:
        threshold = load_best_threshold(model_dir, 0.5)
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=max_length)
    with torch.no_grad():
        out = mdl(**inputs)
        logits = out.logits.squeeze(-1).cpu().numpy()
        prob = float(sigmoid(logits))
        return {{
            'is_vacancy': bool(prob >= threshold),
            'score_vacancy': prob,
            'score_not_vacancy': 1.0 - prob,
            'threshold_used': threshold,
            'model': model_dir
        }}
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python predict_is_vacancy.py \\"text\\" /path/to/model_dir")
        sys.exit(1)
    text = sys.argv[1]; model_dir = sys.argv[2]
    print(predict(text, model_dir))
""")

    print("Saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
