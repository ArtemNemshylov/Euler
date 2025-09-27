# file: vacancy_zeroshot.py
from transformers import pipeline

MODEL_ID = "joeddav/xlm-roberta-large-xnli"

# ВАЖЛИВО: позиційний плейсхолдер `{}`, НЕ {label}
TEMPLATE = "This text is a {}."
CANDIDATE_LABELS = ["vacancy page", "not a vacancy page"]

clf = pipeline(
    "zero-shot-classification",
    model=MODEL_ID,
    device_map="auto",          # ок для MPS/GPU/CPU
)

def predict_is_vacancy(text: str, threshold: float = 0.5) -> dict:
    out = clf(
        text,
        CANDIDATE_LABELS,
        hypothesis_template=TEMPLATE,
        multi_label=False
    )
    scores = dict(zip(out["labels"], out["scores"]))
    is_vacancy = scores.get("vacancy page", 0.0) >= threshold
    return {
        "is_vacancy": bool(is_vacancy),
        "score_vacancy": float(scores.get("vacancy page", 0.0)),
        "score_not_vacancy": float(scores.get("not a vacancy page", 0.0)),
        "model": MODEL_ID
    }

if __name__ == "__main__":
    import json, sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_samples.json"
    samples = json.load(open(path, "r", encoding="utf-8"))
    for s in samples:
        r = predict_is_vacancy(s["text"], threshold=0.5)
        print(json.dumps({**r}, ensure_ascii=False))
