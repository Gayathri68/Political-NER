# ================================================================
# Train_Final.py
# FINAL YEAR PROJECT - MULTILINGUAL PAN-INDIAN POLITICAL NER
#
# Features:
# - Loads Step 3 weak-labeled JSONL data from 6 languages
# - 70/15/15 train/val/test split
# - Trains 2 models: MuRIL + XLM-R
# - Validation during training, best checkpoint saved
# - Final TEST evaluation after training
# - Token-level metrics
# - Entity-level metrics (seqeval)
# - Per-label metrics
# - Per-language test metrics
# - Inference speed benchmark
# - Model size benchmark
# - Sample predictions
# - Error analysis
# - Demo predictions
# - Clean .txt reports for everything
#
# CPU-optimized and final-project friendly
# ================================================================

import os
import json
import time
import math
import random
import shutil
import gc
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# seqeval for entity-level NER metrics
from seqeval.metrics import (
    classification_report as seqeval_classification_report,
    precision_score as seqeval_precision_score,
    recall_score as seqeval_recall_score,
    f1_score as seqeval_f1_score
)

# ================================================================
# GLOBAL CONFIG
# ================================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# CPU / speed-friendly settings
# -----------------------------
# Keep 0 for ALL data
MAX_SENTENCES = 50  # 0 = use all data
EPOCHS = 1
LR = 2e-5
WARMUP_RATIO = 0.1
GRAD_CLIP = 1.0

# Optional early stop if no val improvement
EARLY_STOP_PATIENCE = 2

# Keep dataloader workers 0 for Windows safety
NUM_WORKERS = 0

# ================================================================
# DATA FILES (Step 3 outputs)
# ================================================================

DATA_FILES = [
    "output_files_step3/news_labeled_english.jsonl",
    "output_files_step3/news_labeled_malayalam.jsonl",
    "output_files_step3/news_labeled_odia.jsonl",
    "output_files_step3/news_labeled_punjabi.jsonl",
    "output_files_step3/news_labeled_assamese.jsonl",
    "output_files_step3/news_labeled_kannada.jsonl",
]

# Language display names
LANGUAGE_NAMES = {
    "en": "English",
    "ml": "Malayalam",
    "or": "Odia",
    "pa": "Punjabi",
    "as": "Assamese",
    "kn": "Kannada",
}

# ================================================================
# LABELS
# IMPORTANT:
# This matches your Step 3 style:
# POLITICIAN / PERSON / POLITICAL_PARTY / GOVERNMENT_SCHEME / LOCATION
# BIO format
# ================================================================

LABELS = [
    "O",
    "B-POLITICIAN", "I-POLITICIAN",
    "B-PERSON", "I-PERSON",
    "B-POLITICAL_PARTY", "I-POLITICAL_PARTY",
    "B-GOVERNMENT_SCHEME", "I-GOVERNMENT_SCHEME",
    "B-LOCATION", "I-LOCATION",
]

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# ================================================================
# MODELS
# 2 REAL MODELS (BEST FOR YOU)
# ================================================================

MODELS = [
    {
        "name": "MuRIL",
        "model_id": "google/muril-base-cased",
        "save_dir": "project_outputs/saved_model_muril",
        "max_len": 90,
        "batch_size": 8 if DEVICE.type == "cuda" else 4,
    },
    {
        "name": "XLM-R",
        "model_id": "xlm-roberta-base",
        "save_dir": "project_outputs/saved_model_xlmr",
        "max_len": 90,
        "batch_size": 8 if DEVICE.type == "cuda" else 4,
    },
]

# ================================================================
# OUTPUT PATHS
# ================================================================

BASE_OUT = "project_outputs"
REPORTS_DIR = os.path.join(BASE_OUT, "reports")
PRED_DIR = os.path.join(BASE_OUT, "predictions")
LOGS_DIR = os.path.join(BASE_OUT, "logs")

for d in [BASE_OUT, REPORTS_DIR, PRED_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

RUN_LOG_PATH = os.path.join(LOGS_DIR, "run_log.txt")
MASTER_REPORT_PATH = os.path.join(REPORTS_DIR, "final_training_report.txt")
MODEL_COMPARE_PATH = os.path.join(REPORTS_DIR, "model_comparison_report.txt")
BEST_MODEL_SUMMARY_PATH = os.path.join(REPORTS_DIR, "best_model_summary.txt")
LABEL_DIST_PATH = os.path.join(REPORTS_DIR, "label_distribution_report.txt")
PER_LANGUAGE_PATH = os.path.join(REPORTS_DIR, "per_language_test_report.txt")
CONFUSION_PATH = os.path.join(REPORTS_DIR, "confusion_summary.txt")
SAMPLE_PRED_PATH = os.path.join(PRED_DIR, "sample_predictions.txt")
ERROR_ANALYSIS_PATH = os.path.join(PRED_DIR, "error_analysis.txt")
DEMO_PRED_PATH = os.path.join(PRED_DIR, "demo_predictions.txt")
WEAK_LABEL_SUMMARY_PATH = os.path.join(REPORTS_DIR, "weak_labeling_summary.txt")

# ================================================================
# SIMPLE LOGGER
# ================================================================

def log(msg: str):
    print(msg)
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Reset run log on each run
with open(RUN_LOG_PATH, "w", encoding="utf-8") as f:
    f.write(f"Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ================================================================
# HELPERS
# ================================================================

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def format_pct(x):
    return f"{x * 100:.2f}%"

def get_model_size_mb(folder_path):
    if not os.path.exists(folder_path):
        return 0.0
    total = 0
    for root, _, files in os.walk(folder_path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

def bio_to_entity_spans(tokens, labels):
    """
    Convert BIO labels to entity spans:
    Returns list of tuples: (entity_text, entity_type)
    """
    entities = []
    current_tokens = []
    current_type = None

    for tok, lab in zip(tokens, labels):
        if lab == "O":
            if current_tokens:
                entities.append((" ".join(current_tokens), current_type))
                current_tokens = []
                current_type = None
            continue

        if lab.startswith("B-"):
            if current_tokens:
                entities.append((" ".join(current_tokens), current_type))
            current_type = lab[2:]
            current_tokens = [tok]
        elif lab.startswith("I-"):
            ent_type = lab[2:]
            if current_tokens and current_type == ent_type:
                current_tokens.append(tok)
            else:
                # Broken I- tag, start fresh
                if current_tokens:
                    entities.append((" ".join(current_tokens), current_type))
                current_type = ent_type
                current_tokens = [tok]

    if current_tokens:
        entities.append((" ".join(current_tokens), current_type))

    return entities

def flatten_valid_positions(preds, labels):
    """
    Flatten predictions/labels ignoring -100
    """
    flat_preds, flat_labels = [], []
    for pred_seq, label_seq in zip(preds, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                flat_preds.append(p)
                flat_labels.append(l)
    return flat_preds, flat_labels

# ================================================================
# DATASET
# ================================================================

class NERDataset(Dataset):
    def __init__(self, records, tokenizer, max_len=128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        tokens = rec["tokens"]
        labels = rec["labels"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)

        aligned_labels = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != prev_word_idx:
                aligned_labels.append(label2id.get(labels[word_idx], label2id["O"]))
            else:
                # subword continuation: keep same label (or could set -100)
                # keeping same label gives stronger supervision
                aligned_labels.append(label2id.get(labels[word_idx], label2id["O"]))
            prev_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
            "tokens": tokens,
            "orig_labels": labels,
            "language": rec["language"],
            "sentence": rec.get("sentence", " ".join(tokens)),
        }

# ================================================================
# LOAD DATA
# ================================================================

def load_data():
    log("=" * 80)
    log("LOADING STEP 3 DATA")
    log("=" * 80)

    all_records = []
    per_lang_counts = Counter()
    raw_total_records = 0

    for fpath in DATA_FILES:
        if not os.path.exists(fpath):
            log(f"[SKIP] Missing file: {fpath}")
            continue

        file_count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                raw_total_records += 1

                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                tokens = rec.get("tokens", [])
                labels = rec.get("labels", [])
                language = rec.get("language", "?")
                sentence = rec.get("sentence", " ".join(tokens))

                if not tokens or not labels or len(tokens) != len(labels):
                    continue

                # Keep only labels known to current schema
                cleaned_labels = []
                valid = True
                for lab in labels:
                    if lab not in label2id:
                        valid = False
                        break
                    cleaned_labels.append(lab)

                if not valid:
                    continue

                all_records.append({
                    "tokens": tokens,
                    "labels": cleaned_labels,
                    "language": language,
                    "sentence": sentence,
                })
                per_lang_counts[language] += 1
                file_count += 1

        log(f"Loaded {file_count:>6} from {fpath}")

    if not all_records:
        raise RuntimeError("No valid Step 3 data found. Check output_files_step3/*.jsonl files.")

    # CPU cap if needed
    if MAX_SENTENCES > 0 and len(all_records) > MAX_SENTENCES:
        random.shuffle(all_records)
        all_records = all_records[:MAX_SENTENCES]
        log(f"CPU cap applied: using only {MAX_SENTENCES} sentences")

    # Shuffle once
    random.shuffle(all_records)

    # 70 / 15 / 15 split
    n = len(all_records)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_data = all_records[:train_end]
    val_data = all_records[train_end:val_end]
    test_data = all_records[val_end:]

    log("")
    log(f"TOTAL RECORDS USED : {n}")
    log(f"TRAIN              : {len(train_data)}")
    log(f"VALIDATION         : {len(val_data)}")
    log(f"TEST               : {len(test_data)}")
    log("")

    # Save label distribution + weak label summary
    write_label_distribution(all_records, per_lang_counts, raw_total_records)
    write_weak_label_summary(all_records, raw_total_records)

    return train_data, val_data, test_data, per_lang_counts

# ================================================================
# REPORT: LABEL DISTRIBUTION
# ================================================================

def write_label_distribution(all_records, per_lang_counts, raw_total_records):
    token_counter = Counter()
    entity_counter = Counter()
    total_tokens = 0

    for rec in all_records:
        for lab in rec["labels"]:
            token_counter[lab] += 1
            total_tokens += 1
            if lab != "O":
                entity_counter[lab] += 1

    lines = []
    lines.append("=" * 80)
    lines.append("LABEL DISTRIBUTION REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated at       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Raw Step 3 records : {raw_total_records}")
    lines.append(f"Valid records used : {len(all_records)}")
    lines.append(f"Total tokens       : {total_tokens}")
    lines.append("")

    lines.append("Per-language sentence counts:")
    for lang, cnt in sorted(per_lang_counts.items()):
        lines.append(f"  {LANGUAGE_NAMES.get(lang, lang):<12} ({lang}) : {cnt}")

    lines.append("")
    lines.append("Token label distribution:")
    for lab in LABELS:
        cnt = token_counter[lab]
        pct = safe_div(cnt, total_tokens) * 100
        lines.append(f"  {lab:<25} {cnt:>8}  ({pct:6.2f}%)")

    with open(LABEL_DIST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ================================================================
# REPORT: WEAK LABEL SUMMARY
# ================================================================

def write_weak_label_summary(all_records, raw_total_records):
    has_entity = 0
    for rec in all_records:
        if any(l != "O" for l in rec["labels"]):
            has_entity += 1

    lines = []
    lines.append("=" * 80)
    lines.append("WEAK LABELING SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Generated at                : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Raw Step 3 records read     : {raw_total_records}")
    lines.append(f"Valid records kept          : {len(all_records)}")
    lines.append(f"Records with >=1 entity     : {has_entity}")
    lines.append(f"Entity-bearing %            : {safe_div(has_entity, len(all_records))*100:.2f}%")
    lines.append("")
    lines.append("Note:")
    lines.append("  This dataset comes from Step 3 weak annotation.")
    lines.append("  Weak labels may contain noise, but they enable scalable multilingual training.")

    with open(WEAK_LABEL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ================================================================
# EVALUATION HELPERS
# ================================================================

def evaluate_model(model, dataloader):
    """
    Returns:
      dict with:
        flat_preds, flat_labels,
        seq_true, seq_pred,
        token_records (list of dicts for sample/error/per-language)
    """
    model.eval()

    all_flat_preds = []
    all_flat_labels = []

    seq_true = []
    seq_pred = []

    token_records = []  # per sentence record for later analysis

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()

            bs = preds_cpu.shape[0]

            for i in range(bs):
                pred_seq = preds_cpu[i].tolist()
                label_seq = labels_cpu[i].tolist()

                valid_pred_ids = []
                valid_label_ids = []

                for p, l in zip(pred_seq, label_seq):
                    if l != -100:
                        valid_pred_ids.append(p)
                        valid_label_ids.append(l)
                        all_flat_preds.append(p)
                        all_flat_labels.append(l)

                pred_labels = [id2label[p] for p in valid_pred_ids]
                true_labels = [id2label[l] for l in valid_label_ids]

                seq_pred.append(pred_labels)
                seq_true.append(true_labels)

                tokens = batch["tokens"][i]
                language = batch["language"][i]
                sentence = batch["sentence"][i]

                token_records.append({
                    "sentence": sentence,
                    "tokens": tokens,
                    "true_labels": true_labels,
                    "pred_labels": pred_labels,
                    "language": language,
                })

    return {
        "flat_preds": all_flat_preds,
        "flat_labels": all_flat_labels,
        "seq_true": seq_true,
        "seq_pred": seq_pred,
        "token_records": token_records,
    }

def compute_token_metrics(flat_labels, flat_preds):
    acc = accuracy_score(flat_labels, flat_preds)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        flat_labels, flat_preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        flat_labels, flat_preds, average="weighted", zero_division=0
    )

    # Per-label
    p_per, r_per, f1_per, support_per = precision_recall_fscore_support(
        flat_labels,
        flat_preds,
        labels=list(range(NUM_LABELS)),
        zero_division=0
    )

    clf_report = classification_report(
        flat_labels,
        flat_preds,
        labels=list(range(NUM_LABELS)),
        target_names=LABELS,
        zero_division=0,
        digits=4
    )

    per_label_rows = []
    for i, lab in enumerate(LABELS):
        per_label_rows.append({
            "label": lab,
            "precision": p_per[i],
            "recall": r_per[i],
            "f1": f1_per[i],
            "support": int(support_per[i]),
        })

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": clf_report,
        "per_label_rows": per_label_rows,
    }

def compute_entity_metrics(seq_true, seq_pred):
    """
    Entity-level metrics using seqeval
    """
    ent_p = seqeval_precision_score(seq_true, seq_pred)
    ent_r = seqeval_recall_score(seq_true, seq_pred)
    ent_f1 = seqeval_f1_score(seq_true, seq_pred)
    ent_report = seqeval_classification_report(seq_true, seq_pred, digits=4)

    return {
        "entity_precision": ent_p,
        "entity_recall": ent_r,
        "entity_f1": ent_f1,
        "entity_report": ent_report,
    }

def compute_per_language_metrics(token_records):
    by_lang = defaultdict(list)
    for rec in token_records:
        by_lang[rec["language"]].append(rec)

    rows = []

    for lang, recs in sorted(by_lang.items()):
        flat_true = []
        flat_pred = []
        seq_true = []
        seq_pred = []

        for rec in recs:
            t = [label2id[x] for x in rec["true_labels"]]
            p = [label2id[x] for x in rec["pred_labels"]]
            flat_true.extend(t)
            flat_pred.extend(p)
            seq_true.append(rec["true_labels"])
            seq_pred.append(rec["pred_labels"])

        acc = accuracy_score(flat_true, flat_pred)
        _, _, f1_macro, _ = precision_recall_fscore_support(
            flat_true, flat_pred, average="macro", zero_division=0
        )
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            flat_true, flat_pred, average="weighted", zero_division=0
        )
        ent_f1 = seqeval_f1_score(seq_true, seq_pred)

        rows.append({
            "language_code": lang,
            "language_name": LANGUAGE_NAMES.get(lang, lang),
            "num_sentences": len(recs),
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "entity_f1": ent_f1,
        })

    return rows

def compute_confusion_summary(token_records, top_k=30):
    """
    Human-readable confusion summary excluding O->O
    """
    conf = Counter()

    for rec in token_records:
        for t, p in zip(rec["true_labels"], rec["pred_labels"]):
            if t == p:
                continue
            conf[(t, p)] += 1

    most_common = conf.most_common(top_k)
    return most_common


# ================================================================
# CUSTOM COLLATE FUNCTION
# Fixes variable-length tokens / strings in batch
# ================================================================

def ner_collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "tokens": [x["tokens"] for x in batch],
        "orig_labels": [x["orig_labels"] for x in batch],
        "language": [x["language"] for x in batch],
        "sentence": [x["sentence"] for x in batch],
    }
# ================================================================
# TRAIN ONE MODEL
# ================================================================

def train_one_model(cfg, train_data, val_data, test_data):
    name = cfg["name"]
    model_id = cfg["model_id"]
    save_dir = cfg["save_dir"]
    max_len = cfg["max_len"]
    batch_size = cfg["batch_size"]

    os.makedirs(save_dir, exist_ok=True)

    log("")
    log("#" * 80)
    log(f"TRAINING MODEL: {name}")
    log("#" * 80)
    log(f"Model ID      : {model_id}")
    log(f"Save dir      : {save_dir}")
    log(f"Max len       : {max_len}")
    log(f"Batch size    : {batch_size}")
    log(f"Epochs        : {EPOCHS}")
    log(f"Device        : {DEVICE}")
    log("")

    model_start = time.time()

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)

    # Datasets
    train_dataset = NERDataset(train_data, tokenizer, max_len=max_len)
    val_dataset = NERDataset(val_data, tokenizer, max_len=max_len)
    test_dataset = NERDataset(test_data, tokenizer, max_len=max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=ner_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=ner_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=ner_collate_fn
    )

    optimizer = AdamW(model.parameters(), lr=LR)

    total_steps = max(1, len(train_loader) * EPOCHS)
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    epoch_logs = []
    best_state_dict = None

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        # Validation
        val_eval = evaluate_model(model, val_loader)
        val_token = compute_token_metrics(val_eval["flat_labels"], val_eval["flat_preds"])
        val_entity = compute_entity_metrics(val_eval["seq_true"], val_eval["seq_pred"])

        epoch_time = time.time() - epoch_start

        epoch_info = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_accuracy": val_token["accuracy"],
            "val_f1_macro": val_token["f1_macro"],
            "val_f1_weighted": val_token["f1_weighted"],
            "val_entity_f1": val_entity["entity_f1"],
            "epoch_time_sec": epoch_time,
        }
        epoch_logs.append(epoch_info)

        log(f"[{name}] Epoch {epoch}/{EPOCHS}")
        log(f"  Train Loss      : {avg_train_loss:.4f}")
        log(f"  Val Accuracy    : {val_token['accuracy']:.4f}")
        log(f"  Val F1 Macro    : {val_token['f1_macro']:.4f}")
        log(f"  Val F1 Weighted : {val_token['f1_weighted']:.4f}")
        log(f"  Val Entity F1   : {val_entity['entity_f1']:.4f}")
        log(f"  Epoch Time      : {epoch_time:.1f}s")

        # Best model by VAL ENTITY F1 (better for NER)
        score_for_best = val_entity["entity_f1"]

        if score_for_best > best_val_f1:
            best_val_f1 = score_for_best
            best_epoch = epoch
            epochs_no_improve = 0

            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Save checkpoint
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            # Also save label maps
            with open(os.path.join(save_dir, "label_map.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "labels": LABELS,
                    "label2id": label2id,
                    "id2label": {str(k): v for k, v in id2label.items()}
                }, f, ensure_ascii=False, indent=2)

            log(f"  [OK] BEST MODEL UPDATED (epoch {epoch}, val entity F1 = {best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            log(f"  No improvement. patience = {epochs_no_improve}/{EARLY_STOP_PATIENCE}")

        # Early stopping
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            log(f"  Early stopping triggered for {name}.")
            break

        log("")

    # -----------------------------
    # Load best weights before test
    # -----------------------------
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # -----------------------------
    # Final TEST evaluation
    # -----------------------------
    infer_start = time.time()
    test_eval = evaluate_model(model, test_loader)
    infer_time = time.time() - infer_start

    test_token = compute_token_metrics(test_eval["flat_labels"], test_eval["flat_preds"])
    test_entity = compute_entity_metrics(test_eval["seq_true"], test_eval["seq_pred"])
    per_language_rows = compute_per_language_metrics(test_eval["token_records"])
    confusion_rows = compute_confusion_summary(test_eval["token_records"], top_k=40)

    total_train_time = time.time() - model_start
    sentences_per_sec = safe_div(len(test_data), infer_time)
    avg_ms_per_sentence = safe_div(infer_time, len(test_data)) * 1000.0
    model_size_mb = get_model_size_mb(save_dir)

    # Save per-model report
    per_model_report_path = os.path.join(REPORTS_DIR, f"report_{name.lower().replace('-', '_')}.txt")
    write_per_model_report(
        path=per_model_report_path,
        name=name,
        model_id=model_id,
        cfg=cfg,
        epoch_logs=epoch_logs,
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        test_token=test_token,
        test_entity=test_entity,
        per_language_rows=per_language_rows,
        model_size_mb=model_size_mb,
        total_train_time=total_train_time,
        infer_time=infer_time,
        sentences_per_sec=sentences_per_sec,
        avg_ms_per_sentence=avg_ms_per_sentence
    )

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": name,
        "model_id": model_id,
        "save_dir": save_dir,
        "best_epoch": best_epoch,
        "best_val_entity_f1": best_val_f1,
        "epoch_logs": epoch_logs,
        "test_token": test_token,
        "test_entity": test_entity,
        "per_language_rows": per_language_rows,
        "confusion_rows": confusion_rows,
        "token_records": test_eval["token_records"],
        "model_size_mb": model_size_mb,
        "train_time_sec": total_train_time,
        "infer_time_sec": infer_time,
        "sentences_per_sec": sentences_per_sec,
        "avg_ms_per_sentence": avg_ms_per_sentence,
        "per_model_report_path": per_model_report_path,
    }

# ================================================================
# REPORT: PER MODEL
# ================================================================

def write_per_model_report(
    path,
    name,
    model_id,
    cfg,
    epoch_logs,
    best_epoch,
    best_val_f1,
    test_token,
    test_entity,
    per_language_rows,
    model_size_mb,
    total_train_time,
    infer_time,
    sentences_per_sec,
    avg_ms_per_sentence
):
    lines = []
    lines.append("=" * 100)
    lines.append(f"PER-MODEL REPORT: {name}")
    lines.append("=" * 100)
    lines.append(f"Generated at           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model Name             : {name}")
    lines.append(f"Model ID               : {model_id}")
    lines.append(f"Max Length             : {cfg['max_len']}")
    lines.append(f"Batch Size             : {cfg['batch_size']}")
    lines.append(f"Epochs Configured      : {EPOCHS}")
    lines.append(f"Best Epoch             : {best_epoch}")
    lines.append(f"Best Validation Ent F1 : {best_val_f1:.4f}")
    lines.append("")

    lines.append("-" * 100)
    lines.append("EPOCH-WISE LOGS")
    lines.append("-" * 100)
    for e in epoch_logs:
        lines.append(
            f"Epoch {e['epoch']}: "
            f"loss={e['train_loss']:.4f}, "
            f"val_acc={e['val_accuracy']:.4f}, "
            f"val_f1_macro={e['val_f1_macro']:.4f}, "
            f"val_f1_weighted={e['val_f1_weighted']:.4f}, "
            f"val_entity_f1={e['val_entity_f1']:.4f}, "
            f"time={e['epoch_time_sec']:.1f}s"
        )

    lines.append("")
    lines.append("-" * 100)
    lines.append("FINAL TEST RESULTS (TOKEN LEVEL)")
    lines.append("-" * 100)
    lines.append(f"Accuracy              : {test_token['accuracy']:.4f}")
    lines.append(f"Precision (Macro)     : {test_token['precision_macro']:.4f}")
    lines.append(f"Recall (Macro)        : {test_token['recall_macro']:.4f}")
    lines.append(f"F1 (Macro)            : {test_token['f1_macro']:.4f}")
    lines.append(f"Precision (Weighted)  : {test_token['precision_weighted']:.4f}")
    lines.append(f"Recall (Weighted)     : {test_token['recall_weighted']:.4f}")
    lines.append(f"F1 (Weighted)         : {test_token['f1_weighted']:.4f}")

    lines.append("")
    lines.append("-" * 100)
    lines.append("FINAL TEST RESULTS (ENTITY LEVEL)")
    lines.append("-" * 100)
    lines.append(f"Entity Precision      : {test_entity['entity_precision']:.4f}")
    lines.append(f"Entity Recall         : {test_entity['entity_recall']:.4f}")
    lines.append(f"Entity F1             : {test_entity['entity_f1']:.4f}")

    lines.append("")
    lines.append("-" * 100)
    lines.append("PER-LABEL METRICS")
    lines.append("-" * 100)
    for row in test_token["per_label_rows"]:
        lines.append(
            f"{row['label']:<25} "
            f"P={row['precision']:.4f}  "
            f"R={row['recall']:.4f}  "
            f"F1={row['f1']:.4f}  "
            f"Support={row['support']}"
        )

    lines.append("")
    lines.append("-" * 100)
    lines.append("TOKEN-LEVEL CLASSIFICATION REPORT")
    lines.append("-" * 100)
    lines.append(test_token["classification_report"])

    lines.append("")
    lines.append("-" * 100)
    lines.append("ENTITY-LEVEL CLASSIFICATION REPORT (SEQEVAL)")
    lines.append("-" * 100)
    lines.append(test_entity["entity_report"])

    lines.append("")
    lines.append("-" * 100)
    lines.append("PER-LANGUAGE TEST RESULTS")
    lines.append("-" * 100)
    for row in per_language_rows:
        lines.append(
            f"{row['language_name']:<12} ({row['language_code']}): "
            f"sentences={row['num_sentences']}, "
            f"acc={row['accuracy']:.4f}, "
            f"f1_macro={row['f1_macro']:.4f}, "
            f"f1_weighted={row['f1_weighted']:.4f}, "
            f"entity_f1={row['entity_f1']:.4f}"
        )

    lines.append("")
    lines.append("-" * 100)
    lines.append("SPEED / SIZE")
    lines.append("-" * 100)
    lines.append(f"Training Time (sec)   : {total_train_time:.2f}")
    lines.append(f"Training Time (hrs)   : {total_train_time/3600.0:.2f}")
    lines.append(f"Test Inference (sec)  : {infer_time:.2f}")
    lines.append(f"Sentences / sec       : {sentences_per_sec:.2f}")
    lines.append(f"Avg ms / sentence     : {avg_ms_per_sentence:.2f}")
    lines.append(f"Model Size (MB)       : {model_size_mb:.2f}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ================================================================
# REPORT: MASTER + COMPARISON + BEST MODEL
# ================================================================

def write_master_reports(results, train_data, val_data, test_data):
    # Best model by entity F1, tie-break macro F1
    best = sorted(
        results,
        key=lambda r: (r["test_entity"]["entity_f1"], r["test_token"]["f1_macro"]),
        reverse=True
    )[0]

    # -----------------------------
    # MASTER REPORT
    # -----------------------------
    lines = []
    lines.append("=" * 100)
    lines.append("FINAL TRAINING REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated at          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Device                : {DEVICE}")
    lines.append(f"Train size            : {len(train_data)}")
    lines.append(f"Validation size       : {len(val_data)}")
    lines.append(f"Test size             : {len(test_data)}")
    lines.append(f"Models trained        : {', '.join([r['name'] for r in results])}")
    lines.append("")

    for r in results:
        lines.append("-" * 100)
        lines.append(f"MODEL: {r['name']}")
        lines.append("-" * 100)
        lines.append(f"Best Epoch            : {r['best_epoch']}")
        lines.append(f"Best Val Entity F1    : {r['best_val_entity_f1']:.4f}")
        lines.append(f"Test Accuracy         : {r['test_token']['accuracy']:.4f}")
        lines.append(f"Test F1 Macro         : {r['test_token']['f1_macro']:.4f}")
        lines.append(f"Test F1 Weighted      : {r['test_token']['f1_weighted']:.4f}")
        lines.append(f"Test Entity F1        : {r['test_entity']['entity_f1']:.4f}")
        lines.append(f"Train Time (hrs)      : {r['train_time_sec']/3600.0:.2f}")
        lines.append(f"Sentences/sec         : {r['sentences_per_sec']:.2f}")
        lines.append(f"Avg ms/sentence       : {r['avg_ms_per_sentence']:.2f}")
        lines.append(f"Model Size (MB)       : {r['model_size_mb']:.2f}")
        lines.append(f"Saved model           : {r['save_dir']}")
        lines.append(f"Per-model report      : {r['per_model_report_path']}")
        lines.append("")

    lines.append("=" * 100)
    lines.append("RECOMMENDED FINAL MODEL")
    lines.append("=" * 100)
    lines.append(f"Best Model            : {best['name']}")
    lines.append(f"Reason                : Highest entity-level F1 on final test set")
    lines.append(f"Entity F1             : {best['test_entity']['entity_f1']:.4f}")
    lines.append(f"Macro F1              : {best['test_token']['f1_macro']:.4f}")
    lines.append(f"Accuracy              : {best['test_token']['accuracy']:.4f}")
    lines.append("=" * 100)

    with open(MASTER_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # -----------------------------
    # MODEL COMPARISON REPORT
    # -----------------------------
    lines = []
    lines.append("=" * 100)
    lines.append("MODEL COMPARISON REPORT")
    lines.append("=" * 100)
    header = (
        f"{'Model':<12} | {'BestValEntF1':>11} | {'TestAcc':>8} | {'MacroF1':>8} | "
        f"{'WeightedF1':>10} | {'EntityF1':>8} | {'TrainHrs':>8} | {'Sent/s':>8} | {'SizeMB':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        lines.append(
            f"{r['name']:<12} | "
            f"{r['best_val_entity_f1']:>11.4f} | "
            f"{r['test_token']['accuracy']:>8.4f} | "
            f"{r['test_token']['f1_macro']:>8.4f} | "
            f"{r['test_token']['f1_weighted']:>10.4f} | "
            f"{r['test_entity']['entity_f1']:>8.4f} | "
            f"{r['train_time_sec']/3600.0:>8.2f} | "
            f"{r['sentences_per_sec']:>8.2f} | "
            f"{r['model_size_mb']:>8.2f}"
        )

    lines.append("")
    lines.append(f"Recommended Final Model: {best['name']}")
    lines.append("Reason: Best overall entity-level performance for NER.")

    with open(MODEL_COMPARE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # -----------------------------
    # BEST MODEL SUMMARY
    # -----------------------------
    lines = []
    lines.append("=" * 100)
    lines.append("BEST MODEL SUMMARY")
    lines.append("=" * 100)
    lines.append(f"Generated at          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Best Model            : {best['name']}")
    lines.append(f"Model ID              : {best['model_id']}")
    lines.append(f"Saved Path            : {best['save_dir']}")
    lines.append(f"Best Epoch            : {best['best_epoch']}")
    lines.append(f"Best Validation EntF1 : {best['best_val_entity_f1']:.4f}")
    lines.append(f"Final Test Accuracy   : {best['test_token']['accuracy']:.4f}")
    lines.append(f"Final Test Macro F1   : {best['test_token']['f1_macro']:.4f}")
    lines.append(f"Final Test WeightedF1 : {best['test_token']['f1_weighted']:.4f}")
    lines.append(f"Final Test Entity F1  : {best['test_entity']['entity_f1']:.4f}")
    lines.append(f"Model Size (MB)       : {best['model_size_mb']:.2f}")
    lines.append(f"Sentences / sec       : {best['sentences_per_sec']:.2f}")
    lines.append("")
    lines.append("Label Schema:")
    for lab in LABELS:
        lines.append(f"  - {lab}")

    with open(BEST_MODEL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return best

# ================================================================
# REPORT: PER-LANGUAGE (BEST MODEL ONLY)
# ================================================================

def write_best_per_language_report(best_result):
    lines = []
    lines.append("=" * 100)
    lines.append(f"PER-LANGUAGE TEST REPORT (BEST MODEL = {best_result['name']})")
    lines.append("=" * 100)

    for row in best_result["per_language_rows"]:
        lines.append(
            f"{row['language_name']:<12} ({row['language_code']}): "
            f"sentences={row['num_sentences']}, "
            f"accuracy={row['accuracy']:.4f}, "
            f"macro_f1={row['f1_macro']:.4f}, "
            f"weighted_f1={row['f1_weighted']:.4f}, "
            f"entity_f1={row['entity_f1']:.4f}"
        )

    with open(PER_LANGUAGE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ================================================================
# REPORT: CONFUSION SUMMARY (BEST MODEL ONLY)
# ================================================================

def write_confusion_summary(best_result):
    lines = []
    lines.append("=" * 100)
    lines.append(f"CONFUSION SUMMARY (BEST MODEL = {best_result['name']})")
    lines.append("=" * 100)
    lines.append("Top label confusions (True -> Pred):")
    lines.append("")

    for (true_lab, pred_lab), count in best_result["confusion_rows"]:
        lines.append(f"  {true_lab:<25} -> {pred_lab:<25} : {count}")

    with open(CONFUSION_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ================================================================
# SAMPLE PREDICTIONS / ERROR ANALYSIS
# ================================================================

def write_sample_predictions(best_result, max_samples=30):
    records = best_result["token_records"]

    # Pick first max_samples that contain at least one entity in true or pred
    selected = []
    for rec in records:
        true_entities = bio_to_entity_spans(rec["tokens"], rec["true_labels"])
        pred_entities = bio_to_entity_spans(rec["tokens"], rec["pred_labels"])

        if true_entities or pred_entities:
            selected.append((rec, true_entities, pred_entities))
        if len(selected) >= max_samples:
            break

    lines = []
    lines.append("=" * 100)
    lines.append(f"SAMPLE PREDICTIONS (BEST MODEL = {best_result['name']})")
    lines.append("=" * 100)

    for idx, (rec, true_entities, pred_entities) in enumerate(selected, start=1):
        status = "CORRECT" if set(true_entities) == set(pred_entities) else "PARTIAL/ERROR"

        lines.append(f"[Sample {idx}]")
        lines.append(f"Language : {LANGUAGE_NAMES.get(rec['language'], rec['language'])} ({rec['language']})")
        lines.append(f"Sentence : {rec['sentence']}")
        lines.append(f"True     : {true_entities if true_entities else '[]'}")
        lines.append(f"Pred     : {pred_entities if pred_entities else '[]'}")
        lines.append(f"Status   : {status}")
        lines.append("-" * 100)

    with open(SAMPLE_PRED_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def write_error_analysis(best_result, max_errors=40):
    records = best_result["token_records"]

    errors = []
    for rec in records:
        true_entities = bio_to_entity_spans(rec["tokens"], rec["true_labels"])
        pred_entities = bio_to_entity_spans(rec["tokens"], rec["pred_labels"])

        if set(true_entities) != set(pred_entities):
            # classify error roughly
            if not pred_entities and true_entities:
                err_type = "FALSE NEGATIVE / MISSED ENTITY"
            elif pred_entities and not true_entities:
                err_type = "FALSE POSITIVE / SPURIOUS ENTITY"
            else:
                err_type = "PARTIAL MATCH / WRONG TYPE / BOUNDARY ERROR"

            errors.append((rec, true_entities, pred_entities, err_type))

        if len(errors) >= max_errors:
            break

    lines = []
    lines.append("=" * 100)
    lines.append(f"ERROR ANALYSIS (BEST MODEL = {best_result['name']})")
    lines.append("=" * 100)

    for idx, (rec, true_entities, pred_entities, err_type) in enumerate(errors, start=1):
        lines.append(f"[Error {idx}]")
        lines.append(f"Type     : {err_type}")
        lines.append(f"Language : {LANGUAGE_NAMES.get(rec['language'], rec['language'])} ({rec['language']})")
        lines.append(f"Sentence : {rec['sentence']}")
        lines.append(f"True     : {true_entities if true_entities else '[]'}")
        lines.append(f"Pred     : {pred_entities if pred_entities else '[]'}")
        lines.append("-" * 100)

    with open(ERROR_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ================================================================
# DEMO PREDICTIONS (BEST MODEL ONLY)
# ================================================================

def predict_sentence(model, tokenizer, sentence, max_len=128):
    """
    Simple whitespace-token demo prediction.
    For demo only.
    """
    tokens = sentence.strip().split()
    if not tokens:
        return []

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred_ids = torch.argmax(outputs.logits, dim=-1).squeeze(0).cpu().tolist()

    word_ids = encoding.word_ids(batch_index=0)

    token_labels = []
    seen_word_ids = set()

    for pos, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in seen_word_ids:
            continue
        seen_word_ids.add(word_idx)
        token_labels.append((tokens[word_idx], id2label[pred_ids[pos]]))

    # convert to entities
    out_tokens = [x[0] for x in token_labels]
    out_labels = [x[1] for x in token_labels]
    entities = bio_to_entity_spans(out_tokens, out_labels)
    return token_labels, entities

def write_demo_predictions(best_result):
    model_dir = best_result["save_dir"]
    model_name = best_result["name"]

    # Load best model from disk for demo
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(DEVICE)

    # Demo sentences (edit later if you want)
    demo_sentences = [
        ("en", "Narendra Modi launched PM Kisan in Assam with BJP leaders."),
        ("en", "Rahul Gandhi addressed Congress workers in Karnataka."),
        ("en", "Mamata Banerjee spoke about Ayushman Bharat in West Bengal."),
        ("ml", "നരേന്ദ്ര മോദി അസമിൽ പി എം കിസാൻ പദ്ധതിയെ കുറിച്ച് സംസാരിച്ചു"),
        ("kn", "ನರೇಂದ್ರ ಮೋದಿ ಅಸ್ಸಾಂನಲ್ಲಿ ಪಿಎಂ ಕಿಸಾನ್ ಯೋಜನೆ ಬಗ್ಗೆ ಮಾತನಾಡಿದರು"),
        ("pa", "ਨਰਿੰਦਰ ਮੋਦੀ ਨੇ ਅਸਾਮ ਵਿੱਚ ਪੀਐਮ ਕਿਸਾਨ ਯੋਜਨਾ ਬਾਰੇ ਗੱਲ ਕੀਤੀ"),
        ("or", "ନରେନ୍ଦ୍ର ମୋଦୀ ଅସମରେ ପିଏମ କିସାନ ଯୋଜନା ବିଷୟରେ କହିଲେ"),
        ("as", "নৰেন্দ্ৰ মোদীয়ে অসমত পিএম কিষাণ আঁচনিৰ বিষয়ে কৈছিল"),
    ]

    lines = []
    lines.append("=" * 100)
    lines.append(f"DEMO PREDICTIONS (BEST MODEL = {model_name})")
    lines.append("=" * 100)

    for idx, (lang, sent) in enumerate(demo_sentences, start=1):
        token_labels, entities = predict_sentence(model, tokenizer, sent, max_len=90)

        lines.append(f"[Demo {idx}]")
        lines.append(f"Language : {LANGUAGE_NAMES.get(lang, lang)} ({lang})")
        lines.append(f"Input    : {sent}")
        lines.append(f"Tokens   : {token_labels}")
        lines.append(f"Entities : {entities if entities else '[]'}")
        lines.append("-" * 100)

    with open(DEMO_PRED_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ================================================================
# MAIN
# ================================================================

def main():
    start_all = time.time()

    log("=" * 100)
    log("FINAL YEAR PROJECT - MULTILINGUAL PAN-INDIAN POLITICAL NER")
    log("=" * 100)
    log(f"Started at            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Device                : {DEVICE}")
    log(f"Models                : {', '.join([m['name'] for m in MODELS])}")
    log(f"Epochs                : {EPOCHS}")
    log(f"Learning Rate         : {LR}")
    log(f"Max Sentences Cap     : {'ALL' if MAX_SENTENCES == 0 else MAX_SENTENCES}")
    log(f"Train/Val/Test Split  : 70 / 15 / 15")
    log("")

    # Load data
    train_data, val_data, test_data, per_lang_counts = load_data()

    # Train all models
    results = []
    for idx, cfg in enumerate(MODELS, start=1):
        log("")
        log(f"===== MODEL {idx}/{len(MODELS)} : {cfg['name']} =====")
        result = train_one_model(cfg, train_data, val_data, test_data)
        results.append(result)

    # Write master reports
    best_result = write_master_reports(results, train_data, val_data, test_data)

    # Best-model specific reports
    write_best_per_language_report(best_result)
    write_confusion_summary(best_result)
    write_sample_predictions(best_result, max_samples=30)
    write_error_analysis(best_result, max_errors=40)
    write_demo_predictions(best_result)

    total_time = time.time() - start_all

    log("")
    log("=" * 100)
    log("ALL DONE SUCCESSFULLY")
    log("=" * 100)
    log(f"Finished at           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Total runtime (sec)   : {total_time:.2f}")
    log(f"Total runtime (hrs)   : {total_time/3600.0:.2f}")
    log("")
    log("OUTPUT FILES CREATED:")
    log(f"  {MASTER_REPORT_PATH}")
    log(f"  {MODEL_COMPARE_PATH}")
    log(f"  {BEST_MODEL_SUMMARY_PATH}")
    log(f"  {LABEL_DIST_PATH}")
    log(f"  {WEAK_LABEL_SUMMARY_PATH}")
    log(f"  {PER_LANGUAGE_PATH}")
    log(f"  {CONFUSION_PATH}")
    log(f"  {SAMPLE_PRED_PATH}")
    log(f"  {ERROR_ANALYSIS_PATH}")
    log(f"  {DEMO_PRED_PATH}")
    log(f"  {RUN_LOG_PATH}")
    for r in results:
        log(f"  {r['per_model_report_path']}")
    log("=" * 100)

if __name__ == "__main__":
    main()