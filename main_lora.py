import os
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Tuple

from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)

from peft import LoraConfig, TaskType, get_peft_model


# -----------------------
# Config
# -----------------------
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
SEED = 42
EPOCHS = 3
LR_FULL = 2e-5
LR_LORA = 2e-4          # adapters can usually take a higher LR
BS_TRAIN = 16
BS_EVAL = 32
WEIGHT_DECAY = 0.01
OUTPUT_DIR_FULL = "outputs/bert_full"
OUTPUT_DIR_LORA = "outputs/bert_lora"

set_seed(SEED)


# -----------------------
# Data
# -----------------------
print("Loading GLUE/SST-2…")
raw = load_dataset("glue", "sst2")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def preprocess(examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=MAX_LENGTH)

tokenized = raw.map(preprocess, batched=True, remove_columns=["sentence", "idx"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics (accuracy + F1)
acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
    }


# -----------------------
# Utilities
# -----------------------
def count_trainable_params(model: torch.nn.Module) -> Tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total
    return trainable, total, pct


def common_args(output_dir: str, lr: float) -> TrainingArguments:
    # fp16 if CUDA supports it; bf16 if AMPere+ and torch supports
    use_fp16 = torch.cuda.is_available()
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        num_train_epochs=EPOCHS,
        learning_rate=lr,
        per_device_train_batch_size=BS_TRAIN,
        per_device_eval_batch_size=BS_EVAL,
        weight_decay=WEIGHT_DECAY,
        logging_steps=50,
        report_to="none",
        fp16=use_fp16 and not use_bf16,
        bf16=use_bf16,
        seed=SEED,
        dataloader_num_workers=2,
    )


# -----------------------
# 1) Full fine-tuning
# -----------------------
print("\n=== Training: Full BERT-base fine-tune ===")
model_full = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

t_full, T_full, pct_full = count_trainable_params(model_full)
print(f"FULL FT trainable params: {t_full:,} / {T_full:,} ({pct_full:.2f}%)")

args_full = common_args(OUTPUT_DIR_FULL, LR_FULL)

trainer_full = Trainer(
    model=model_full,
    args=args_full,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer_full.train()
metrics_full = trainer_full.evaluate()
print("Full FT validation metrics:", metrics_full)


# -----------------------
# 2) LoRA fine-tuning (adapters)
# -----------------------
print("\n=== Training: BERT-base with LoRA adapters ===")
model_lora_base = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Enable gradient checkpointing to save memory (optional)
model_lora_base.gradient_checkpointing_enable()

# LoRA config — target common projection names inside BERT
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    target_modules=["query", "key", "value", "dense"],
)

model_lora = get_peft_model(model_lora_base, lora_cfg)

t_lora, T_lora, pct_lora = count_trainable_params(model_lora)
print(f"LoRA trainable params: {t_lora:,} / {T_lora:,} ({pct_lora:.2f}%)")

args_lora = common_args(OUTPUT_DIR_LORA, LR_LORA)

trainer_lora = Trainer(
    model=model_lora,
    args=args_lora,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer_lora.train()
metrics_lora = trainer_lora.evaluate()
print("LoRA validation metrics:", metrics_lora)


# -----------------------
# Side-by-side comparison
# -----------------------
def fmt(metrics: Dict[str, float]) -> Dict[str, float]:
    return {k: float(f"{v:.4f}") for k, v in metrics.items() if isinstance(v, (int, float))}

print("\n=== Comparison (Validation) ===")
print(f"Params (trainable / total):")
print(f"  Full FT: {t_full:,} / {T_full:,}  ({pct_full:.2f}%)")
print(f"  LoRA   : {t_lora:,} / {T_lora:,}  ({pct_lora:.2f}%)")

print("\nMetrics:")
print(f"  Full FT -> {fmt(metrics_full)}")
print(f"  LoRA    -> {fmt(metrics_lora)}")

# Save a tiny summary file
os.makedirs("outputs", exist_ok=True)
with open("outputs/summary.txt", "w") as f:
    f.write("BERT-base on GLUE/SST-2: Full FT vs LoRA\n")
    f.write(f"Seed: {SEED}, Epochs: {EPOCHS}\n\n")
    f.write("Trainable parameters:\n")
    f.write(f"  Full FT: {t_full} / {T_full} ({pct_full:.2f}%)\n")
    f.write(f"  LoRA   : {t_lora} / {T_lora} ({pct_lora:.2f}%)\n\n")
    f.write("Validation metrics:\n")
    f.write(f"  Full FT: {metrics_full}\n")
    f.write(f"  LoRA   : {metrics_lora}\n")

print("\nDone. Best models saved to:")
print(f"  {OUTPUT_DIR_FULL}")
print(f"  {OUTPUT_DIR_LORA}")
print("A textual summary is at outputs/summary.txt")
