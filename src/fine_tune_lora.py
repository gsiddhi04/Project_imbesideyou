"""
fine_tune_lora.py
-----------------
LoRA fine-tuning script for the Smart Wellness Planner+ project.
"""

import json
import torch
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


# ----------------------- CONFIG ------------------------
BASE_MODEL = "distilgpt2"   # small & VRAM friendly
DATA_PATH = "Data/train.jsonl"  # ✅ matches your folder
OUTPUT_DIR = "models/smart-wellness-lora"
EPOCHS = 5
BATCH_SIZE = 2
LR = 2e-4
MAX_LEN = 256
# --------------------------------------------------------


# -------------------- LOAD DATASET -----------------------
def load_jsonl_dataset(path):
    with open(path, "r") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    return {"input": [x["input"] for x in lines], "output": [x["output"] for x in lines]}

print("📦 Loading dataset...")
raw_data = load_jsonl_dataset(DATA_PATH)
dataset = Dataset.from_dict(raw_data)

# ------------------- TOKENIZER --------------------------
print("🔠 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize(example):
    text = f"Input: {example['input']}\nOutput: {example['output']}"
    return tokenizer(text, truncation=True, max_length=MAX_LEN, padding="max_length")

print("🔍 Tokenizing...")
tokenized = dataset.map(tokenize, batched=False).shuffle(seed=42)

# ------------------ BASE MODEL --------------------------
print("🧠 Loading base model...")
device_map = "auto" if torch.cuda.is_available() else None
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map=device_map
)
model.resize_token_embeddings(len(tokenizer))

# ------------------ PREPARE FOR LORA ---------------------
print("🔧 Applying LoRA configuration...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],  # ✅ safer for distilgpt2
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ------------------ TRAINING SETUP -----------------------
print("⚙️ Setting up Trainer...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    learning_rate=LR,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ------------------- TRAIN ------------------------------
print("🚀 Starting LoRA fine-tuning...")
trainer.train()

# ------------------- SAVE -------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"💾 Saving model to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Fine-tuning complete. LoRA model saved.")
