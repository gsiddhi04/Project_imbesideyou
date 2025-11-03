import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------- PATHS --------------------
base_model = "distilgpt2"
lora_model_path = os.path.join("models", "smart-wellness-lora")

# -------------------- LOAD MODEL --------------------
print(f"🔍 Loading LoRA model from: {lora_model_path}")

# ✅ Load tokenizer from the LoRA directory (not base model)
tokenizer = AutoTokenizer.from_pretrained(lora_model_path)

# ✅ Load base model
model = AutoModelForCausalLM.from_pretrained(base_model)

# ✅ Attach LoRA weights
model = PeftModel.from_pretrained(model, lora_model_path, ignore_mismatched_sizes=True)

model.eval()
print("✅ Model loaded successfully.")


prompt = "Give me a short healthy meal suggestion."
output = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), max_new_tokens=50)
print("\n🧠 Model response:\n", tokenizer.decode(output[0], skip_special_tokens=True))
