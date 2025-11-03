"""
planner_executor.py
-------------------
AI agent logic for Smart Wellness Planner+.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.logger import log_interaction


class SmartWellnessAgent:
    def __init__(self, base_model="mistralai/Mistral-7B-v0.1"):
        # dynamically locate the model folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        adapter_path = os.path.join(base_dir, "models", "smart-wellness-lora")

        print(f"🔍 Loading adapter from: {adapter_path}")

        # Load tokenizer and models
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.mode = "Normal"

    def set_mode(self, mode):
        if mode.lower() in ["exam", "normal"]:
            self.mode = mode.capitalize()
            print(f"🧩 Mode set to: {self.mode}")
        else:
            print("Invalid mode. Choose 'Normal' or 'Exam'.")

    def analyze(self, text):
        prefix = "You are a wellness assistant. "
        if self.mode == "Exam":
            prefix += "This is exam week. Focus on concentration, alertness, and nutrition for energy.\n"
        else:
            prefix += "Focus on general wellness, healthy balance, and recovery.\n"

        prompt = prefix + f"Analyze: {text}\nGive nutrition + one short insight."
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=80)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        log_interaction(self.mode, text, response)
        return response


if __name__ == "__main__":
    agent = SmartWellnessAgent()
    agent.set_mode("Exam")
    print(agent.analyze("Had maggi and coffee, slept 4 hours."))
