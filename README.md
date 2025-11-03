Smart Wellness Planner+

An AI Agent that automates student wellness tracking, nutrition analysis, and exam-time focus planning using a fine-tuned LoRA model.

Project Information

Name: Siddhi Gupta
University: Indian Institute of Technology, Kanpur (IITK)
Department: Material Science and Engineering
Submission: DS Internship Assignment – AI Agent Prototype

Overview

Smart Wellness Planner+ is an AI-driven wellness companion that helps students maintain health and focus by analyzing their food, sleep, and study habits.
It supports two reasoning modes:

Normal Mode: gives general diet and sleep recommendations for balanced health.

Exam Mode: adapts to exam weeks with focus-boosting and energy-management suggestions.

The AI agent can reason over natural-language input (e.g., “Had maggi and coffee, slept 4 hours”) and produce a nutrition summary with insights.

Architecture
Component	Role
Fine-Tuned LoRA Model	Maps free-text food descriptions → structured nutrition output + insight
Planner-Executor Logic	Decides response type based on current mode (Normal / Exam)
Logger	Saves all interactions in JSON session logs
Streamlit App	Simple UI for user interaction and demo

Manual Task Automated

Students usually track meals, sleep, and study schedules manually to stay healthy during semester and exam periods.
This project automates that manual reasoning and planning process — the AI reads the text, evaluates nutrition, and gives focused advice instantly.

Dataset

File: data/train.jsonl

Size: ~50–600 text–nutrition pairs

Content: Indian meals + exam-context examples

Format Example:

{"input": "Had dal rice and tea in the evening.",
 "output": "Calories: 550 kcal; Protein: 15 g; Carbs: 80 g; Fat: 10 g; Insight: Balanced Indian meal."}

Fine-Tuning Setup
Parameter	Value
Base Model	EleutherAI/gpt-neo-1.3B (lightweight)
Method	LoRA (PEFT)
Learning Rate	2e-4
Epochs	5
Batch Size	2
Output Dir	models/smart-wellness-lora/

Evaluation

MAE (Calories): ≈ 60 kcal

Insight Similarity: ≈ 84 %

Qualitative: advice aligns with nutrition facts and exam context reasoning.

Tech Stack

Python 3.10+

Transformers, PEFT, Datasets, Torch, Streamlit

JSON logging for reproducible experiments.

Run Instructions

Install dependencies

pip install -r requirements.txt


Fine-tune LoRA model

python src/fine_tune_lora.py


Launch demo app

streamlit run app/app.py


Check logs

All sessions saved in /logs/ (Normal & Exam mode).

Repository Structure
Smart-Wellness-Planner/
 ├── app/                → Streamlit UI
 ├── src/                → Training + agent + logger
 ├── data/               → train.jsonl dataset
 ├── evaluation/         → eval.py metrics
 ├── logs/               → session logs
 ├── models/             → fine-tuned LoRA adapter (after training)
 ├── requirements.txt
 └── README.md


Future Work

Integrate wearable data (Fitbit / phone sensors)

Add mood & stress detection

Personalized weekly wellness dashboard
