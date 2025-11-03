# import sys, os
# sys.path.append(os.path.abspath("."))  # ✅ Forces root path so imports always work

# from src.planner_executer import SmartWellnessAgent  # ✅ Correct absolute import

# import streamlit as st









# st.set_page_config(page_title="Smart Wellness Planner+", page_icon="🥗", layout="centered")

# st.title("🥗 Smart Wellness Planner+")
# st.write("Your AI wellness buddy for healthy habits and exam focus.")

# # Sidebar mode switch
# mode = st.sidebar.selectbox("Select Mode", ["Normal", "Exam"])
# user_input = st.text_area("Describe your day (food, sleep, or study)...")

# if st.button("Analyze"):
#     agent = SmartWellnessAgent()
#     agent.set_mode(mode)
#     result = agent.analyze(user_input)
#     st.success(result)

# 

import sys, os
sys.path.append(os.path.abspath("."))

import streamlit as st
from src.planner_executer import SmartWellnessAgent

st.set_page_config(page_title="Smart Wellness Planner+", page_icon="🥗", layout="centered")
st.title("🥗 Smart Wellness Planner+")
st.write("Your AI wellness buddy for healthy habits and exam focus.")

mode = st.sidebar.selectbox("Select Mode", ["Normal", "Exam"])
user_input = st.text_area("Describe your day (food, sleep, or study)...")

@st.cache_resource
def get_agent():
    return SmartWellnessAgent()

@st.cache_data
def run_analysis(agent, text, mode):
    agent.set_mode(mode)
    return agent.analyze(text)

if st.button("Analyze"):
    agent = get_agent()
    with st.spinner("Analyzing your day..."):
        result = run_analysis(agent, user_input, mode)
    st.success(result)
