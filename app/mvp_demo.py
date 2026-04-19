import streamlit as st
import pandas as pd
import sys
import os

# Ensure Python can find the 'models' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.rl_feedback_loop import RLFeedbackLoop

# ==========================================
# 1. Initialize RL Engine (Stored in session_state)
# ==========================================
if 'rl_engine' not in st.session_state:
    st.session_state.rl_engine = RLFeedbackLoop()

rl_engine = st.session_state.rl_engine

# ==========================================
# 2. Updated Mock Database with Stable Image IDs
# ==========================================
MOCK_DATABASE = {
    "explorer": [
        {
            "name": "The Vinyl Records Cafe", 
            "stars": 3.8, "dist": "4.5 km", 
            "desc": "Dim lighting, perfect vintage vibe for music lovers.", 
            "img": "https://images.unsplash.com/photo-1554118811-1e0d58224f24?w=500&q=80"
        },
        {
            "name": "Secret Jazz Diner", 
            "stars": 4.2, "dist": "5.2 km", 
            "desc": "Hidden gem with amazing live performances.", 
            "img": "https://images.unsplash.com/photo-1525610553991-2bede1a236e2?w=400&q=80"
        }
    ],
    "reputation": [
        {
            "name": "The Grand Steakhouse", 
            "stars": 4.9, "dist": "8.0 km", 
            "desc": "World-class dry-aged ribeye with impeccable service.", 
            "img": "https://images.unsplash.com/photo-1594041680534-e8c8cdebd659?w=500&q=80"
        },
        {
            "name": "Michelin Starred Bistro", 
            "stars": 4.8, "dist": "6.5 km", 
            "desc": "Local favorite with over 5,000 top-rated reviews.", 
            "img": "https://images.unsplash.com/photo-1559339352-11d035aa65de?w=500&q=80"
        }
    ],
    "convenience": [
        {
            "name": "Corner Quick Burger", 
            "stars": 3.4, "dist": "0.2 km", 
            "desc": "Speedy service, right around the corner.", 
            "img": "https://images.unsplash.com/photo-1550547660-d9450f859349?w=500&q=80"
        },
        {
            "name": "Street Food Express", 
            "stars": 4.1, "dist": "0.1 km", 
            "desc": "The most famous hotdog cart on your street.", 
            "img": "https://images.unsplash.com/photo-1612392061787-2d078b3e573c?w=500&q=80"
        }
    ]
}

# ==========================================
# 3. UI Layout and Interaction Logic
# ==========================================
st.set_page_config(layout="wide", page_title="Tourist Mode MVP")

# Lock the currently displayed strategy in the session_state.
if 'current_arm' not in st.session_state:
    st.session_state.current_arm = rl_engine.select_strategy()

# Define the callback function. This function will settle the accounts before the page is re-rendered!
def handle_feedback(reward_value):
    # 1. Add the score to the old strategy that was just presented.
    rl_engine.log_user_feedback(st.session_state.current_arm, reward=reward_value, query="cozy dinner")
    # 2. After adding the score, select a new strategy for the next page render
    st.session_state.current_arm = rl_engine.select_strategy()

# Sidebar: Performance Monitoring
with st.sidebar:
    st.header("🧠 RL Engine Dashboard")
    st.caption("Live Strategy Confidence (from q_values.json)")
    
    q_data = pd.DataFrame.from_dict(rl_engine.q_values, orient='index', columns=['Q-Value'])
    st.bar_chart(q_data)
    
    st.divider()
    if st.button("Clear RL Memory"):
        import json
        initial_q = {arm: 0.0 for arm in rl_engine.arms.keys()}
        with open(rl_engine.q_path, 'w') as f:
            json.dump(initial_q, f)
        rl_engine.q_values = initial_q
        # When clearing the memory, also reinsert the card once
        st.session_state.current_arm = rl_engine.select_strategy()
        st.success("History Reset!")
        st.rerun()

st.title("🍽️ Smart Tourist Recommender")
st.markdown("Search Query: **'A cozy place for dinner'**")

# Strategy Decision Step
current_arm = st.session_state.current_arm
recommendations = MOCK_DATABASE[current_arm]

st.info(f"🤖 **RL Decision:** Using the `{current_arm}` arm based on current rewards.")

# Rendering Comparison Cards
col1, col2 = st.columns(2)

with col1:
    r1 = recommendations[0]
    st.image(r1["img"], use_container_width=True)
    st.subheader(r1["name"])
    st.write(f"⭐ {r1['stars']} | 📍 {r1['dist']}")
    st.write(f"*{r1['desc']}*")
    st.button("👍 Select Option A", use_container_width=True, key="btn_a", 
              on_click=handle_feedback, args=(1.0,))

with col2:
    r2 = recommendations[1]
    st.image(r2["img"], use_container_width=True)
    st.subheader(r2["name"])
    st.write(f"⭐ {r2['stars']} | 📍 {r2['dist']}")
    st.write(f"*{r2['desc']}*")
    st.button("👍 Select Option B", use_container_width=True, key="btn_b", 
              on_click=handle_feedback, args=(1.0,))

st.divider()

# Negative Feedback Button
st.markdown("### Not what I'm looking for?")
st.button("👎 Refresh Results (-0.1 penalty)", type="primary", use_container_width=True, 
          on_click=handle_feedback, args=(-0.1,))