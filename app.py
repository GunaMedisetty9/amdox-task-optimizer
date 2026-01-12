
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Page config
st.set_page_config(
    page_title="AI Task Optimizer",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown('''
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    </style>
    ''', unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">🎯 AI-Powered Task Optimizer</p>', unsafe_allow_html=True)
st.markdown("### *By NIT Trichy Student*")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    try:
        with open('duration_model.pkl', 'rb') as f:
            duration_model = pickle.load(f)
        with open('recommendation_model.pkl', 'rb') as f:
            recommendation_model = pickle.load(f)
        return duration_model, recommendation_model
    except:
        return None, None

duration_model, recommendation_model = load_models()

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('task_dataset.csv')
        df_stress = pd.read_csv('stress_monitoring.csv')
        df_team = pd.read_csv('team_analytics.csv')
        return df, df_stress, df_team
    except:
        return None, None, None

df, df_stress, df_team = load_data()

# Sidebar
st.sidebar.header("🎮 Navigation")
page = st.sidebar.radio("Go to", 
    ["🏠 Dashboard", "🔮 Predict Task", "📊 Analytics", "👥 Team View", "🚨 Stress Monitor"])

# ==================== HOME DASHBOARD ====================
if page == "🏠 Dashboard":
    st.header("📊 Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Total Tasks", "1000", "+50 this week")
    with col2:
        st.metric("⏱️ Avg Duration", "10.6 hrs", "-0.5 hrs")
    with col3:
        st.metric("🎯 Model Accuracy", "85.5%", "+2.3%")
    with col4:
        st.metric("👥 Team Health", "Good", "4.7/10 stress")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Task Priority Distribution")
        if df is not None:
            priority_counts = df['priority'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['red', 'orange', 'yellow', 'green']
            priority_counts.plot(kind='bar', ax=ax, color=colors)
            ax.set_title('Task Distribution by Priority')
            ax.set_ylabel('Count')
            plt.xticks(rotation=0)
            st.pyplot(fig)
    
    with col2:
        st.subheader("😊 Mood Impact on Duration")
        if df is not None:
            mood_duration = df.groupby('mood')['estimated_duration'].mean().sort_values()
            fig, ax = plt.subplots(figsize=(8, 5))
            mood_duration.plot(kind='barh', ax=ax, color='teal')
            ax.set_title('Average Duration by Mood')
            ax.set_xlabel('Hours')
            st.pyplot(fig)

# ==================== TASK PREDICTION ====================
elif page == "🔮 Predict Task":
    st.header("🔮 Task Duration Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        task_desc = st.selectbox("📝 Task Type", 
            ["Bug fix", "Code review", "API development", "Database migration", 
             "Research", "Client call", "Documentation"])
        priority = st.selectbox("⚡ Priority", ["Low", "Medium", "High", "Critical"])
        mood = st.selectbox("😊 Current Mood", 
            ["Happy", "Motivated", "Calm", "Neutral", "Tired", "Anxious", "Stressed"])
    
    with col2:
        deadline = st.slider("📅 Days Until Deadline", 1, 30, 7)
        workload = st.slider("💼 Current Workload (hrs)", 2.0, 12.0, 6.0, 0.5)
    
    if st.button("🚀 Predict Duration", type="primary"):
        priority_map = {'Low': 0, 'Medium': 2, 'High': 3, 'Critical': 1}
        mood_map = {'Anxious': 0, 'Calm': 1, 'Happy': 2, 'Motivated': 3, 
                    'Neutral': 4, 'Stressed': 5, 'Tired': 6}
        task_map = {t: i for i, t in enumerate(df['task_description'].unique())}
        
        priority_enc = priority_map.get(priority, 2)
        mood_enc = mood_map.get(mood, 1)
        task_enc = task_map.get(task_desc, 0)
        
        urgency_score = priority_enc * (30 - deadline) / 30
        stress_factor = mood_enc * workload
        
        features = [[priority_enc, mood_enc, task_enc, deadline, 
                     workload, urgency_score, stress_factor]]
        
        if duration_model:
            duration = duration_model.predict(features)[0]
            st.success(f"### ⏱️ Predicted Duration: **{duration:.1f} hours**")
            
            if duration > 15:
                st.warning("⚠️ Long task! Consider breaking into chunks.")
            elif duration < 5:
                st.info("✅ Quick task! Good for filling gaps.")

# ==================== ANALYTICS ====================
elif page == "📊 Analytics":
    st.header("📊 Advanced Analytics")
    
    if df is not None:
        tab1, tab2 = st.tabs(["📈 Trends", "📋 Data"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Workload Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                df['workload_hours'].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                ax.set_xlabel('Hours')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Deadline Urgency")
                fig, ax = plt.subplots(figsize=(8, 5))
                df['days_until_deadline'].hist(bins=30, ax=ax, color='coral', edgecolor='black')
                ax.set_xlabel('Days')
                st.pyplot(fig)
        
        with tab2:
            st.dataframe(df.head(100), use_container_width=True)

# ==================== TEAM VIEW ====================
elif page == "👥 Team View":
    st.header("👥 Team Performance Dashboard")
    
    if df_team is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📈 Avg Productivity", f"{df_team['productivity'].mean():.1f}%")
        with col2:
            st.metric("🚨 Avg Stress", f"{df_team['stress_level'].mean():.1f}/10")
        with col3:
            st.metric("✅ Total Tasks", df_team['tasks_completed'].sum())
        
        st.markdown("---")
        st.dataframe(df_team, use_container_width=True)

# ==================== STRESS MONITOR ====================
elif page == "🚨 Stress Monitor":
    st.header("🚨 Stress Management System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        work_hours = st.slider("⏰ Hours Worked Today", 0.0, 12.0, 6.0, 0.5)
        tasks_pending = st.slider("📋 Pending Tasks", 0, 20, 5)
    
    with col2:
        current_mood = st.selectbox("😊 Mood", 
            ["Happy", "Motivated", "Calm", "Neutral", "Tired", "Anxious", "Stressed"])
        deadline_pressure = st.slider("📅 Days to Deadline", 1, 15, 5)
    
    if st.button("Calculate Stress", type="primary"):
        mood_scores = {'Happy': 1, 'Motivated': 2, 'Calm': 2, 'Neutral': 5, 
                       'Tired': 6, 'Anxious': 8, 'Stressed': 9}
        
        stress = (
            (work_hours / 12) * 3 +
            (tasks_pending / 10) * 2 +
            (mood_scores.get(current_mood, 5) / 10) * 3 +
            (1 / max(deadline_pressure, 1)) * 2
        )
        stress = min(round(stress, 1), 10)
        
        if stress >= 8:
            st.error(f"🚨 CRITICAL STRESS: {stress}/10")
        elif stress >= 6:
            st.warning(f"⚠️ HIGH STRESS: {stress}/10")
        else:
            st.success(f"✅ HEALTHY: {stress}/10")

st.markdown("---")
st.markdown("*🎓 NIT Trichy Student | AI Task Management System*")
