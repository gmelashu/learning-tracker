
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
import csv


# 🚨 Auto-repair corrupted CSV
csv_file = "learning_log.csv"
expected_cols = 6
if os.path.exists(csv_file):
    try:
        # Try reading the file with expected columns
        df_check = pd.read_csv(csv_file)
        if df_check.shape[1] != expected_cols:
            raise ValueError("Malformed CSV: incorrect number of columns")
    except Exception as e:
        # Rename the corrupted file
        os.rename(csv_file, csv_file.replace(".csv", "_broken.csv"))
        st.warning("⚠️ Previous CSV was corrupted and has been renamed. A new one will be created.")


# ----------------------------
# ✅ INITIAL TRAINING DATA
# ----------------------------

sample_data = {
    'Name': ['ChildA', 'ChildB', 'ChildA', 'ChildB'],
    'Activity': ['Reading', 'Math', 'Puzzle', 'Reading'],
    'Time_Spent_Min': [15, 10, 20, 12],
    'Mood': ['Focused', 'Frustrated', 'Happy', 'Focused'],
    'Completed': [1, 0, 1, 1]
}
df_train = pd.DataFrame(sample_data)

# Encode features
le_mood = LabelEncoder()
df_train['Mood_Encoded'] = le_mood.fit_transform(df_train['Mood'])
df_train = pd.get_dummies(df_train, columns=['Activity'])

# Prepare features
required_columns = ['Time_Spent_Min', 'Mood_Encoded', 'Activity_Math', 'Activity_Puzzle']
for col in required_columns:
    if col not in df_train.columns:
        df_train[col] = 0

X = df_train[required_columns]
y = df_train['Completed']

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# ----------------------------
# 🧠 APP STARTS
# ----------------------------

st.title("🎓 Personalized Learning Tracker")
csv_file = "learning_log.csv"

# Ensure new CSV with correct headers
header = ['Timestamp', 'Name', 'Activity', 'Mood', 'Time_Spent_Min', 'Completed']
if not os.path.isfile(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

# ----------------------------
# 📝 DATA ENTRY FORM
# ----------------------------

with st.form("log_form"):
    st.subheader("📝 Log a New Session")
    name = st.text_input("Child's Name")
    activity = st.selectbox("Activity", ["Reading", "Math", "Puzzle"])
    mood = st.selectbox("Mood", le_mood.classes_)
    time_spent = st.slider("Time Spent (minutes)", 5, 60, 15)
    completed = st.radio("Completed?", [1, 0])
    submitted = st.form_submit_button("Submit")

    if submitted:
        entry = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Name': name,
            'Activity': activity,
            'Mood': mood,
            'Time_Spent_Min': time_spent,
            'Completed': completed
        }
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow(entry)
        st.success(f"✅ Entry for {name} saved.")

# ----------------------------
# 🔮 PREDICTION INTERFACE
# ----------------------------

st.subheader("🔮 Predict Task Completion")
if name:
    mood_val = le_mood.transform([mood])[0]
    activity_data = {
        'Activity_Math': 1 if activity == 'Math' else 0,
        'Activity_Puzzle': 1 if activity == 'Puzzle' else 0
    }
    input_features = [[time_spent, mood_val, activity_data['Activity_Math'], activity_data['Activity_Puzzle']]]

    if st.button("Predict Now"):
        result = model.predict(input_features)[0]
        if result == 1:
            st.success("✅ Likely to complete!")
        else:
            st.warning("⚠️ Might need support.")

# ----------------------------
# 📊 DASHBOARD
# ----------------------------

if os.path.isfile(csv_file):
    st.subheader("📊 Progress Dashboard")

    df_log = pd.read_csv(csv_file)

    if not df_log.empty and "Name" in df_log.columns:
        child_names = df_log['Name'].dropna().unique()
        selected_name = st.selectbox("Select a child", child_names)
        df_kid = df_log[df_log['Name'] == selected_name]

        st.markdown(f"### 📚 History for {selected_name}")
        st.dataframe(df_kid.sort_values("Timestamp", ascending=False), use_container_width=True)

        # Mood vs Completion
        st.markdown("#### 😊 Completion Rate by Mood")
        mood_chart = df_kid.groupby('Mood')['Completed'].mean().reset_index()
        fig1, ax1 = plt.subplots()
        sns.barplot(data=mood_chart, x='Mood', y='Completed', palette='coolwarm', ax=ax1)
        ax1.set_ylabel("Completion Rate")
        st.pyplot(fig1)

        # Time vs Completion
        st.markdown("#### ⏱ Time Spent vs Completion")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df_kid, x='Time_Spent_Min', y='Completed', hue='Activity', ax=ax2)
        ax2.set_ylabel("Completed (1 = Yes, 0 = No)")
        st.pyplot(fig2)
    else:
        st.info("No child data available yet.")
else:
    st.info("Log some sessions to unlock insights.")

    # 🎯 Goal Tracking Setup
from datetime import timedelta

GOAL_PER_WEEK = 5  # Sessions per child per week

if not df_kid.empty:
    st.subheader("🏆 Weekly Goal Progress")

    # Get current week's start (Monday)
    today = datetime.now()
    monday = today - timedelta(days=today.weekday())
    df_kid['Timestamp'] = pd.to_datetime(df_kid['Timestamp'], errors='coerce')
    this_week = df_kid[df_kid['Timestamp'] >= monday]

    completed_this_week = this_week[this_week['Completed'] == 1].shape[0]

    st.progress(min(completed_this_week / GOAL_PER_WEEK, 1.0))

    if completed_this_week >= GOAL_PER_WEEK:
        st.success(f"🎉 {selected_name} reached their weekly goal! {completed_this_week} sessions logged.")
    else:
        remaining = GOAL_PER_WEEK - completed_this_week
        st.info(f"{selected_name} has completed {completed_this_week} out of {GOAL_PER_WEEK} sessions this week. {remaining} to go!")

    with st.expander("🧹 Clean & Review CSV Data"):
        if os.path.exists(csv_file):
            df_raw = pd.read_csv(csv_file)

            # Show raw data before cleaning
            st.write("📦 Raw Data Preview:", df_raw)

            # Strip whitespace from names (important for filtering)
            if 'Name' in df_raw.columns:
                df_raw['Name'] = df_raw['Name'].astype(str).str.strip()

            # Fix Timestamp formatting
            if 'Timestamp' in df_raw.columns:
                df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'], errors='coerce')

            # Drop rows with missing name or timestamp
            cleaned_df = df_raw.dropna(subset=['Name', 'Timestamp'])

            st.markdown("### ✅ Cleaned Data Snapshot")
            st.dataframe(cleaned_df.sort_values("Timestamp", ascending=False), use_container_width=True)

            st.info(f"Cleaned {len(df_raw) - len(cleaned_df)} rows (blank names or bad timestamps removed).")
        else:
            st.warning("No CSV found yet. Add an entry first to begin tracking.")