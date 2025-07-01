import streamlit as st
import pandas as pd
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from jinja2 import Template
#from weasyprint import HTML

# --- CONFIG ---
SHEET_ID = "1VBGZR3BYCXRrKJrgN8LntrqPHmwEdQ-ThQmTn1vIKJI"
SHEET_NAME = "Sheet1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# --- AUTHENTICATE GOOGLE SHEETS ---
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)

def load_data():
    df = pd.DataFrame(sheet.get_all_records())
    if "Points" in df.columns:
        df["Points"] = pd.to_numeric(df["Points"], errors="coerce").fillna(0).astype(int)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df

def save_entry(entry_dict):
    base_points = entry_dict["Time_Spent_Min"] // 5
    bonus = 5 if entry_dict["Completed"] else 0
    entry_dict["Points"] = int(base_points + bonus)
    df = load_data()
    df.loc[len(df)] = entry_dict
    set_with_dataframe(sheet, df)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Family Learning Tracker", layout="wide")
st.title("🎓 Family Learning Tracker")

ACTIVITIES = ["Reading", "Writing", "Math", "Science"]
BOOK_FIELD = "Book_Title"
HEADER = ["Timestamp", "Name", "Activity", "Mood", "Time_Spent_Min", "Completed", BOOK_FIELD]

le_mood = LabelEncoder()
le_mood.fit(["Focused", "Frustrated", "Happy", "Distracted"])

# --- MODEL SETUP ---
sample_data = pd.DataFrame({
    "Activity": ["Reading", "Math", "Writing", "Science"],
    "Mood": ["Happy", "Frustrated", "Focused", "Distracted"],
    "Time_Spent_Min": [15, 20, 10, 25],
    "Completed": [1, 0, 1, 1]
})
sample_data["Mood_Encoded"] = le_mood.transform(sample_data["Mood"])
sample_data = pd.get_dummies(sample_data, columns=["Activity"])
required_columns = ['Time_Spent_Min', 'Mood_Encoded'] + [f"Activity_{a}" for a in ACTIVITIES]
for col in required_columns:
    if col not in sample_data.columns:
        sample_data[col] = 0
X = sample_data[required_columns]
y = sample_data["Completed"]
model = DecisionTreeClassifier(random_state=42).fit(X, y)


def generate_weekly_report(df_kid, child_name, age, chart_path, coach_message, output_path="weekly_report.pdf"):
        today = datetime.today()
        start_date = (today - timedelta(days=7)).strftime('%b %d, %Y')
        end_date = today.strftime('%b %d, %Y')

        df_kid = df_kid.copy()
        df_kid["Completed"] = pd.to_numeric(df_kid["Completed"], errors="coerce")
        df_kid["Time_Spent_Min"] = pd.to_numeric(df_kid["Time_Spent_Min"], errors="coerce")
        df_kid["Timestamp"] = pd.to_datetime(df_kid["Timestamp"], errors="coerce")

        df_week = df_kid[df_kid["Timestamp"] >= today - timedelta(days=7)]
        weekly_completion = int(df_week["Completed"].sum() / len(df_week) * 100) if len(df_week) else 0
        total_minutes = int(df_week["Time_Spent_Min"].sum())
        active_days = df_week["Timestamp"].dt.date.nunique()
        sessions_completed = df_week["Completed"].sum()

        mood_stats = df_week.groupby("Mood")["Completed"].mean().reset_index()
        if not mood_stats.empty:
            best_mood_row = mood_stats.loc[mood_stats["Completed"].idxmax()]
            best_mood = best_mood_row["Mood"]
            best_rate = int(best_mood_row["Completed"] * 100)
        else:
            best_mood, best_rate = "—", "—"

        bins = [0, 10, 20, 30, 40, 50, 60]
        labels = ["0–10", "11–20", "21–30", "31–40", "41–50", "51–60"]
        df_week["Time_Bin"] = pd.cut(df_week["Time_Spent_Min"], bins=bins, labels=labels)
        time_stats = df_week.groupby("Time_Bin")["Completed"].mean().reset_index()
        if not time_stats.empty:
            sweet_spot = time_stats.loc[time_stats["Completed"].idxmax()]
            sweet_spot_time = sweet_spot["Time_Bin"]
            sweet_rate = int(sweet_spot["Completed"] * 100)
        else:
            sweet_spot_time, sweet_rate = "—", "—"

        df_reading = df_week[df_week["Activity"] == "Reading"]
        books_read = df_reading["Book_Title"].nunique()
        goal = 4 if age == 6 else 8
        reading_reward = "🎉 Goal Met!" if books_read >= goal else f"{goal - books_read} book(s) to go"

        subject_counts = df_week[df_week["Completed"] == 1]["Activity"].value_counts()
        top_subject = subject_counts.idxmax() if not subject_counts.empty else "—"

        total_points = int(df_kid["Points"].sum())
        badges = []
        if df_kid.shape[0] >= 5: badges.append("Consistent Learner 🧠")
        if (df_kid["Timestamp"].dt.hour < 9).any(): badges.append("Early Riser ☀️")
        if books_read >= goal: badges.append("Reading Star 📚")
        if total_points >= 100: badges.append("Point Pro 🎯")
        if df_kid["Timestamp"].dt.date.nunique() >= 3: badges.append("Streak Streaker 🔥")

        with open("report_template.html", "r") as f:
            html_template = Template(f.read())

        rendered_html = html_template.render(
            child_name=child_name,
            age=age,
            start_date=start_date,
            end_date=end_date,
            weekly_completion=weekly_completion,
            sessions_completed=sessions_completed,
            total_minutes=total_minutes,
            active_days=active_days,
            best_mood=best_mood,
            best_rate=best_rate,
            sweet_spot_time=sweet_spot_time,
            sweet_rate=sweet_rate,
            top_subject=top_subject,
            books_read=books_read,
            goal=goal,
            reading_reward=reading_reward,
            total_points=total_points,
            badges=badges,
            activity_chart_path=chart_path,
            coach_message=coach_message
        )

        HTML(string=rendered_html).write_pdf(output_path)
        return output_path


# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📋 Log Session", "📈 Dashboard", "🧹 Inspect & Clean Data"])

with tab1:
    st.subheader("Log a New Learning Session")

    # Layout: Form (left) | Prediction (right)
    left_col, right_col = st.columns([2, 1])

    with left_col:
        name = st.text_input("Child's Name")
        activity = st.selectbox("Activity", ACTIVITIES)
        mood = st.selectbox("Mood", le_mood.classes_)
        time_spent = st.slider("Time Spent (minutes)", 5, 60, 15)
        completed = st.radio("Completed?", [1, 0])
        book_title = ""
        if activity == "Reading":
            book_title = st.text_input("Book Title (optional)").strip()

        if st.button("Submit Session"):
            if name:
                base_points = time_spent // 5
                bonus = 5 if completed else 0
                entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Name": name.strip(),
                    "Activity": activity,
                    "Mood": mood,
                    "Time_Spent_Min": time_spent,
                    "Completed": completed,
                    "Book_Title": book_title if activity == "Reading" else "",
                    "Points": int(base_points + bonus)
                }
                save_entry(entry)
                st.success("✅ Entry saved successfully!")
            else:
                st.error("🚫 Please enter a name to save this session.")

    with right_col:
        st.markdown("### 🎯 Completion Prediction")
        if name:
            mood_val = le_mood.transform([mood])[0]
            activity_features = {f"Activity_{a}": 1 if a == activity else 0 for a in ACTIVITIES}
            input_features = [[time_spent, mood_val] + list(activity_features.values())]
            prediction = model.predict(input_features)[0]

            if st.button("🔮 Predict"):
                if prediction:
                    st.success("✅ Likely to complete the session!")
                else:
                    st.warning("⚠️ Might need support or a break.")
        else:
            st.info("Fill in the child's name to enable prediction.")

with tab2:
    st.subheader("📊 Learning Dashboard")

    # Top-right aligned button
    header_cols = st.columns([9, 1])
    with header_cols[1]:

        df = load_data()

    if df.empty:
        st.info("No data logged yet.")
    else:
        top_row = st.columns([1.5, 1, 1, 1, 1, 1])
        with top_row[0]:
            selected_name = st.selectbox("👤 Choose a child", df["Name"].dropna().unique())
        with top_row[1]:
            age = st.selectbox("🎂 Age", [6, 9], key="age_box")

        df_kid = df[df["Name"] == selected_name].copy()
        df_kid["Completed"] = pd.to_numeric(df_kid["Completed"], errors="coerce")
        df_kid["Timestamp"] = pd.to_datetime(df_kid["Timestamp"], errors="coerce")
        df_kid["Time_Spent_Min"] = pd.to_numeric(df_kid["Time_Spent_Min"], errors="coerce")

        now = datetime.now()
        df_week = df_kid[df_kid["Timestamp"] >= now - timedelta(days=7)]
        weekly_completion = df_week["Completed"].sum() / df_week.shape[0] if not df_week.empty else 0
        reading_sessions = df_kid[df_kid["Activity"] == "Reading"].shape[0]
        reading_pct = reading_sessions / df_kid.shape[0] if not df_kid.empty else 0
        longest_session = df_kid["Time_Spent_Min"].max()
        df_month = df_kid[(df_kid["Timestamp"].dt.month == now.month) & (df_kid["Timestamp"].dt.year == now.year)]
        active_days = df_month["Timestamp"].dt.date.nunique()

        with top_row[2]:
            st.metric("✅ Weekly Completion", f"{weekly_completion:.0%}")
        with top_row[3]:
            st.metric("📖 % Reading", f"{reading_pct:.0%}")
        with top_row[4]:
            st.metric("⏱️ Longest Session", f"{int(longest_session) if pd.notna(longest_session) else 0} min")
        with top_row[5]:
            st.metric("📅 Active Days", active_days)
        #if st.button("📄 Generate Report"):
         #   chart_path = "activity_chart.png"
          #  coach_note = ""
           # pdf_path = generate_weekly_report(df_kid, selected_name, age, chart_path, coach_note)

            #with open(pdf_path, "rb") as f:
             #   st.download_button(
              #      "📥 Download Weekly Report",
               #     f,
                #    file_name=f"{selected_name}_JimmyAcademy_Report.pdf",
                 #   mime="application/pdf"
                #)

        # Weekly Highlights section
        st.markdown("### 🏁 Weekly Highlights")
        col1, col_div1, col2, col_div2, col3 = st.columns([1, 0.02, 1, 0.02, 1])

        # ⏳ Time by Subject + Completion Insights
        with col1:
            st.markdown("<div style='padding:1rem;border:1px solid #ddd;border-radius:0.5rem;background:#f9f9f9'>", unsafe_allow_html=True)
            st.markdown("#### ⏳ By Subject")
            subject_totals = df_week.groupby("Activity")["Time_Spent_Min"].sum().reindex(ACTIVITIES, fill_value=0)
            if df_week.empty:
                st.info("No sessions this week.")
            else:
                for subject, minutes in subject_totals.items():
                    st.markdown(f"**{subject}** — {int(minutes)} min")
                    st.progress(min(minutes / 60, 1.0))

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("#### 📈 Completion Insights")

            mood_stats = df_kid.groupby("Mood")["Completed"].mean().reset_index()
            if not mood_stats.empty:
                best_mood = mood_stats.loc[mood_stats["Completed"].idxmax()]["Mood"]
                best_rate = mood_stats["Completed"].max()
                st.info(f"🧠 Highest Completion Mood: **{best_mood}** ({best_rate:.0%})")
            else:
                st.info("🧠 Not enough mood data to analyze.")

            bins = [0, 10, 20, 30, 40, 50, 60]
            labels = ["0–10", "11–20", "21–30", "31–40", "41–50", "51–60"]
            df_kid["Time_Bin"] = pd.cut(df_kid["Time_Spent_Min"], bins=bins, labels=labels)
            time_stats = df_kid.groupby("Time_Bin")["Completed"].mean().reset_index()
            if not time_stats.empty:
                sweet_spot = time_stats.loc[time_stats["Completed"].idxmax()]
                st.success(f"⏱️ Sweet Spot: **{sweet_spot['Time_Bin']} min** ({sweet_spot['Completed']:.0%})")
            else:
                st.info("⏱️ Not enough time data to analyze.")

            st.markdown("</div>", unsafe_allow_html=True)
                # 🏅 Points & Rewards + Badges
        with col2:
            st.markdown("<div style='padding:1rem;border:1px solid #ddd;border-radius:0.5rem;background:#f9f9f9'>", unsafe_allow_html=True)
            st.markdown("#### 🏅 Points & Reward")
            df_reading = df_kid[
                (df_kid["Activity"] == "Reading") &
                (df_kid["Book_Title"].notnull()) &
                (df_kid["Timestamp"].dt.month == now.month) &
                (df_kid["Timestamp"].dt.year == now.year)
            ]
            books_read = df_reading["Book_Title"].nunique()
            goal = 4 if age == 6 else 8
            total_points = df_kid["Points"].sum()
            st.markdown(f"**Total Points:** {int(total_points)}")

            if total_points >= 200:
                reward = "📦 Mystery box"
            elif total_points >= 100:
                reward = "🍿 Movie token"
            elif total_points >= 50:
                reward = "🧩 Learning game"
            elif total_points >= 25:
                reward = "🎨 Choose activity"
            else:
                reward = "📚 Keep logging!"
            st.markdown(f"**Unlocked:** {reward}")

            # Badges
            st.markdown("<hr style='margin:0.75rem 0'>", unsafe_allow_html=True)
            st.markdown("#### 🧱 Badges")
            badge_cols = st.columns(5)
            with badge_cols[0]:
                if df_kid.shape[0] >= 5:
                    st.markdown("🧠")
                    st.caption("Consistent Learner")
            with badge_cols[1]:
                if (df_kid["Timestamp"].dt.hour < 9).any():
                    st.markdown("☀️")
                    st.caption("Early Riser")
            with badge_cols[2]:
                if books_read >= goal:
                    st.markdown("📚")
                    st.caption("Reading Star")
            with badge_cols[3]:
                if total_points >= 100:
                    st.markdown("🎯")
                    st.caption("Point Pro")
            with badge_cols[4]:
                if df_kid["Timestamp"].dt.date.nunique() >= 3:
                    st.markdown("🔥")
                    st.caption("Streak Streaker")
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div style='padding:1rem;border:1px solid #ddd;border-radius:0.5rem;background:#f9f9f9'>", unsafe_allow_html=True)
            st.markdown("#### 📚 Book Progress")

            st.markdown(f"**Books This Month:** {books_read} / {goal}")
            st.progress(min(books_read / goal, 1.0))
            if books_read >= goal:
                st.success("🎉 Goal Met!")
                st.markdown("- 🎮 Game of choice  \n- 🎬 Theater movie  \n- 💵 $20 reward")
            else:
                st.info(f"📖 {goal - books_read} more to go!")

            # 🧩 Activity Mix chart
            st.markdown("#### 🧩 Activity Mix (Past 7 Days)")
            completed_week = df_week[df_week["Completed"] == 1]
            activity_counts = completed_week["Activity"].value_counts().reindex(ACTIVITIES, fill_value=0)
            total = activity_counts.sum()
            activity_pct = (activity_counts / total * 100).round(1) if total else activity_counts

            fig_bar, ax_bar = plt.subplots(figsize=(3.5, 1.5))
            sns.barplot(x=activity_pct.values, y=activity_pct.index, ax=ax_bar, palette="pastel")
            for i, (pct, label) in enumerate(zip(activity_pct.values, activity_pct.index)):
                ax_bar.text(pct + 1, i, f"{pct:.0f}%" if total else "0%", va="center")

            ax_bar.set_xlim(0, 100)
            ax_bar.set_xlabel("")
            ax_bar.set_ylabel("")
            ax_bar.set_title("")
            ax_bar.spines["top"].set_visible(False)
            ax_bar.spines["right"].set_visible(False)
            ax_bar.spines["bottom"].set_visible(False)
            ax_bar.spines["left"].set_visible(False)
            ax_bar.tick_params(left=False, bottom=False)
            ax_bar.get_xaxis().set_visible(False)

            st.pyplot(fig_bar)
            fig_bar.savefig("activity_chart.png", bbox_inches="tight")
            st.markdown("</div>", unsafe_allow_html=True)

        # 🗃️ Logged Sessions Table
        st.markdown("### 🗃️ Logged Sessions")
        st.dataframe(
            df_kid.sort_values("Timestamp", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=300
        )
  ##  tab3 = st.tabs(["🧹 Inspect & Clean Data"])[0]

with tab3:
    st.subheader("📦 Raw Data from Google Sheet")

    try:
        df_inspect = load_data()

        if df_inspect.empty:
            st.info("ℹ️ No data found. Try logging a session first.")
        else:
            st.dataframe(df_inspect, use_container_width=True)

            malformed = df_inspect[df_inspect.isnull().any(axis=1)]
            if not malformed.empty:
                st.warning(f"⚠️ Found {len(malformed)} incomplete rows:")
                st.dataframe(malformed)

                if st.button("💾 Download Cleaned CSV"):
                    cleaned = df_inspect.fillna("")
                    cleaned.to_csv("learning_log_cleaned.csv", index=False)
                    st.success("File saved as 'learning_log_cleaned.csv'")
            else:
                st.success("✅ All entries look complete.")
    except Exception as e:
        st.error(f"Something went wrong while reading the sheet: {e}")

    # One-time patch utility
    def calculate_points(row):
        try:
            time_spent = int(row["Time_Spent_Min"])
            base = time_spent // 5
        except (ValueError, TypeError):
            base = 0
        completed = row.get("Completed", 0)
        bonus = 5 if str(completed).strip().lower() in ["1", "true", "yes"] else 0
        return base + bonus

    def backfill_points():
        df = load_data()
        df["Time_Spent_Min"] = pd.to_numeric(df["Time_Spent_Min"], errors="coerce")
        df["Completed"] = pd.to_numeric(df["Completed"], errors="coerce")
        df["Points"] = df.apply(calculate_points, axis=1)
        set_with_dataframe(sheet, df)
        st.success("✅ Points successfully added to all rows!")

    if st.button("🔁 Recalculate Points for All Sessions"):
        backfill_points()

    def generate_weekly_report(df_kid, child_name, age, chart_path, coach_message, output_path="weekly_report.pdf"):
        today = datetime.today()
        start_date = (today - timedelta(days=7)).strftime('%b %d, %Y')
        end_date = today.strftime('%b %d, %Y')

        df_kid = df_kid.copy()
        df_kid["Completed"] = pd.to_numeric(df_kid["Completed"], errors="coerce")
        df_kid["Time_Spent_Min"] = pd.to_numeric(df_kid["Time_Spent_Min"], errors="coerce")
        df_kid["Timestamp"] = pd.to_datetime(df_kid["Timestamp"], errors="coerce")

        df_week = df_kid[df_kid["Timestamp"] >= today - timedelta(days=7)]
        weekly_completion = int(df_week["Completed"].sum() / len(df_week) * 100) if len(df_week) else 0
        total_minutes = int(df_week["Time_Spent_Min"].sum())
        active_days = df_week["Timestamp"].dt.date.nunique()
        sessions_completed = df_week["Completed"].sum()

        mood_stats = df_week.groupby("Mood")["Completed"].mean().reset_index()
        if not mood_stats.empty:
            best_mood_row = mood_stats.loc[mood_stats["Completed"].idxmax()]
            best_mood = best_mood_row["Mood"]
            best_rate = int(best_mood_row["Completed"] * 100)
        else:
            best_mood, best_rate = "—", "—"

        bins = [0, 10, 20, 30, 40, 50, 60]
        labels = ["0–10", "11–20", "21–30", "31–40", "41–50", "51–60"]
        df_week["Time_Bin"] = pd.cut(df_week["Time_Spent_Min"], bins=bins, labels=labels)
        time_stats = df_week.groupby("Time_Bin")["Completed"].mean().reset_index()
        if not time_stats.empty:
            sweet_spot = time_stats.loc[time_stats["Completed"].idxmax()]
            sweet_spot_time = sweet_spot["Time_Bin"]
            sweet_rate = int(sweet_spot["Completed"] * 100)
        else:
            sweet_spot_time, sweet_rate = "—", "—"

        df_reading = df_week[df_week["Activity"] == "Reading"]
        books_read = df_reading["Book_Title"].nunique()
        goal = 4 if age == 6 else 8
        reading_reward = "🎉 Goal Met!" if books_read >= goal else f"{goal - books_read} book(s) to go"

        subject_counts = df_week[df_week["Completed"] == 1]["Activity"].value_counts()
        top_subject = subject_counts.idxmax() if not subject_counts.empty else "—"

        total_points = int(df_kid["Points"].sum())
        badges = []
        if df_kid.shape[0] >= 5: badges.append("Consistent Learner 🧠")
        if (df_kid["Timestamp"].dt.hour < 9).any(): badges.append("Early Riser ☀️")
        if books_read >= goal: badges.append("Reading Star 📚")
        if total_points >= 100: badges.append("Point Pro 🎯")
        if df_kid["Timestamp"].dt.date.nunique() >= 3: badges.append("Streak Streaker 🔥")

        with open("report_template.html", "r", encoding="utf-8") as f:  # ✅ FIXED LINE
            html_template = Template(f.read())


        rendered_html = html_template.render(
            child_name=child_name,
            age=age,
            start_date=start_date,
            end_date=end_date,
            weekly_completion=weekly_completion,
            sessions_completed=sessions_completed,
            total_minutes=total_minutes,
            active_days=active_days,
            best_mood=best_mood,
            best_rate=best_rate,
            sweet_spot_time=sweet_spot_time,
            sweet_rate=sweet_rate,
            top_subject=top_subject,
            books_read=books_read,
            goal=goal,
            reading_reward=reading_reward,
            total_points=total_points,
            badges=badges,
            activity_chart_path=chart_path,
            coach_message=coach_message
        )

        HTML(string=rendered_html).write_pdf(output_path)
        return output_path
