import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================
# CONFIG
# ==========================

st.set_page_config(
    page_title="Teen Mental Health Prediction",
    page_icon="🧠",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================
# HEADER
# ==========================

st.title("🧠 Teen Mental Health Prediction")

st.markdown("""
Prediksi kemungkinan depresi pada remaja berdasarkan penggunaan media sosial,
kualitas tidur, tingkat stres, tingkat kecemasan, dan faktor kesehatan mental lainnya.
""")

st.divider()

# ==========================
# SIDEBAR INPUT
# ==========================

st.sidebar.header("📝 Input Data")

age = st.sidebar.slider(
    "Age",
    13,
    19,
    16
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

daily_social_media_hours = st.sidebar.slider(
    "Daily Social Media Hours",
    1.0,
    8.0,
    4.0
)

platform_usage = st.sidebar.selectbox(
    "Platform Usage",
    [
        "All Platforms",
        "Facebook",
        "Instagram",
        "TikTok",
        "YouTube"
    ]
)

sleep_hours = st.sidebar.slider(
    "Sleep Hours",
    4.0,
    9.0,
    6.5
)

screen_time_before_sleep = st.sidebar.slider(
    "Screen Time Before Sleep",
    0.5,
    3.0,
    1.5
)

academic_performance = st.sidebar.slider(
    "Academic Performance",
    2.0,
    4.0,
    3.0
)

physical_activity = st.sidebar.slider(
    "Physical Activity",
    0.0,
    2.0,
    1.0
)

social_interaction_level = st.sidebar.selectbox(
    "Social Interaction Level",
    ["High", "Low", "Medium"]
)

stress_level = st.sidebar.slider(
    "Stress Level",
    1,
    10,
    5
)

anxiety_level = st.sidebar.slider(
    "Anxiety Level",
    1,
    10,
    5
)

addiction_level = st.sidebar.slider(
    "Addiction Level",
    1,
    10,
    5
)

sleep_quality = st.sidebar.selectbox(
    "Sleep Quality",
    ["Fair", "Good", "Poor"]
)

# ==========================
# LABEL ENCODER MAPPING
# ==========================

gender_map = {
    "Female": 0,
    "Male": 1
}

platform_map = {
    "All Platforms": 0,
    "Facebook": 1,
    "Instagram": 2,
    "TikTok": 3,
    "YouTube": 4
}

social_map = {
    "High": 0,
    "Low": 1,
    "Medium": 2
}

sleep_quality_map = {
    "Fair": 0,
    "Good": 1,
    "Poor": 2
}

# ==========================
# PREDICTION
# ==========================

if st.sidebar.button("🔍 Prediksi"):

    input_data = pd.DataFrame(
        [[
            age,
            gender_map[gender],
            daily_social_media_hours,
            platform_map[platform_usage],
            sleep_hours,
            screen_time_before_sleep,
            academic_performance,
            physical_activity,
            social_map[social_interaction_level],
            stress_level,
            anxiety_level,
            addiction_level,
            sleep_quality_map[sleep_quality]
        ]],
        columns=[
            'age',
            'gender',
            'daily_social_media_hours',
            'platform_usage',
            'sleep_hours',
            'screen_time_before_sleep',
            'academic_performance',
            'physical_activity',
            'social_interaction_level',
            'stress_level',
            'anxiety_level',
            'addiction_level',
            'sleep_quality'
        ]
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    st.header("📈 Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Tidak Depresi",
            f"{probability[0]*100:.2f}%"
        )

    with col2:
        st.metric(
            "Depresi",
            f"{probability[1]*100:.2f}%"
        )

    st.divider()

    if prediction == 1:
        st.error("⚠️ Terindikasi Depresi")
    else:
        st.success("✅ Tidak Terindikasi Depresi")

# ==========================
# MODEL COMPARISON
# ==========================

st.divider()

st.header("📊 Perbandingan Model")

comparison = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Naive Bayes'
    ],
    'Accuracy': [
        0.988,
        0.904
    ],
    'Precision': [
        1.000,
        0.207
    ],
    'Recall': [
        0.500,
        1.000
    ],
    'F1-Score': [
        0.667,
        0.343
    ]
})

st.dataframe(
    comparison,
    use_container_width=True
)

fig, ax = plt.subplots(figsize=(8, 5))

comparison.set_index('Model').plot(
    kind='bar',
    ax=ax
)

ax.set_title("Perbandingan Performa Model")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)

plt.xticks(rotation=0)

st.pyplot(fig)

# ==========================
# BEST MODEL
# ==========================

st.divider()

st.success("""
🏆 Model Terbaik: Logistic Regression

Accuracy : 98.8%
Precision : 100%
Recall : 50%
F1-Score : 66.7%

Model Logistic Regression dipilih sebagai model utama karena memiliki
akurasi dan performa keseluruhan yang lebih baik dibandingkan Naive Bayes.
""")

# ==========================
# FOOTER
# ==========================

st.divider()

st.caption(
    "Project Data Mining | Teen Mental Health Prediction Dashboard"
)