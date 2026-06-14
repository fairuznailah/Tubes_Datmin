import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Teen Mental Health Prediction",
    page_icon="🧠"
)

# ==========================
# HEADER
# ==========================

st.title("🧠 Teen Mental Health Prediction")

st.write("""
Aplikasi ini digunakan untuk memprediksi kemungkinan depresi pada remaja
berdasarkan aktivitas media sosial, kualitas tidur, tingkat stres,
tingkat kecemasan, dan faktor kesehatan mental lainnya.

Model yang digunakan:
- Logistic Regression
- Naive Bayes
""")

st.divider()

# ==========================
# INFORMASI DATASET
# ==========================

st.subheader("📊 Informasi Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jumlah Data", "1200")

with col2:
    st.metric("Tidak Depresi", "1170")

with col3:
    st.metric("Depresi", "30")

st.divider()

# ==========================
# INPUT DATA
# ==========================

st.header("📋 Input Data")

age = st.slider(
    "Age",
    13,
    19,
    16
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

daily_social_media_hours = st.slider(
    "Daily Social Media Hours",
    1.0,
    8.0,
    4.0
)

platform_usage = st.selectbox(
    "Platform Usage",
    [
        "All Platforms",
        "Facebook",
        "Instagram",
        "TikTok",
        "YouTube"
    ]
)

sleep_hours = st.slider(
    "Sleep Hours",
    4.0,
    9.0,
    6.5
)

screen_time_before_sleep = st.slider(
    "Screen Time Before Sleep",
    0.5,
    3.0,
    1.5
)

academic_performance = st.slider(
    "Academic Performance",
    2.0,
    4.0,
    3.0
)

physical_activity = st.slider(
    "Physical Activity",
    0.0,
    2.0,
    1.0
)

social_interaction_level = st.selectbox(
    "Social Interaction Level",
    ["High", "Low", "Medium"]
)

stress_level = st.slider(
    "Stress Level",
    1,
    10,
    5
)

anxiety_level = st.slider(
    "Anxiety Level",
    1,
    10,
    5
)

addiction_level = st.slider(
    "Addiction Level",
    1,
    10,
    5
)

sleep_quality = st.selectbox(
    "Sleep Quality",
    ["Fair", "Good", "Poor"]
)

# ==========================
# MAPPING LABEL ENCODER
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
# PREDIKSI
# ==========================

if st.button("🔍 Prediksi"):

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

    st.divider()

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

    if prediction == 1:
        st.error("⚠️ Terindikasi Depresi")
    else:
        st.success("✅ Tidak Terindikasi Depresi")

    st.info("""
    Prediksi dilakukan menggunakan model Logistic Regression
    yang memiliki performa terbaik berdasarkan hasil evaluasi model.
    """)

# ==========================
# PERBANDINGAN MODEL
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

fig, ax = plt.subplots(figsize=(8,5))

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
# KESIMPULAN
# ==========================

st.divider()

st.header("📝 Kesimpulan")

st.write("""
Berdasarkan hasil evaluasi model, Logistic Regression memberikan performa terbaik
dengan accuracy sebesar 98.8%.

Model ini memiliki precision sebesar 100%, recall sebesar 50%, dan F1-score sebesar 66.7%.

Oleh karena itu, Logistic Regression dipilih sebagai model utama untuk melakukan
prediksi kondisi depresi pada remaja.
""")

# ==========================
# FOOTER
# ==========================

st.divider()

st.caption(
    "Project Data Mining 2025 | Teen Mental Health Prediction Dashboard"
)