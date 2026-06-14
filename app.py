import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model dan scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Teen Mental Health Prediction",
    page_icon="🧠"
)

st.title("🧠 Teen Mental Health Prediction")

st.write(
    "Prediksi kemungkinan depresi pada remaja berdasarkan "
    "penggunaan media sosial dan kondisi kesehatan mental."
)

st.divider()

# ======================
# INPUT DATA
# ======================

age = st.slider("Age", 13, 19, 16)

gender = st.selectbox(
    "Gender",
    ["female", "male"]
)

daily_social_media_hours = st.slider(
    "Daily Social Media Hours",
    1.0, 8.0, 4.0
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
    4.0, 9.0, 6.5
)

screen_time_before_sleep = st.slider(
    "Screen Time Before Sleep",
    0.5, 3.0, 1.5
)

academic_performance = st.slider(
    "Academic Performance",
    2.0, 4.0, 3.0
)

physical_activity = st.slider(
    "Physical Activity",
    0.0, 2.0, 1.0
)

social_interaction_level = st.selectbox(
    "Social Interaction Level",
    ["high", "low", "medium"]
)

stress_level = st.slider(
    "Stress Level",
    1, 10, 5
)

anxiety_level = st.slider(
    "Anxiety Level",
    1, 10, 5
)

addiction_level = st.slider(
    "Addiction Level",
    1, 10, 5
)

sleep_quality = st.selectbox(
    "Sleep Quality",
    ["Fair", "Good", "Poor"]
)

# ======================
# MAPPING LABEL ENCODER
# ======================

gender_map = {
    "female": 0,
    "male": 1
}

platform_map = {
    "All Platforms": 0,
    "Facebook": 1,
    "Instagram": 2,
    "TikTok": 3,
    "YouTube": 4
}

social_map = {
    "high": 0,
    "low": 1,
    "medium": 2
}

sleep_quality_map = {
    "Fair": 0,
    "Good": 1,
    "Poor": 2
}

# ======================
# PREDIKSI
# ======================

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

    if prediction == 1:
        st.error("⚠️ Terindikasi Depresi")
    else:
        st.success("✅ Tidak Terindikasi Depresi")

    st.write(
        f"Probabilitas Tidak Depresi : {probability[0]*100:.2f}%"
    )

    st.write(
        f"Probabilitas Depresi : {probability[1]*100:.2f}%"
    )

    st.subheader("📊 Perbandingan Model")

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

st.dataframe(comparison)

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