import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import subprocess

# -------------------------------------------------
# SafeFusion AI X - Premium Autonomous Dashboard
# -------------------------------------------------

st.set_page_config(
    page_title="SafeFusion AI X",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# Premium CSS
# -------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg,#050816,#0B1220,#111827);
        color: #E5E7EB;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(90deg,#2563EB,#06B6D4);
        border-radius: 22px;
        padding: 28px;
        color: white;
        box-shadow: 0 20px 45px rgba(37,99,235,.35);
        margin-bottom: 22px;
    }

    .hero h1 {
        margin:0;
        font-size:2.4rem;
        font-weight:800;
    }

    .hero p {
        margin-top:10px;
        opacity:.92;
        font-size:1rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(17,24,39,.92);
        border:1px solid rgba(96,165,250,.22);
        border-radius:18px;
        padding:18px;
        box-shadow:0 12px 30px rgba(0,0,0,.35);
    }

    div[data-testid="stMetricLabel"] {
        color:#AFC6FF;
        font-weight:600;
    }

    div[data-testid="stMetricValue"] {
        color:#22D3EE;
        font-size:2rem;
        font-weight:800;
    }

    .stButton > button {
        background: linear-gradient(90deg,#2563EB,#06B6D4);
        color:white;
        border:none;
        border-radius:12px;
        padding:12px 22px;
        font-weight:700;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }

    .stTable table {
        border-radius:16px;
        overflow:hidden;
    }

    .stTable th {
        background:#111827;
        color:#E5E7EB;
    }

    .stTable td {
        background:#0B1220;
        color:#E5E7EB;
    }

    hr {
        border:none;
        border-top:1px solid rgba(148,163,184,.18);
        margin:24px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Hero Section
# -------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🚗 SafeFusion AI X</h1>
        <p>
            Predictive Explainable Collision Prevention for Autonomous Vehicles.
            YOLO11m + ByteTrack + Time-to-Collision + Explainable AI + Threat Intelligence.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
# Default stats (so dashboard works before processing)
stats = {
    "unique_pedestrians": 0,
    "unique_vehicles": 0,
    "cars": 0,
    "trucks": 0,
    "buses": 0,
    "motorcycles": 0,
    "bicycles": 0,
    "highest_risk": "LOW",
    "collision_warnings": 0,
    "video_frames": 0,
    "estimated_ttc": None,
    "fps": 0.0,
    "average_confidence": 0.0,
}

input_path = "videos/input_video.mp4"
output_video = "output/detected_video.mp4"
stats_path = "output/stats.json"

uploaded_file = st.file_uploader(
    "Choose a driving video",
    type=["mp4", "avi", "mov"],
    key="main_video_uploader"
)

if uploaded_file is not None:
    os.makedirs("videos", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Delete only old output files
    for path in [output_video, stats_path]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except PermissionError:
            pass

    # Save uploaded video
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("▶ Run SafeFusion AI"):
        with st.spinner("Processing video..."):
            result = subprocess.run(
                ["python", "detector.py"],
                capture_output=True,
                text=True,
            )

        st.code(result.stdout)

        if result.returncode != 0:
            st.error(result.stderr)
        else:
            st.success("Processing completed successfully!")

            if os.path.exists(stats_path):
                with open(stats_path, "r") as f:
                    stats = json.load(f)

                st.json(stats)

            if os.path.exists(output_video):
                with open(output_video, "rb") as video_file:
                    video_bytes = video_file.read()

                st.video(video_bytes)

                st.download_button(
                    label="Download Processed Video",
                    data=video_bytes,
                    file_name="SafeFusion_Output.mp4",
                    mime="video/mp4",
                )

            if os.path.exists(stats_path):
                with open(stats_path, "rb") as report_file:
                    st.download_button(
                        label="Download Analytics Report (JSON)",
                        data=report_file.read(),
                        file_name="SafeFusion_Report.json",
                        mime="application/json",
                    )
# Dashboard Header
# -------------------------------------------------
st.markdown(
    """
    <div style="background:linear-gradient(90deg,#111827,#0B1220);
                border:1px solid rgba(96,165,250,.18);
                border-radius:20px;
                padding:18px;
                margin-bottom:18px;
                box-shadow:0 10px 30px rgba(0,0,0,.25);">
        <h2 style="margin:0;color:#FFFFFF;">📊 Autonomous Perception Dashboard</h2>
        <p style="margin:8px 0 0 0;color:#CBD5E1;">
            Real-time tracking, collision analytics, trajectory prediction, and explainable autonomous safety intelligence
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Object Analytics (Dynamic)
# -------------------------------------------------
left, right = st.columns([1.2, 1])

# -----------------------------
# Object Analytics
# -----------------------------
with left:
    st.subheader("Object Analytics")

    analytics = pd.DataFrame(
        {
            "Object": [
                "Pedestrians",
                "Cars",
                "Trucks",
                "Buses",
                "Motorcycles",
                "Bicycles",
            ],
            "Count": [
                stats.get("unique_pedestrians", 0),
                stats.get("cars", 0),
                stats.get("trucks", 0),
                stats.get("buses", 0),
                stats.get("motorcycles", 0),
                stats.get("bicycles", 0),
            ],
        }
    )

    fig = px.bar(
        analytics,
        x="Object",
        y="Count",
        color="Object",
        text="Count",
        template="plotly_dark",
        title="Detected Objects by Category",
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(color="white", size=13),
        title_font=dict(size=18, color="white"),
        xaxis_title="",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="object_analytics_chart",
    )

# -----------------------------
# System Status
# -----------------------------
with right:
    st.subheader("System Status")

    fps_value = stats.get("fps", 30)
    confidence = stats.get("average_confidence", 0.0)

    risk = stats.get("highest_risk", "LOW")

    if risk == "HIGH":
        sensor_conf = "96%"
        status = "Critical"
    elif risk == "MEDIUM":
        sensor_conf = "92%"
        status = "Warning"
    else:
        sensor_conf = "89%"
        status = "Normal"

    st.metric("Video Frames", stats.get("video_frames", 0))
    st.metric("Average FPS", f"{fps_value:.2f}")
    st.metric("Detection Confidence", f"{confidence:.1f}%")
    st.metric("Sensor Confidence", sensor_conf)
    st.metric("System Status", status)

# -------------------------------------------------
# Threat Ranking (Dynamic)
# -------------------------------------------------
# -------------------------------------------------
# Collision Summary (Real Data)
# -------------------------------------------------

st.subheader("Collision Summary")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Collision Warnings",
    stats.get("collision_warnings", 0),
)

c2.metric(
    "Highest Risk",
    stats.get("highest_risk", "LOW"),
)

ttc = stats.get("estimated_ttc")
c3.metric(
    "Estimated TTC",
    f"{ttc:.2f} s" if ttc is not None else "N/A",
)

# -------------------------------------------------
# Risk Assessment Gauge (Dynamic)
# -------------------------------------------------

st.subheader("Risk Assessment Gauge")

risk_value = 20

if stats.get("highest_risk") == "MEDIUM":
    risk_value = 60
elif stats.get("highest_risk") == "HIGH":
    risk_value = 90

gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=risk_value,
        title={"text": "Collision Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red"},
            "steps": [
                {"range": [0, 40], "color": "green"},
                {"range": [40, 70], "color": "orange"},
                {"range": [70, 100], "color": "red"},
            ],
        },
    )
)

gauge.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
)

st.plotly_chart(
    gauge,
    use_container_width=True,
    key="risk_gauge_chart",
)

risk = stats.get("highest_risk", "LOW")

if risk == "HIGH":
    st.error(
        "Immediate braking or evasive steering is recommended. Collision risk is critical."
    )
elif risk == "MEDIUM":
    st.warning(
        "Reduce speed and maintain a safe following distance. Collision risk is moderate."
    )
else:
    st.success(
        "No immediate collision threat detected. Continue monitoring the environment."
    )

# -------------------------------------------------
# Explainable AI (Dynamic)
# -------------------------------------------------
st.subheader("Explainable AI Decision")

risk = stats["highest_risk"]
ped = stats["unique_pedestrians"]
veh = stats["unique_vehicles"]
warnings = stats["collision_warnings"]
ttc = stats["estimated_ttc"]

if risk == "HIGH":
    explanation = (
        f"The system detected **{ped} pedestrians** and **{veh} vehicles**. "
        f"A **HIGH collision risk** was identified with **{warnings} collision warnings**. "
        f"Estimated Time-to-Collision: **{ttc if ttc is not None else 'N/A'} seconds**. "
        "Immediate braking or avoidance is recommended."
    )
elif risk == "MEDIUM":
    explanation = (
        f"The system detected **{ped} pedestrians** and **{veh} vehicles**. "
        f"A **MEDIUM collision risk** was identified with **{warnings} warnings**. "
        f"Estimated Time-to-Collision: **{ttc if ttc is not None else 'N/A'} seconds**. "
        "The vehicle should slow down and prepare for possible braking."
    )
else:
    explanation = (
        f"The system detected **{ped} pedestrians** and **{veh} vehicles**. "
        "No significant collision threat was identified. Continue normal driving while monitoring the environment."
    )

st.info(explanation)

if ttc is not None:
    st.metric("Estimated Time to Collision", f"{ttc} s")
else:
    st.metric("Estimated Time to Collision", "N/A")



# -------------------------------------------------
# Safety Recommendation (Real Output)
# -------------------------------------------------

st.subheader("Safety Recommendation")

risk = stats.get("highest_risk", "LOW")
ttc = stats.get("estimated_ttc")

if risk == "HIGH":
    st.error(
        f"High collision risk detected. {stats.get('collision_warnings', 0)} warning events were triggered. Estimated TTC: {ttc:.2f} s. Immediate braking and right-lane avoidance are recommended."
    )

elif risk == "MEDIUM":
    st.warning(
        f"Moderate collision risk detected. Estimated TTC: {ttc:.2f} s. Reduce speed and maintain a safe following distance."
    )

else:
    st.success(
        "No immediate collision threat detected. Continue normal driving while monitoring the environment."
    )

st.markdown("---")
# -------------------------------------------------
# Safety Recommendation (Dynamic)
# -------------------------------------------------
st.subheader("Safety Recommendation")

risk = stats["highest_risk"]
warnings = stats["collision_warnings"]
ttc = stats["estimated_ttc"]

if risk == "HIGH":
    st.error(
        f"High collision risk detected. {warnings} warning events were triggered. "
        f"Estimated TTC: {ttc if ttc is not None else 'N/A'} s. "
        "Immediate braking and right-lane avoidance are recommended."
    )

elif risk == "MEDIUM":
    st.warning(
        f"Medium collision risk detected. {warnings} warning events were triggered. "
        f"Estimated TTC: {ttc if ttc is not None else 'N/A'} s. "
        "Reduce speed and prepare for braking."
    )

else:
    st.success(
        "No significant collision threat detected in this video. Continue normal driving and maintain lane awareness."
    )
# -------------------------------------------------
# Footer
# -------------------------------------------------
# -------------------------------------------------
# Premium Footer
# -------------------------------------------------

st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric("Model", "YOLO11 + ByteTrack")
col2.metric("Processing FPS", f"{stats.get('fps', 0):.2f}")
col3.metric(
    "Detection Confidence",
    f"{stats.get('average_confidence', 0):.1f}%"
)

st.markdown(
    """
    <div style="text-align:center; padding:20px 0;">
        <h1 style="color:white; font-size:52px; margin-bottom:8px;">
            SafeFusion AI X
        </h1>
        <p style="color:#9CA3AF; font-size:20px; margin-bottom:18px;">
            Predictive Explainable Collision Prevention for Autonomous Vehicles
        </p>
        <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
            <span style="background:#111827; color:#E5E7EB; padding:8px 12px; border-radius:999px;">YOLO11</span>
            <span style="background:#111827; color:#E5E7EB; padding:8px 12px; border-radius:999px;">ByteTrack</span>
            <span style="background:#111827; color:#E5E7EB; padding:8px 12px; border-radius:999px;">Time-to-Collision</span>
            <span style="background:#111827; color:#E5E7EB; padding:8px 12px; border-radius:999px;">Explainable AI</span>
            <span style="background:#111827; color:#E5E7EB; padding:8px 12px; border-radius:999px;">Autonomous Safety Intelligence</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)