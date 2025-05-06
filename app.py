import streamlit as st
import tempfile
import cv2
import numpy as np
import os
from ultralytics import YOLO
import mediapipe as mp
import requests
from streamlit_lottie import st_lottie
from twilio.rest import Client
from sendgrid.helpers.mail import Mail
import base64

def set_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# --- CONFIG ---
st.set_page_config(page_title="ResQDrone AI", layout="wide", page_icon="🚨")
# Example: Using an online image
set_bg_from_local("img.jpeg")


# Or: Using a local image (less reliable in Streamlit, best for deployed apps with static hosting)
# set_bg_image("background.jpg")

os.environ["PYTORCH_NO_CUSTOM_CLASS"] = "1"

# --- LOAD MODELS ---
model = YOLO("yolov8s.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# --- LOTTIE UTILS ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- LOGIN SYSTEM ---
USER_CREDENTIALS = {
    "admin@example.com": "admin123",
    "user@example.com": "user123"
}

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'email' not in st.session_state:
    st.session_state.email = ""

# Login form
if not st.session_state.logged_in:
    st.title("Login to RescueVision AI")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email in USER_CREDENTIALS and USER_CREDENTIALS[email] == password:
            st.session_state.logged_in = True
            st.session_state.email = email
            st.rerun()
        else:
            st.error(" Invalid credentials")

else:
    # --- SIDEBAR NAV ---
    st.sidebar.title("ResQDrone AI")
    page = st.sidebar.radio("Navigate", ["Home", "About Us", "Profile"])
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update({'logged_in': False, 'email': ""}))

    # --- ALERT HTML ---
    def play_alert():
        audio_path = "alert.mp3"
        if os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
                b64_audio = base64.b64encode(audio_bytes).decode()
                audio_html = f"""
                <audio autoplay loop>
                    <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
    # --- HOME PAGE ---
    if page == "Home":
        st.title(" ResQDrone AI - Posture Detection with Thermal View")
        uploaded_file = st.file_uploader("Upload a drone video (MP4)", type=["mp4"])

        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()
            alert_triggered = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                annotated_frame = frame.copy()
                urgent_detected = False

                for result in results:
                    for box in result.boxes:
                        if int(box.cls[0]) == 0:  # person class
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            person_crop = frame[y1:y2, x1:x2]
                            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                            pose_result = pose.process(person_rgb)

                            label = "Unknown"
                            color = (0, 255, 0)

                            if pose_result.pose_landmarks:
                                landmarks = pose_result.pose_landmarks.landmark
                                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
                                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y

                                vertical_span = abs((left_shoulder + right_shoulder)/2 - (left_hip + right_hip)/2)
                                if vertical_span < 0.2:
                                    label = " URGENT - Lying"
                                    color = (0, 0, 255)
                                    urgent_detected = True
                                else:
                                    label = "Standing"
                                    color = (0, 255, 0)

                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                thermal = cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
                combined = np.hstack((annotated_frame, thermal))
                display_frame = cv2.resize(combined, (960, 480))
                stframe.image(display_frame, channels="BGR", caption="Processed Frame", use_container_width=True)

                if urgent_detected:
                    play_alert()
            cap.release()
            st.success("Video processing completed.")
        else:
            st.info("Please upload a drone surveillance video to begin.")

    # --- ABOUT US PAGE ---
    elif page == "About Us":
        lottie_ai = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_jcikwtux.json")

        st.markdown('<div style="font-size: 2.5rem; color:#0a58ca; font-weight:600;">About RescueVision AI</div>', unsafe_allow_html=True)
        st.markdown("Saving lives through intelligent posture detection and thermal vision technology.")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("""
            RescueVision AI is a smart danger detection system designed to:
            - Detect unconscious or fallen individuals via posture estimation
            - Support real-time surveillance with simulated thermal imaging
            - Aid emergency services in tough terrains or night missions
            """)
        with col2:
            st_lottie(lottie_ai, height=280, key="about-anim")

    # --- PROFILE PAGE ---
    elif page == "Profile":
        lottie_user = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_pprxh53t.json")

        col1, col2 = st.columns([1, 2])
        with col1:
            st_lottie(lottie_user, height=180, key="profile-anim")
        with col2:
            st.title("Your Profile")
            st.markdown("Update your details. Stored in your browser (localStorage).")

        name = st.text_input("Name")
        email = st.text_input("Email")

        if st.button(" Save"):
            st.success("Saved!")
            st.markdown(f"""
            <script>
            window.localStorage.setItem("rescue_name", "{name}");
            window.localStorage.setItem("rescue_email", "{email}");
            alert("Info saved in your browser!");
            </script>
            """, unsafe_allow_html=True)
