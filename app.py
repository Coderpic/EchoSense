import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import numpy as np
from src.predict import predict_emotion

# ------------------ CONFIG ------------------
st.set_page_config(page_title="EchoSense | Emotion AI", layout="wide", page_icon="🎙️")

# ------------------ SESSION STATE & AUTO-RESET ------------------
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "last_mode" not in st.session_state:
    st.session_state.last_mode = "Upload Audio"

# ------------------ ELITE DARK BLUE CSS ------------------
st.markdown("""
<style>
    /* Dark Navy Body Background */
    .stApp {
        background-color: #0a192f;
    }

    /* Sidebar - Near Black for contrast */
    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1e293b;
    }

    /* Text Colors - Light Grey and Off-White */
    h1, h2, h3, b, strong {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }
    
    .stMarkdown p, .stCaption, label {
        color: #94a3b8 !important; /* Soft Light Grey */
    }

    /* The Main Prediction Card - Deep Blue with Glow */
    .prediction-container {
        background-color: #112240;
        padding: 40px;
        border-radius: 15px;
        border: 1px solid #233554;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .emotion-result {
        font-size: 64px;
        font-weight: 800;
        color: #64ffda !important; /* Teal accent for readability */
        text-shadow: 0 0 10px rgba(100, 255, 218, 0.3);
    }

    /* Sidebar Title Styling */
    .sidebar-title {
        color: #64ffda !important;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR & AUTO-RESET ------------------
with st.sidebar:
    st.markdown('<p class="sidebar-title">🎙️ EchoSense</p>', unsafe_allow_html=True)
    
    # Mode selection
    mode = st.radio(
        "Choose Audio Source",
        ["Upload Audio", "Record Audio"],
        key="mode_selection"
    )

    # AUTO-RESET TRIGGER: If user changes radio button, clear data
    if mode != st.session_state.last_mode:
        st.session_state.file_path = None
        st.session_state.last_mode = mode
        st.rerun()
    
    st.markdown("---")
    st.caption("Intelligence: Random Forest")
    if st.button("Clear Session", use_container_width=True):
        st.session_state.file_path = None
        st.rerun()

# ------------------ MAIN CONTENT ------------------
st.title("Speech Emotion Recognition")
st.markdown("Automated sentiment analysis through MFCC feature extraction.")

# --- INPUT AREA ---
col_in, _ = st.columns([2, 1])
with col_in:
    if mode == "Upload Audio":
        uploaded_file = st.file_uploader("Upload .wav file", type=["wav"], label_visibility="collapsed")
        if uploaded_file:
            # ✅ FIX: Play the audio using the uploaded file object directly (bytes)
            st.audio(uploaded_file, format="audio/wav")
            
            # Save to temp for your prediction function to use
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.getbuffer()) # Use getbuffer for better performance
                st.session_state.file_path = tmp.name
    else:
        recorded_audio = st.audio_input("Record voice snippet", label_visibility="collapsed")
        if recorded_audio:
            # ✅ FIX: Play the recording directly
            st.audio(recorded_audio)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(recorded_audio.read())
                st.session_state.file_path = tmp.name

# --- RESULTS SECTION ---
if st.session_state.file_path:
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.spinner("Decoding audio features..."):
        emotion = predict_emotion(st.session_state.file_path)
        audio, sr = librosa.load(st.session_state.file_path)

    # Hero Prediction Display
    st.markdown(f"""
        <div class="prediction-container">
            <p style="text-transform: uppercase; letter-spacing: 3px; font-size: 12px; color: #8892b0;">Primary Emotion</p>
            <div class="emotion-result">{emotion.upper()}</div>
        </div>
    """, unsafe_allow_html=True)

    # Visualization Grid
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.subheader("🔊 Waveform")
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0a192f')
        librosa.display.waveshow(audio, sr=sr, ax=ax, color="#64ffda") # Teal wave
        ax.set_facecolor('#0a192f')
        ax.tick_params(colors='#8892b0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#233554')
        st.pyplot(fig)

    with col_r:
        st.subheader("🎼 MFCC Features")
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        fig2, ax2 = plt.subplots(figsize=(10, 4), facecolor='#0a192f')
        img = librosa.display.specshow(mfcc, x_axis='time', ax=ax2, cmap='magma')
        ax2.set_facecolor('#0a192f')
        ax2.tick_params(colors='#8892b0')
        # Custom colorbar for dark theme
        cbar = fig2.colorbar(img, ax=ax2)
        cbar.ax.yaxis.set_tick_params(color='#8892b0')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8892b0')
        st.pyplot(fig2)

# --- FOOTER ---
st.markdown(
    "<div style='text-align: center; margin-top: 50px; border-top: 1px solid #233554; padding-top: 20px; color: #495670;'>"
    "EchoSense Dashboard • Built with Random Forest Classifier</div>", 
    unsafe_allow_html=True
)