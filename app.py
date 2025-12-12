import streamlit as st
import torch
import librosa
import numpy as np
import soundfile as sf
from jointmodel import JointModel

N_MELS = 64
FRAMES = 96
SR = 16000
WINDOW_LENGTH = 400
HOP_LENGTH = 160
SEGMENT_SAMPLES = 16000  # 1 second
FMIN = 125
FMAX = 7500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = JointModel().to(device)
model.load_state_dict(torch.load("joint_model.pth", map_location=device))
model.eval()

def processing_data(path):
    wav,_ = librosa.load(path,sr=SR)
    if len(wav) <= SEGMENT_SAMPLES:
        pad = SEGMENT_SAMPLES - len(wav)
        wav = np.pad(wav,(0,pad))
    else:
        wav = wav[:SEGMENT_SAMPLES]
    
    wav = wav/np.max(np.abs(wav))
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=SR,
        n_fft=WINDOW_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    logmel = librosa.power_to_db(mel,ref=np.max)

    if logmel.shape[1] < FRAMES:
        pad = np.zeros((N_MELS, FRAMES - logmel.shape[1]))
        logmel = np.hstack([logmel,pad])
    else:
        logmel = logmel[:,:FRAMES]
    
    wav_tensor = torch.tensor(wav,dtype=torch.float32).unsqueeze(0)
    logmel_tensor = torch.tensor(logmel,dtype=torch.float32).unsqueeze(0)

    return wav_tensor, logmel_tensor

def predict(wav_tensor, logmel_tensor):  

    wav_tensor = wav_tensor.to(device)
    logmel_tensor = logmel_tensor.to(device)

    with torch.no_grad():
        output = model(wav_tensor, logmel_tensor)
        prob = torch.sigmoid(output).item()
    
    label = "Female" if prob > 0.5 else "Male"
    confidence = prob if prob > 0.5 else 1 - prob

    return label, confidence

st.title("GENDER RECOGNITION FROM VOICE 🎤")
st.markdown("Upload audio OR record using microphone, then press **Predict Audio**.")

st.markdown("### Example Audio")
col1, col2 = st.columns(2)
with col1:
    st.audio("male_example.wav")
    male_example_btn = st.button("Predict Male Example")
with col2:
    st.audio("female_example.wav")
    female_example_btn = st.button("Predict Female Example")

st.markdown("---")

# -------------------------------------------------------
# Example Male/Female Buttons
# -------------------------------------------------------

if male_example_btn:
    wav_tensor, logmel_tensor = processing_data("male_example.wav")
    label, conf = predict(wav_tensor, logmel_tensor)
    st.success(f"**Prediction:** {label} ({conf*100:.2f}%)")
    st.audio("male_example.wav")

if female_example_btn:
    wav_tensor, logmel_tensor = processing_data("female_example.wav")
    label, conf = predict(wav_tensor, logmel_tensor)
    st.success(f"**Prediction:** {label} ({conf*100:.2f}%)")
    st.audio("female_example.wav")

# -------------------------------
# FILE UPLOAD SECTION
# -------------------------------
st.header("UPLOAD AUDIO FILE")
audio_file = st.file_uploader("Upload WAV file", type=["wav"], key="upload_file")
predict_upload_btn = st.button("Predict Uploaded Audio", key="predict_upload_btn")

if predict_upload_btn:
    if audio_file is None:
        st.error("Please upload a WAV file first!")
    else:
        wav_path = "temp_upload.wav"
        with open(wav_path, "wb") as f:
            f.write(audio_file.read())

        wav_tensor, logmel_tensor = processing_data(wav_path)
        label, conf = predict(wav_tensor, logmel_tensor)

        st.success(f"### Prediction: **{label}**")
        st.write(f"Confidence: **{conf*100:.2f}%**")
        st.audio(wav_path)


# -------------------------------
# MICROPHONE SECTION
# -------------------------------
st.header("RECORD USING MICROPHONE")
mic_audio = st.audio_input("Record your voice here", key="mic_input")
predict_mic_btn = st.button("Predict Microphone Audio", key="predict_mic_btn")

if predict_mic_btn:
    if mic_audio is None:
        st.error("Please record audio before predicting!")
    else:
        wav_path = "temp_mic.wav"
        with open(wav_path, "wb") as f:
            f.write(mic_audio.read())

        wav_tensor, logmel_tensor = processing_data(wav_path)
        label, conf = predict(wav_tensor, logmel_tensor)

        st.success(f"### Prediction: **{label}**")
        st.write(f"Confidence: **{conf*100:.2f}%**")
        st.audio(wav_path)
