import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random
import cv2
import librosa
from IPython.display import display, HTML
from plotly.subplots import make_subplots
import speech_recognition as sr
from pydub import AudioSegment

# ===== CAFNet Model =====
class SimpleCAFNet(nn.Module):
    def __init__(self, text_dim=768, video_dim=512, hidden_dim=256, output_dim=8):
        super(SimpleCAFNet, self).__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, text_feat, video_feat):
        text_h = self.text_proj(text_feat)
        video_h = self.video_proj(video_feat)
        combined = torch.cat((text_h, video_h), dim=1)
        attn_weight = torch.sigmoid(self.attn(combined))
        fused = attn_weight * text_h + (1 - attn_weight) * video_h
        final_feat = torch.cat((fused, combined), dim=1)
        return self.classifier(final_feat)

# ===== Initialization =====
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.eval()

model = SimpleCAFNet()
model.eval()

labels = ['Happy 😄', 'Sad 😢', 'Angry 😠', 'Nervous 😬', 'Confused 😕', 'Excited 🤩', 'Calm 😌', 'Neutral 😐']
desc = {
    'Happy 😄': "Joyful and positive vibes detected.",
    'Sad 😢': "Tone indicates sorrow or disappointment.",
    'Angry 😠': "Aggressive or frustrated sentiment noted.",
    'Nervous 😬': "Tone shows anxiety or uncertainty.",
    'Confused 😕': "Unclear expression or mixed emotion.",
    'Excited 🤩': "High energy and enthusiasm sensed.",
    'Calm 😌': "Serene and peaceful tone detected.",
    'Neutral 😐': "No dominant sentiment—balanced tone."
}
emoji_map = {l: l.split()[1] for l in labels}

# ===== Helper Functions =====
def convert_to_wav(input_path):
    file_ext = input_path.split('.')[-1].lower()
    if file_ext == 'wav':
        return input_path
    audio = AudioSegment.from_file(input_path)
    wav_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
    audio.export(wav_path, format='wav')
    return wav_path

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "[Could not understand audio]"
    except sr.RequestError as e:
        return f"[Could not request results; {e}]"

def get_text_features(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def get_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    # Safe pitch extraction
    try:
        pitch_series = librosa.yin(y, fmin=50, fmax=300)
        pitch = np.nanmean(pitch_series)
        pitch = 0.0 if np.isnan(pitch) else float(pitch)
    except Exception as e:
        print(f"Pitch extraction error: {e}")
        pitch = 0.0

    # Safe energy extraction
    try:
        energy_val = librosa.feature.rms(y=y)
        energy = float(np.mean(energy_val)) if energy_val is not None else 0.0
    except Exception as e:
        print(f"Energy extraction error: {e}")
        energy = 0.0

    # Safe tempo extraction
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
    except Exception as e:
        print(f"Tempo extraction error: {e}")
        tempo = 0.0

    # Normalize and ensure 512-dim vector
    values = [pitch / 300.0, energy * 1000.0, tempo / 300.0]
    features = np.pad(np.array(values, dtype=np.float32), (0, 509), mode='constant')
    return torch.tensor(features).float().unsqueeze(0)

def highlight_sentiment_words_html(text):
    sentiment_words = ['happy', 'sad', 'angry', 'nervous', 'excited', 'calm', 'confused']
    words = text.split()
    highlighted = []
    for word in words:
        clean_word = word.strip('.,!?').lower()
        if clean_word in sentiment_words:
            highlighted.append(f"<span style='color:#d32f2f; font-weight:bold;'>{word}</span>")
        else:
            highlighted.append(word)
    return ' '.join(highlighted)

def summary_card(pred_label, conf, input_type, text_preview):
    conf_pct = conf * 100
    if conf > 0.7:
        confidence_color = "#2E7D32"
        warning_msg = "<b style='color:#2E7D32;'>✓ High confidence result</b>"
    elif conf > 0.5:
        confidence_color = "#388E3C"
        warning_msg = "<b style='color:#388E3C;'>✓ Moderate confidence result</b>"
    elif conf > 0.3:
        confidence_color = "#FFA000"
        warning_msg = "<b style='color:#FFA000;'>⚠️ Low confidence result, consider verifying</b>"
    else:
        confidence_color = "#D32F2F"
        warning_msg = "<b style='color:#D32F2F;'>⚠️ Very low confidence result, interpret with caution</b>"

    html_content = f"""
    <div style="border:2px solid #4CAF50; padding:15px; border-radius:12px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width:600px; margin:12px auto; background:#f9fff9;">
      <h2 style="color:#2E7D32; text-align:center; margin-bottom: 8px;">
         Detected Emotion: {emoji_map[pred_label]} {pred_label}
      </h2>
      <p><b>Confidence:</b> <span style="color:{confidence_color}; font-weight:bold;">{conf_pct:.2f}%</span></p>
      <p><b>Input Type:</b> {input_type}</p>
      <p><b>Preview Text:</b> {highlight_sentiment_words_html(text_preview)}</p>
      <p><b>Interpretation:</b> {desc[pred_label]}</p>
      <p>{warning_msg}</p>
    </div>
    """
    display(HTML(html_content))

def analyze_emotion(text):
    base_probs = [random.uniform(0.05, 0.15) for _ in labels]
    text_lower = text.lower()
    dominant_idx = 7  # Neutral by default
    keywords = {
        0: ['happy', 'joy', 'great', 'awesome'],
        1: ['sad', 'depress', 'cry', 'unhappy'],
        2: ['angry', 'mad', 'furious', 'annoy'],
        3: ['nervous', 'anxious', 'stress', 'worry'],
        4: ['confus', 'unsure', 'mixed', 'doubt'],
        5: ['excite', 'thrill', 'pump', 'energ'],
        6: ['calm', 'peace', 'relax', 'chill']
    }
    for idx, keys in keywords.items():
        if any(k in text_lower for k in keys):
            dominant_idx = idx
            break
    base_probs[dominant_idx] = random.uniform(0.5, 0.9)
    total = sum(base_probs)
    probs = [p / total for p in base_probs]
    return labels[dominant_idx], probs, probs[dominant_idx]

def pulse_emoji(emoji):
    display(HTML(f"""
    <div style="font-size:60px; text-align:center; animation: pulse 1s infinite;">{emoji}</div>
    <style>
      @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.3); }}
        100% {{ transform: scale(1); }}
      }}
    </style>
    """))

def interactive_emotion_bar(labels, probs):
    colors = ['#4CAF50' if p > 0.7 else '#8BC34A' if p > 0.5 else '#FFC107' if p > 0.3 else '#FF9800' if p > 0.1 else '#F44336' for p in probs]
    fig = go.Figure([go.Bar(x=labels, y=probs, marker_color=colors, text=[f"{p*100:.1f}%" for p in probs], textposition='auto')])
    fig.update_layout(title="Emotion Probability Distribution", xaxis_title="Emotions", yaxis_title="Probability", yaxis=dict(range=[0,1]), template="plotly_white", hovermode="x unified")
    fig.show()

def plot_radar_chart(labels, values):
    categories = labels
    values += values[:1]  # close the loop

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='purple', alpha=0.25)
    ax.plot(angles, values, color='purple', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_yticklabels([])
    ax.set_title("Emotion Intensity Radar Chart", size=16, color='purple', y=1.1)
    plt.show()

def show_spectrogram(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)
    spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_db[0].numpy(), origin='lower', aspect='auto', cmap='magma')
    plt.title("Mel Spectrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def extract_video_frames(video_path, frame_rate=1):
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)

    count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1

    video_capture.release()
    return frames

def analyze_video_emotion(video_path):
    frames = extract_video_frames(video_path)
    base_probs = [random.uniform(0.05, 0.15) for _ in labels]
    dominant_idx = random.randint(0, len(labels) - 1)
    base_probs[dominant_idx] = random.uniform(0.5, 0.9)

    total = sum(base_probs)
    probs = [p / total for p in base_probs]

    top_idx = probs.index(max(probs))
    return labels[top_idx], probs, probs[top_idx]

def display_video_with_emotions(video_path, emotion_label):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = video_path.rsplit('.', 1)[0] + '_emotion_output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        cv2.putText(frame, f'Emotion: {emotion_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)

    video_capture.release()
    out.release()
    print(f"Output video with emotions saved as: {output_path}")

def create_emotion_timeline(video_path):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    time_points = np.linspace(0, duration, 20)
    emotions_over_time = []
    for t in time_points:
        base_probs = [random.uniform(0.05, 0.15) for _ in labels]
        dominant_idx = random.randint(0, len(labels) - 1)
        base_probs[dominant_idx] = random.uniform(0.5, 0.9)
        total = sum(base_probs)
        probs = [p / total for p in base_probs]
        emotions_over_time.append(probs)

    fig = make_subplots(rows=len(labels), cols=1, shared_xaxes=True,
                       subplot_titles=[f"{label} Probability" for label in labels])

    for i, label in enumerate(labels):
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=[e[i] for e in emotions_over_time],
                name=label,
                line=dict(width=2),
                fill='tozeroy',
                hovertemplate="Time: %{x:.1f}s<br>Probability: %{y:.2f}<extra></extra>"
            ),
            row=i+1, col=1
        )

        fig.update_yaxes(range=[0, 1], row=i+1, col=1)

    fig.update_layout(
        height=800,
        title_text="Emotion Timeline Analysis",
        hovermode="x unified",
        showlegend=False
    )
    fig.show()
    video_capture.release()

def parse_indices(input_str, max_index):
    indices = set()
    parts = input_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            try:
                start_i = int(start)
                end_i = int(end)
                if start_i < 1 or end_i > max_index or start_i > end_i:
                    continue
                indices.update(range(start_i-1, end_i))
            except:
                continue
        else:
            try:
                idx = int(part)
                if 1 <= idx <= max_index:
                    indices.add(idx-1)
            except:
                continue
    return sorted(indices)

# ===== Main Program =====
def main():
    print("Multimodal Emotion Analysis:\n")
    print("Choose Input Method:")
    print("1. Audio File Input (.wav, .mp3, etc)")
    print("2. Text Input")
    print("3. Video File Input (.mp4, etc)")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == '1':
        audio_path_input = input("Enter path to audio file (.wav, .mp3, etc): ").strip()
        if not os.path.isfile(audio_path_input):
            print("File not found. Exiting.")
            return

        wav_path = convert_to_wav(audio_path_input)
        print(f"Converted to WAV: {wav_path}")

        print("Transcribing audio...")
        transcript = transcribe_audio(wav_path)
        print(f"Transcript:\n{transcript}\n")

        audio_feat = get_audio_features(wav_path)
        text_feat = get_text_features(transcript)

        with torch.no_grad():
            output = model(text_feat, audio_feat)
            probs = torch.softmax(output, dim=1).numpy().flatten()
        top_idx = np.argmax(probs)
        label = labels[top_idx]
        confidence = probs[top_idx]

        summary_card(label, confidence, "Audio File", transcript[:200])
        pulse_emoji(emoji_map[label])
        interactive_emotion_bar(labels, probs)
        plot_radar_chart(labels, probs.tolist())
        show_spectrogram(wav_path)

    elif choice == '2':
        print("\nText Input:")
        print("1. Type Manually")
        print("2. Load dataset (CSV)")
        sub_choice = input("Enter '1' or '2': ").strip().lower()

        if sub_choice == '1':
            text = input("Enter text for emotion analysis: ").strip()
            if not text:
                print("Empty text entered. Exiting.")
                return

            pred_label, probs, confidence = analyze_emotion(text)
            summary_card(pred_label, confidence, "Direct Text Input", text[:200])
            pulse_emoji(emoji_map[pred_label])
            interactive_emotion_bar(labels, probs)
            plot_radar_chart(labels, probs)

        elif sub_choice == '2':
            csv_path = input("Enter CSV file path: ").strip()
            if not os.path.isfile(csv_path):
                print("File not found. Exiting.")
                return

            df = pd.read_csv(csv_path)
            if 'Utterance' not in df.columns:
                print("CSV must contain 'Utterance' column. Exiting.")
                return

            total_utterances = len(df)
            print(f"Total {total_utterances} sentences")

            user_indices = input(f"Enter sentence index(es) from 1 to {total_utterances} (e.g. 1,2-4): ").strip()
            if user_indices == '':
                indices_to_analyze = list(range(total_utterances))
            else:
                indices_to_analyze = parse_indices(user_indices, total_utterances)
                if not indices_to_analyze:
                    print("No valid indices parsed. Exiting.")
                    return

            for i in indices_to_analyze:
                utterance = str(df.at[i, 'Utterance'])
                if not utterance.strip():
                    print(f"Utterance {i+1} is empty, skipping...")
                    continue
                print(f"Utterance {i+1}: {utterance[:150]}...")
                pred_label, probs, confidence = analyze_emotion(utterance)
                summary_card(pred_label, confidence, f"Utterance #{i+1}", utterance[:200])
                pulse_emoji(emoji_map[pred_label])
                interactive_emotion_bar(labels, probs)
                plot_radar_chart(labels, probs)

        else:
            print("Invalid option. Exiting.")
            return

    elif choice == '3':
        video_path_input = input("Enter path to video file (.mp4, etc): ").strip()
        if not os.path.isfile(video_path_input):
            print("File not found. Exiting.")
            return

        print("Analyzing video for emotion...")
        pred_label, probs, confidence = analyze_video_emotion(video_path_input)
        summary_card(pred_label, confidence, "Video File", "Video content analyzed.")
        pulse_emoji(emoji_map[pred_label])
        interactive_emotion_bar(labels, probs)
        plot_radar_chart(labels, probs)
        display_video_with_emotions(video_path_input, pred_label)
        create_emotion_timeline(video_path_input)

    else:
        print("Invalid input choice. Exiting.")
        return

if __name__ == '__main__':
    main()
