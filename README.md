# Multimodal Emotion Analysis

> **Project Environment Notice:** This project was developed, trained, and executed on **Google Colab**.

---

## 1. Project Overview
The **Multimodal Emotion Analysis** system identifies human emotional states from multiple input modalities: **Text**, **Audio**, and **Video**. It extracts textual contextual embeddings using a pretrained BERT model (`bert-base-uncased`) and acoustic/spectral features using Librosa and Torchaudio. These cross-modal features are dynamically fused using **CAFNet** (Cross-Attention Fusion Network).

### Emotion Classes
The classification pipeline evaluates inputs across 8 distinct categories:
* **Happy 😄**: Joyful and positive vibes detected.
* **Sad 😢**: Tone indicates sorrow or disappointment.
* **Angry 😠**: Aggressive or frustrated sentiment noted.
* **Nervous 😬**: Tone shows anxiety or uncertainty.
* **Confused 😕**: Unclear expression or mixed emotion.
* **Excited 🤩**: High energy and enthusiasm sensed.
* **Calm 😌**: Serene and peaceful tone detected.
* **Neutral 😐**: No dominant sentiment—balanced tone[cite: 1].

---

## 2. Core Architecture & Pipeline
* **Text Representation:** Tokenized and embedded via `BertModel.from_pretrained('bert-base-uncased')` with mean-pooling over sequence hidden states ($768\text{-dim}$)[cite: 1].
* **Audio Representation:** Acoustic properties (pitch via Yin, RMS energy, and tempo) are extracted and padded into a $512\text{-dimensional}$ feature vector via `librosa`[cite: 1].
* **Cross-Attention Fusion (SimpleCAFNet):** 
  * Linear projections align text ($768 \to 256$) and audio/video feature representations ($512 \to 256$)[cite: 1].
  * Cross-attention gating weights the modalities dynamically[cite: 1]:
    $$\text{fused} = \alpha \cdot h_{\text{text}} + (1 - \alpha) \cdot h_{\text{audio/video}}$$
  * A multilayer classifier evaluates the concatenated representations to produce class logits across the 8 emotion classes[cite: 1].
* **Interactive Visualizations:**
  * Animated pulsing emoji display[cite: 1].
  * HTML Summary Card with colored confidence metrics and highlighted sentiment keywords[cite: 1].
  * Plotly interactive probability bar chart[cite: 1].
  * Polar radar chart displaying multidimensional emotional intensity[cite: 1].
  * Mel spectrogram generation via `torchaudio`[cite: 1].
  * Frame-by-frame video emotion labeling and multi-class temporal probability timelines[cite: 1].

---

## 3. How to Run in Google Colab (Primary Platform)

1. Open a new notebook on [Google Colab](https://colab.research.google.com/).
2. In the first cell, install system dependencies and required Python packages:
   ```bash
   !apt-get install -y ffmpeg portaudio19-dev
   !pip install torch torchvision torchaudio transformers opencv-python pytesseract matplotlib wordcloud pyaudio SpeechRecognition librosa pydub plotly pandas numpy
