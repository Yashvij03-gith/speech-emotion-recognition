# 🎤 Speech Emotion Recognition

> A machine learning system that detects emotions in human speech with **77% accuracy**

Automatically classify voice recordings into **happy**, **sad**, **angry**, and **neutral** emotions using deep learning.

---

## 🏗️ Project Structure

```
speech-emotion-recognition/
├── train.py              # Training pipeline
├── predict.py            # Inference script
├── features.py           # Feature extraction module
└── data/                # Audio dataset
    ├── Actor_01/
    ├── Actor_02/
    └── ... (24 actors total)
├── emotion_model.pkl     # Trained model (generated)
├── scaler.pkl           # Feature scaler (generated)
└── label_encoder.pkl    # Emotion encoder (generated)
```

---

## 📦 Installation

### Requirements
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
   ```bash
   cd speech-emotion-recognition
   ```

2. **Install dependencies**
   ```bash
   pip install librosa numpy scikit-learn soundfile matplotlib joblib
   ```

---

## 🚀 Quick Start

### Train the Model

```bash
python train.py
```

This will:
- Scan all `.wav` files in the `data/` directory
- Extract 180-dimensional feature vectors from each file
- Train a 2-layer neural network (128 → 64 neurons)
- Evaluate on test set and save artifacts

**Output:**
```
Found 1440 audio files. Extracting features...
Loaded 672 samples across emotions: {angry, happy, neutral, sad}
Training on 537 samples, testing on 135 samples.
...
✅ Accuracy: 77.04%
Model saved to emotion_model.pkl
```

### Make Predictions

```bash
python predict.py path/to/audio.wav
```

**Example output:**
```
🎤 File: data/Actor_01/03-01-01-01-01-01-01.wav
🎯 Predicted emotion: HAPPY

Confidence scores:
  happy       ████████████████      87.5%
  neutral     ███████                42.3%
  sad         ████                   18.9%
  angry       ██                     11.2%
```

---

## 📊 Model Performance

### Overall Accuracy
- **Test Accuracy: 77.04%**
- Training: 537 samples | Testing: 135 samples

### Per-Emotion Breakdown

| Emotion | Precision | Recall | F1-Score | Notes |
|---------|-----------|--------|----------|-------|
| 😠 Angry   | **91%** | 87% | 0.89 | Best recognized |
| 😊 Happy   | 72% | 66% | 0.69 | |
| 😐 Neutral | 73% | 76% | 0.74 | |
| 😢 Sad     | 67% | 74% | 0.70 | Most challenging |

---
## 📁 Dataset Information

Uses the **RAVDESS dataset** format:
- 24 professional actors
- 1,440 total audio files (60 per actor)
- 4 emotions: Neutral, Happy, Sad, Angry
- Clean studio recordings (~1-5 seconds each)

**Filename Pattern:** `XX-XX-emotion_code-XX-XX-XX-XX.wav`

| Code | Emotion |
|------|---------|
| 01   | Neutral |
| 03   | Happy   |
| 04   | Sad     |
| 05   | Angry   |

---

## 🔧 Tech Stack

- **Python 3.8+** — Programming language
- **librosa** — Audio feature extraction
- **scikit-learn** — Machine learning (MLPClassifier, preprocessing)
- **NumPy** — Numerical computations
- **joblib** — Model persistence
- **SoundFile** — Audio I/O

---

## 📝 Usage Notes

- **Audio Format**: WAV files (.wav)
- **Feature extraction**: ~2-3 seconds per audio file
- **Model training**: ~2-5 minutes on typical CPU
- **Inference**: ~10-50ms per prediction

---

## 📄 License

This project uses the RAVDESS dataset for educational purposes.

---

## 📚 References

- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [librosa Documentation](https://librosa.org/)
- [scikit-learn ML Workflows](https://scikit-learn.org/)

---

**Built with ❤️ for emotion recognition**
