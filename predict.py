import joblib
import sys
import numpy as np
from features import extract_features

def predict_emotion(audio_file):
    model = joblib.load("emotion_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")

    features = extract_features(audio_file)
    features = scaler.transform(features.reshape(1, -1))

    pred_encoded = model.predict(features)[0]
    emotion = le.inverse_transform([pred_encoded])[0]

    probs = model.predict_proba(features)[0]

    print(f"\n🎤 File: {audio_file}")
    print(f"🎯 Predicted emotion: {emotion.upper()}")
    print("\nConfidence scores:")
    for cls, prob in sorted(zip(le.classes_, probs), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"  {cls:<10} {bar:<20} {prob*100:.1f}%")

    return emotion

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/audio.wav")
    else:
        predict_emotion(sys.argv[1])