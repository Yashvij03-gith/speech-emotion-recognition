import os
import glob
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from features import extract_features

TARGET_EMOTIONS = {"03": "happy", "04": "sad", "05": "angry", "01": "neutral"}

def load_dataset(data_path="data/"):
    X, y = [], []

    files = glob.glob(os.path.join(data_path, "**/*.wav"), recursive=True)

    if not files:
        print("No .wav files found. Check your data/ folder.")
        return None, None

    print(f"Found {len(files)} audio files. Extracting features...")

    for file in files:
        filename = os.path.basename(file)
        parts = filename.split("-")

        if len(parts) < 3:
            continue

        emotion_code = parts[2]

        if emotion_code not in TARGET_EMOTIONS:
            continue

        label = TARGET_EMOTIONS[emotion_code]

        try:
            features = extract_features(file)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=str)  # <-- KEY FIX: force plain str


def train():
    X, y = load_dataset()

    if X is None or len(X) == 0:
        print("Dataset loading failed. Exiting.")
        return

    print(f"\nLoaded {len(X)} samples across emotions: {set(y)}")

    # Scale features (important for neural networks)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")

    # Encode labels as integers (fixes the np.str_ issue entirely)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)          # e.g. angry=0, happy=1, neutral=2, sad=3
    joblib.dump(le, "label_encoder.pkl")     # save so predict.py can decode back

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print(f"Emotion classes: {list(le.classes_)}\n")

    # Build and train
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
        early_stopping=False,
        verbose=True
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed report:")
    # Decode integers back to emotion names for the report
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(model, "emotion_model.pkl")
    print("\nModel saved to emotion_model.pkl")


if __name__ == "__main__":
    train()
