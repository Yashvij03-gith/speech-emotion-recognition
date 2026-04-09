import librosa
import numpy as np

def extract_features(file_path):
    """
    Extract audio features from a .wav file.
    Returns a 1D numpy array of 180 features.
    """
    # Load audio file (sr=None keeps original sample rate)
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    # 1. MFCCs — captures timbre (most important feature for emotion)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)          # shape: (40,)

    # 2. Chroma — captures pitch class (musical tone)
    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    chroma_mean = np.mean(chroma.T, axis=0)         # shape: (12,)

    # 3. Mel spectrogram — frequency energy over time
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_mean = np.mean(mel.T, axis=0)               # shape: (128,)

    # Combine all features into one vector
    combined = np.hstack([mfccs_mean, chroma_mean, mel_mean])  # shape: (180,)
    return combined
    