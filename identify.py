import numpy as np
import joblib
from resemblyzer import preprocess_wav, VoiceEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# load the embeddings and labels
data = np.load("/home/jawabreh/Desktop/voice-recognition/voice_embeddings.npz")
embeddings = data["embeddings"]
labels = data["labels"]

# train the random forest classifier
scaler = StandardScaler()
X_train = scaler.fit_transform(embeddings)
y_train = labels
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# save the classifier and scaler to files
joblib.dump(clf, "random_forest.joblib")
joblib.dump(scaler, "scaler.joblib")

# load the classifier and scaler from files
clf = joblib.load("random_forest.joblib")
scaler = joblib.load("scaler.joblib")

# load the voice encoder model
encoder = VoiceEncoder()

# define a function to predict the identity of a voice recording
def predict_identity(audio_path, threshold=0.8):
    wav = preprocess_wav(audio_path)
    embedding = encoder.embed_utterance(wav)
    embedding_scaled = scaler.transform(embedding.reshape(1, -1))
    prediction = clf.predict_proba(embedding_scaled)
    identity = clf.classes_[np.argmax(prediction)]
    confidence = np.max(prediction)
    if confidence >= threshold:
        return identity, confidence
    else:
        return "Unknown", confidence

# usage example
audio_path = "/home/jawabreh/Desktop/voice-recognition/data/val/Unknown/2.flac"
identity, confidence = predict_identity(audio_path)
print("Predicted identity:", identity)
print("Confidence:", confidence)
