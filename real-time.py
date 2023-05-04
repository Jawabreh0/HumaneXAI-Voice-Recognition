import numpy as np
import joblib
import pyaudio
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
def predict_identity(embedding, threshold=0.8):
    embedding_scaled = scaler.transform(embedding.reshape(1, -1))
    prediction = clf.predict_proba(embedding_scaled)
    identity = clf.classes_[np.argmax(prediction)]
    confidence = np.max(prediction)
    if confidence >= threshold:
        return identity, confidence
    else:
        return "Unknown", confidence

# define a function to record audio from the microphone
def record_audio():
    chunk = 1024
    sample_format = pyaudio.paFloat32
    channels = 1
    fs = 16000
    seconds = 5

    p = pyaudio.PyAudio()

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []

    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio = np.frombuffer(b"".join(frames), dtype=np.float32)
    return audio

# define a function to predict the identity of the speaker in real-time
def predict_identity_realtime():
    # start recording audio
    print("Recording started")
    audio = record_audio()

    # extract voice embedding
    wav = preprocess_wav(audio)
    embedding = encoder.embed_utterance(wav)

    # predict speaker identity
    identity, confidence = predict_identity(embedding)

    # print results
    print(f"Recording success with confidence rate: {confidence:.2f}")
    print(f"Predicted identity: {identity}")

# usage example
predict_identity_realtime()
