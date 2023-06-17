import numpy as np
import resemblyzer
from sklearn.svm import SVC

# Load the pre-trained model
encoder = resemblyzer.VoiceEncoder()

# Define the npz file path
npz_file = "/home/jawabreh/Desktop/HumaneX_AI/voice-recognition/voice-embeddings.npz"

# Load the training set embeddings and labels from the npz file
data = np.load(npz_file)
X_train = data['embeddings']
y_train = data['labels']

# Process the input voice recording and extract its embeddings
def process_voice_recording(file_path):
    wav = resemblyzer.preprocess_wav(file_path)
    emb = encoder.embed_utterance(wav)
    return emb

# Train an SVM classifier on the training set embeddings and labels
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Set the threshold for the confidence score
threshold = 0.8

# Predict the identity of the input voice recording using the trained SVM classifier
def predict_identity(input_embeddings):
    predicted_identity = svm.predict(input_embeddings.reshape(1, -1))
    confidence_scores = svm.predict_proba(input_embeddings.reshape(1, -1))
    max_confidence_score = np.max(confidence_scores)
    
    if max_confidence_score < threshold:
        predicted_identity = "Unknown"
    
    return predicted_identity, max_confidence_score

# Provide the path of the input voice recording file
input_file_path = "/home/jawabreh/Desktop/HumaneX_AI/voice-recognition/data/test/998800274/3.wav"

# Process the input voice recording
input_embeddings = process_voice_recording(input_file_path)

# Predict the identity and confidence score of the input voice recording
predicted_identity, confidence_score = predict_identity(input_embeddings)

# Check if the predicted identity is "Unknown"
if predicted_identity == "Unknown":
    print("The identity is unknown.")
else:
    print("Predicted Identity:", predicted_identity)
    print("Confidence Score:", confidence_score)
