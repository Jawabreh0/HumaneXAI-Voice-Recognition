import os
import csv
import numpy as np
import joblib
from resemblyzer import preprocess_wav, VoiceEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Define the directories and files
parent_directory = "/home/jawabreh/Desktop/voice-recognition/data/val"
subdirectories = ["Milena", "Ravilya", "Unknown"]
output_file = "/home/jawabreh/Desktop/CrossValidation2.csv"

# Load the embeddings and labels
data = np.load("/home/jawabreh/Desktop/voice-recognition/voice-embeddings.npz")
embeddings = data["embeddings"]
labels = data["labels"]

# Train the random forest classifier
scaler = StandardScaler()
X_train = scaler.fit_transform(embeddings)
y_train = labels
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the classifier and scaler to files
joblib.dump(clf, "random_forest.joblib")
joblib.dump(scaler, "scaler.joblib")

# Load the classifier and scaler from files
clf = joblib.load("random_forest.joblib")
scaler = joblib.load("scaler.joblib")

# Load the voice encoder model
encoder = VoiceEncoder()

# Define a function to predict the identity of a voice recording
def predict_identity(embedding, threshold=0.63):
    embedding_scaled = scaler.transform(embedding.reshape(1, -1))
    prediction = clf.predict_proba(embedding_scaled)
    identity = clf.classes_[np.argmax(prediction)]
    confidence = np.max(prediction)
    if confidence >= threshold:
        return identity, confidence
    else:
        return "Unknown", confidence

# Open the output file for writing
with open(output_file, "w") as f:
    writer = csv.writer(f)

    # Write the header row
    writer.writerow(["File Name", "Ground Truth", "Predicted Identity", "Confidence"])

    # Loop through each subdirectory
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(parent_directory, subdirectory)

        # Loop through each file in the subdirectory
        for file_name in os.listdir(subdirectory_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(subdirectory_path, file_name)

                try:
                    # Load the audio and extract the voice embedding
                    audio = preprocess_wav(file_path)
                    embedding = encoder.embed_utterance(audio)

                    # Predict the speaker identity and confidence
                    predicted_identity, confidence = predict_identity(embedding)

                    # Write the results to the output file
                    writer.writerow([file_name, subdirectory, predicted_identity, confidence])
                except:
                    # Skip the file if it cannot be processed
                    pass
