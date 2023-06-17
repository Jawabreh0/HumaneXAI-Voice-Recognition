import numpy as np
import resemblyzer
import os

# Load the pre-trained model
encoder = resemblyzer.VoiceEncoder()

# Define directories
train_dir = "/home/jawabreh/Desktop/HumaneX_AI/voice-recognition/data/train"
npz_file = "/home/jawabreh/Desktop/HumaneX_AI/voice-recognition/voice-embeddings.npz"

# Initialize lists to hold embeddings and labels
embeddings = []
labels = []

# Iterate over subdirectories and wav files
for subdir in os.listdir(train_dir):
    subdir_path = os.path.join(train_dir, subdir)
    if os.path.isdir(subdir_path):
        for wav_file in os.listdir(subdir_path):
            wav_file_path = os.path.join(subdir_path, wav_file)
            label = subdir
            wav = resemblyzer.preprocess_wav(wav_file_path)
            emb = encoder.embed_utterance(wav)
            embeddings.append(emb)
            labels.append(label)

# Convert lists to numpy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Save embeddings and labels to npz file
try:
    np.savez(npz_file, embeddings=embeddings, labels=labels)
    print("\n\n\t\tEmbeddings saved successfully!\n\n")
except:
    print("\n\n\t\tError saving embeddings!\n\n")
