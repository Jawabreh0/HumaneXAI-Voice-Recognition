import numpy as np
import resemblyzer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model
encoder = resemblyzer.VoiceEncoder()

# Encode speech into embeddings
wav_file_1 = "/home/jawabreh/Desktop/voice-recognition/data/val/Milena/milena-7.wav"
wav_file_2 = "/home/jawabreh/Desktop/voice-recognition/data/val/Ravilya/ravilya-7.wav"
emb_1 = encoder.embed_utterance(resemblyzer.preprocess_wav(wav_file_1))
emb_2 = encoder.embed_utterance(resemblyzer.preprocess_wav(wav_file_2))

# Compare embeddings for speaker verification
similarity = cosine_similarity(emb_1.reshape(1, -1), emb_2.reshape(1, -1))[0][0]

# Determine confidence level based on similarity
if similarity >= 0.85:
    print(f"\n\n\t\tSame person with high confidence: {similarity}\n\n")
elif similarity >= 0.78:
    print(f"\n\n\t\tSame person with low confidence: {similarity}\n\n")
else:
    print("\n\n\t\tTwo different voiceprints\n\n")
    
    
    
    
    
