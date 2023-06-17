import numpy as np
import resemblyzer
from sklearn.svm import SVC
from openpyxl import Workbook
import os 
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
threshold = 0.9

# Predict the identity of the input voice recording using the trained SVM classifier
def predict_identity(input_embeddings):
    predicted_identity = svm.predict(input_embeddings.reshape(1, -1))
    confidence_scores = svm.predict_proba(input_embeddings.reshape(1, -1))
    max_confidence_score = np.max(confidence_scores)
    
    if max_confidence_score < threshold:
        predicted_identity = "Unknown"
    else:
        predicted_identity = predicted_identity[0]  # Extract the string value
    
    return predicted_identity, max_confidence_score

# Provide the directory path containing the input voice recordings
directory_path = "/home/jawabreh/Desktop/HumaneX_AI/voice-recognition/data/test/793434506/"

# Create an Excel workbook and sheet
workbook = Workbook()
sheet = workbook.active

# Iterate over the voice recordings in the directory
for file_name in os.listdir(directory_path):
    if file_name.endswith(".wav"):
        # Construct the full file path
        input_file_path = os.path.join(directory_path, file_name)
        
        # Process the input voice recording
        input_embeddings = process_voice_recording(input_file_path)
        
        # Predict the identity and confidence score of the input voice recording
        predicted_identity, confidence_scores = predict_identity(input_embeddings)
        max_confidence_score = np.max(confidence_scores)
        
        # Write the results to the Excel sheet
        sheet.append([file_name, predicted_identity, max_confidence_score.item()])
        # Convert max_confidence_score to a scalar value using .item() method
        
# Save the Excel workbook
output_file_path = "793434506.xlsx"
workbook.save(output_file_path)

# Print completion message
print("Results saved to:", output_file_path)
