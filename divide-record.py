import os
import soundfile as sf
import librosa

# Define the input file path
input_file = "//home/jawabreh/Desktop/voice-recognition/data/scan/Milena/training-one.wav"

# Define the output directory
output_directory = "/home/jawabreh/Desktop/sds"

# Define the length of each clip in seconds
clip_length = 10

# Load the audio file
y, sr = librosa.load(input_file, sr=None)

# Calculate the total number of samples in the audio file
total_samples = len(y)

# Calculate the number of samples in each clip
clip_samples = int(sr * clip_length)

# Calculate the number of clips that can be extracted from the audio file
total_clips = int(total_samples / clip_samples)

# Loop through each clip and save it to a separate file
for clip_index in range(total_clips):
    # Calculate the start and end samples for the current clip
    start_sample = clip_index * clip_samples
    end_sample = start_sample + clip_samples

    # Extract the current clip from the audio file
    clip = y[start_sample:end_sample]

    # Define the output file path for the current clip
    output_file = os.path.join(output_directory, f"clip_{clip_index + 1}.wav")

    # Save the current clip to a separate file using soundfile
    sf.write(output_file, clip, sr)
