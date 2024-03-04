import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import librosa
import librosa.util
import audioread


# Function to load and preprocess audio files
def load_and_process_audio(file_path, target_sr=15000):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    except audioread.exceptions.NoBackendError:
        print(f"No audio backend available to read file: {file_path}")
        return None, None
    
    # Normalize audio
    if audio is not None:
        audio = librosa.util.normalize(audio)
    
    return audio, sr


# Function to extract PaSST embeddings from audio
def extract_passt_embeddings(audio, passt_model):
    # Convert audio to PyTorch tensor
    audio_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Forward pass through the model
    with torch.no_grad():
        embeddings = passt_model(audio_tensor)
    
    return embeddings


def main(args):
    # Initialize the PaSST model
    passt_model = nn.Sequential(
        nn.Conv1d(1, 1, kernel_size=3, padding=1),  # Kernel size 3, padding 1
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Conv1d(1, 1, kernel_size=3, padding=1),  # Kernel size 3, padding 1
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2)
    )

    # Traverse through each audio file
    for root, _, files in os.walk(args.folder_path):
        for file_name in files:
            # Check if file is an audio file (you can modify this check as needed)
            if file_name.endswith(('.wav', '.mp3', '.ogg')):
                file_path = os.path.join(root, file_name)
                print("Processing:", file_path)
                
                # Load and preprocess audio
                audio, _ = load_and_process_audio(file_path)
                
                # Handle case where audio load failed
                if audio is None:
                    print(f"Skipping file: {file_path}")
                    continue
                
                # Extract PaSST embeddings
                embeddings = extract_passt_embeddings(audio, passt_model)
                
                # Saving the embeddings
                output_name = os.path.splitext(file_name)[0] + "_passt.npy"
                output_file = os.path.join(args.output_folder, output_name)
                np.save(output_file, embeddings.numpy())

    print("PaSST embeddings extracted and saved successfully in folder:", args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PaSST embeddings from audio files")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    parser.add_argument("output_folder", type=str, help="Name of the folder to save PaSST embeddings")
    args = parser.parse_args()

    main(args)
