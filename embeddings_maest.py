import os
import argparse
import librosa
import ssl
import numpy as np
from models.maest import maest
import time
ssl._create_default_https_context = ssl._create_unverified_context


def process_audio(audio_path, sr=16000, duration=30):
        # Load audio file
        data, _ = librosa.load(audio_path, sr=sr, mono=True)

        # Trim silence from the beginning and end
        trimmed_data, _ = librosa.effects.trim(data)

        # Extract the central 30 seconds
        if len(trimmed_data) >= sr * duration:
            start = (len(trimmed_data) - sr * duration) // 2
            end = start + sr * duration
            trimmed_data = trimmed_data[start:end]

        return trimmed_data


def extract_embeddings(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.mp3', '.wav')):
            t_s = time.time()
            audio_path = os.path.join(input_folder, file_name)
            model = maest(arch="discogs-maest-30s-pw-129e")
            data = process_audio(audio_path)
            _, embeddings = model(data)
            embeddings = embeddings.detach().numpy()

            if embeddings.shape[0] == 0:
                print(f"Embeddings for {file_name} is empty. Skipping.")
            else:
                print(f"Embedding shape for {file_name}: {embeddings.shape}")
                print(f"First 5 embedding values for {file_name}:\n{embeddings[0, :5]}")

            # Save embeddings to a file
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_embeddings.npy")
            np.save(output_file_path, embeddings)
            del model
            print(f"Embeddings saved for {file_name}.")
            print("time elapsed: ", time.time() - t_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings from audio files.")
    parser.add_argument("input_folder", help="Path to the folder containing audio files.")
    parser.add_argument("output_folder", help="Path to the folder for saving embeddings.")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the vision transformer.")
    args = parser.parse_args()

    extract_embeddings(args.input_folder, args.output_folder)