import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import subprocess
import matplotlib.cm as cm
import mplcursors
from pydub import AudioSegment
import simpleaudio as sa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


def play_audio(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path, format="wav")
    play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
    play_obj.wait_done()

def on_click(event):
    if event.ind is not None and len(event.ind) > 0:
        index = event.ind[0]
        selected_label = labels[index]
        selected_folder_path = os.path.join(main_folder_path, selected_label)

        audio_file_path = os.path.join(selected_folder_path, f"{selected_label}.wav")

        if os.path.exists(audio_file_path):
            play_audio(audio_file_path)
        else:
            print(f"Audio file not found for {selected_label}")

main_folder_path = input("Enter the path to the main folder containing dataset folders: ")
output_folder = input("Enter the path to the output folder for saving PNG files: ")

if not os.path.exists(main_folder_path):
    print("Error: The specified main folder path does not exist.")
    exit()

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize lists to store data and labels
all_embeddings = []
labels = []

# Load all embeddings from dataset folders
for dataset_folder in os.listdir(main_folder_path):
    dataset_folder_path = os.path.join(main_folder_path, dataset_folder)
    if os.path.isdir(dataset_folder_path):
        # Load all embeddings in the dataset folder
        embeddings_list = []
        for filename in os.listdir(dataset_folder_path):
            if filename.endswith('.npy'):
                embedding = np.load(os.path.join(dataset_folder_path, filename))
                embeddings_list.append(embedding)

        if not embeddings_list:
            print(f"Warning: No embedding files found in the dataset folder {dataset_folder}. Skipping.")
            continue

        # Concatenate embeddings and add labels
        dataset_embeddings = np.vstack(embeddings_list)
        all_embeddings.append(dataset_embeddings)
        labels.extend([dataset_folder] * dataset_embeddings.shape[0])

# Convert string labels to numeric values
label_to_numeric = {label: i for i, label in enumerate(set(labels))}
numeric_labels = [label_to_numeric[label] for label in labels]

# Concatenate all data and apply t-SNE
all_embeddings = np.vstack(all_embeddings)
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(all_embeddings)

# Plot the 2D embeddings with different colors for each dataset folder
plt.figure(figsize=(10, 8))
for dataset_folder in set(labels):
    indices = np.array([i for i, label in enumerate(labels) if label == dataset_folder])
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], alpha=0.5, label=dataset_folder)

# Enable cursor to display labels when hovering over points
mplcursors.cursor(hover=True)

# Connect the click event to the callback function
plt.gcf().canvas.mpl_connect('pick_event', on_click)
plt.legend()

plt.title('t-SNE Visualization of Embeddings for All Music Genres')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# Save the plot as a PNG file in the specified output folder
output_filename = os.path.join(output_folder, 'tsne_visualization_all_datasets.png')
plt.savefig(output_filename)
print(f"Plot saved as: {output_filename}")

plt.show()