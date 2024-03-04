# Music Embeddings Visualization with MAEST and t-SNE #

This repository explores the process of extracting music embeddings using the MAEST model and visualizing them using t-SNE (t-Distributed Stochastic Neighbor Embedding). 
Music embeddings are compact representations of audio signals, extracted from audio files utilizing the MAEST model, a convolution-free transformer architecture.

### Introduction ###

This project was completed during my internship at TMC2, where I explored the extraction of music embeddings using the MAEST model and their visualization with t-SNE (t-Distributed Stochastic Neighbor Embedding). 
Music embeddings provide a concise representation of audio signals, allowing us to understand the underlying patterns and similarities between different music tracks.

### Overview ###

Music embeddings provide a concise representation of audio signals, allowing us to understand the underlying patterns and similarities between different music tracks. 
In this project, we use the MAEST model to extract these embeddings from a dataset of music files. 
We then apply t-SNE for dimensionality reduction, creating 2D visualizations that reveal relationships between music tracks based on their similarity.

### Methodology ###
MAEST Model: The MAEST model is a transformer architecture designed for audio signal processing. It enables us to efficiently extract music embeddings, capturing the essence of each audio track in a compact form.

t-SNE Visualization: After obtaining the music embeddings, we utilize t-SNE for dimensionality reduction. This technique allows us to project the high-dimensional embeddings into a 2D space while preserving the pairwise similarities between them.

### Key Features ###

- Extraction of music embeddings using the MAEST model.
- Visualization of music embeddings using t-SNE.
- Plotting 2D embeddings with distinct colors representing different music genres or categories.
- Interactive visualization for exploration and analysis of individual data points.

## How to Use ##
### Data Preparation: ###

* Ensure your music dataset is in a compatible format (audio files, genres/categories).
* Update the file paths in data_preparation.py to point to your dataset.

### Extracting Music Embeddings:###

* Run embeddings_maest.py to extract music embeddings using the MAEST model.
* Adjust parameters such as batch size, embedding size, etc., as needed.
### Visualizing Embeddings with t-SNE:###

After extracting embeddings, run tsne_visualization.py to perform t-SNE dimensionality reduction and create visualizations.
Customize the visualization by specifying colors for different genres or categories in the dataset.
#### Requirements ####
Python 3.x
Libraries: TensorFlow, NumPy, matplotlib, scikit-learn
## Usage ##
### Clone this repository: ###

```
git clone https://Reby_magomere@bitbucket.org/tmc-2/music_learning.git
```
### Install dependencies: ###


```
pip install -r requirements.txt
```

### Prepare your data: ###

* Place your music dataset in the data/ directory.
* Update paths and settings according to your dataset.

### Extract Music Embeddings:###


```
python embeddings_maest.py
```

This script will extract music embeddings using the MAEST model.

### Visualize with t-SNE:###

```
python tsne_visualization.py
```

### Results ###
The output/ directory will contain:
* embeddings.csv: Extracted music embeddings.
* tsne_plot.png: t-SNE visualization plot.

### Visualization ###
The resulting 2D visualizations offer an insightful look into the music dataset:

Color-Coded Genres: Each data point on the t-SNE plot represents a music track. We use distinct colors to represent different music genres or categories, providing a clear visual separation of genres.

Interactive Exploration: The visualization is interactive, enabling users to explore and analyze individual data points. By hovering over a point, users can view metadata such as track name, artist, and genre.

![2D visualization with t-sne](master/plot.png)

### Conclusion ###
This repository offers a powerful framework for understanding and analyzing music data. By combining the capabilities of the MAEST model for extracting music embeddings and t-SNE for visualizing these embeddings, we gain valuable insights into the underlying structure of music tracks. Whether you're a music enthusiast, researcher, or data scientist, this project provides a versatile tool for exploring the rich world of music through advanced machine learning and visualization techniques.

### Acknowledgements ###
MAEST Model: [https://github.com/palonso/maest]

