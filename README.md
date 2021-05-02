# LaTeX Transcription

Project to transcribe a picture or screenshot of a LaTeX equation into its source code.

### 0. Data Generation and Augmentation

#### Generation

The `generate.py` module creates a base dataset of 405 unique grayscale 120x120 LaTeX symbols. It requires LaTeX to be installed in the system. The dataset is located in the `data` folder, and the labels are in the `meta.csv` file.

#### Augmentation

The model is trained with randomized augmented data. The augmentation transformations are specified in `transforms.py`. The default training augmentation consists of the following transformations (all of which are randomly applied with a probability of 0.5):

* Rotation
* Scaling
* Translation
* Brightness
* Gaussian Noise
* Gaussian Blur

### 1. Single Character Classification

#### Basic Convolutional Model

This model, defined as `SingleCharacterClassifier` in `model.py` and saved as `scc.model`, consists of three convolutional layers, one max pooling layer, and three fully connected layers.

The current version has an accuracy of 98.32% and was trained for 30 epochs, with each epoch consisting of 100 augmented replications of the dataset. In practice, the model performs well on screenshots, but very poorly in photographs.

### 2. Basic Equation Classification 

