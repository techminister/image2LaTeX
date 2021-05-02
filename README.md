# LaTeX Transcription

Project to transcribe a picture or screenshot of a LaTeX equation into its source code.

### 0. Data Generation and Augmentation

#### Generation

The `generate.py` module creates a base dataset of 405 unique grayscale 120x120 LaTeX symbols. It requires LaTeX to be installed in the system. The dataset is located in the `data` folder, and the labels are in the `meta.csv` file.

#### Dataset for Complex Equation Classifier 

The dataset used in the complex equation classifier can be found in the link below:

 https://zenodo.org/record/56198#.YI5G3BQzZhE

#### Augmentation

The model is trained with randomized augmented data. The augmentation transformations are specified in `transforms.py`. The default training augmentation consists of the following transformations (all of which are randomly applied with a probability of 0.5):

* Rotation
* Scaling
* Translation
* Brightness
* Gaussian Noise

### 1. Single Character Classification

#### Basic Convolutional Model

The base model, defined as `SingleCharacterClassifier` in `model.py` and saved as `scc.model`, consists of three convolutional layers, one max pooling layer, and three fully connected layers.

Following the development of the base model, additional features were explored to identify if they led to a increase in the model's performance. The files relating to the models will be written in the following format (saved model, training plot) and can be found in the `part_1-2` folder There are as follows: 

1. Addition of batch norm and ReLU between convolution layers (`custom_bn.pt,custom_bn.png`)
2. Addition of an extra convolution layer (`custom_excon.pt, custom_excon.png`)
3. Addition of dropout (`custom_drop.pt, custom_drop.png`)

The best performing version was that of addition of an extra convolution layer that has an accuracy of 98.27% and was trained for 20 epochs, with each epoch consisting of 100 augmented replications of the dataset. In practice, the model performs well on screenshots, but very poorly in photographs.

### 2. Basic Equation Classification 

The next iteration was to test the model performance on short equations. The equations were segmented using pre-defined criterion into individual characters before being passed into the model. A comparison was made between the best performing model from Single Character Classification and other pre-trained models. The files relating to the models will be written in the following format (saved model, training plot) and can be found in the `part_1-2` folder There are as follows: 

1. Best performing custom model (`custom_excon.pt, custom_excon.png`)
2. Resnet-18 pre-trained model (`resent.pt, resent.png`)
3. VGG-16 pre-trained model (`vgg_16.pt, vgg_16.png`) 
   1. The model for VGG has not been uploaded to this repository due to its large file size, do drop an email to suhas_sahu@mymail.sutd.edu.sg if the file is required.
4. Densenet121 pre-trained model (`densenet.pt, densenet.png`)

The best performing model was that of the custom model, which followed the same architecture as the model used in Single Character Classification, achieving an accuracy of 90.67%. Each of the models were trained for 20 epochs. The performance was measured for equations that were screenshots, but there were problems found when fractions or subscripts were present in the equations. 

### 3. Complex Equation Classification

The files for the complex equation classifier can be found in part_3 folder. The models used for the task can be found in `models.py` file and the saved model that has been run for 60 epochs can be found in `epochs60.pt`. 

The complex equation classifier was done using a standard encoder-decoder model, in which the features of an image are encoded by a number of convolutional layers, and run through a recurrent neural network (RNN) that further processes the feature map. Initial model implementations could only train with one sample at a time, and combined with the large size of the dataset, required 50 hours to train on a single epoch.

To make the implementation of this model feasible, we reduced the complexity of the model, and also restricted training to only include images that were associated with LaTeX equations of less than 30 tokens. This shortened the training time per epoch to 90 minutes. 

The model achieved a final training loss of 0.7596, with a best validation loss of 1.1708. 

### 4. Reproducibility 

1. A jupyter notebook, `CV Complied.ipynb` has been made to facilitate the reproducibility of the project. The file can be found under the directory, `image2LaTeX/part_1-2/CV Compiled.ipynb`
   1. The second block of code has to be changed to the present working directory prior to running the notebook. A comment has been made in the notebook as well
2. For the complex equation classification the `predict.py` file can be run. It is found in the following directory, `image2LaTeX/part_3/predict.py

