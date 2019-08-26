# Flower Classification with PyTorch


### Installation
No additional libraries are needed to run the code beyond the Anaconda distribution of Python. There should be no issues running the code using Python versions 3.x

### Project Summary:
The goal of this project was to build and train an image classifier that is able to recognize different species of flowers. We relied on the so-called transfer learning approach - using pre-trained deep convolutional neural networks (e.g. VGG, DenseNet, AlexNet imported from _Pytorch torchvision_) as fixed feature extractors and combining them with custom fully-connected classifiers for solving the given task. Training and testing was done on the publicly available [dataset] (http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) with images of 102 flower categories (each class consists of between 40 and 256 images).

We implemented the model training and prediction as a generic command line application so it can be used on any set of labelled images, and can be easily adapted to other image recognition tasks.

### File Descriptions
- cat_to_name.json : mappings from category labels to category names
- classifier_util.py : utility functions for building a convolutional neural network with transfer learning
- predict.py : loads a trained neural network and use it to predict the k most likely classes for a given input image
- train.py : loads the input data, trains a new convolutional neural network and saves the model as a checkpoint
- util.py : utility functions for loading data and pre-processing images

### Instructions
1. To load the input data from a given directory and train a convolutional neural network run: `python train.py data_directory`. This will print out training loss, validation loss, and validation accuracy as the network trains. The following optional arguments can be set:
- save_dir : directory to save checkpoints
- arch : convolutional neural network architecture
- hidden_units : number of hidden units in each layer of the custom fully connected classifier
- learning_rate : learning rate used in the Adam optimizer
- epochs : number of training epochs
- gpu : flag to run on GPU
For example: `python train.py data_directory --save_dir save_directory --arch "vgg13" --epochs 20 --gpu`

2. To load the trained neural network and use it to predict the k most likely classes for a given input image run:
`python predict.py path/to/image checkpoint`. The following optional arguments can be set:
- top_k : number of most likely categories displayed
- category_names : mapping of categories to category names
- gpu : flag to run on GPU
For example: `python predict.py path/to/image checkpoint --top_k 5 --category_names cat_to_name.json --gpu`

### Results
The default `vgg11` model with default parameters (see `train.py`) achieved a solid 91.34% accuracy on the flower classification testing set (102 categories).

### Acknowledgments
We thank the Visual Geometry Group, Department of Engineering Science, University of Oxford for providing the dataset.
