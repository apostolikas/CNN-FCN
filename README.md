# CNN - FCN

This repository contains the code for my Master's thesis at the Democritus University of Thrace. The title of the thesis is "Development of a hybrid classifier using Convolutional Neural Networks and Fuzzy Cognitive Networks".  As a convolutional neural network, I will use the LeNet-5 architecture and its aim is to extract features from the input images. The classification is done by the Fuzzy Cognitive Network. In this way, the last layer of the LeNet-5 is removed and replaced by the FCN. 

## Main goal

The aim of this project is to show the potential of the FCNs in the field of computer vision and pattern recognition. For this reason, I am using the MNIST digit dataset and compare the results between four different architectures. For more complicated datasets, the LeNet-5 can be replaced with a deeper architecture (e.g. ResNet). 

## Content

In the folder models, you can find all the different architectures used.
In order to run the experiments, you can run "/scripts/train_mnist_cnn.py"
