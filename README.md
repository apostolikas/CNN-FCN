# Development of a hybrid classifier using Convolutional Neural Networks and Fuzzy Cognitive Networks

## Contents 

This repository contains the code for my Master's thesis at the Democritus University of Thrace. The title of the thesis is "Development of a hybrid classifier using Convolutional Neural Networks and Fuzzy Cognitive Networks".  As a convolutional neural network, I will use the LeNet-5 architecture and its aim is to extract features from the input images. The classification is done by the Fuzzy Cognitive Network. In this way, the last layer of the LeNet-5 is removed and replaced by the FCN. 


In order to run the experiment, you can run "train_mnist_lenet.py" which will write the output of the second to last layer in a csv file. Then you can run "FCN-classifier.py" which will take as input these csv files and perform the classification.

## Main goal

The aim of this project is to show the potential of the FCNs in the field of computer vision and pattern recognition. For this reason, I am using the MNIST digit dataset and compare the results between four different architectures. For more complicated datasets, the LeNet-5 can be replaced with a deeper architecture (e.g. ResNet). 


