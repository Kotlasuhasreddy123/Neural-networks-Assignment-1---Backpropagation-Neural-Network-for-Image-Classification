Backpropagation Neural Network for Image Classification
Assignment #1: Multi-Layer Perceptron (MLP) from Scratch
Author: Suhas Reddy Kotla (L30122169)
Project Overview
This project implements a Multi-Layer Perceptron (MLP) neural network from scratch using Python and NumPy. The network is trained using the backpropagation algorithm to perform pixel-wise color classification on an image of Jupiter’s moon, Io. The goal is to distinguish between different surface features such as background, general surface, volcanic craters, and bright spots based on RGB values.
Technical Features
Architecture: Multi-Layer Perceptron (MLP) with one hidden layer.
Activation Functions:
Sigmoid: Used for the hidden layer to map inputs to a probability-like range.
Softmax: Applied to the output layer for multi-class classification.
Optimization: standard two-phase cycle consisting of:
Forward Pass: Data flows from input pixels through hidden layers to generate predictions.
Backward Pass: Uses the chain rule to propagate error and update weights based on a set learning rate.
Loss Function: Cross-entropy loss.
Dataset and Preprocessing
The model uses an image of Jupiter's moon, Io (Source: NASA/JPL via Gonzalez and Woods).
Training Points: Specific coordinate points were selected to represent four classes: Background, Surface, Volcano, and Bright spots.
Normalization: Pixel RGB values are normalized to a range of $[0, 1]$ before being fed into the network.
Getting Started
Prerequisites
Python 3.x
NumPy
OpenCV (cv2)
Matplotlib
Installation
Bash
pip install numpy opencv-python matplotlib

Usage
Ensure the image file Fig0628(a)(jupiter-moon.-Io).tif is in your working directory.
Run the notebook or script to initialize and train the network.
The model will output loss values every 100 epochs and display a final Classification Map.
Results and Analysis
Performance: The model effectively minimizes loss over time (starting around 1.38 and dropping significantly over 5,000 epochs).
Observations: The network identifies the black background with high precision due to its numerical distinctness.
Challenges: Some volcanic craters were misclassified as background because both consist of very low-intensity dark pixels, showing the limitation of color classification without spatial context.
Resource Citation
Image Source: Jupiter Moon Io, NASA/JPL. From Digital Image Processing, 3rd ed, by Gonzalez and Woods.
