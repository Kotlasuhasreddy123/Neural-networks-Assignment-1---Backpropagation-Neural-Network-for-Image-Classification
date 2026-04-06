# Backpropagation Neural Network for Image Classification

Multi-Layer Perceptron (MLP) from Scratch — Assignment #1
Lewis University | MS in Artificial Intelligence | Suhas Reddy Kotla (L30122169)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)
![Assignment](https://img.shields.io/badge/Assignment-MLP%20from%20Scratch-7C3AED?style=flat-square)

---

## What This Project Does

Can a neural network learn to read the surface of Jupiter's moon?

This project implements a Multi-Layer Perceptron from scratch using only Python and NumPy — no PyTorch, no TensorFlow. The network is trained via backpropagation to perform pixel-wise color classification on a NASA image of Io, distinguishing between four surface features based purely on RGB values.

**Four classes:**
- `0` — Background (black space)
- `1` — General surface
- `2` — Volcanic craters
- `3` — Bright spots

---

## Architecture
Input (3 RGB values) → Hidden Layer (Sigmoid) → Output Layer (Softmax) [3] [N] [4 classes]

| Component | Detail |
|-----------|--------|
| Type | Multi-Layer Perceptron (MLP) |
| Hidden activation | Sigmoid |
| Output activation | Softmax |
| Loss function | Cross-entropy |
| Optimizer | Backpropagation + SGD |
| Training epochs | 5,000 |

---

## How It Works

**Forward Pass** — data flows from input RGB pixels through the hidden layer to generate class probabilities.

**Backward Pass** — the chain rule propagates error back through the network, updating weights at each layer based on the learning rate.


# Forward pass
z1 = X @ W1 + b1
a1 = sigmoid(z1)          # hidden layer activation
z2 = a1 @ W2 + b2
a2 = softmax(z2)          # output probabilities

# Backward pass
dz2 = a2 - y_onehot                        # output error
dW2 = a1.T @ dz2 / m                       # output weight gradient
dz1 = (dz2 @ W2.T) * sigmoid_grad(z1)     # hidden error (chain rule)
dW1 = X.T @ dz1 / m                        # hidden weight gradient

Dataset
Image: Jupiter's moon Io — NASA/JPL
(Source: Digital Image Processing, 3rd ed., Gonzalez & Woods)

Preprocessing:

RGB pixel values normalized to [0, 1]
Specific coordinate points hand-selected to represent each of the 4 classes
Training points cover background, surface, volcanic craters, and bright spots
Results
Metric	Value
Initial loss	~1.38
Final loss (5,000 epochs)	Significantly reduced
Background classification	High precision
Key challenge	Volcanic craters misclassified as background — both have very low-intensity dark pixels, showing the limitation of color-only classification without spatial context
Loss is printed every 100 epochs. Final output is a full Classification Map overlaid on the original image.

Getting Started
Prerequisites

Python 3.x
NumPy
OpenCV (cv2)
Matplotlib
Install

pip install numpy opencv-python matplotlib
Run

# Place the image file in your working directory:
# Fig0628(a)(jupiter-moon.-Io).tif

# Then run the notebook or script
python mlp_classifier.py
Key Takeaway
Color-based classification without spatial context has clear limits — dark volcanic craters and black background share similar RGB values, causing misclassification. This motivates the use of convolutional architectures (CNNs) that capture spatial relationships, not just pixel color.

Author
Suhas Reddy Kotla
MS in Artificial Intelligence — Lewis University (GPA: 4.0)
Graduate AI Research Assistant
