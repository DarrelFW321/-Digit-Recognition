# Feedforward Neural Network for Handwritten Digit Recognition

This project implements a Feedforward Neural Network (FFN) with two hidden layers to classify handwritten digits from the MNIST dataset. The model achieves a training accuracy of approximately **94%**, making it a reliable tool for digit recognition.

---

## Features
- **Two-layer FFN**: Custom-built using NumPy, demonstrating fundamental neural network principles.
- **Softmax activation**: For output layer, to handle multi-class classification.
- **Training and testing**: Trained on MNIST data with high accuracy.
- **Custom digit input**: Draw your own digit and test it with the model.

---

## Dataset
The project uses the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), a benchmark dataset for handwritten digit recognition. The dataset contains:
- **Training data**: 60,000 images.
- **Testing data**: 10,000 images.

Each image is a grayscale 28x28 pixel representation of a single digit (0-9).

---

## How It Works
1. **Preprocessing**: The input images are normalized to values between 0 and 1 for faster convergence.
2. **Neural Network Architecture**:
    - **Input layer**: 784 neurons (28x28 pixels).
    - **Hidden layer 1**: Configurable (default: 128 neurons).
    - **Hidden layer 2**: Configurable (default: 128 neurons).
    - **Output layer**: 10 neurons (one for each digit).
3. **Activation Functions**:
    - ReLU for hidden layers.
    - Softmax for the output layer.
4. **Training**:
    - Gradient descent with backpropagation.
    - Cross-entropy loss function.
5. **Testing**:
    - Test set accuracy reported.
    - Visualizations of correctly and incorrectly classified digits.
6. **Custom Input**:
    - Draw your own digit using a graphical tool, save it as an image, and test it.

