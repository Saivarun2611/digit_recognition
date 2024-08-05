# Handwritten Digit Recognizer

This is a basic handwritten digit recognizer project that I created just for fun. The project uses the MNIST dataset to train a simple neural network to recognize handwritten digits (0-9).

## Dataset

The model is trained using the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits. Each image is 28x28 pixels.

## Model

The neural network model is built using TensorFlow and Keras. It consists of a single dense layer with 10 neurons and a sigmoid activation function.

## Code

Here's a snippet of the code used to create the model:

```python
import tensorflow as tf
from tensorflow import keras

# Creating Neural Network
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

# Optimizer allows us to reach global minima during backpropagation efficiently
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Assuming X_train_flat and y_train are already defined and preprocessed
model.fit(X_train_flat, y_train, epochs=5)
