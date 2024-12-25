#training the FFN

import ffn
import pandas as pd
import matplotlib.pyplot as plt

#images are 28x28 grayscale with corresponding labels (0-9), where the label is in the first column
train_file = "datasets/mnist_train.csv"

train_data = pd.read_csv(train_file)

x_train = train_data.iloc[:,1:].values
y_train = train_data.iloc[:,0].values
x_train = x_train/255.

print(f"X_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

# example_digit = x_train[0].reshape(28, 28)
# plt.imshow(example_digit, cmap='gray')
# plt.title(f"Label: {y_train[0]}")
# plt.show()

layer1 = 128  # Number of neurons in the hidden layer
layer2 = 128

# Train the model
#ffn.train(x_train, y_train, layer1, layer2, 0.01, 500)
ffn.continue_train(x_train,y_train,layer1,layer2, 0.01, 500)

