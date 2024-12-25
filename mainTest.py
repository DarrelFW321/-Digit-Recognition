#running the ffn

import ffn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_file = "datasets/mnist_test.csv"

test_data = pd.read_csv(test_file)

x_test = test_data.iloc[:,1:].values
y_test = test_data.iloc[:,0].values

x_test = x_test/255.

W1 = np.load('parameters/W1.npy')
b1 = np.load('parameters/b1.npy')
W2 = np.load('parameters/W2.npy')
b2 = np.load('parameters/b2.npy')
W3 = np.load('parameters/W3.npy')
b3 = np.load('parameters/b3.npy')

Z1, A1, Z2, A2,Z3,A3 = ffn.forward_propagation(x_test, W1, b1, W2,b2, W3,b3)
print("Shape of A3:", A3.shape)
predictions = ffn.get_predictions(A3)
print(ffn.get_accuracy(predictions, y_test))

# random_index = np.random.randint(0, x_test.shape[0])

# sample_image = x_test[random_index].reshape(28, 28)  
# true_label = y_test[random_index]
# predicted_label = predictions[random_index]

# plt.imshow(sample_image, cmap='gray')
# plt.title(f"True: {true_label}, Predicted: {predicted_label}")
# plt.show()

# for i, label in enumerate (y_test):
#     if predictions[i] != label:
#         image = x_test[i].reshape(28,28)
#         plt.imshow(image, cmap = 'gray')
#         plt.title(f"True: {label}, Predicted: {predictions[i]}")
#         plt.show()

# misclassified_indices = np.where(predictions != y_test)[0]

# # Display the first 10 misclassified images
# for i, idx in enumerate(misclassified_indices[:10]):  # Limit to first 10
#     image = x_test[idx].reshape(28, 28)  # Reshape to 28x28
#     true_label = y_test[idx]  # True label
#     predicted_label = predictions[idx]  # Model's prediction
    
#     plt.figure()
#     plt.imshow(image, cmap='gray')
#     plt.title(f"True: {true_label}, Predicted: {predicted_label}")
#     plt.axis('off')  # Remove axes for clarity
#     plt.show()