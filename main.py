import pygame
import numpy as np
from PIL import Image
import ffn

def draw_digit():
    pygame.init()
    window = pygame.display.set_mode((280, 280))  # Create a 280x280 window
    pygame.display.set_caption("Draw a Digit")
    clock = pygame.time.Clock()

    drawing = False
    window.fill((0, 0, 0))  # Black background

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:  # Press 'S' to save the image
                    pygame.image.save(window, "digit.png")
                    pygame.quit()
                    return "digit.png"

        if drawing:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            pygame.draw.circle(window, (255, 255, 255), (mouse_x, mouse_y), 15)  # White brush

        pygame.display.update()
        clock.tick(30)
        
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img)
    img_array = 255 - img_array  # Invert colors (white background)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array.flatten().reshape(1, -1)  # Flatten and reshape for FFN

def predict_digit(image_path, W1, b1, W2, b2, W3, b3):
    X = preprocess_image(image_path)
    _, _, _, _, _, A3 = ffn.forward_propagation(X, W1, b1, W2, b2, W3, b3)
    prediction = ffn.get_predictions(A3)
    return prediction[0]

W1 = np.load("parameters/W1.npy")
b1 = np.load("parameters/b1.npy")
W2 = np.load("parameters/W2.npy")
b2 = np.load("parameters/b2.npy")
W3 = np.load("parameters/W3.npy")
b3 = np.load("parameters/b3.npy")

# Step 1: Draw a digit
image_path = draw_digit()
if image_path:
    # Step 2: Predict the digit
    prediction = predict_digit(image_path, W1, b1, W2, b2, W3, b3)
    print(f"Predicted Digit: {prediction}")
