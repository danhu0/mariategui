# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import MeanSquaredError


# Load the saved model
model = tf.keras.models.load_model('style_transfer_model.h5', custom_objects={'mse': MeanSquaredError()})

# Define image dimensions (should match those used during training)
IMG_HEIGHT = 512
IMG_WIDTH = 386

def load_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256, 192))  # Resize the image to (256, 192)
    img = img / 255.0  # Normalize the image to the range [0, 1]
    return img

def apply_style(model, test_path):
    # Ensure image is resized properly
    test_img = load_and_resize(test_path)
    
    # Add batch dimension and make prediction
    pred = model.predict(tf.expand_dims(test_img, 0), verbose=0)[0]
    
    # Display input and output
    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(test_img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Stylized")
    plt.imshow(pred)
    plt.axis('off')
    plt.show()

# Example usage:
# Apply the style transfer four times in sequence
test_path = '../mariateguis/maria1.jpg'
test_img = load_and_resize(test_path)

# Create a figure to display all results
plt.figure(figsize=(20, 5))

for i in range(4):
    # Add batch dimension and make prediction
    pred = model.predict(tf.expand_dims(test_img, 0), verbose=0)[0]
    
    # Update the test_img with the prediction for the next iteration
    test_img = pred
    
    # Display the result
    plt.subplot(1, 4, i + 1)
    plt.title(f"Iteration {i + 1}")
    plt.imshow(pred)
    plt.axis('off')

plt.show()
