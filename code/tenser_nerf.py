import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.keras import TqdmCallback  # For progress bar

# Load the full image and normalize to [-1, 1]
img_path = "José_Carlos_Mariátegui_in_1929.jpg"
image = Image.open(img_path).convert("L")
img_array = (np.array(image) / 255.0) * 2.0 - 1.0  # Normalize to [-1, 1]

# Get the original image shape
H, W = img_array.shape

# Create the coordinate grid
x = np.linspace(-1, 1, W)
y = np.linspace(-1, 1, H)
xx, yy = np.meshgrid(x, y)
coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)

# Flatten image pixels and prepare for training
pixels = img_array.ravel().reshape(-1, 1)

# Positional encoding for better representation of spatial frequencies
def positional_encoding(x, num_frequencies=16):
    enc = [x]
    for i in range(num_frequencies):
        for fn in [np.sin, np.cos]:
            enc.append(fn((2.0 ** i) * x))
    return np.concatenate(enc, axis=-1)

encoded_coords = positional_encoding(coords, num_frequencies=16)

# Define the MLP model with higher capacity
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(encoded_coords.shape[-1],)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='tanh')  # Output in [-1, 1] range
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train with progress bar
model.fit(
    encoded_coords, pixels,
    epochs=100,
    batch_size=1024,
    verbose=0,
    callbacks=[TqdmCallback(verbose=1)]
)

# Visualize the reconstruction of the original image
reconstructed = model.predict(encoded_coords).reshape(H, W)
diff = np.abs(reconstructed - pixels.reshape(H, W))  # Show the difference
plt.imshow(diff, cmap='hot')
plt.title("Difference: Reconstruction vs. Original")
plt.axis('off')
plt.show()

# Create new coordinates for extrapolation (expanded region)
pad = 16  # Padding around the original image
new_H, new_W = H + 2 * pad, W + 2 * pad
new_x = np.linspace(-1.5, 1.5, new_W)
new_y = np.linspace(-1.5, 1.5, new_H)
new_xx, new_yy = np.meshgrid(new_x, new_y)
new_coords = np.stack([new_xx.ravel(), new_yy.ravel()], axis=-1)

# Apply positional encoding to the new coordinates
encoded_new_coords = positional_encoding(new_coords, num_frequencies=16)

# Predict the new pixels on the expanded grid
new_pixels = model.predict(encoded_new_coords, batch_size=2048).reshape(new_H, new_W)

# Embed the original image in the center of the expanded grid
final_image = new_pixels.copy()
final_image[pad:pad+H, pad:pad+W] = img_array

# Plot the final outpainted image
plt.figure(figsize=(8, 8))
plt.imshow((final_image + 1) / 2, cmap='gray')  # Convert from [-1, 1] to [0, 1]
plt.title("Outpainted Image with Original Preserved")
plt.axis("off")
plt.tight_layout()
plt.show()
