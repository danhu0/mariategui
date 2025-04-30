import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

# --- Image dimensions ---
IMG_HEIGHT = 256
IMG_WIDTH = 192  # divisible by 8

# --- Load and preprocess style image (grayscale but converted to RGB) ---
STYLE_PATH = '../mariateguis/maria1.jpg'

def load_and_resize(path):
    img = load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')  # grayscale
    img = img_to_array(img) / 255.0  # shape: (H, W, 1)
    img = np.repeat(img, 3, axis=-1)  # convert to (H, W, 3)
    return img.astype(np.float32)

style_img = load_and_resize(STYLE_PATH)

# --- Augmentation (optional) ---
def random_augment(image):
    # image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, max_delta=0.2)
    # image = tf.image.random_contrast(image, 0.8, 1.2)
    # image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])  # Ensure consistent shape
    return image

# --- Dataset of augmented style images ---
def make_dataset(img, batch_size=8):
    def gen():
        while True:
            yield random_augment(img), img
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32)
        )
    ).batch(batch_size)

dataset = make_dataset(style_img, batch_size=2)

# --- Encoder-decoder model that works with any size ---
def build_model():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = inputs

    # Downsample less
    for f in [32, 64]:  # was [32, 64, 128]
        x = layers.Conv2D(f, 3, strides=2, padding='same', activation='relu')(x)

    # Bottleneck
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    # Upsample
    for f in [32, 16]:  # was [64, 32, 16]
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)

    outputs = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)
    return models.Model(inputs, outputs)

model = build_model()
model.compile(optimizer='adam', loss='mse')

# --- Train ---
# model.fit(dataset, steps_per_epoch=100, epochs=5)
EPOCHS = 5
STEPS_PER_EPOCH = 20

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    pbar = tqdm(dataset.take(STEPS_PER_EPOCH), total=STEPS_PER_EPOCH)
    for step, (x_batch, y_batch) in enumerate(pbar):
        loss = model.train_on_batch(x_batch, y_batch)
        pbar.set_description(f"Loss: {loss:.4f}")

# --- Apply style to test image ---
def apply_style(model, test_path):
    test_img = load_and_resize(test_path)
    pred = model.predict(tf.expand_dims(test_img, 0), verbose=0)[0]

    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(test_img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Stylized")
    plt.imshow(pred)
    plt.axis('off')
    plt.show()

# Example:
# Save the trained model
model.save('style_transfer_model.h5')
# apply_style(model, 'guzman.jpg')
