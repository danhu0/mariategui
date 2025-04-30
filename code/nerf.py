import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load and grayscale image
img_path = "/mnt/data/sample_image.jpg"  # placeholder, user can replace
image = Image.open(img_path).convert("L")
original_size = image.size

# Resize to smaller image for demo
transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])
img_tensor = transform(image).squeeze(0)  # Shape: [H, W]

H, W = img_tensor.shape
coords = torch.stack(torch.meshgrid(
    torch.linspace(-1, 1, steps=H),
    torch.linspace(-1, 1, steps=W),
    indexing="ij"
), dim=-1).reshape(-1, 2)  # Shape: [H*W, 2]
pixels = img_tensor.reshape(-1, 1)  # Shape: [H*W, 1]

# Define NeRF-like 2D MLP
class NeRF2D(nn.Module):
    def __init__(self, hidden_dim=128, depth=4):
        super().__init__()
        layers = []
        for i in range(depth):
            in_dim = 2 if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

model = NeRF2D()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Train model
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(coords)
    loss = loss_fn(out, pixels)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Generate extrapolated image from larger coordinate range
new_H, new_W = 96, 96
new_coords = torch.stack(torch.meshgrid(
    torch.linspace(-1.5, 1.5, steps=new_H),
    torch.linspace(-1.5, 1.5, steps=new_W),
    indexing="ij"
), dim=-1).reshape(-1, 2)

with torch.no_grad():
    new_pixels = model(new_coords).reshape(new_H, new_W).clip(0, 1)

# Plot original and extrapolated
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_tensor.numpy(), cmap="gray")
axs[0].set_title("Original (64x64)")
axs[1].imshow(new_pixels.numpy(), cmap="gray")
axs[1].set_title("Extrapolated (96x96)")
for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.show()
