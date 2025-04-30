from PIL import Image

def downsample_image(path, target_size=512):
    img = Image.open(path)
    img = img.convert('L')
    img.thumbnail((target_size, target_size), Image.LANCZOS)  # maintains aspect ratio
    img.save("downsampled_style.jpg")
    return "downsampled_style.jpg"

if __name__ == "__main__":
    downsampled_path = downsample_image("../mariateguis/maria0.jpg")
