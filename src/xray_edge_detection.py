import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_xray_image(image_path):
    image = Image.open(image_path).convert("L")
    return np.array(image, dtype=float)


def display_image(image):
    plt.imshow(image, cmap="gray")
    plt.title("Input X-ray Patch")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    image_path = "dataset/xray_patch.png"

    image = load_xray_image(image_path)

    print("Image shape:", image.shape)
    print("Pixel values:")
    print(image)

    display_image(image)