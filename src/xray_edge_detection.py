from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_xray(image_path):
    image = Image.open(image_path).convert("L")
    return np.array(image, dtype=float)


if __name__ == "__main__":

    image_path = "dataset/xray_patch.png"

    image = load_xray(image_path)

    print("X-ray loaded successfully.")
    print("Image dimensions:", image.shape)
    print("Minimum pixel value:", image.min())
    print("Maximum pixel value:", image.max())

    plt.imshow(image, cmap="gray")
    plt.title("Original X-ray")
    plt.axis("off")
    plt.show()