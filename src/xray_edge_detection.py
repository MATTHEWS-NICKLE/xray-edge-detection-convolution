from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_xray(image_path):
    image = Image.open(image_path).convert("L")
    return np.array(image, dtype=float)


def extract_patch(image, x, y, width, height):
    patch = image[
        y:y + height,
        x:x + width
    ]

    return patch


if __name__ == "__main__":

    image_path = "dataset/xray_original.png"

    image = load_xray(image_path)

    print("Original X-ray dimensions:")
    print(image.shape)

    # Temporary coordinates.
    # We will change these after checking your X-ray.
    x = 100
    y = 100

    patch = extract_patch(
        image,
        x,
        y,
        15,
        15
    )

    print("\nSelected patch dimensions:")
    print(patch.shape)

    print("\nSelected patch pixel values:")
    print(patch)

    plt.figure(figsize=(6, 6))
    plt.imshow(patch, cmap="gray")
    plt.title("Selected 15 x 15 X-ray Patch")
    plt.axis("off")
    plt.show()