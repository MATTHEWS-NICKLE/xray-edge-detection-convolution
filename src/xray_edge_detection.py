from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_xray(image_path):
    image = Image.open(image_path).convert("L")
    return np.array(image, dtype=float)


def extract_patch(image, x, y, width, height):
    return image[
        y:y + height,
        x:x + width
    ]


def create_edge_kernel():
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=float)

    return kernel


if __name__ == "__main__":

    image_path = "dataset/xray_original.png"

    image = load_xray(image_path)

    x = 100
    y = 100

    patch = extract_patch(
        image,
        x,
        y,
        15,
        15
    )

    print("Patch dimensions:")
    print(patch.shape)

    print("\nPixel matrix:")
    print(patch)

    kernel = create_edge_kernel()

    print("\nEdge Detection Kernel:")
    print(kernel)

    plt.figure(figsize=(6, 6))
    plt.imshow(patch, cmap="gray")
    plt.title("Selected X-ray Patch")
    plt.axis("off")
    plt.show()