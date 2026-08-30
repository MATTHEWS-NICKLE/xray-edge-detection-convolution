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
    return np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=float)


def apply_convolution(image, kernel):

    image_height, image_width = image.shape

    kernel_height, kernel_width = kernel.shape

    output_height = (
        image_height - kernel_height + 1
    )

    output_width = (
        image_width - kernel_width + 1
    )

    output = np.zeros(
        (output_height, output_width)
    )

    for i in range(output_height):

        for j in range(output_width):

            region = image[
                i:i + kernel_height,
                j:j + kernel_width
            ]

            response = np.sum(
                region * kernel
            )

            output[i, j] = response

    return output
def save_response_matrix(response):

    np.savetxt(
        "results/response_matrix.csv",
        response,
        delimiter=",",
        fmt="%.2f"
    )

    print(
        "\nResponse matrix saved to "
        "results/response_matrix.csv"
    )

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

    kernel = create_edge_kernel()

    response = apply_convolution(
        patch,
        kernel
    )

    print("Input patch dimensions:")
    print(patch.shape)

    print("\nKernel dimensions:")
    print(kernel.shape)

    print("\nResponse dimensions:")
    print(response.shape)

    print("\nConvolution Response Matrix:")
    print(response)
    save_response_matrix(response)