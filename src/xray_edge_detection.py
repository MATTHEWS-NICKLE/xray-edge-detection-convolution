from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_xray(image_path):

    image = Image.open(
        image_path
    ).convert("L")

    return np.array(
        image,
        dtype=float
    )


def extract_patch(
    image,
    x,
    y,
    width,
    height
):

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


def apply_convolution(
    image,
    kernel
):

    image_height, image_width = (
        image.shape
    )

    kernel_height, kernel_width = (
        kernel.shape
    )

    output_height = (
        image_height
        - kernel_height
        + 1
    )

    output_width = (
        image_width
        - kernel_width
        + 1
    )

    output = np.zeros(
        (
            output_height,
            output_width
        )
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


def save_response_matrix(
    response
):

    np.savetxt(
        "results/response_matrix.csv",
        response,
        delimiter=",",
        fmt="%.2f"
    )


def find_emphasized_locations(
    response
):

    magnitude = np.abs(
        response
    )

    threshold = (
        np.mean(magnitude)
        + np.std(magnitude)
    )

    locations = np.argwhere(
        magnitude >= threshold
    )

    print(
        "\nEdge Response Threshold:"
    )

    print(threshold)

    print(
        "\nEmphasized Locations:"
    )

    for row, column in locations:

        print(
            f"Location ({row}, {column}) "
            f"Response = "
            f"{response[row, column]:.2f}"
        )

    return magnitude, threshold


def visualize_response(
    magnitude
):

    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        magnitude,
        cmap="gray"
    )

    plt.colorbar(
        label="Response Magnitude"
    )

    plt.title(
        "X-ray Convolution Edge Response"
    )

    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.savefig(
        "results/edge_response.png",
        bbox_inches="tight"
    )

    plt.show()


def visualize_emphasized_edges(
    magnitude,
    threshold
):

    edge_map = (
        magnitude >= threshold
    )

    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        edge_map,
        cmap="gray"
    )

    plt.title(
        "Emphasized Bone Boundary Locations"
    )

    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.savefig(
        "results/emphasized_edges.png",
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":

    image_path = (
        "dataset/xray_original.png"
    )

    image = load_xray(
        image_path
    )

    print(
        "Original image dimensions:"
    )

    print(image.shape)

    # Change these coordinates
    # according to the selected
    # bone region.
    x = 100
    y = 100

    patch = extract_patch(
        image,
        x,
        y,
        15,
        15
    )

    print(
        "\nPatch dimensions:"
    )

    print(patch.shape)

    print(
        "\nPixel Matrix:"
    )

    print(patch)

    kernel = create_edge_kernel()

    print(
        "\nEdge Detection Kernel:"
    )

    print(kernel)

    response = apply_convolution(
        patch,
        kernel
    )

    print(
        "\nResponse Dimensions:"
    )

    print(response.shape)

    print(
        "\nConvolution Response Matrix:"
    )

    print(response)

    save_response_matrix(
        response
    )

    magnitude, threshold = (
        find_emphasized_locations(
            response
        )
    )

    visualize_response(
        magnitude
    )

    visualize_emphasized_edges(
        magnitude,
        threshold
    )

    print(
        "\nProcessing completed successfully."
    )