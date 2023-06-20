import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def init_kmeans(data, k: int) -> KMeans:
    model = KMeans(n_clusters=k, n_init="auto")
    model.fit(data)
    return model


def compress_image_with_kmeans(
    image, original_shape: tuple, amount_of_clusters: int
) -> np.ndarray:
    kmeans = init_kmeans(image, amount_of_clusters)
    compressed_image = kmeans.cluster_centers_[kmeans.labels_] * 255
    compressed_image = np.clip(compressed_image.astype("uint8"), 0, 255)
    compressed_image = np.reshape(compressed_image, original_shape)
    return compressed_image


def plot_images(images, parameters: list[int]) -> None:
    fig = plt.figure(figsize=(2, 4))
    fig.tight_layout()
    for i, image in enumerate(images):
        ax = fig.add_subplot(2, 4, i + 1)
        ax.set_title(f"parameter={parameters[i]}")
        plt.imshow(image)
    plt.show()


def compress_image_with_pca(
    image, original_shape: tuple, n_components: int
) -> np.ndarray:
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(image)
    return (
        (pca.inverse_transform(transformed) * 255)
        .reshape(original_shape)
        .astype("uint8")
        .clip(0, 255)
    )


def compress_image(
    image, original_shape: tuple, parameters: list[int], mode="KMEANS"
) -> list[np.ndarray]:
    images = []
    for p in parameters:
        if mode == "KMEANS":
            new_image = compress_image_with_kmeans(image, original_shape, p)
        if mode == "PCA":
            new_image = compress_image_with_pca(image, original_shape, p)
        images.append(new_image)
    return images


if __name__ == "__main__":
    # kmeans clusters
    ks = [2, 4, 8, 16, 32, 64, 128, 256]
    # primcipal components
    components = [2, 6, 20, 50, 100, 200, 400]

    im1 = imread("pics/butterfly.jpg") / 255
    original_shape = im1.shape
    size = im1.shape[0], im1.shape[1] * im1.shape[2]
    images = compress_image(im1.reshape(size), original_shape, ks)
    plot_images(images, ks)

    components.append(original_shape[0])
    images = compress_image(im1.reshape(size), original_shape, components, mode="PCA")
    plot_images(images, components)
    components.pop()

    im2 = imread("pics/flower.jpg") / 255
    original_shape = im2.shape
    size = im2.shape[0], im2.shape[1] * im2.shape[2]
    images = compress_image(im2.reshape(size), original_shape, ks)
    plot_images(images, ks)

    components.append(original_shape[0])
    images = compress_image(im2.reshape(size), original_shape, components, mode="PCA")
    plot_images(images, components)
    components.pop()

    im3 = imread("pics/nasa.jpg") / 255
    original_shape = im3.shape
    size = im3.shape[0], im3.shape[1] * im3.shape[2]
    images = compress_image(im3.reshape(size), original_shape, ks)
    plot_images(images, ks)

    components.append(original_shape[0])
    images = compress_image(im3.reshape(size), original_shape, components, mode="PCA")
    plot_images(images, components)
    components.pop()
