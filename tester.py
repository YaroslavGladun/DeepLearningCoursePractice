import numpy as np
import os
import matplotlib.pyplot as plt

IMAGES_DIR = "./"


def imshow(image: np.ndarray, title: str) -> None:
    plt.imshow(image)
    plt.title(title)
    plt.show()


def test_solution_1_1(f) -> bool:
    for n in range(5, 10):
        for m in range(5, 10):
            if not (f(n, m) == 0).all():
                return False
    return True


def test_solution_1_2(f) -> bool:
    for a in range(5, 10):
        for b in range(a + 1, 11):
            segments = f(a, b)
            if np.unique(np.gradient(segments).round(decimals=4)).shape[0] != 1:
                return False
    return True


def test_solution_1_3(f) -> bool:
    for n in range(40, 51):
        a = f(n)
        if not np.allclose(a[:-1, 1:], np.eye(n - 1)):
            return False
        if (a[:, 0] != 0).all():
            return False
        if (a[-1, :] != 0).all():
            return False
    return True


def test_solution_3_1(f) -> bool:
    rgb_image = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[255, 255, 0], [0, 255, 255], [255, 0, 255]]
    ])

    bgr_image = f(rgb_image)

    expected_bgr_image = np.array([
        [[0, 0, 255], [0, 255, 0], [255, 0, 0]],
        [[0, 255, 255], [255, 255, 0], [255, 0, 255]]
    ])

    return np.array_equal(bgr_image, expected_bgr_image)


def test_solution_3_2(f) -> bool:
    image = np.load(IMAGES_DIR + "image.npy")
    imshow(image, "input image")

    expected_image = image[:, ::-1, :]
    imshow(expected_image, "expected image")

    f_image = f(image)
    imshow(f_image, "your image")

    return np.array_equal(f_image, expected_image)


def test_solution_3_3(f) -> bool:
    image = np.load(IMAGES_DIR + "image.npy")
    imshow(image, "input image")

    expected_image = np.load(IMAGES_DIR + "resized_image.npy")
    imshow(expected_image, "expected image")

    f_image = f(image)
    print(f_image.shape)
    imshow(f_image, "your image")

    return f_image.shape == (1600, 1600, 3)
