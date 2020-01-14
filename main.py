import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
from skimage.io import imshow as show_image
import skimage
import matplotlib.pyplot as plt


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle(img):
    # Fill in this function
    return 100, 100, 30


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def main():
    results = []
    # n = 5
    size = 200
    radius = 50
    noise = 2
    n = 1
    radius_list = np.zeros(n)
    row_list = np.zeros(n)
    column_list = np.zeros(n)
    for index in range(n):
        params, img = noisy_circle(size, radius, noise)
        print(img.shape)

        radius_list[index] = params[2]
        row_list[index] = params[0]
        column_list[index] = params[1]
        # draw_circle(img,params[0],params[1],params[2])
        # plt.figure()
        # skimage.io.imshow(img)

        # plt.imshow(img)
        # plt.show()
        detected = find_circle(img)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())

    print("min radius : ",np.min(radius_list))
    print("max radius : ",np.max(radius_list))
    print("")

# main()