import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import cv2
import os
import h5py

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



def pre_process_image(img):
    """
    this function will normalize the image with level 2 noise.
    It will make value of each pixel between 0 and 1
    :param img: image input
    :return: output image with pixel value between (0,1)
    """
    img = img - 1
    img = img / 2.0
    img = np.clip(img, 0, 1)

    return img

def clean_noise(img):
    """
    This function will remove the noise level in the image and make the circle
    to be more visible
    :param img: input image
    :return:
    """
    if (np.max(img) <=1.0):
        img = img * 255.0
        img = np.round(img)
        img = img.astype(np.uint8)
    result = cv2.fastNlMeansDenoising(img, None, h=27.0, templateWindowSize=7, searchWindowSize=21)

    return result / 255.0

def create_dataset(samples_size,image_size, max_radius, noise_level):
    """
    create dataset to train model
    :param samples_size: number of image to create
    :param image_size: size of image
    :param max_radius: max radius
    :param noise_level:  noise level
    :return: image dataset (sample_size,image_size,image_size), image labels (sample_size,3)
    """
    training_images = np.zeros((samples_size, image_size, image_size))
    training_labels = np.zeros((samples_size, 3), dtype=np.float64)
    for index in range(samples_size):
        params, img = noisy_circle(image_size, max_radius, noise_level)

        #transform image


        img = pre_process_image(img)

        #denoise image:
        # print(img.shape)
        img = clean_noise(img)


        training_images[index] = img
        training_labels[index] = params

    return [training_images,training_labels]

def save_data_into_h5(train_img,train_labels,test_img,test_labels,valid_img,valid_labels,path = None):
    if path is None:
        path = "dataset"
        if not os.path.isdir(path):
            os.makedirs(path)
        path = os.path.join(path,'data.hdf5')
    hdf5_file = h5py.File(path, mode='w')
    hdf5_file.create_dataset("train_img", shape = train_img.shape, dtype=np.float32,data=train_img)
    hdf5_file.create_dataset("valid_img", valid_img.shape, np.float32,valid_img)
    hdf5_file.create_dataset("test_img", test_img.shape, np.float32,test_img)
    hdf5_file.create_dataset("train_labels", train_labels.shape, np.float32,train_labels)
    hdf5_file.create_dataset("valid_labels", valid_labels.shape, np.float32,valid_labels)
    hdf5_file.create_dataset("test_labels", test_labels.shape, np.float32,test_labels)

def load_h5_data(hdf5_path = None):
    if hdf5_path is None:
        hdf5_path = "dataset/data.hdf5"
    # open the hdf5 file
    hdf5_file = h5py.File(hdf5_path, "r")
    train_img = hdf5_file["train_img"][:]
    train_label = hdf5_file["train_labels"][:]

    test_img = hdf5_file["test_img"][:]
    test_label = hdf5_file["test_labels"][:]

    valid_img = hdf5_file["valid_img"][:]
    valid_label = hdf5_file["valid_labels"][:]
    return [train_img,train_label,test_img,test_label,valid_img,valid_label]

