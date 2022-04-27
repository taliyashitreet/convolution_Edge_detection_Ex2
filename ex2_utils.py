import math
import numpy as np
import cv2
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters


def myID() -> np.int:
    return 314855099


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    flip = np.flip(k_size)  # flip the vector kernel
    result = np.zeros(in_signal.size + k_size.size - 1)  # init the result array with the new size ("full")
    num_of_zeros = k_size.size - 1  # number of zeros to padding each side of the signal
    zero_padding = np.zeros(num_of_zeros)
    new_signal = np.append(zero_padding, np.append(in_signal, zero_padding))

    for i in range(result.size):  # from the starting of new_signal to the last element of inSignal
        result[i] = np.dot(new_signal[i: i + num_of_zeros + 1], flip)

    return result


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    result = np.zeros_like(in_image)
    x_width, y_width = pad_width(kernel)
    img_pad = np.pad(in_image, ((x_width,), (y_width,)), 'constant', constant_values=0)  # zero padding the image
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.multiply(img_pad[i:i + kernel.shape[0], j:j + kernel.shape[1]], kernel).sum()

    return result


def pad_width(kernel: np.ndarray) -> (int, int):
    # The width should be half of the kernel width
    x_width = np.floor(kernel.shape[0] / 2).astype(int)
    if x_width < 1: x_width = 1
    y_width = np.floor(kernel.shape[1] / 2).astype(int)
    if y_width < 1: y_width = 1
    return x_width, y_width


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    kernel_x = np.array([[1, 0, -1]])
    kernel_y = kernel_x.reshape(3, 1)
    im_derive_x = cv2.filter2D(in_image, -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
    im_derive_y = cv2.filter2D(in_image, -1, kernel_y, borderType=cv2.BORDER_REPLICATE)
    # MagG = ||G|| = (Ix^2 + Iy^2)^(0.5)
    mag = np.sqrt(np.square(im_derive_x) + np.square(im_derive_y))
    # DirectionG = tan^(-1) (Iy/ Ix)
    Direction = np.arctan2(im_derive_y, im_derive_x)
    return mag, Direction


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    g_ker1d = np.array([1, 1])
    while len(g_ker1d) != k_size:
        g_ker1d = conv1D(g_ker1d, g_ker1d)
    g_ker1d = g_ker1d / g_ker1d.sum()
    g_ker1d = g_ker1d.reshape((1, len(g_ker1d)))
    g_ker2d = g_ker1d.T @ g_ker1d
    img = conv2D(in_image, g_ker2d)

    return img


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    kernel = cv2.getGaussianKernel(k_size, 0)
    blur = cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return blur


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """
    # I didnt implement
    pass


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> (np.ndarray):
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(img, (11, 11), 0)

    # Apply Laplacian operator in some higher datatype
    lap_img = cv2.Laplacian(blur, cv2.CV_64F)
    lap_img = lap_img / lap_img.max()
    ans = zeroCrossing(lap_img)
    return ans


def zeroCrossing(image: np.ndarray) -> np.ndarray:
    zc_image = np.zeros(image.shape)
    row = zc_image.shape[0]
    col = zc_image.shape[1]
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            neighbour = [image[i + 1, j - 1],
                         image[i + 1, j],
                         image[i + 1, j + 1],  # 5    6     7
                         image[i, j - 1],  # 3  [i,j]   4
                         image[i, j + 1],  # 0    1     2
                         image[i - 1, j - 1],
                         image[i - 1, j],
                         image[i - 1, j + 1]]
            only_pos = [num for num in neighbour if num > 0]  # for counting positive neighbourhood
            only_neg = [num for num in neighbour if num < 0]  # for counting negative neighbourhood
            pixel = image[i, j]
            if pixel < 0:
                if len(only_pos) > 0:  # {-,+} or {+,-}
                    zc_image[i][j] = 1.0
            if pixel > 0:  # {-,+} or {+,-}
                if len(only_neg) > 0:
                    zc_image[i][j] = 1.0
            else:  # pixel == 0, {+,0,-}
                comp_list = [neighbour[6] < 0 and neighbour[1] > 0, neighbour[6] > 0 and neighbour[1] < 0,
                             neighbour[7] < 0 and neighbour[0] > 0, neighbour[7] > 0 and neighbour[0] < 0,
                             neighbour[3] < 0 and neighbour[4] > 0, neighbour[3] > 0 and neighbour[4] < 0,
                             neighbour[5] < 0 and neighbour[2] > 0, neighbour[5] > 0 and neighbour[2] < 0]
                if any(comp_list):
                    zc_image[i][j] = 1.0
    return zc_image


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    if min_radius <= 0 or max_radius <= 0 or min_radius >= max_radius:
        print("There is some problem with the given radius values")
        return []
    img = cv2.GaussianBlur(img, (11, 11), 1)
    # find the edges with canny
    edged_img = cv2.Canny((img * 255).astype(np.uint8), 255 / 3, 255)
    circles_list = list()  # the answer to return
    edges = []  # where edges only
    circlesPoints = [] # points that are on the circle
    height, width = edged_img.shape
    for i in range(height):
        for j in range(width):
            if edged_img[i, j] == 255:
                edges.append((i, j))

    for r in range(min_radius, max_radius + 1):
        for theta in range(1, 361, 5):  # for each possible theta
            x = int(r * np.cos(theta * np.pi / 180))
            y = int(r * np.sin(theta * np.pi / 180))
            circlesPoints.append((x, y, r))
    size_radius = max_radius - min_radius + 1
    accumulator = np.zeros((height, width, size_radius))  # 2D arr to vote for the circles centers
    for i, j in edges:
        for x, y, r in circlesPoints:  # find the possible a and b on the hough Circle space
            b = j - y
            a = i - x
            if 0 <= a < height and 0 <= b < width:  # if in borders
                accumulator[a, b, r - min_radius] += 1

    find_by_thresh(accumulator, circles_list, min_radius, max_radius)

    return circles_list


def find_by_thresh(accumulator: np.ndarray, circles_list: list, min_radius: int, max_radius: int):
    """
    find the local maximums in the accumulator
    :param max_radius:
    :param min_radius:
    :param accumulator:
    :param circles:
    :param radius: curr radius
    :return: none
    """
    (h, w, rad) = accumulator.shape
    threshold = np.median([np.amax(accumulator[:, :, radius]) for radius in range(rad)])
    print("threshold = np.median([np.amax(accumulator[:, :, radius]) for radius in range(rad)])")
    print("After many attempts - brought the best result")
    for r in range(rad):
        for i in range(h):
            for j in range(w):
                if accumulator[i, j, r] >= threshold:
                    circles_list.append((j, i, r + min_radius))


def maximum_by_radius(r: int, circles_list: list, acc_mat: np.ndarray):
    # find the local maximum by 5 neighborhood
    neighborhood_size = 5
    threshold = np.max(acc_mat) * 0.8
    data_max = filters.maximum_filter(acc_mat, neighborhood_size)  # change tha all neighborhood to the max value
    maxima = (acc_mat == data_max)
    data_min = filters.minimum_filter(acc_mat, neighborhood_size)  # change tha all neighborhood to the min value
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    # find tha x and y of the circle center
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        y_center = (dy.start + dy.stop - 1) / 2
        circles_list.append((x_center, y_center, r))


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    img_filter = np.zeros_like(in_image)
    width = int(np.floor(k_size / 2))  # width for padding

    img_pad = np.pad(in_image, ((width,), (width,)), 'constant', constant_values=0)  # zero padding the image
    if k_size % 2 != 0:  # k_size must be odd number
        Gaus_kernel = cv2.getGaussianKernel(abs(k_size), sigma_space)
    else:
        Gaus_kernel = cv2.getGaussianKernel(abs(k_size) + 1, sigma_space)

    for x in range(in_image.shape[0]):
        for y in range(in_image.shape[1]):
            pivot_v = in_image[x, y]
            neighbor_hood = img_pad[x:x + k_size,
                            y:y + k_size]

            diff = pivot_v - neighbor_hood
            diff_gau = np.exp(-np.power(diff, 2) / (2 * sigma_color))
            combo = Gaus_kernel * diff_gau
            result = (combo * neighbor_hood / combo.sum()).sum()
            img_filter[x][y] = result

    #  Bilateral of cv2
    cv2_image = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
    return cv2_image, img_filter
