import math
import numpy as np
import cv2


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

    kernel = np.zeros((k_size, k_size))
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    for i in range(k_size):
        for j in range(k_size):
            kernel[i][j] = ((1 / 2 * np.pi * np.square(sigma)) * np.e) - ((i ** 2 + j ** 2) / 2 * np.square(sigma))
    return conv2D(in_image, kernel)


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

    return


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
                if len(only_pos) > 0:
                    zc_image[i][j] = 1.0
            if pixel > 0:
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
    # find the edges with canny
    edged_img = cv2.Canny((img * 255).astype(np.uint8), 550, 100)
    circles_list = list()  # the answer to return

    height, width = edged_img.shape
    count_radius = max_radius - min_radius  # the radius range
    acc_mat = np.zeros((height, width, count_radius))  # Accumulator Matrix - hough Circle space

    for x in range(height):
        for y in range(width):
            if edged_img[x, y] == 255:  # if its edge
                for r in range(count_radius):  # for each possible radius
                    for theta in range(361):  # for each possible theta
                        # find the possible a and b on the hough Circle space
                        a = int(y - r * np.cos((theta * np.pi) / 180))
                        b = int(x - r * np.sin((theta * np.pi) / 180))
                        # put this a, b on the Accumulator Matrix
                        if 0 < a < len(acc_mat) and 0 < b < len(acc_mat):
                            acc_mat[a, b, r] += 1
    # find the cell with maximum value - this point suspected of being the center of the circle
    for i in range(count_radius):
        thresh = np.max(acc_mat[:, :, i])  # for each radius find the maximum
        x, y = np.where(acc_mat[:, :, i] == thresh)
        for j in range(len(x)):
            if x[j] != 0 and y[j] != 0:
                circles_list.append((x[j], y[j], i))
    return circles_list


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
    width = int (np.floor(k_size / 2)) # width for padding
    img_pad = np.pad(in_image, ((width,), (width,)), 'constant', constant_values=0)  # zero padding the image
    if k_size % 2 != 0:  # k_size must be odd number
        Gaus_kernel = cv2.getGaussianKernel(abs(k_size), 1)
    else:
        Gaus_kernel = cv2.getGaussianKernel(abs(k_size) + 1, 1)

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
