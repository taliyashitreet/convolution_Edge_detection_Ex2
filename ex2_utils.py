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

    return


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


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
    for i in range(1,row - 1):
        for j in range(1,col - 1):
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


# def houghCircle(img:np.ndarray, min_radius:float, max_radius:float) -> list:
#     if min_radius <= 0 or max_radius <= 0 or min_radius >= max_radius:
#         print("There is some problem with the given radius values")
#         return []
#
#     blur_img = cv2.GaussianBlur(img, (5, 5), 1)
#     edged_img = cv2.Canny(blur_img, 75, 150)
#     circles_list = list()  # the answer to return
#
#     height, width = edged_img.shape
#     radii = 100
#
#     output = img.copy()
#
#     acc_array = np.zeros((height, width, radii))
#
#     filter3D = np.ones((30, 30, radii))
#
#     edges = np.where(edged_img == 255)
#     for i in range(0, len(edges[0])):
#         x = edges[0][i]
#         y = edges[1][i]
#         for radius in range(20, 55):
#             fill_acc_array(x, y, radius, height, width, acc_array)
#
#     i = 0
#     j = 0
#     while (i < height - 30):
#         while (j < width - 30):
#             filter3D = acc_array[i:i + 30, j:j + 30, :] * filter3D
#             max_pt = np.where(filter3D == filter3D.max())
#             a = max_pt[0]
#             b = max_pt[1]
#             c = max_pt[2]
#             b = b + j
#             a = a + i
#             if (filter3D.max() > 90):
#                 cv2.circle(output, (b, a), c, (0, 255, 0), 2)
#             j = j + 30
#             filter3D[:, :, :] = 1
#         j = 0
#         i = i + 30
#
#     return circles_list
#
#
# def fill_acc_array(x0, y0, radius, height, width, acc_array):
#     x = radius
#     y = 0
#     decision = 1 - x
#
#     while (y < x):
#         if (x + x0 < height and y + y0 < width):
#             acc_array[x + x0, y + y0, radius] += 1;  # Octant 1
#         if (y + x0 < height and x + y0 < width):
#             acc_array[y + x0, x + y0, radius] += 1;  # Octant 2
#         if (-x + x0 < height and y + y0 < width):
#             acc_array[-x + x0, y + y0, radius] += 1;  # Octant 4
#         if (-y + x0 < height and x + y0 < width):
#             acc_array[-y + x0, x + y0, radius] += 1;  # Octant 3
#         if (-x + x0 < height and -y + y0 < width):
#             acc_array[-x + x0, -y + y0, radius] += 1;  # Octant 5
#         if (-y + x0 < height and -x + y0 < width):
#             acc_array[-y + x0, -x + y0, radius] += 1;  # Octant 6
#         if (x + x0 < height and -y + y0 < width):
#             acc_array[x + x0, -y + y0, radius] += 1;  # Octant 8
#         if (y + x0 < height and -x + y0 < width):
#             acc_array[y + x0, -x + y0, radius] += 1;  # Octant 7
#         y += 1
#         if (decision <= 0):
#             decision += 2 * y + 1
#         else:
#             x = x - 1;
#             decision += 2 * (y - x) + 1


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return
