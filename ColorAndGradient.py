import numpy as np
import cv2



def ColorandGradient(img, s_thresh=(170, 255), sx_thresh=(70, 255)):

    """Use color transforms, gradients, etc., to create a thresholded binary image."""

    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    threshold_binary = np.zeros_like(sxbinary)
    threshold_binary[((sxbinary == 1) | (s_binary == 1))] = 1
    threshold_binary = np.uint8(255*threshold_binary)
    #color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return threshold_binary


