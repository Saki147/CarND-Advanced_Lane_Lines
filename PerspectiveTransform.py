import pickle
import cv2
import numpy as np




def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)


    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def bird_eye(img):
    """Apply the perspective transform to the image """
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[243, 690], [572, 466], [711, 466], [1067, 690]])

    #dest = np.float32([[243, 690], [243, 0], [1067, 0], [1067, 690]])
    dest = np.float32([[(img_size[0] / 4), img_size[1]],
    [(img_size[0] / 4), 0],
    [(img_size[0] *3/ 4), 0],
    [(img_size[0] *3 / 4), img_size[1]]])

    # Use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dest)

    # Use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M

def unwarp(img):
    """Warp the detected lane boundaries back onto the original image """
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[243, 690], [572, 466], [711, 466], [1067, 690]])

    # dest = np.float32([[243, 690], [243, 0], [1067, 0], [1067, 690]])
    dest = np.float32([[(img_size[0] / 4), img_size[1]],
                       [(img_size[0] / 4), 0],
                       [(img_size[0] * 3 / 4), 0],
                       [(img_size[0] * 3 / 4), img_size[1]]])
    # Use cv2.getPerspectiveTransform() to get Minv, the inverse transform matrix
    Minv = cv2.getPerspectiveTransform(dest, src)

    # Use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    return warped, Minv