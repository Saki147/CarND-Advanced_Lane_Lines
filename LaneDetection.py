import numpy as np
import cv2
import matplotlib.pyplot as plt
import helpFunc as hf
import pickle



def LaneDetection(image):
    """
    Detect the lane lines for the given image.
    This function is the combination of the whole detection process.
    Returns the fit lane lines and the marked lane image.
    """
    # TO DO: Apply a distortion to a raw image
    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    dst = cv2.undistort(image, mtx, dist, None, mtx)

    # TO DO: Use color transforms, gradients, etc., to create a thresholded binary image.
    img = np.copy(dst)
    # Use color transforms and gradients in HLS color space, to create a thresholded binary image
    threshold_binary = hf.ColorandGradient(img, s_thresh=(170, 255), sx_thresh=(30, 255))


    # TO DO: Apply a mask of region of interest
    vertices = np.array([[(100,720), (565,460), (725, 460), (1180, 720)]], dtype=np.int32)
    masked_img = hf.region_of_interest(threshold_binary, vertices)


    # TO DO: Apply a perspective transform to rectify binary image ("birds-eye view")
    warped, M = hf.bird_eye(masked_img)


    # TO DO: Detect lane pixels and fit to find the lane boundary.
    binary_warped = np.copy(warped)
    leftx, lefty, rightx, righty, out_img = hf.find_lane_pixels(binary_warped)

    # Fit the polynomial equation and get the parameters for left and right lane lines
    lane_lines, left_fit, right_fit, left_fitx, right_fitx = hf.fit_polynomial(binary_warped)


    # TO DO: Determine the curvature of the lane and vehicle position with respect to center.
    ynum = lane_lines.shape[0]
    ploty = np.linspace(0, ynum-1, num=ynum)
    # Calculate the lane lines curvature at the bottom of the image in real world
    # Convert the pixels into the distance and fit in the polynomial equations
    ympp = 3 / 100  # meters per pixel in y dimension
    xmpp = 3.7 / 800  # meters per pixel in x dimension
    ploty = ympp*ploty # distance in y in real world
    left_fit, right_fit, left_base, right_base = hf.fit_real_world_polynomial(binary_warped, ympp, xmpp)
    # Calculate the curvature in the real world
    left_curverad, right_curverad = hf.measure_curvature_pixels(ploty, left_fit, right_fit)
    avg_curvature = (left_curverad + right_curverad)/2
    #print('The left curvature is ' + str(left_curverad) + ' m, and the right curvature is ' + str(right_curverad) + ' m.')
    # Calculate the lane lines center and the car center
    lane_lines_center = (left_base + right_base)/2
    car_center = xmpp*image.shape[1]/2
    # vehicle position with respect to center
    offset = car_center - lane_lines_center
    #print('The vehicle position with respect to the lane lines center is: ' + str(offset) +' m.')


    # TO DO: Warp the detected lane boundaries back onto the original image.
    warped_img = np.zeros_like(img)
    warped_img[lefty, leftx] = [255, 0, 0]
    warped_img[righty, rightx] = [0, 0, 255]
    y = np.linspace(0, ynum-1, num=ynum)

    # Define the lane lines points
    left = np.round(left_fitx)
    right = np.round(right_fitx)
    left_boundary = np.dstack((left, y))
    right_boundary = np.dstack((right, y))
    # Flip the right lane detected points to connect with the left points
    right_boundary = np.flip(right_boundary, axis = 1)
    vertices = np.hstack((left_boundary, right_boundary))
    mask = np.copy(warped_img).astype(np.uint8)
    # Fill the lane part with the green color
    fill_lane = cv2.fillPoly(mask, np.int32(vertices), (0, 200, 0))
    # Add the information text on the image
    cv2.putText(fill_lane,'|',(int(fill_lane.shape[1]/2), fill_lane.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),8)
    cv2.putText(fill_lane,'|',(int(lane_lines_center/xmpp), fill_lane.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),8)
    #cv2.imwrite('output_images/fill_lane_straight_lines1.jpg', fill_lane, None)
    unwarped_lane, Minv = hf.unwarp(fill_lane)
    result = cv2.addWeighted(img, 1., unwarped_lane, 0.6, 0.)


    # TO DO: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    cv2.putText(result,'Vehicle is ' + str(round(offset,3))+'m'+' with respect to the lane center', (50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),thickness=2)
    cv2.putText(result,'Radius of curvature: '+str(round(avg_curvature))+'m',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),thickness=2)
    #cv2.imwrite('output_images/final_unwarped_straight_lines1.jpg', result, None)
    #result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    #plt.title('Final Result')
    #plt.imshow(result)
    #plt.axis('off')
    #plt.show()
    return result

