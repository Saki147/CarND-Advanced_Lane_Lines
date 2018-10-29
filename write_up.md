## Writeup 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg  "Original Chessboard" 
[image2]: ./output_images/dst_chessboard.jpg "Undistorted Chessboard"
[image3]: ./test_images/test1.jpg "Original image"
[image4]: ./output_images/dst_test1.jpg "Undistorted"
[image5]: ./output_images/threshold_binary_straight_lines1.jpg "Binary image"
[image6]: ./output_images/masked_straight_lines1.jpg "Masked image"
[image7]: ./output_images/warped_straight_lines1.jpg "Perspective transform"
[image10]: ./output_images/lane_lines_straight_lines1.jpg "Marked lane pixels with 
window"
[image8]: ./output_images/lane_lines_project_frame.jpg "Marked lane lines with window"
[image9]: ./output_images/final_unwarped_test1.jpg "Output"
[video1]: ./output_images/project_video.mp4 "Output video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
Note: The functions are finally combined in `helperFunction.py`, and the 
implementation code is included in function file `plotLane.py`. The file 
`new_main.py` can either process the image or video detection, while the `main.py` can only 
process and save the image.
---

### Writeup / README

### Camera Calibration

#### 1. Brief state of the camera matrix and distortion coefficients computation. Provide an example of a distortion corrected calibration image.

The Code for this step is contained in the file called 'CameraCalibration.py', and the 'CameraCalibration(image)' is also included in the 'helperFunction.py'.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original chessboard:
![alt text][image1]
Undistorted chessboard:
![alt text][image2]

### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]
The image above was undistorted by `cv2.undistort()` function using the camera matrix *mtx* and distortion coefficient *dist*. I got the undistorted image as below:
![alt text][image4]

#### 2. Use color transforms, gradients or other methods to create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary 
image (thresholding steps at lines # through # in `ColorAndGradient.py`). The
 image was converted to HLS space first, and the s channel was filtered with 
 threshold (170, 255). The l channel was used to calculate the x gradient
  with `cv2.Sobel()` function. Then the gradient values were filtered with 
  the threshold (70, 255). Finally, the two layers of image were stacked 
  together to form a threshold binary image. Here's an example of my output for this step. 

![alt text][image5]


#### 3. Description of  how to perform a perspective transform and provide an example of a transformed image.

I chose the region of interest to detect the lane more precisely. The 
funciton is called `region_of_interest()` in the `PerspectiveTransform.py`,
which takes two input as img and vertices. The image after this process is:

![alt text][image6]

The code in `PerspectiveTransform.py` also includes a function called `bird_eye()`, which is to show a bird eye view. The `bird_eye()` function takes as inputs an image (`img`). The source (`src`) and destination (`dest`) points were defined in the function. I chose 
the source and destination points in the following manner:

```python
src = src = np.float32([[243, 690], [572, 466], [711, 466], [1067, 690]])

dest =  dest = np.float32([[(img_size[0] / 4), img_size[1]],
    [(img_size[0] / 4), 0],
    [(img_size[0] *3/ 4), 0],
    [(img_size[0] *3 / 4), img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 243, 690      | 320, 720      | 
| 572, 466      | 320, 0        |
| 711, 466      | 960, 0        |
| 1067, 690     | 960, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]

#### 4. Description of how to identify lane-line pixels and fit their positions with a polynomial.

I built a histogram to count the none zero pixels on the vertical axis 
both in the left half side and right half side. In the `find_lane_pixels()`
function, I chose the two positions with the largest pixels numbers as window
 centers. From these two positions, I used 9 windows with 80 height and +/-60
  margins (120x80) size from the image bottom to count the none zero pixels, and modify the window center position. Then, I
 used `fit_polynomial()` function in the `FindLines.py` to fit the lane lines
  with a 2nd order polynomial which returns the fit coefficients and fitted 
  points. The image is shown below. I also used an another image with larger 
  curvature so that it can be demonstrated more clearly.

![alt text][image10]![alt text][image8]

#### 5. Discription of how to calculate the curvature of the lane and the position of the vehicle with respect to center. 

I did this in lines # through # in my code in `FindLines.py`which includes
`fit_real_world_polynomial()` function with the input `image`, `xmpp`(3.7 
meters per 100 pixel in x axis) and `ympp`(3 meters per 630 pixels in y axis),
 to convert the pixels to meters in the real world and fit the polynomial and calculate the curvature using the 
function called `measure_curvature_pixels()`. The car center is calculated by
 the *real world image center(car center) - lane center*.

#### 6. An example image of my result plotted back down onto the road.

I combined all the functions in my code in `helperFunction
.py` in the function `helerFunction()`. Then, I implemented the pipelines in 
lines # through # in `new_main.py`. Here is an example of my result on a test
 image:

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to my final video output.  

Here's a [link to my video result](./output_images/project_video.mp4). The class 
`Line()` is used to save the previous lane lines data, current lane lines 
position, curvatures, and start and end positions, etc. After the 10 
iterations of the lane detection, the fitted positions are the average values 
of the previous 10 iterations, thus the wobbly situation can be reduced. The 
code to lane detection for the video is in `new_main.py` where you should 
change the `file` in line 17 to 'video'.

---

### Discussion

#### 1. Briefly discuss any problems / issues I faced in my implementation of this project. 

I used the color and gradient threshold to show the binary image, and 
perspective transform to get a bird eye view image. The window sliding method
 was applied to detect the lane lines pixels, which was then fitted into the 
 2nd polynomial. This approach shows good performances in the small curvature 
 lines detection on a clear road. However, this method performs not well 
 enough in the large turns and quick changes (such as `harder_challenge_video
 .mp4`). In addition, if there  are some similar color or shape things on the
  road (such as `challenge_video.mp4`), the algorithm will be confused. 
  Moreover, the other vehicles in front of the car could hinder the detection.
 
 To improve the performance, it is important to tune the color and 
 gradient threshold to fit different situation, as well as to set a standard 
 relationship between the two lane lines. In the window sliding process, it 
 can be improved by using [deep-learning-based semantic segmentation](https://arxiv.org/pdf/1605.06211.pdf) to find 
 pixels which might be lane markers, and fit the polynomial on them.
