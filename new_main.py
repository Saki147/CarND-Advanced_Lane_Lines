import LaneDetection as ld
# Import everything needed to edit/save/watch video clips
import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import helperFunction as hf
import plotLane as pl
import pickle



pl.left_line = hf.Line()
pl.right_line = hf.Line()

file = 'video' #'image' #'video'


if file == 'image':
    image =cv2.imread('test_images/test5.jpg') #cv2.imread('test_images/straight_lines1.jpg')  # cv2.imread('test_images/challenge_video_frame.jpg')
    result = pl.plotLane(image)
    plt.imshow(result)
    cv2.imwrite('output_images/try.jpg', result, None)
    plt.show()

elif file == 'video':
    clip1 = VideoFileClip("harder_challenge_video.mp4")
    #clip1 = VideoFileClip("challenge_video.mp4").subclip(0, 15)
    #cap = cv2.VideoCapture("challenge_video.mp4")
    #while (cap.isOpened()):

       # _, frame = cap.read()
       # cv2.imwrite('output_images/challeng_video_frame.jpg',frame, None)
       # plt.imshow(frame)
       # plt.show()
    white_output = 'output_images/hard_challenge_video.mp4'
    white_clip = clip1.fl_image(pl.plotLane) #NOTE: this function expects color images!!
    #%time white_clip.write_videofile(white_output, audio=False)
    white_clip.write_videofile(white_output)

    #display the result video
    white_clip.ipython_display(width="1280", height="720")

    HTML("""
    <video width="1280" height="720" controls>
      <source src="{0}">
    """.format(white_output))

