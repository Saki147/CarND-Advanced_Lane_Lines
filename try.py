import numpy as np


a = np.array([1, 2, 3, 4, 5 ])
b =np.array([2, 3, 4, 5, 6])
c = np.dstack((a,b))
c=np.flip(c, axis =1)
print(c)

ignore_mask_color = (255,) * 3
print(ignore_mask_color)

m = np.array([[(100,720), (565,460), (725, 460), (1180, 720)]], dtype=np.int32)
n= np.copy(m)*2
n = np.flip(n, axis =1)
vertices = np.hstack((m,n))



print(vertices)

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

left_line = hf.Line()
right_line = hf.Line()
print(left_line.startx)

def jj():
    print(left_line.startx)
    aa(left_line)

def aa(left_line):
    print(left_line.detected)

jj()

image =cv2.imread('output_images/challenge_video_frame.jpg')


