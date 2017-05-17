from lanedetection.pipeline import LaneDetectPipeline
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from lanedetection.pipeline import LaneDetectPipeline

debug = False

calibration_files = glob.glob("./camera_cal/calibration*.jpg")
lane_detect = LaneDetectPipeline(calibration_files)
if debug is True:
    # undistort an image
    img = mpimg.imread("./camera_cal/calibration1.jpg")
    und_img = lane_detect.undistort(img)
    plt.imshow(und_img)
    plt.show()

# read a video

cap = cv2.VideoCapture('project_video.mp4')

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    outframe = lane_detect.run(frame)
    byteframe = outframe.astype('uint8')
    cv2.imwrite("output/IMG/" + str(i) + ".jpg", byteframe)
    i += 1
    
cap.release()
cv2.destroyAllWindows()


