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

[image1]: ./output/markdown/undistorted.png "Undistorted"
[image2]: ./output/markdown/transformed.png "Road Transformed"
[image3]: ./output/markdown/warped.png "Warp Example"
[image4]: ./output/markdown/detected.png "Detected Example"
[image5]: ./output/markdown/unwarped_lane.png "Lane Unwarped"
[image6]: ./output/markdown/final.png "Output"
[video1]: ./output/IMG.mp4 "Video"
[image7]: ./output/markdown/original.png "Original"
[image8]: ./output/markdown/original1.png "Original 1"
[image9]: ./output/markdown/original1_undistorted.png "Original 1 Undistorted"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Code is in  `lanedetection/pipeline.py` class  `LaneDetectPipeline` initialization method. I store the camera matrix and distortion coefficient in a file so that I can just read them instead of computing for every time I run pipeline

Basically to calibrate for camera's distortion, we have to have a few images of a chessboard from that camera from different angles. Thats because corners of a chessboards are easier to find using opencv function and moreover we can also specify each of the camera's (x,y,z) coordinate with z=0 for 2D image and x,y numeric numbers representing number of columns and rows in chessboard respectively. Those numeric coordinates will be called `objpoints` and for each image whatever coordinates `cv2.findChessboardCorners` finds will be `imagepoints`

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Following is distorted image and subsequent to that is an undistorted image
![Original][image8]
![Undistorted][image9]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

`lanedetection/pipeline.py` class `LaneDetectPipeline` implements different color transforms, sobel gradient, direction and magnitude thresholds. Function `transform_img` basically makes use of all these cv2 functions to convert to an image which detects lanes pretty sharply. 

![Transformed Image][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

`lanedetection/pipeline.py` class `LaneDetectPipeline` has a method `get_perspective_transform_matrix` which basically returns warp matrix and unwarp matrix. Points used to find those matrix are as given below. And these points I derived from applying perspective transform on straight line example image and making sure perspective transformed image has parallel lines

```
        src = np.float32(
            [[200, 720],
            [1100, 720],
            [595, 450],
            [685, 450]])
        dst = np.float32(
            [[300, 720],
            [980, 720],
            [300, 0],
            [980, 0]])
```

Here's a warped image of the straight road

![Warped Image][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

`_finding_lines` function has logic for finding a lane line from the perspective transformed image. It has two branches, one where if we have detected lines in a previous frame. If that is the case we try to find line within +/- 100 pixels from the line in previous frame. If we havent detected a line in previous frame we basically detect line by using histogram and windowing algorithm. Basically dividing image in 9 horizontal windows. Starting from bottom window we take the histogram of black/white pixel and we find two peaks in a histogram and call them part of left and right lane. We keep on doing this for all the windows to figure out lane points in each window. Then we fit a polynomial through all those points. And once we have a polynomial fit for each of the line calculate x points for all the y points. Example image is shown below where polynomial fit is shown as a red and blue line for left and right respectively 

![Detected Image][image4]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

radius of curvature and position of the vehicle in `get_curvature` and `get_offset` functions of Lane class.  Curvature I find using a mean of right and left curvature. Each individual curvature is calculated in `Line` class. Similarly offset is calculated by taking a mean of base points for both the lines and subtracting it from midpoint of the frame and multiplying it with meters/pixels to convert it to meters unit.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

 You can see those values calculated in output image as well as lane region marked as a green polynomial on top of the image.

![Output][image6]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output/IMG.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One major problem I faced is detecting white lanes very reliably. I used hls colorspace and mostly use light value to detect white lane however under very bright light or direct sunlight it makes many other parts of the frame detected as white. Other issue I faced is reliably coming up with sanity check. Current values are heuristically driven, tried many a times and what seems like a threshold to ignore or accept. I need to make changes to some of the perspective transform related code to have sharper lanes in the image. Currently warped image has lanes which are very thick and sometimes minor detection in actual images appear as a big splash throwing off the polynomial fit to a very reliable line. And because of that curvature seems to fluctuate a lot. Other approach I would like to try is storing and averaging polynomial equation average instead of current approach of storing recent xpoints for both lines and averaging them and finding a polynomial fit there. Other thing I have struggled with is straight line curvature. For some reason curvature difference of left and right is really huge when the lines are straight.