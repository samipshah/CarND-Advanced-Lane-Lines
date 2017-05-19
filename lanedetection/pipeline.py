import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import os
import math
from collections import deque

# with every image we detect a left and a right line
class Line():
    def __init__(self):
        # x points
        self.allx = None
        # y points
        self.ally = None
        # fit equation
        self.fit = None
        # curvature
        self.curvature = None
        # meters per pexel in y dimension
        self.ym_per_pix = 30./720
        # meters per pixel in x dimension
        self.xm_per_pix = 3.7/500
        # slope
        self.slope = None

    def _update_curvature(self):
        y_eval = np.max(self.ally)
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.ally*self.ym_per_pix, self.allx*self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        numerator = ((1 + (2*fit_cr[0]*y_eval*self.ym_per_pix + fit_cr[1])**2)**1.5)
        self.curvature = (numerator / np.absolute(2*fit_cr[0]))

    def _update_slope(self):
        # 2Ay + b
        midpoint = int(len(self.ally)/2)
        y_eval = self.ally[midpoint]
        slope = (2*self.fit[0]*y_eval) + self.fit[1]
        self.slope = slope

    def _update_fit(self):
        self.fit = np.polyfit(self.ally, self.allx, 2)

    def basex(self):
        yeval = np.max(self.ally).astype(np.uint8)
        return self.allx[yeval]

    def update_xy(self, allx, ally):
        self.allx = allx
        self.ally = ally

        self._update_fit()
        self._update_curvature()
        self._update_slope()

    def calculate_x(self, nonzeroy):
        return self.fit[0]*(nonzeroy**2) + self.fit[1]*nonzeroy + self.fit[2]

class LineAggregate():
    def __init__(self, N):
        self.N = N
        self.aggregate = deque(maxlen=N)
        self._best_line_count = 0

    def add_line(self, line):
        self.aggregate.append(line)

        # if len(self.aggregate) == self.N:
        #     print(line)
        #     self.aggregate[1:].append(line)
        #     print(self.aggregate)
        # else:
        #     self.aggregate.append(line)

    def best_line(self, ysize=720, debug=False):
        self._best_line_count += 1
        # average last N allx
        if len(self.aggregate) == 0:
            if debug is True:
                print("aggregate zero")
            return None

        recent_xfitted = np.array([x.allx for x in self.aggregate])
        avg_allx = np.mean(recent_xfitted, axis=0)
        ally = np.linspace(0, ysize-1, ysize)
        line = Line()
        line.update_xy(avg_allx, ally)
        if debug is True:
            print("returned best_line: ", self._best_line_count)
        # print(line.fit[0])
        return line

class Lane():
    def __init__(self):
        # right line
        self.right = Line()
        # left line
        self.left = Line()
        # right line aggregate
        self.right_agg = LineAggregate(5)
        # left line aggregate
        self.left_agg = LineAggregate(5)
        # detected
        self.detected = False
        # sanity check failure accepted
        self.failure_count = 0

    def _sanity_check(self, left, right):
        # compare with existing curvature
        # if self.left.curvature is not None:
        #     percent = abs(self.left.curvature - left.curvature)/self.left.curvature
        #     if percent > 2:
        #         print("Left curvature diff: ", percent)
        #         return False

        # # compare with existing right curvature
        # if self.right.curvature is not None:
        #     percent = abs(self.right.curvature - right.curvature)/self.right.curvature
        #     if percent > 2:
        #         print("Right curvature diff: ", percent)
        #         return False

        # curvature_diff = abs(left.curvature - right.curvature)
        # # check curvature is more or less similar of both lanes
        # if (curvature_diff/np.min([left.curvature, right.curvature])) > 2:
        #     print("Distance: ", curvature_diff, " left:", left.curvature, " right:", right.curvature)
        #     return False

        # compare with lane width
        lanewidth = abs(left.basex() - right.basex())*self.right.xm_per_pix
        # 25% error in m width accepted
        if lanewidth > 5 or lanewidth < 2.5:
            print("Lanewidth: ", lanewidth)
            return False

        # roughly parallel
        div = abs(left.slope/right.slope)
        if div > 2 or div < .5:
            print("Div: ", div, " left-slope:", left.slope, " right-slope:", right.slope)
            return False

        # if offset from center is way out reject the frame
        offset = self.get_offset(1280, left=left, right=right)
        if offset > 1.5:
            print("Offset: ", offset)
            return False

        return True

    def line_detected_current_frame(self, left, right):
        # make a sanity check on the line if line passes
        if self._sanity_check(left, right):
            # set them as right and left and add it to aggregate
            self.right_agg.add_line(right)
            self.left_agg.add_line(left)
            self.detected = True
            self.failure_count = 0
        else:
            self.failure_count += 1
            if self.failure_count > 5:
                self.detected = False

        self.left = self.left_agg.best_line()
        if self.left is None:
            self.left = left
        self.right = self.right_agg.best_line()
        if self.right is None:
            self.right = right
        return self.left, self.right

    def get_curvature(self):
        return np.mean([self.right.curvature + self.left.curvature])

    def get_offset(self, xshape, left=None, right=None, debug=False):
        if left is None:
            left = self.left

        if right is None:
            right = self.right

        if debug is True:
            print("mean: ", np.mean([right.basex(), left.basex()]))
            print("midpoint: ", xshape/2)
        return abs((xshape/2) - np.mean([right.basex(), left.basex()]))*right.xm_per_pix

    def get_lanewidth(self, debug=False):
        if debug is True:
            print("right basex:", self.right.basex())
            print("left basex:", self.left.basex())
        return abs(self.right.basex() - self.left.basex())*self.right.xm_per_pix

class LaneDetectPipeline():
    def __init__(self, calibration_files=None):
        # right, left line
        self.lane = Lane()
        # camera matrix
        self.mat = None
        # distortion coefficient
        self.dist = None
        # perspective transform
        self.warp_mat, self.unwarp_mat = self._get_perspective_transform_matrix()
        camera_file = "cam.npz"
        if os.path.exists(camera_file):
            npload = np.load(camera_file)
            self.mat = npload['mat']
            self.dist = npload['dist']
        else:
            if calibration_files != None:
                self.mat, self.dist = self._get_camera_distortion_matrix(calibration_files)
                np.savez(camera_file, mat=self.mat, dist=self.dist)


    def _get_perspective_transform_matrix(self):
        src = np.float32(
            [[200, 720],
            [1100, 720],
            [595, 450],
            [685, 450]])
        dst = np.float32(
            [[400, 720],
            [900, 720],
            [400, 0],
            [900, 0]])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    def _get_camera_distortion_matrix(self, image_paths, pattern=(9,6), debug=False):
        imgpoints = []
        objpoints = []
        imgsize = None
        objp = np.zeros((pattern[1]*pattern[0],3), dtype=np.float32)
        objp[:,:2] = np.mgrid[0:pattern[0],0:pattern[1]].T.reshape(-1,2)
        if debug is True:
            print(objp)
        for filepath in image_paths:
            # read image
            img = mpimg.imread(filepath)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if imgsize is None:
                imgsize = gray.shape[::-1]
            
            ret, corners = cv2.findChessboardCorners(img, pattern)
            if ret is True:
                objpoints.append(objp)
                imgpoints.append(corners)
                if debug is True:
                    cv2.drawChessboardCorners(img, pattern, corners, ret)
                    plt.imshow(img)
                    plt.show()

        ret, mat, dist, rvects, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgsize, None, None)
        return mat, dist

    def undistort(self, img, debug=False):
        if self.mat is None:
            raise "camera matrix cant be none"

        if self.dist is None:
            raise "distortion coefficient cant be None"

        return cv2.undistort(img, self.mat, self.dist, None, self.mat)

    def sat_threshold(self, img, thresh=(0, 255)):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = img[:, :, 2]
        s_binary = np.zeros_like(s_channel)
        mask = [(s_channel > thresh[0]) & (s_channel < thresh[1])]
        s_binary[mask] = 1
        return s_binary

    def hue_threshold(self, img, thresh=(0,255)):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h_channel = img[:, :, 0]
        hue_binary = np.zeros_like(h_channel)
        mask = [(h_channel > thresh[0]) & (h_channel < thresh[1])]
        hue_binary[mask] = 1
        return hue_binary

    def lightness_threshold(self, img, thresh=(0, 255)):
        ing = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = img[:, :, 1]
        l_binary = np.zeros_like(l_channel)
        mask = [(l_channel > thresh[0]) & (l_channel < thresh[1])]
        l_binary[mask] = 1
        return l_binary

    def yellow_thresh(self, img, debug=False):
        huedetect = self.hue_threshold(img, thresh=(18, 25))
        satdetect = self.sat_threshold(img, thresh=(100, 255))
        yellowdetect = huedetect & satdetect
        if debug is True:
            plt.imshow(yellowdetect, cmap='gray')
            plt.title('Yellow Detect')
            plt.show()
        return yellowdetect

    def value_threshold(self, img, thresh=(0,255)):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        v_binary = np.zeros_like(v_channel)
        mask = [(v_channel > thresh[0]) & (v_channel < thresh[1]) & (h_channel == 0) & (s_channel == 0)]
        v_binary[mask] = 1
        return v_binary

    def white_thresh(self, img, debug=False):
        # whitedetect = self.value_threshold(img, thresh=(70,255))
        whitedetect = self.lightness_threshold(img, thresh=(195, 255))
        if debug is True:
            plt.imshow(whitedetect, cmap='gray')
            plt.title('White Detect')
            plt.show()
        return whitedetect

    def red_channel_thresh(self, img, sobel_kernel=3, thresh=(0, 255)):
        r_channel = img[:, :, 0]
        r_sobel = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        abs_sobel = np.absolute(r_sobel)
        min = np.min(abs_sobel)
        max = np.max(abs_sobel)
        scaled_abs_sobel = np.array((abs_sobel - min)*255/(max - min), dtype=np.uint8)

        grad_binary = np.zeros(scaled_abs_sobel.shape)
        grad_binary[(scaled_abs_sobel > thresh[0]) & (scaled_abs_sobel < thresh[1])] = 1
        return grad_binary

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        else:
            sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

        abs_sobel = np.absolute(sobel)
        min = np.min(abs_sobel)
        max = np.max(abs_sobel)
        scaled_abs_sobel = np.array((abs_sobel - min)*255/(max - min), dtype=np.uint8)

        grad_binary = np.zeros(scaled_abs_sobel.shape)
        grad_binary[(scaled_abs_sobel > thresh[0]) & (scaled_abs_sobel < thresh[1])] = 1
        return grad_binary

    def mag_thresh(self, img, sobel_kernel=3, thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        mag = np.sqrt(np.square(gradx) + np.square(grady))

        minmag = np.min(mag)
        maxmag = np.max(mag)
        scaled_mag = np.array(((mag-minmag)/(maxmag-minmag)*255), dtype=np.uint8)

        mag_binary = np.zeros_like(scaled_mag)
        mask = ((scaled_mag > thresh[0]) & (scaled_mag < thresh[1]))
        mag_binary[mask] = 1
        return mag_binary.astype(np.uint8)

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        absgradx = np.absolute(gradx)
        absgrady = np.absolute(grady)
        direction = np.arctan2(absgrady, absgradx)
        mask = [(direction > thresh[0]) & (direction < thresh[1])]
        dir_binary = np.zeros_like(direction)
        dir_binary[mask] = 1
        return dir_binary.astype(np.uint8)

    def transform_img(self, und_img, debug=False):
        yellow = self.yellow_thresh(und_img)
        if debug is True:
            plt.imshow(yellow, cmap='gray')
            plt.title('yellow lane')
            plt.show()

        white = self.white_thresh(und_img)
        if debug is True:
            plt.imshow(white, cmap='gray')
            plt.title('white lane')
            plt.show()
        
        combined_colors = white | yellow
        if debug is True:
            plt.imshow(combined_colors, cmap='gray')
            plt.title('yellow and white lanes')
            plt.show()

        dirt = self.dir_threshold(und_img, sobel_kernel=15, thresh=(.7, 1.3))
        if debug is True:
            plt.imshow(dirt, cmap='gray')
            plt.title('direction')
            plt.show()

        magt = self.mag_thresh(und_img, sobel_kernel=3, thresh=(50, 255))
        if debug is True:
            plt.imshow(magt, cmap='gray')
            plt.title('magnitude')
            plt.show()

        both_dir_mag = dirt & magt
        if debug is True:
            plt.imshow(both_dir_mag, cmap='gray')
            plt.title('both direction and magnitude')
            plt.show()

        result = combined_colors | both_dir_mag
        # result = combined_colors & dirt
        if debug is True:
            plt.imshow(result, cmap='gray')
            plt.title('result')
            plt.show()
        return result
        
    def draw(self, img, left, right, debug=False):
        warp_zero = np.zeros_like(img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left.allx, left.ally]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right.allx, right.ally])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.unwarp_mat, img.shape[0:2][::-1])
        return newwarp
    
    def warp_img(self, img, debug=False):
        warped_img = cv2.warpPerspective(img, self.warp_mat, img.shape[0:2][::-1],flags=cv2.INTER_LINEAR)
        binarywarped_img = np.copy(warped_img)
        binarywarped_img[warped_img.nonzero()] = 1
        return binarywarped_img

    def unwarp_img(self, img):
        return cv2.warpPerspective(img, self.unwarp_mat, img.shape[0:2][::-1])

    def _finding_lines(self, binary_warped):

        def _line_already_detected():
            # Assume you now have a new warped binary image
            # from the next frame of video (also called "binary_warped")
            # It's now much easier to find line pixels!
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            margin = 100
            calculated_left_pos = self.lane.left.calculate_x(nonzeroy)
            calculated_right_pos = self.lane.right.calculate_x(nonzeroy)
            left_lane_inds = ((nonzerox > (calculated_left_pos - margin)) & (nonzerox < (calculated_left_pos + margin)))
            right_lane_inds = ((nonzerox > (calculated_right_pos - margin)) & (nonzerox < (calculated_right_pos + margin)))

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            left = Line()
            left.update_xy(left_fitx, ploty)
            right = Line()
            right.update_xy(right_fitx, ploty)
            return left, right

        def _line_not_detected():
            # Assuming you have created a warped binary image called "binary_warped"
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            nwindows = 9
            # Set height of windows
            window_height = np.int(binary_warped.shape[0]/nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Set the width of the windows +/- margin
            margin = 100
            # Set minimum number of pixels found to recenter window
            minpix = 50
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2) 
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2) 
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            left = Line()
            left.update_xy(left_fitx, ploty)
            right = Line()
            right.update_xy(right_fitx, ploty)
            return left, right

        if self.lane.detected:
            left, right = _line_already_detected()
        else:
            left, right = _line_not_detected()

        return left, right

    def find_lines(self, img, debug=False):
        return self._finding_lines(img)

    def write_on_image(self, img, text, location=(719, 500)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, location, font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def run(self, img, debug=False):
        und_img = self.undistort(img)
        if debug is True:
            plt.imshow(und_img)
            plt.title('undistorted image')
            plt.show()

        res_img = self.transform_img(und_img, debug)
        if debug is True:
            plt.imshow(res_img, cmap='gray')
            plt.title('transformed image')
            plt.show()

        warped_img = self.warp_img(res_img, debug)
        if debug is True:
            plt.imshow(warped_img, cmap='gray')
            plt.title('warped image')
            plt.show()
            print(warped_img.shape)

        # detect line within the img
        left, right = self.find_lines(warped_img, debug)
        if debug is True:
            #draw line on top of warped_img
            temp_warpimg = np.copy(warped_img)
            tempcolor_warp = (np.dstack((temp_warpimg, temp_warpimg, temp_warpimg))*255).astype(np.uint8)

            pts_left = np.array([np.transpose(np.vstack([left.allx, left.ally]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right.allx, right.ally])))])
            # pts = np.hstack((pts_left, pts_right)
            pts_left = pts_left.reshape((-1, 1, 2))
            pts_right = pts_right.reshape((-1, 1, 2))
            cv2.polylines(tempcolor_warp, np.int_([pts_left]), False, (0, 0, 255), thickness=5)
            cv2.polylines(tempcolor_warp, np.int_([pts_right]), False, (255, 0, 0), thickness=5)
            plt.imshow(tempcolor_warp)
            plt.title('detected lines')
            plt.show()

        left, right = self.lane.line_detected_current_frame(left, right)
        lane_detected = self.draw(warped_img, left, right, debug)
        if debug is True:
            plt.imshow(lane_detected)
            plt.title('lane detected')
            plt.show()

        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, lane_detected, 0.3, 0)

        # add text on image
        curve_text = "curvature:" + '{:.3f}'.format(self.lane.get_curvature())
        self.write_on_image(result, curve_text, location=(10, 50))
        road_width = "lane width:" + '{:.3f}'.format(self.lane.get_lanewidth())
        self.write_on_image(result, road_width, location=(10,150))
        offset_text = "offset from center:" + '{:.3f}'.format(self.lane.get_offset(result.shape[1]))
        self.write_on_image(result, offset_text, location=(10, 100))

        return result



if __name__ == "__main__":
    files = glob.glob("../test_images/*.jpg")
    calibration_files = glob.glob("../camera_cal/calibration*.jpg")
    pipeline = LaneDetectPipeline(calibration_files)
    files = ["../output/IMG/570.jpg"]

    for filename in files:
        testimg = mpimg.imread(filename)
        plt.imshow(testimg)
        plt.title('Original ' + filename)
        plt.show()

        result = pipeline.run(testimg, debug=True)
        plt.imshow(result)
        plt.title('Result')
        plt.show()
