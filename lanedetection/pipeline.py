import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob

class Line():
    def __init__(self):
        self.N = 10
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def update(self, current_fit, fitx):
        self.diffs = np.subtract(current_fit, self.current_fit)
        self.current_fit = current_fit
        if len(self.recent_xfitted) == self.N:
            self.recent_xfitted[1:].append(fitx)
        self.bestx = np.average(self.recent_xfitted, axis=0)
        


class LaneDetectPipeline():
    def __init__(self, calibration_files=None):
        # left lane
        self.left = Line()
        # right lane
        self.right = Line()
        # camera matrix 
        self.mat = None
        # distortion coefficient
        self.dist = None
        # perspective transform
        self.warp_mat, self.unwarp_mat = self._get_perspective_transform_matrix()
        if calibration_files != None:
            self.mat, self.dist = self._get_camera_distortion_matrix(calibration_files)

    def _get_perspective_transform_matrix(self):
        src = np.float32([
            [205, 720],
            [1075, 720],
            [590, 450],
            [690, 450]
        ])
        dst = np.float32([
            [205, 720],
            [1075, 720],
            [205, 0],
            [1075, 0]
        ])
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

    def undistort(self, img):
        if self.mat is None:
            raise "camera matrix cant be none"

        if self.dist is None:
            raise "distortion coefficient cant be None"

        return cv2.undistort(img, self.mat, self.dist, None, self.mat)

    def hue_threshold(self, img, thresh=(0,255)):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h_channel = img[:,:,0]
        hue_binary = np.zeros_like(h_channel)
        mask = [(h_channel > thresh[0]) & (h_channel < thresh[1])]
        hue_binary[mask] = 1
        return hue_binary
    
    def lightness_threshold(self, img, thresh=(0,255)):
        ing = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = img[:,:,1]
        l_binary = np.zeros_like(l_channel)
        mask = [(l_channel > thresh[0]) & (l_channel < thresh[1])]
        l_binary[mask] = 1
        return l_binary

    def yellow_thresh(self, img):
        return self.hue_threshold(img, thresh=(18, 25))

    def white_thresh(self, img):
        return self.lightness_threshold(img, thresh=(195,255))

    def red_channel_thresh(self, img, sobel_kernel=3, thresh=(0,255)):
        r_channel = img[:,:,0]
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
        return mag_binary

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
        return dir_binary

    def transform_img(self, und_img):
        yellow = self.yellow_thresh(und_img)
        white = self.white_thresh(und_img)
        combined_colors = np.zeros_like(yellow)
        combined_colors[(yellow == 1) | (white == 1)] = 1
        dirt = self.dir_threshold(und_img, sobel_kernel=13, thresh=(.7, 1.3))
        magt = self.mag_thresh(und_img, sobel_kernel=9, thresh=(30, 100))
        both_dir_mag = np.zeros_like(dirt)
        both_dir_mag[(dirt == 1) & (magt == 1)] = 1
        result = np.zeros_like(dirt)
        result[(combined_colors == 1) | (both_dir_mag == 1)] = 1
        return result

    def draw(self, img, ploty, left_fitx, right_fitx):
        warp_zero = np.zeros_like(img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.unwarp_mat, img.shape[0:2][::-1]) 
        return newwarp
    
    def warp_img(self, img):
        return cv2.warpPerspective(img, self.warp_mat, img.shape[0:2][::-1],flags=cv2.INTER_LINEAR)

    def unwarp_img(self, img):
        return cv2.warpPerspective(img, self.unwarp_mat, img.shape[0:2][::-1])

    def _finding_lines(self, binary_warped):
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
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
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

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        self.left.update(left_fit, left_fitx)
        self.right.update(right_fit, right_fitx)

        return ploty, left_fitx, right_fitx

    def _sliding_window_search():
        window_width = 50 
        window_height = 80 # Break image into 9 vertical layers since image height is 720
        margin = 100 # How much to slide left and right for searching

        def window_mask(width, height, img_ref, center,level):
            output = np.zeros_like(img_ref)
            output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
            return output

        def find_window_centroids(image, window_width, window_height, margin):
            
            window_centroids = [] # Store the (left,right) window centroid positions per level
            window = np.ones(window_width) # Create our window template that we will use for convolutions
            
            # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
            # and then np.convolve the vertical image slice with the window template 
            
            # Sum quarter bottom of image to get slice, could use a different ratio
            l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
            l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
            r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
            r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
            
            # Add what we found for the first layer
            window_centroids.append((l_center,r_center))
            
            # Go through each layer looking for max pixel locations
            for level in range(1,(int)(warped.shape[0]/window_height)):
                # convolve the window into the vertical slice of the image
                image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
                conv_signal = np.convolve(window, image_layer)
                # Find the best left centroid by using past left center as a reference
                # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
                offset = window_width/2
                l_min_index = int(max(l_center+offset-margin,0))
                l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
                l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
                # Find the best right centroid by using past right center as a reference
                r_min_index = int(max(r_center+offset-margin,0))
                r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
                r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
                # Add what we found for that layer
                window_centroids.append((l_center,r_center))

            return window_centroids

        window_centroids = find_window_centroids(warped, window_width, window_height, margin)

        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(warped)
            r_points = np.zeros_like(warped)

            # Go through each level and draw the windows 	
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
                r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
                # Add graphic points from window mask here to total pixels found 
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channel
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
        
        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

    def lane_points(self, img):
        return self._finding_lines(img)

    def find_curve(self, ploty, leftx, rightx):
        y_eval = np.max(ploty)
        # left_curverad = ((1 + (2*self.left.current_fit[0]*y_eval + self.left.current_fit[1])**2)**1.5) / np.absolute(2*self.left.current_fit[0])
        # right_curverad = ((1 + (2*self.right.current_fit[0]*y_eval + self.right.current_fit[1])**2)**1.5) / np.absolute(2*self.right.current_fit[0])
        # print(left_curverad, right_curverad)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')
        self.left.radius_of_curvature = left_curverad
        self.right.radius_of_curvature = right_curverad
        return left_curverad, right_curverad  

    def sanity_check(self):
        if abs(self.left.radius_of_curvature - self.right.radius_of_curvature) < abs(self.left.radius_of_curvature)*0.2:
            print("line detected does not have similar curvature")
        
        
    def run(self, img, debug=False):
        und_img = self.undistort(img)
        if debug is True:
            plt.imshow(und_img)
            plt.title('undistorted image')
            plt.show()

        res_img = self.transform_img(und_img)
        if debug is True:
            plt.imshow(res_img, cmap='gray')
            plt.title('transformed image')
            plt.show()

        warped_img = self.warp_img(res_img)
        if debug is True:
            plt.imshow(warped_img, cmap='gray')
            plt.title('warped image')
            plt.show()
            print(warped_img.shape)

        # detect line within the img
        ypts, left_fitx, right_fitx = self.lane_points(warped_img)
        lane_detected = self.draw(warped_img, ypts, left_fitx, right_fitx)
        if debug is True:
            plt.imshow(lane_detected)
            plt.title('lane detected')
            plt.show()

        # find curvature
        left_curve, right_curve = self.find_curve(ypts, left_fitx, right_fitx)
        if debug is True:
            print(left_curve, 'm', right_curve, 'm')

        self.sanity_check()

        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, lane_detected, 0.3, 0)
        return result



if __name__ == "__main__":
    testimg = mpimg.imread("../test_images/test2.jpg")
    calibration_files = glob.glob("../camera_cal/calibration*.jpg")
    pipeline = LaneDetectPipeline(calibration_files)
    plt.imshow(testimg)
    plt.title('Original')
    plt.show()

    result = pipeline.run(testimg, debug=True)
    plt.imshow(result)
    plt.title('Result')
    plt.show()
    # undistort = pipeline.undistort(testimg)
    # plt.imshow(undistort)
    # plt.title('Undistort')
    # plt.show()

    # tranx = pipeline.transform_img(undistort)
    # plt.imshow(tranx, cmap='gray')
    # plt.title('Transformed')
    # plt.show()
    
    # warped = pipeline.warp_img(tranx)
    # plt.imshow(warped, cmap='gray')
    # plt.title('Warped')
    # plt.show()

    # unwarped = pipeline.unwarp_img(warped)
    # combined = cv2.addWeighted(testimg, 1, unwarped, 0.3, 0)
    # plt.imshow(combined)
    # plt.title('Original')
    # plt.show()
    # yellow_detection = pipeline.yellow_thresh(testimg)
    # plt.imshow(yellow_detection, cmap='gray')
    # plt.title('yellow lane')
    # plt.show()

    # white_detection = pipeline.white_thresh(testimg)
    # plt.imshow(white_detection, cmap='gray')
    # plt.title('white lane')
    # plt.show()

    # white_and_yellow = np.zeros_like(white_detection)
    # white_and_yellow[(yellow_detection == 1) | (white_detection == 1)] = 1
    # plt.imshow(white_and_yellow, cmap='gray')
    # plt.title('white and yellow lane')
    # plt.show()

    # sobelx = pipeline.abs_sobel_thresh(testimg, sobel_kernel=9, thresh=(20,150))
    # plt.imshow(sobelx, cmap='gray')
    # plt.title('sobelx')
    # plt.show()

    # sobely = pipeline.abs_sobel_thresh(testimg, orient='y', sobel_kernel=9, thresh=(20,150))
    # plt.imshow(sobely, cmap='gray')
    # plt.title('sobely')
    # plt.show()

    # direction = pipeline.dir_threshold(testimg, sobel_kernel=13, thresh=(.7, 1.3))
    # plt.imshow(direction, cmap='gray')
    # plt.title('direction')
    # plt.show()

    # magnitude = pipeline.mag_thresh(testimg, sobel_kernel=9, thresh=(30, 100))
    # plt.imshow(magnitude, cmap='gray')
    # plt.title('magnitude')
    # plt.show()

    # basic = np.zeros_like(magnitude)
    # basic[(white_and_yellow == 1) | ((magnitude == 1) & (direction == 1))] = 1
    # plt.imshow(basic, cmap='gray')
    # plt.title('Basic Implementation')
    # plt.show()

    # # for better yellow line detection
    # rchan = pipeline.red_channel_thresh(testimg, sobel_kernel=7, thresh=(10, 100))
    # plt.imshow(rchan, cmap='gray')
    # plt.title('red channel')
    # plt.show()

    # combined = np.zeros_like(rchan)
    # mask = [(direction == 1) & (rchan == 1)]
    # combined[mask] = 1
    # plt.imshow(combined, cmap='gray')
    # plt.title('Yellow Detection')
    # plt.show()

    # white = np.zeros_like(rchan)
    # mask = [(rchan == 1) & (sobelx == 1) & (magnitude == 1)]
    # white[mask] = 1
    # plt.imshow(white, cmap='gray')
    # plt.title('White Detection')
    # plt.show()


    # final = np.zeros_like(white)
    # mask = [(white == 1) | (combined == 1)]
    # mask = [(another == 1) | (white == 1)]
    # final[mask] = 1
    # plt.imshow(final, cmap='gray')
    # plt.title('Final')
    # plt.show()