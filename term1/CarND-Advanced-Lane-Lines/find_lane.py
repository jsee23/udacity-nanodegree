#/usr/bin/python

import numpy as np
import glob
import cv2
import os
import pickle
from moviepy.editor import VideoFileClip

calibration_file_path   = 'calibration.p'
calibration_img_path    = './output_images/calibration/'

mtx     = []
dist    = []

# define conversions in x and y from pixels space to meters
ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

SMOOTHING_FACTOR = 10

def check_and_create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def find_chessboard_corners(root_path):
    print("camera calibration started...", end=' ')

    images = glob.glob(root_path)

    columns = 9
    rows = 6

    raw_obj_points = np.zeros((rows * columns, 3), np.float32)
    raw_obj_points[:,:2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    check_and_create_directory(calibration_img_path)

    for index, file_name in enumerate(images):
        img = cv2.imread(file_name)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray_img, (columns, rows), None)
        if (ret == True):
            obj_points.append(raw_obj_points)
            img_points.append(corners)

            cv2.drawChessboardCorners(img, (columns, rows), corners, True)
            cv2.imwrite('./output_images/calibration/corners_found' + str(index) + '.jpg', img)

    print("OK")
    return (obj_points, img_points)

class LaneImage:
    def __init__(self):
        self.wasDetected = False
        self.outputDir = ""
        self.outputFile = ""
        self.saveSteps = False
        self.previousLeftFit = []
        self.previousRightFit = []
        self.isVideo = False
        self.curveRadius = None
        self.laneWidth = (0.0, 0.0, 0.0)
        self.distanceToCenter = 0.0
        self.errorCounter = 0
        self.leftFits = []
        self.rightFits = []
        self.bestLeftFit = []
        self.bestRightFit = []

    def undistoreImage(self, img):
        undistored_img = cv2.undistort(img, mtx, dist, None, mtx)
        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_01_undistored.jpg", undistored_img)
        return undistored_img

    def transformPerspective(self, img):
        img_size = (img.shape[1], img.shape[0])
        x_center = img_size[0]/2
        x_offset = 200

        source = np.float32([
            [(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
            [(img_size[0] / 6) + 10, img_size[1]],
            [(img_size[0] * 5 / 6) - 45, img_size[1]],
            [(img_size[0] / 2) + 55, img_size[1] / 2 + 100]
        ])
        destination = np.float32([
            [(img_size[0] / 4), 0],
            [(img_size[0] / 4), img_size[1]],
            [(img_size[0] * 3 / 4), img_size[1]],
            [(img_size[0] * 3 / 4), 0]
        ])

        M = cv2.getPerspectiveTransform(source, destination)
        Minv = cv2.getPerspectiveTransform(destination, source)
        warped_img = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)

        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_02_transformed.jpg", warped_img)
        return warped_img, Minv

    def colorThreshold(self, img, hls_threshold=(150,255), rgb_threshold=(180, 255)):
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls_img[:,:,2]
        r_channel = img[:,:,0]
        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_03_s_channel.jpg", s_channel)
            cv2.imwrite(self.outputDir + self.outputFile + "_04_r_channel.jpg", r_channel)

        r_binary = np.zeros_like(r_channel)
        r_binary[(r_channel >= rgb_threshold[0]) & (r_channel <= rgb_threshold[1])] = 1
        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_06_r_channel_binary.jpg", r_binary)

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= hls_threshold[0]) & (s_channel <= hls_threshold[1])] = 1
        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_05_s_channel_binary.jpg", s_binary)

        color_binary = np.zeros_like(s_channel)
        color_binary[(s_binary == 1) | (r_binary == 1)] = 1
        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_07_combined_color_binary.jpg", color_binary)

        return s_binary

    def absoluteSobelThreshold(self, img, orientation='x', sobel_kernel_size=3, threshold=(20,100)):
        x = 1 if orientation == 'x' else 0
        y = 1 if orientation == 'y' else 0

        sobel = cv2.Sobel(img, cv2.CV_64F, x, y, ksize=sobel_kernel_size)
        sobel_absolute = np.absolute(sobel)
        sobel_scaled = np.uint8(255 * sobel_absolute / np.max(sobel_absolute))

        sobel_binary = np.zeros_like(sobel_scaled)
        sobel_binary[(sobel_scaled >= threshold[0]) & (sobel_scaled <= threshold[1])] = 1
        return sobel_binary

    def magnitudeThreshold(self, img, sobel_kernel_size=3, threshold=(10,100)):
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
        grad_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        scale_factor = np.max(grad_magnitude) / 255
        grad_magnitude = (grad_magnitude / scale_factor).astype(np.uint8)

        magnitude_binary = np.zeros_like(grad_magnitude)
        magnitude_binary[(grad_magnitude >= threshold[0]) & (grad_magnitude <= threshold[1])] = 1
        return magnitude_binary

    def directionThreshold(self, img, sobel_kernel_size=7, threshold=(0.0, 0.3)):
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)

        sobel_absolute_x = np.absolute(sobel_x)
        sobel_absolute_y = np.absolute(sobel_y)
        direction = np.arctan2(sobel_absolute_y, sobel_absolute_x)

        direction_binary = np.zeros_like(direction)
        direction_binary[(direction >= threshold[0]) & (direction <= threshold[1])] = 1
        return direction_binary


    def combinedColorGradientThreshold(self, img):
        color_binary = self.colorThreshold(img)
       
        gradient_img = gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gradient_x = self.absoluteSobelThreshold(gradient_img, orientation='x')
        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_08_gradient_x.jpg", gradient_x)
        
        gradient_y = self.absoluteSobelThreshold(gradient_img, orientation='y')
        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_09_gradient_y.jpg", gradient_y)

        magnitude_binary = self.magnitudeThreshold(gradient_img)
        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_10_magnitude.jpg", magnitude_binary)

        direction_binary = self.directionThreshold(gradient_img)
        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_11_direction.jpg", direction_binary)

        # sum up everything
        gradient_binary = np.zeros_like(direction_binary)
        gradient_binary[((gradient_x == 1) & (gradient_y == 1) | (magnitude_binary == 1) & (direction_binary == 1))] = 1
        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_12_combined_gradient.jpg", gradient_binary)

        combined_binary = np.zeros_like(direction_binary)
        combined_binary[(gradient_binary == 1) | (color_binary == 1)] = 1
        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_13_combined_color_gradient.jpg", combined_binary)

        return combined_binary

    def slidingWindowDetection(self, gradient, use_previous_line=False, window_size=9):
        nonzero = gradient.nonzero()
        nonzero_x = np.array(nonzero[1])
        nonzero_y = np.array(nonzero[0])

        output_img = np.dstack((gradient, gradient, gradient)) * 255

        margin = 100
        min_pixel_per_try = 50
        window_height = np.int(gradient.shape[0] / window_size)

        left_lane_inds = []
        right_lane_inds = []

        if (not use_previous_line) or len(self.previousLeftFit) == 0 or len(self.previousLeftFit) == 0:

            histogram = np.sum(gradient[int(gradient.shape[0]/2):,:], axis=0)
            midpoint = np.int(histogram.shape[0] / 2)

            left_x_base = np.argmax(histogram[:midpoint])
            right_x_base = np.argmax(histogram[midpoint:]) + midpoint

            left_x_current = left_x_base
            right_x_current = right_x_base

            for window in range(window_size):
                window_y_low    = gradient.shape[0] - (window + 1) * window_height
                window_y_high   = gradient.shape[0] - (window * window_height)
                window_x_left_low   = left_x_current - margin
                window_x_left_high  = left_x_current + margin
                window_x_right_low  = right_x_current - margin
                window_x_right_high = right_x_current + margin

                cv2.rectangle(output_img, (window_x_left_low, window_y_low), (window_x_left_high, window_y_high),
                              (0, 255, 0), 2)
                cv2.rectangle(output_img, (window_x_right_low, window_y_low), (window_x_right_high, window_y_high),
                              (0, 255, 0), 2)

                good_left_inds  = ((nonzero_y >= window_y_low) & (nonzero_y < window_y_high) &
                                   (nonzero_x >= window_x_left_low) & (nonzero_x < window_x_left_high)).nonzero()[0]
                good_right_inds = ((nonzero_y <= window_y_low) & (nonzero_y < window_y_high) &
                                   (nonzero_x >= window_x_right_low) & (nonzero_x < window_x_right_high)).nonzero()[0]

                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                if len(good_left_inds) > min_pixel_per_try:
                    left_x_current = np.int(np.mean(nonzero_x[good_left_inds]))
                if len(good_right_inds) > min_pixel_per_try:
                    right_x_current = np.int(np.mean(nonzero_x[good_right_inds]))

        else:
            previous_x_left     = self.bestLeftFit[0] * nonzero_y ** 2 + self.bestLeftFit[1] * nonzero_y + self.bestLeftFit[2]
            window_x_left_low   = previous_x_left - margin
            window_x_left_high  = previous_x_left + margin

            previous_x_right    = self.bestRightFit[0] * nonzero_y ** 2 + self.bestRightFit[1] * nonzero_y + self.bestRightFit[2]
            window_x_right_low  = previous_x_right - margin
            window_x_right_high = previous_x_right + margin

            good_left_inds = ((nonzero_x >= window_x_left_low) & (nonzero_x < window_x_left_high)).nonzero()[0]
            good_right_inds = ((nonzero_x >= window_x_right_low) & (nonzero_x < window_x_right_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        left_x = nonzero_x[left_lane_inds]
        left_y = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]

        for y, x in zip(left_y, left_x):
            cv2.circle(output_img, (x, y), 2, (255, 0, 0), 2)

        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_14_sliding_window.jpg", output_img)

        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        return output_img, left_fit, right_fit, left_lane_inds, right_lane_inds

    def sanityCheck(self, left_fit, right_fit):
        lane_is_ok = True

	# compute lane width
        top_y           = 10
        bottom_y        = img.shape[0]
        top_left_x      = left_fit[0] * bottom_y ** 2 + left_fit[1] * bottom_y + left_fit[2]
        bottom_left_x   = left_fit[0] * bottom_y ** 2 + left_fit[1] * bottom_y + left_fit[2]
        top_right_x     = right_fit[0] * bottom_y ** 2 + right_fit[1] * bottom_y + right_fit[2]
        bottom_right_x  = right_fit[0] * bottom_y ** 2 + right_fit[1] * bottom_y + right_fit[2]

        bottom_lane_width = abs(bottom_left_x - bottom_right_x) * xm_per_pix
        top_lane_width = abs(top_left_x - top_right_x) * xm_per_pix

        if len(self.previousLeftFit) > 0 or len(self.previousRightFit) > 0:
            top_lane_width_diff = self.laneWidth[0] - top_lane_width
            if top_lane_width_diff > 0.07 or top_lane_width_diff < -0.07:
                lane_is_ok = False

        if self.errorCounter >= 3:
            lane_is_ok = True
            self.errorCounter = 0

        if lane_is_ok:
            self.laneWidth = ((top_lane_width + bottom_lane_width) / 2, top_lane_width, bottom_lane_width)
            self.previousLeftFit = left_fit
            self.previousRightFit = right_fit

            self.leftFits.append(left_fit)
            self.rightFits.append(right_fit)
            if len(self.leftFits) > SMOOTHING_FACTOR:
                del self.leftFits[0]
                del self.rightFits[0]
            self.bestLeftFit = np.mean(self.leftFits, axis=0)
            self.bestRightFit = np.mean(self.rightFits, axis=0)
            self.errorCounter = 0
        else:
            self.errorCounter += 1

        return lane_is_ok

    def computeLaneStatistics(self, img, left_lane_inds, right_lane_inds):
        nonzero = img.nonzero()
        nonzero_x = np.array(nonzero[1])
        nonzero_y = np.array(nonzero[0])

        left_x = nonzero_x[left_lane_inds]
        left_y = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]

        plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        y_eval = np.max(plot_y)

        left_fit_converted = np.polyfit(left_y * ym_per_pix, left_x * xm_per_pix, 2)
        right_fit_converted = np.polyfit(right_y * ym_per_pix, right_x * xm_per_pix, 2)

        # compute curvature radius
        if len(left_x) != 0 and len(right_x) != 0:
            left_curv_radius = ((1 + (2 * left_fit_converted[0] * (y_eval * ym_per_pix) + left_fit_converted[1]) ** 2) ** 1.5) / (2 * np.absolute(left_fit_converted[0]))
            right_curv_radius = ((1 + (2 * right_fit_converted[0] * (y_eval * ym_per_pix) + right_fit_converted[1]) ** 2) ** 1.5) / (2 * np.absolute(right_fit_converted[0]))
            self.curveRadius = (left_curv_radius, right_curv_radius)
        else:
            self.curveRadius = (0, 0)

        # compute distance in meters of vehicle center from the line
        car_center = img.shape[1]/2  # we assume the camera is centered in the car
        lane_center = ((self.previousLeftFit[0] * y_eval ** 2 + self.previousLeftFit[1] * y_eval + self.previousLeftFit[2]) +
                       (self.previousRightFit[0] * y_eval ** 2 + self.previousRightFit[1] * y_eval + self.previousRightFit[2])) / 2
        self.distanceToCenter = (lane_center - car_center) * xm_per_pix

    def drawStatistics(self, img):
        font = cv2.FONT_HERSHEY_DUPLEX
        text = 'left radius curvature: ' + '{:04.2f}'.format(self.curveRadius[0]) + 'm'
        cv2.putText(img, text, (50, 70), font, 1, (255,255,255), 2, cv2.LINE_AA)

        text = 'right radius curvature: ' + '{:04.2f}'.format(self.curveRadius[1]) + 'm'
        cv2.putText(img, text, (50, 100), font, 1, (255,255,255), 2, cv2.LINE_AA)

        text = 'lane width: ' + '{:04.2f}'.format(self.laneWidth[1]) + 'm'
        cv2.putText(img, text, (50, 130), font, 1, (255,255,255), 2, cv2.LINE_AA)

        if self.distanceToCenter > 0:
            text = 'vehicle position: {:04.2f}'.format(self.distanceToCenter) + 'm left of center'
        else:
            text = 'vehicle position: {:04.2f}'.format(self.distanceToCenter) + 'm right of center'
        cv2.putText(img, text, (50, 160), font, 1, (255,255,255), 2, cv2.LINE_AA)

        return img

    def drawLinePolygon(self, img, warped_img, minv):
        plot_y_values = np.linspace(0, img.shape[1] - 1, img.shape[1])
        left_fit_x_values = self.previousLeftFit[0] * plot_y_values ** 2 + self.previousLeftFit[1] * plot_y_values + self.previousLeftFit[2]
        right_fit_x_values = self.previousRightFit[0] * plot_y_values ** 2 + self.previousRightFit[1] * plot_y_values + self.previousRightFit[2]

        warped_line_img_zero = np.zeros_like(warped_img).astype(np.uint8)
        warped_line_img = np.dstack((warped_line_img_zero, warped_line_img_zero, warped_line_img_zero))

        points_left = np.array([np.transpose(np.vstack([left_fit_x_values, plot_y_values]))])
        points_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x_values, plot_y_values])))])
        points = np.hstack((points_left, points_right))

        cv2.fillPoly(warped_line_img, np.int_([points]), (0, 255, 0))
        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_15_lines.jpg", warped_line_img)

        unwarped_lane_img = cv2.warpPerspective(warped_line_img, minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        weighted_img = cv2.addWeighted(img, 1, unwarped_lane_img, 0.3, 0)
        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_16_unwarped.jpg", weighted_img)

        return weighted_img

    def processImage(self, img, output_dir = "", file_name = "",  save_steps = False):
        self.outputDir = output_dir
        self.outputFile = file_name
        self.saveSteps = save_steps

        undistored_img = self.undistoreImage(img)
        gradient = self.combinedColorGradientThreshold(undistored_img)
        warped_img, Minv = self.transformPerspective(gradient)

        window_img, left_fit, right_fit, left_lane_inds, right_lane_inds = self.slidingWindowDetection(warped_img, use_previous_line=self.isVideo)
        detected = self.sanityCheck(left_fit, right_fit)
        if detected:
            self.computeLaneStatistics(warped_img, left_lane_inds, right_lane_inds)
        final = self.drawLinePolygon(undistored_img, warped_img, Minv)
        final = self.drawStatistics(final)
        if self.saveSteps:
            cv2.imwrite(self.outputDir + self.outputFile + "_17_final.jpg", final)
        return final


# calibrate camera
if os.path.exists(calibration_file_path):
    print("reading already calculated calibration data...", end=' ')
    calibration_data = pickle.load(open(calibration_file_path, "rb"))
    mtx = calibration_data["mtx"]
    dist = calibration_data["dist"]
    print("OK")
else:
    obj_points, img_points = find_chessboard_corners('camera_cal/calibration*.jpg')
    img = cv2.imread('./test_images/test1.jpg')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img.shape[1::-1], None, None)

    # undistort the chessboard images
    chessboard_img = cv2.imread('./camera_cal/calibration1.jpg')
    undistorted_chessboard_img = cv2.undistort(chessboard_img, mtx, dist, None, mtx)
    cv2.imwrite('./output_images/calibration/undistorted_calibration1.jpg', undistorted_chessboard_img) 

    print("saving calibration data to file...", end=' ')
    calibration_pickle = {}
    calibration_pickle["mtx"]   = mtx
    calibration_pickle["dist"]  = dist
    pickle.dump(calibration_pickle, open(calibration_file_path, "wb"))
    print("OK")

check_and_create_directory('./output_videos')
check_and_create_directory('./output_images')

# process test images
test_img_dir = './test_images/'
for file_path in os.listdir(test_img_dir):
    print("Running pipeline for '" + file_path + "'...", end=' ')
    img = cv2.imread(test_img_dir + file_path)
    lane_img = LaneImage()
    lane_img.processImage(img, './output_images/', file_path, True)
    print("OK")

# process videos
print("Processing video...", end=" ")

lane_img = LaneImage()
lane_img.isVideo = True
video_input = VideoFileClip("./test_videos/project_video.mp4")
processed_video = video_input.fl_image(lane_img.processImage)
processed_video.write_videofile("./output_videos/project_video.mp4")

print("OK")
