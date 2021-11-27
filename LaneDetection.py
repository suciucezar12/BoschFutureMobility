import os
import time

import cv2
import keyboard
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from utils import set_save_files, save_dir, ensure_dir, get_save_files

perspective_trapezoid = None
perspective_destination = None
perspective_correction = None
perspective_correction_inverse = None
warp_size = None
orig_size = None
left_fit_avg = None
right_fit_avg = None
MIN_DETECTIONS = 8
MAX_DETECTIONS = 10


def compute_perspective(width, height, pt1, pt2, pt3, pt4):
    global perspective_trapezoid, perspective_destination

    perspective_trapezoid = [(pt1[0], pt1[1]), (pt2[0], pt2[1]), (pt3[0], pt3[1]), (pt4[0], pt4[1])]
    source = np.float32([pt1, pt2, pt3, pt4])
    # widest side on the trapezoid
    x1 = pt1[0]
    x2 = pt4[0]
    # height of the trapezoid
    y1 = pt1[1]
    y2 = pt2[1]
    h = y1 - y2
    # The destination is a rectangle with the height of the trapezoid and the width of the widest side
    destination = np.float32([[x1, h], [x1, 0], [x2, 0], [x2, h]])
    perspective_destination = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]

    global perspective_correction, perspective_correction_inverse
    global warp_size, orig_size

    perspective_correction = cv2.getPerspectiveTransform(source, destination)   # returns the transformation matrix
    perspective_correction_inverse = cv2.getPerspectiveTransform(destination, source)
    warp_size = (width, h)
    orig_size = (width, height)


def warp(image, filename):
    # image = original image (in BGR)
    # filename = where the image is located
    image_perspective = image.copy()
    # draw the lines to visualize the trapezoid and the rectangle perspectives
    cv2.line(image_perspective, perspective_destination[0], perspective_destination[1], (255, 255, 255), 3)
    cv2.line(image_perspective, perspective_destination[1], perspective_destination[2], (255, 255, 255), 3)
    cv2.line(image_perspective, perspective_destination[2], perspective_destination[3], (255, 255, 255), 3)
    cv2.line(image_perspective, perspective_destination[3], perspective_destination[0], (255, 255, 255), 3)

    cv2.line(image_perspective, perspective_trapezoid[0], perspective_trapezoid[1], (0, 192, 0), 3)
    cv2.line(image_perspective, perspective_trapezoid[1], perspective_trapezoid[2], (0, 192, 0), 3)
    cv2.line(image_perspective, perspective_trapezoid[2], perspective_trapezoid[3], (0, 192, 0), 3)
    cv2.line(image_perspective, perspective_trapezoid[3], perspective_trapezoid[0], (0, 192, 0), 3)

    save_dir(image_perspective, "persp_", filename)

    return save_dir(cv2.warpPerspective(image, perspective_correction, warp_size, flags=cv2.INTER_LANCZOS4), "warp_",
                    filename)


def edge_detection(channel, filename):  # channel = original image but only the green(in our case) channel
    edge_x = cv2.Scharr(channel, cv2.CV_64F, 1, 0)  # Edge detection using the Scharr operator
                                                    # result is returned in 64-bit float
                                                    # calculates derivatives in the X axis
    edge_x = np.absolute(edge_x)    # we are not interested on the sign, but only in the fact that there is an edge

    return save_dir(np.uint8(255 * edge_x / np.max(edge_x)), "edge_", filename) # we except the values (after Scharr) range in 8-bit integer: [0, 255]
                                                                        # we convert by using the "simple rule of 3"


def threshold(channel_threshold, channel_edge, filename):   # channel_threshold = image in HLS color space (particular the lightness channel)
                                                            # channel_edge = image after edge detection(=Scharr) (for green channel)
    binary = np.zeros_like(channel_edge)    # a matrix of zeros with size like the channel_edge
    height = binary.shape[0]

    threshold_up = 15
    threshold_down = 50
    threshold_delta = threshold_down = threshold_up

    for y in range(height):
        binary_line = binary[y, :]
        edge_line = channel_edge[y, :]
        threshold_line = threshold_up + threshold_delta * y / height
        binary_line[edge_line >= threshold_line] = 255

    save_dir(binary, "threshold_edge_only", filename)   # binary = image (a copy) after applying the interpolate threshold on the image on green channel only
    save_dir(channel_threshold, "channel_only_", filename)  # the Lightness(HLS) channel of the image

    binary[(channel_threshold >= 140) & (channel_threshold <= 255)] = 255   # combine the previous "binary" with
                                                                            # image in HLS after applying a threshold of 140 on the Lightness channel
    binary_threshold = np.zeros_like(channel_threshold)
    binary_threshold[(channel_threshold >= 140) & (channel_threshold <= 255)] = 255     # the Lightness (HLS) channel after applying T = 140 threshold

    return (save_dir(binary, "threshold_", filename), save_dir(binary_threshold, "threshold_other", filename))


def histogram(image, prefix, filename): # creates the histogram for the image
    partial_image = image[image.shape[0] * 2 // 3:, :]  # Select the bottom part (one third of the image)
    hist = np.sum(partial_image, axis=0)  # sum on each column (count the number of white pixels in that particular column)

    if get_save_files() and filename:
        plt.plot(hist)
        plt.savefig(ensure_dir(filename) + "//" + prefix + filename.replace(".jpg", ".png"))
        plt.clf()

    return hist


class Line:
    lane_indexes = None
    # pixel positions
    x = None
    y = None

    # Fit a second order polynomial to each
    fit = None  # holds the three parameters of the second order polynomial
    # Plotting parameters
    fitx = None  # fitx = X = f(Y)

    # Histogram
    hist_x = None   # the point X = peak in the histogram


# Data collected during the sliding windows phase
class SlideWindow:
    left = Line()
    right = Line()
    hist = None  # hist = is a HistLane object -> it has the two peaks and their confidence

    ploty = None    # Range of [0, height]

    def __init__(self, hist, left_lane_indexes, right_lane_indexes, non_zero_x, non_zero_y):
        self.left.lane_indexes = np.concatenate(left_lane_indexes)
        self.right.lane_indexes = np.concatenate(right_lane_indexes)
        self.left.hist_x = hist.x_left
        self.right.hist_x = hist.x_right
        # Extract left and right positions
        self.left.x = non_zero_x[self.left.lane_indexes]
        self.left.y = non_zero_y[self.left.lane_indexes]
        self.right.x = non_zero_x[self.right.lane_indexes]
        self.right.y = non_zero_y[self.right.lane_indexes]

    def plot_lines(self, image, color_left=(0, 255, 255), color_right=(0, 255, 255)):   # visualize the lanes on the image
        left = []
        right = []
        for i in range(0, len(self.ploty)):
            left.append((self.left.fitx[i], self.ploty[i]))
            right.append((self.right.fitx[i], self.ploty[i]))

        cv2.polylines(image, np.int32([left]), False, color_left)
        cv2.polylines(image, np.int32([right]), False, color_right)

        return image


def slide_window(image, binary_warped, hist, num_windows, filename=None, prefix=None):   # image = image_warped(detect_lanes)
    # hist = is of HistLane object -> it has the two peaks and their confidence
    image_height = binary_warped.shape[0]   #  binary_warped = image_binary_combined(detect_lanes)
    window_height = np.uint8(image_height / num_windows)
    # Indices (e. g. coordinates) of the pixels that are not zero
    non_zero = binary_warped.nonzero()  # nonzero returns the indices where value(pixels) != zero
    non_zero_y = np.array(non_zero[0])
    non_zero_x = np.array(non_zero[1])
    # Current position, to be updated while sliding the window; we start with the ones identified with the histogram
    left_x = hist.x_left
    right_x = hist.x_right
    margin = 80
    # Set minimum number of pixels found to recenter window
    min_pixels = 25
    left_lane_indexes = []  # contains the white pixels within the left lane
    right_lane_indexes = []  # -||- right lane

    out_image = image.copy() if (filename and prefix) else None     # image where we will visualize the sliding windows

    for idx_window in range(num_windows):
        # X range where we expect the left lane to land
        win_x_left_min = left_x - margin
        win_x_left_max = left_x + margin
        # X range where we expect the right lane to land
        win_x_right_min = right_x - margin
        win_x_right_max = right_x + margin
        # Y range that we are analyzing
        win_y_top = image_height - idx_window * window_height
        win_y_bottom = win_y_top - window_height

        # Show the windows
        if (filename and prefix):
            cv2.rectangle(out_image, (win_x_left_min, win_y_bottom), (win_x_left_max, win_y_top), (255, 255, 255), 2)
            cv2.rectangle(out_image, (win_x_right_min, win_y_bottom), (win_x_right_max, win_y_top), (255, 255, 255), 2)

        # Non zero pixels in x and y    -> Select only the white pixels within the current sliding window
        non_zero_left = ((non_zero_y >= win_y_bottom) & (non_zero_y < win_y_top) & (non_zero_x >= win_x_left_min) & (
                non_zero_x < win_x_left_max)).nonzero()[0]
        non_zero_right = ((non_zero_y >= win_y_bottom) & (non_zero_y < win_y_top) & (non_zero_x >= win_x_right_min) & (
                non_zero_x < win_x_right_max)).nonzero()[0]

        left_lane_indexes.append(non_zero_left)     # add them to the left_lane_indexes
        right_lane_indexes.append(non_zero_right)
        # If you found > min_pixels, recenter next window on the mean position
        if len(non_zero_left) > min_pixels:
            left_x = np.int_(np.mean(non_zero_x[non_zero_left]))

        if len(non_zero_right) > min_pixels:
            right_x = np.int_(np.mean(non_zero_x[non_zero_right]))

    # valid = signals that already exists previous frames
    valid, sw = fit_slide_window(binary_warped, hist, left_lane_indexes, right_lane_indexes, non_zero_x, non_zero_y)

    if valid and filename and prefix and get_save_files():
        out_image[non_zero_y[sw.left.lane_indexes], non_zero_x[sw.left.lane_indexes]] = [0, 255, 192]
        out_image[non_zero_y[sw.right.lane_indexes], non_zero_x[sw.right.lane_indexes]] = [0, 255, 192]
        image_plot = sw.plot_lines(out_image)
        cv2.imwrite(ensure_dir(filename) + "//" + prefix + filename, image_plot)

    return valid, sw


def moving_average(prev_average, new_value, beta):
    return beta * prev_average + (1 - beta) * new_value if prev_average is not None else new_value


def fit_slide_window(binary_warped, hist, left_lane_indexes, right_lane_indexes, non_zero_x, non_zero_y):
    # hist = is of HistLane object -> it has the two peaks and their confidence
    #  binary_warped = image_binary_combined(detect_lanes)
    # left/right_lane_indexes = contains the coords of the white pixels found within the slide window
    # non_zero_x/y = coords (grouped by X and Y) of all white pixels
    sw = SlideWindow(hist, left_lane_indexes, right_lane_indexes, non_zero_x, non_zero_y)

    # y coordinates
    sw.ploty = np.array([float(x) for x in range(binary_warped.shape[0])])

    if len(sw.left.y) == 0:  # if exist points (cords of pixels) in the left lane
        return False, sw

    # Fit a second order polynomial to approximate the points
    left_fit = np.polynomial.polynomial.polyfit(sw.left.y, sw.left.x, 2)
    right_fit = np.polynomial.polynomial.polyfit(sw.right.y, sw.right.x, 2)

    global left_fit_avg, right_fit_avg

    left_fit_avg = moving_average(left_fit_avg, left_fit, 0.92)     # Rolling average
    right_fit_avg = moving_average(right_fit_avg, right_fit, 0.92)

    # Generate list of x and y values, using the terms of the polynomial
    # x = Ay^2 + BY + C;
    sw.left.fitx = left_fit_avg[2] * sw.ploty ** 2 + left_fit_avg[1] * sw.ploty + left_fit_avg[0]
    sw.right.fitx = right_fit_avg[2] * sw.ploty ** 2 + right_fit_avg[1] * sw.ploty + right_fit_avg[0]

    return True, sw


class HistLanes:
    def __init__(self, x_left, x_right, left_confidence, right_confidence):
        self.x_left = x_left    # index of peak on left side
        self.x_right = x_right
        self.left_confidence = left_confidence  # value on histogram[x_left] => bigger it is => confident that there is a lane there
        self.right_confidence = right_confidence


def avg_x(hist_lanes):
    return sum([lane.hist_x for lane in hist_lanes]) // len(hist_lanes)


def show_lanes(sw, image_warped, image_original, filename):
    # image_warped = the section of the image (trapezoid) after it's being stretched
    image = image_warped.copy()

    if sw.left:
        fitx_points_warped = np.float32([np.transpose(np.vstack([sw.left.fitx, sw.ploty]))])
        fitx_points = cv2.perspectiveTransform(fitx_points_warped, perspective_correction_inverse)
        left_line_warped = np.int_(fitx_points_warped[0])
        left_line = np.int_(fitx_points[0])
        n = len(left_line)

        for i in range(n - 1):
            cv2.line(image_original, (left_line[i][0], left_line[i][1]), (left_line[i + 1][0], left_line[i + 1][1]),
                     (0, 255, 0), 5)
            cv2.line(image, (left_line_warped[i][0], left_line_warped[i][1]),
                     (left_line_warped[i + 1][0], left_line_warped[i + 1][1]), (0, 255, 0), 5)

    if sw.right:
        fitx_points_warped = np.float32([np.transpose(np.vstack([sw.right.fitx, sw.ploty]))])
        fitx_points = cv2.perspectiveTransform(fitx_points_warped, perspective_correction_inverse)
        right_line_warped = np.int_(fitx_points_warped[0])
        right_line = np.int_(fitx_points[0])

        for i in range(len(right_line) - 1):
            cv2.line(image_original, (right_line[i][0], right_line[i][1]), (right_line[i + 1][0], right_line[i + 1][1]),
                     (0, 0, 255), 5)
            cv2.line(image, (right_line_warped[i][0], right_line_warped[i][1]),
                     (right_line_warped[i + 1][0], right_line_warped[i + 1][1]), (0, 0, 255), 5)

    save_dir(image, "lanes_warped_", filename)

    return save_dir(image_original, "lanes_orig_", filename)


def lanes_full_histogram(histogram):
    size = len(histogram)
    max_index_left = np.argmax(histogram[0:size // 2])  # choose the index of the peak on the left half of the histogram
    max_index_right = np.argmax(histogram[size // 2:]) + size // 2  # same thing on right half

    return HistLanes(max_index_left, max_index_right, histogram[max_index_left], histogram[max_index_right])


def lanes_partial_histogram(histogram, left_lanes, right_lanes, tolerance):
    max_index_left = partial_lane(histogram, left_lanes, tolerance)
    max_index_right = partial_lane(histogram, right_lanes, tolerance)

    return HistLanes(max_index_left, max_index_right, histogram[max_index_left], histogram[max_index_right])


def partial_lane(histogram, lanes_previous_frames, tolerance):
    lane_avg = avg_x(lanes_previous_frames)
    lane_min = max(lane_avg - tolerance, 0)
    lane_max = min(lane_avg + tolerance, len(histogram))

    return np.argmax(histogram[lane_min:lane_max]) + lane_min


def reset():
    global left_fit_avg, right_fit_avg

    left_fit_avg = None
    right_fit_avg = None


def detect_lanes(image_bgr, filename, left_lanes, right_lanes):
    image_warped = warp(image_bgr, filename)    # image_warped = the section of the image (trapezoid) after it's being streched
    image_hls = cv2.cvtColor(image_warped, cv2.COLOR_BGR2HLS).astype(np.float64)
    image_edge = edge_detection(image_warped[:, :, 1], filename)    # index 1 => green channel
                                                                    # image_edge = image after edge detection(=Scharr) (for green channel)
    (image_binary_combined, image_binary_solo) = threshold(image_hls[:, :, 1], image_edge, filename)
    #image_binary_combined = HLS at threshold T = 140 +(combined) Green Channel at interpolate threshold
    #image_binary_solo = HLS at threshold T = 140
    hist = histogram(image_binary_combined, "hist_", filename)  # hist = the histogram for image_binary_combined
    histogram(image_binary_solo, "hist_solo_", filename)    # also plot the histogram for image_binary_solo

    if len(left_lanes) > MIN_DETECTIONS:
        lanes = lanes_partial_histogram(hist, left_lanes, right_lanes, 30)
    else:
        lanes = lanes_full_histogram(hist)
    # ret = signals that already exists previous frames
    ret, sw = slide_window(image_warped, image_binary_combined, lanes, 15, filename, prefix="window_")

    if ret:
        left_lanes.append(deepcopy(sw.left))
        right_lanes.append(deepcopy(sw.right))
    else:
        # In case of problems, use the previous detection
        sw.left = left_lanes[len(left_lanes) - 1]
        sw.right = right_lanes[len(right_lanes) - 1]

        left_lanes.append(sw.left)
        right_lanes.append(sw.right)

    image_lane = show_lanes(sw, image_warped, image_bgr, filename)

    return image_lane


# compute_perspective(1024, 600, [160, 425], [484, 310], [546, 310], [877, 425])
# for filename in os.listdir("test_images_sd"):
#     reset()
#     print(filename)
#     image = cv2.imread("test_images_sd//" + filename)
#     detect_lanes(image, filename, [], [])

set_save_files(False)
# cap = cv2.VideoCapture("video-sd.mp4")
cap = cv2.VideoCapture(0)
time.sleep(0.1)
writer = cv2.VideoWriter('video-sd-out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (640, 480))

n = 0
left_lanes = []
right_lanes = []

# compute_perspective(1024, 600, [140, 415], [445, 315], [596, 315], [867, 415])
compute_perspective(640, 480, [0, 480], [190, 260], [460, 260], [640, 480])
reset()

while True:
    ret, frame = cap.read()
    if not (ret):
        break

    if n % 50 == 0:
        set_save_files(True)
    # set_save_files(False)
    
    frame = detect_lanes(frame, "frame_" + str(n) + ".jpg", left_lanes, right_lanes)

    scale_percent = 50
    w = int(frame.shape[1] * scale_percent / 100)
    h = int(frame.shape[0] * scale_percent / 100)
    dim = (w, h)
    frame_resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Frame", frame_resized)
    cv2.waitKey(1)
    writer.write(frame)

    if (len(left_lanes) > MAX_DETECTIONS):
        left_lanes = left_lanes[1:]
        right_lanes = right_lanes[1:]

    # if (n % 5 == 0):
    #     print("Saving frame", n)
    n = n + 1
    # if keyboard.is_pressed('q'):
    #     break
    set_save_files(False)

writer.release()
cap.release()