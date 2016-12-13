import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import sys

from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    # defining a 3 channel or 1 channel color to fill the mask with
    # depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    # filling pixels inside the polygon defined by "vertices"
    # with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #cv2.imshow("Region of Interest", mask)
    #cv2.waitKey(0)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_line(img, line, color=[255,0,255], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    cv2.line(img, line[0], line[1], color, thickness)

def hough(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(
                img,
                rho,
                theta,
                threshold,
                np.array([]),
                minLineLength=min_line_len,
                maxLineGap=max_line_gap)
    return lines

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

def split_lines(lines):
    left, right = [], []
    if lines is None:
        return left, right
    for line in lines:
        for x1,y1, x2,y2 in line:
            sp, ep = 0, 0
            if x1 > x2 or (x1 == x2 and y1 < y2):
                sp, ep = (x2, y2), (x1, y1)
            else:
                sp, ep = (x1, y1), (x2, y2)
            if y1 < y2:
                left.append((sp, ep))
            else:
                right.append((sp, ep))
    return left, right

def line_equation(start, end):
    m = (end[1] - start[1]) / (end[0] - start[0])
    b = start[1] - m * start[0]
    return m, b

def intersection_point(line1, line2):
    x = (line2[1] - line1[1]) / (line1[0] - line2[0])
    y = x * line1[0] + line1[1]
    return (x, y)

def stabilize_line(sublines, ulimit, dlimit):
    uline = line_equation((0, ulimit), (1, ulimit))
    dline = line_equation((0, dlimit), (1, dlimit))
    uxes, dxes = [], []
    for start, end in sublines:
        line = line_equation(start, end)
        ux, _ = intersection_point(uline, line)
        dx, _ = intersection_point(dline, line)
        uxes.append(ux)
        dxes.append(dx)
    ux = np.int32(np.median(uxes))
    dx = np.int32(np.median(dxes))
    return ((ux, ulimit), (dx, dlimit))

def draw_stable_lines(img, lines, ulimit, dlimit):
     lefts, rights = split_lines(lines)
     left = stabilize_line(lefts, ulimit, dlimit)
     right = stabilize_line(rights, ulimit, dlimit)
     draw_line(img, left, thickness=10)
     draw_line(img, right, thickness=10)

def process_image(image, kernel_size=5, low_thr=50, high_thr=150):
    """
    `kernel_size` should be odd number 3, 5, 7 and etc.
    `low_thr` is lower threshold bound, which is used by
              Canny edge detector
    `high_thr` is high threshold bound, which is used by
              Canny edge detector
    `
    """
    # Part 1. Convert image to grayscale, blurring it and 
    #         searching edges using Canny algorithm with
    #         specified parameters
    im = grayscale(image)
    # NOTE: To check. Should we apply threshold method before
    #       before continue. 
    #th, im = cv2.threshold(im, 150, 225, cv2.THRESH_BINARY)
    #cv2.imshow("Threashold", im)
    #cv2.waitKey(0)
    im = gaussian_blur(im, kernel_size)
    im = canny(im, low_thr, high_thr)
    
    # Part 2. Cut only interesting regions from the image with edges.
    ylen, xlen, _ = image.shape
    xm, x8th, x20th = xlen // 2, xlen // 8, xlen // 25
    xl, xr = xm - x8th, xm + x8th
    ym = ylen // 2 + ylen // 10
    vertices = np.array([
            # Left polygon
            [(0, ylen),
             (xm - x20th, ym),
             (xm, ym),
             (xl, ylen)],
            # RIght polygon
            [(xr, ylen),
             (xm, ym),
             (xm + x20th, ym),
             (xlen, ylen)]
        ], dtype=np.int32)
    im = region_of_interest(im, vertices)
    
    # Part 3. Enhance edge quality
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    im = cv2.dilate(im, elem, iterations=3)
    #cv2.imshow("Dilation", im)
    #cv2.waitKey(0)

    #elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    im = cv2.erode(im, elem, iterations=2)
    #cv2.imshow("Skeleton", im)
    #cv2.waitKey(0)
    
    # Part 4. Apply Hough algorithm for finding lines. Then draw lines
    #        on color image.
    rho = 1
    theta = np.pi/180
    threshold = 100
    min_len = 40
    max_gap = 50
    lines = hough(im, rho, theta, threshold, min_len, max_gap)
    
    im_with_lines = np.zeros((*im.shape, 3), dtype=np.uint8)
    
    # Part 5. Choose best right and left lane lines and draw them on image
    draw_stable_lines(im_with_lines, lines, ylen, ym + 20)

    # Part 5. Mix source image with drawn lines.
    final_image = weighted_img(im_with_lines, image, 1.0, 0.9)
    return final_image

################################################################
################################################################
################################################################

def read_and_process(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    result = process_image(image)
    cv2.imshow(image_file, result)
    cv2.waitKey(0)

def handle_images(path='.'):
    if os.path.isfile(path):
        read_and_process(path)
    elif os.path.isdir(path):
        for sub in os.listdir(path):
            fimage = "/".join([path, sub])
            read_and_process(fimage)
    else:
        print("Unknown path", file=sys.stderr)
        sys.exit(1)

def handle_video(clip_path):
    output_path = 'output_video.mp4'
    clip = VideoFileClip(clip_path)
    out_clip = clip.fl_image(process_image)
    out_clip.write_videofile(output_path, audio=False)

if __name__ == '__main__':
    first = sys.argv[1]
    if first is not None:
        if first == "video":
            assert(sys.argv[2])
            handle_video(sys.argv[2])
            sys.exit(0)
        handle_images(first)
    else:
        handle_images()
