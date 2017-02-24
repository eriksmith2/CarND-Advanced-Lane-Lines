# Advanced Lane Finding Project

### Project Goals:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: /output_images/Undistorted.png "Distortion Correction"
[image2]: /output_images/Input_Image.png "Pipeline Input"
[image3]: /output_images/Input_Image.png "Undistorted"
[image4]: /output_images/Thresholding.png "Thresholding Example"
[image5]: /output_images/Warped.png "Birds-Eye-View Transform"
[image6]: /output_images/histogram.png "Histogram"
[image7]: /output_images/Output.png "Pipeline Output"
[video1]: /output_images/project.mp4 "Video"

---

## Project:

### Camera Calibration

The first step of this project was to calibrate for the camera used for video and image capture in order to create distortion-corrected images. Using distortion-corrected images ensures that the camera and lens used to capture the images do not skew the results of the image processing that happens later.
The following code was used to compute the camera calibration coefficients for this lens/camera combination.  

```python
images = glob.glob('camera_cal/calibration*.jpg')
nx = 9
ny = 6

objpoints = []
imgpoints = []

objp = np.zeros((nx*ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)


# Read in an image
for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
    else:
        print('error: corners not found in ' + fname + ' - image not used')

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

```
I used an array called `objpoints` to represent the expected coordinates of the corners on a chessboard calibration image after it is undistorted and a second array, named `imgpoints`, to hold the corner locations detected in the distorted calibration images. The `imgpoints` array was populated using OpenCV's `cv2.findChessboardCorners` function. `findChessboardCorners` takes in each calibration image and is told to find a 9x6 grid of corners on the chessboard - the coordinates for each detected corner is appended to `imgpoints`. Any images where a 9x6 grid cannot be found are discarded.

With the `objpoints` and `imgpoints` arrays, I was able to calculate the camera calibration (`mtx`) and distortion coefficients (`dist`) using the `cv2.calibrateCamera` function. By using the `cv2.undistort()` function and the previously calculated coefficients, images can be undistorted:
![alt text][image1]

### Pipeline (single images)

For testing my pipeline, I found a portion of video that looked particularly difficult and saved out the frame as a test image. I picked the frame at 23 seconds into the project video:

![alt text][image2]

This image is a good test image because it lacks the distinct color difference between the pavement and the lane line paint that is typical of the sections with darker pavement. This particular frame should be relatively difficult to fit lines to, so if the pipeline works on this image, it should also work on images where the gradient is more pronounced.

The first step of the pipeline is, as described above, to undistort the input image such that the particular camera and lens used to capture the image will not impact the pipeline results. Assuming the input image is loaded into the variable `img`, undistorted image can be generated using `cv2.undistort` with the input image and the previously calculated camera calibration and distortion coefficients:

```python
undistorted = cv2.undistort(img, mtx, dist, None, mtx)
```

This results in the undistorted image shown below:

![alt text][image3]

Next, I used various methods of image processing and thresholding to try and isolate the lane lines from the superfluous image data. The idea of thresholding is to create a binary image where any pixel in an input image with a value between a maximum and minimum threshold is set to 1. All pixels outside the threshold range are set to 0. I used the following code to convert images to thresholded binary:

```python
def threshold(image, thresh_min, thresh_max):
    # convert image to binary with given min and max threshold
    binary = np.zeros_like(image)
    binary[(image >= thresh_min) & (image <= thresh_max)] = 1
    return binary
```

 The first image processing method I performed was using the Sobel operator to find the gradient derivatives in the x and y directions across the image. I then take the absolute value of that derivative and threshold based on that value.

 ```python
 def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh_min=0, thresh_max=255): #code from Udacity lectures
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output = threshold(scaled_sobel, thresh_min, thresh_max)
    # Return the result
    return binary_output
```
I also made use of isolating color channels and thresholding based on those values. Based on lots of experimentation, I found that the red channel of RGB colorspace and the S channel of HLS colorspace did the best job of isolating the lane lines. I made two functions to isolate and threshold these channels:

```python
def s_chan(img, thresh_min=170, thresh_max=250):
    # Convert image to HLS colour space and return the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    output = threshold(s_channel, thresh_min, thresh_max)
    return output

def r_chan(img, thresh_min=170, thresh_max=250):
    # return the R channel
    r_channel = img[:,:,2]
    output = threshold(r_channel, thresh_min, thresh_max)
    return output
```

Finally, I combined all of these thresholded images into one by filtering for areas where the r-channel or s-channel binary images had a value of 1 or areas where both the x and y gradient thrsholds were 1.

```python
def combined_sobel(image):
    #combine the various thresholded images through filtering
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=9, thresh_min=30, thresh_max=200)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=9, thresh_min=30, thresh_max=200)
    s_channel = s_chan(image, thresh_min=170, thresh_max=255)
    r_channel = r_chan(image, thresh_min=220, thresh_max=255)

    combined = np.zeros_like(gradx)
    combined[((s_channel == 1) | (r_channel == 1)) | ((gradx == 1) & (grady == 1))] = 1

    return combined
```
Below are samples of the various images created during the image processing pipeline:

![alt text][image4]

The next step in the pipeline is to warp the image into a birds-eye-view. This birds-eye-view image is easier to work with when trying to detect, measure, and draw the lane lines. The warping of the image is done by choosing 4 source points that lie along the lane lines in a straight image of road and performing a perspective transform to destination points that create the birds-eye-view image. I chose the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 200, 700      | 200, 700      |
| 600, 450      | 200, 0        |
| 700, 450      | 950, 0        |
| 1080, 700     | 950, 700      |

I made a `warp` function to perform the perspective transform and return both the warped image and the inverse matrix required to unwarp the image again:

```python
def warp(img):
    #transform image to birds-eye-view
    source = [(200, 700), (600, 450), (700, 450), (1080, 700)]    
    dest = [(200, 700), (200, 0), (950, 0), (950, 700)]

    img_size = (img.shape[1], img.shape[0]) # Grab the image shape
    src = np.float32(source)
    dst = np.float32(dest)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)
    Minv = np.linalg.inv(M)

    return warped, Minv
```

With this function I was able to transform input images to a birds-eye-view:
![alt text][image5]

The next task was to identify the lane line pixels and fit a polynomial to them. This was done with the `find_lane_lines` function. This function takes in the birds-eye-view combined threshold image and fits polynomials to pixels detected as being part of each lane line.

```python
def find_lane_lines(binary_warped):
    #code from Udacity lectures
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

    return left_fit, leftx, lefty, right_fit, rightx, righty
```
`find_lane_lines` functions by taking a histogram of the bottom half of the image and finding the left and right peaks.

![alt text][image6]

The locations of these peaks is the starting point of each lane line. A series of sliding window searchs is then performed and non-zero pixels are appended to a list that has a 2nd order polynomial fit to it. The two polynomials represent the lane lines and are returned by `find_lane_lines`.  


For use in the video processing pipeline, a second function was created. `quick_find_lines` is used once the lane lines have been detected by `find_lane_lines`.

```python
def quick_find_lines(binary_warped, left_fit, right_fit):
    #code from Udacity lectures
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fit, leftx, lefty, right_fit, rightx, righty
```
`quick_find_lines` is a more efficient means of detecting the lane lines since it uses the lines detected by `find_lane_lines` as a starting point and looks for the lines in a margin around this starting line.

I set up the pipeline such that, whenever `quick_find_lines` fails to create reasonable lane lines three times in a row, a full `find_lane_lines` is completed again.  

The check to see if lines detected by `quick_find_lines` are real requires some means of determining that their geometry is reasonable. My approach was to check that the lane width is roughly correct, and that the lane lines have roughly the same curvature. Before running this check, the curvature and lane width need to be calculated - I did this in a function named `curve_pos`:

```python
def curve_pos(left_fit, right_fit):

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    ploty = np.linspace(0, 719, num=720)
    y_max = np.max(ploty)

    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    img_width = 1280*xm_per_pix
    img_height = 720*ym_per_pix

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = round(((1 + (2*left_fit_cr[0]*img_height + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0]), 1)
    right_curverad = round(((1 + (2*right_fit_cr[0]*img_height + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0]), 1)

    left_bot = left_fit_cr[0] * img_height ** 2 + left_fit_cr[1] * img_height + left_fit_cr[2]
    right_bot = right_fit_cr[0] * img_height ** 2 + right_fit_cr[1] * img_height + right_fit_cr[2]

    lane_center = (left_bot+right_bot)/2
    car_pos = round(lane_center - img_width / 2.0, 2)
    lane_width = right_bot - left_bot

    return left_curverad, right_curverad, car_pos, lane_width
```
`curve_pos` takes in the left and right lane line polynomials and returns the radius of curvature for each lane, the position of the car within the lane, and the width of the lane.

With this information, the detected lines are checked for reasonableness in the pipeline:

```python
#check line found is reasonable
if first_run:
    left_fits = [left_fit, left_fit, left_fit]
    left_best_fit = np.mean(left_fits, axis=0)

    right_fits = [right_fit, right_fit, right_fit]
    right_best_fit = np.mean(right_fits, axis=0)

    left_best_curverad, right_best_curverad, car_best_pos, lane_best_width = curve_pos(left_best_fit, right_best_fit)

    first_run = False

elif 0.8 <= lane_width/3.7 <= 1.2 and 0.5 <= left_curverad/left_curverad <=2:
    line_found = True
    fails = 0

    left_fits = np.roll(left_fits, -1, axis = 0)
    left_fits[2] = left_fit
    left_best_fit = np.mean(left_fits, axis=0)

    right_fits = np.roll(right_fits, -1, axis = 0)
    right_fits[2] = right_fit
    right_best_fit = np.mean(right_fits, axis=0)

else:
    line_found = False
    fails += 1
```

If only one image is processed, or the first frame of a video is being processed, the line check test is skipped. Otherwise, lines that pass this test are averaged with the previous two *good* lines to avoid overly jittery lines when processing videos.

Finally, when lane lines have been found, the lane can be drawn back onto the input image and returned as the pipeline output. I did so with the function `draw_lines`:

```python
def draw_lines(img, warped, Minv, left_fitx, right_fitx):
    #code from Udacity lectures
    # Generate x and y values for plotting
    ploty = np.linspace(0, 719, num=720)
    left_fitx = left_fitx[0]*ploty**2 + left_fitx[1]*ploty + left_fitx[2]
    right_fitx = right_fitx[0]*ploty**2 + right_fitx[1]*ploty + right_fitx[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    output = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return output
```

`draw_lines` takes in the original input image, the warped binary threshold image, and the polynomials representing the lane lines and returns the input image with the lane drawn on top:

![alt text][image7]

---

### Pipeline (video)

The pipeline works the same way on video streams.

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

The main problem I faced in implementing my pipeline was getting the right combination of binay thresholding images. Between the various thresholding values, kernal sizes, and combinations in the combined threshold there was a lot of trial and error required. The pipeline still jitters a little when the car gets to the lighter colored pavement so further refinement of the image preprocessing may be required to smooth out that area. The pipeline also doesn't work well on the challenge videos where the curvature on some of the turns is much too sharp for the histogram method of finding the lines to function. A better algorithm would be required to successfully draw the lanes on the challenge videos.
