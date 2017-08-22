## Writeup - Submission N. 2 

---

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

## Required N.1 

Position of the car with respect to center is not correct. As the width of the lane is about 3.7 meter and center of the lane would be 3.7/2=1.85 approximately. With high values>abs(1.0) , car should be driving outside the lanes but it is not the case in the video. After, detecting the lanes correctly, please make sure that value of car position witmeterh respect to center is in range (-0.7,0.7)

### How it was handled 
Please, see cells #30 of Advanced_Lane_Finding_Notebook.ipynb (lines 170-171) where specific log has been added and it was never invoked 


## Required N.2 
Lanes are drawn on the distorted image instead of undistorted one. It is obvious by looking at the car front hood. Please, make sure to undistort the image in the beginning of the pipeline, use that image for further processing and at the end lanes should also be drawn on the undistorted image not the original image.

### How it was handled 
Please, see cells #30 of Advanced_Lane_Finding_Notebook.ipynb 

## Required N.3 
Good job in detecting and drawing the lanes. Overall the results are very good.
Lanes are drawn on the distorted frame instead of undistorted one. It is obvious by looking at the car front hood. Please, make sure to undistort the frame in the beginning of the pipeline, use that frame for further processing and at the end lanes should also be drawn on the undistorted frame not the original frame.
In few frames,for example under the shadow area right lane is not well detected.
Scenarios where the algorithm fails are shared below. You can click on individual image to see and zoom the results. As it is a second project about lanes detection in this term, for this video very good results are expected.

### How it was handled 
Please, see the new [video](https://youtu.be/Y94e3LvBfyM)  


## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.  

CRITERIA | MEETS SPECIFICATIONS | HOW I ADDRESSED THE POINT | 
--- | --- | --- |
Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. Here is a template writeup for this project you can use as a guide and a starting point. | The writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled. |   Please, refer to [writeup](https://github.com/gtesei/SelfDrivingCars/blob/master/P4-CarND-Advanced-Lane-Lines/writeup.md)  |
Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image. | OpenCV functions or other methods were used to calculate the correct camera matrix and distortion coefficients using the calibration chessboard images provided in the repository (note these are 9x6 chessboard images, unlike the 8x6 images used in the lesson). The distortion matrix should be used to un-distort one of the calibration images provided as a demonstration that the calibration is correct. Example of undistorted calibration image is Included in the writeup (or saved to a folder). |   Please, refer to the section _Camera Calibration_ of [writeup](https://github.com/gtesei/SelfDrivingCars/blob/master/P4-CarND-Advanced-Lane-Lines/writeup.md) |
Provide an example of a distortion-corrected image. | Distortion correction that was calculated via camera calibration has been correctly applied to each image. An example of a distortion corrected image should be included in the writeup (or saved to a folder) and submitted with the project. |  Please, refer to the section _1. Provide an example of a distortion-corrected image._ of [writeup](https://github.com/gtesei/SelfDrivingCars/blob/master/P4-CarND-Advanced-Lane-Lines/writeup.md)   |
Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result. | A method or combination of methods (i.e., color transforms, gradients) has been used to create a binary image containing likely lane pixels. There is no "ground truth" here, just visual verification that the pixels identified as part of the lane lines are, in fact, part of the lines. Example binary images should be included in the writeup (or saved to a folder) and submitted with the project. |   Please, refer to _2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result._  of [writeup](https://github.com/gtesei/SelfDrivingCars/blob/master/P4-CarND-Advanced-Lane-Lines/writeup.md) |
Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image. | OpenCV function or other method has been used to correctly rectify each image to a "birds-eye view". Transformed images should be included in the writeup (or saved to a folder) and submitted with the project. |   Please, refer to _3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image._ of [writeup](https://github.com/gtesei/SelfDrivingCars/blob/master/P4-CarND-Advanced-Lane-Lines/writeup.md) |
Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial? | Methods have been used to identify lane line pixels in the rectified binary image. The left and right line have been identified and fit with a curved functional form (e.g., spine or polynomial). Example images with line pixels identified and a fit overplotted should be included in the writeup (or saved to a folder) and submitted with the project. |   Please, refer to section _ 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?_ of [writeup](https://github.com/gtesei/SelfDrivingCars/blob/master/P4-CarND-Advanced-Lane-Lines/writeup.md)|
Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center. | Here the idea is to take the measurements of where the lane lines are and estimate how much the road is curving and where the vehicle is located with respect to the center of the lane. The radius of curvature may be given in meters assuming the curve of the road follows a circle. For the position of the vehicle, you may assume the camera is mounted at the center of the car and the deviation of the midpoint of the lane from the center of the image is the offset you're looking for. As with the polynomial fitting, convert from pixels to meters. |   Please, refer to _5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center_ of [writeup](https://github.com/gtesei/SelfDrivingCars/blob/master/P4-CarND-Advanced-Lane-Lines/writeup.md) |
Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly. | The fit from the rectified image has been warped back onto the original image and plotted to identify the lane boundaries. This should demonstrate that the lane boundaries were correctly identified. An example image with lanes, curvature, and position from center should be included in the writeup (or saved to a folder) and submitted with the project. |   Please, refer to section _6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly._ of [writeup](https://github.com/gtesei/SelfDrivingCars/blob/master/P4-CarND-Advanced-Lane-Lines/writeup.md)|
Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!) | The image processing pipeline that was established to find the lane lines in images successfully processes the video. The output here should be a new video where the lanes are identified in every frame, and outputs are generated regarding the radius of curvature of the lane and vehicle position within the lane. The pipeline should correctly map out curved lines and not fail when shadows or pavement color changes are present. The output video should be linked to in the writeup and/or saved and submitted with the project. |  [output video](https://youtu.be/pf64NCrA7eY)  |
Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust? | Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail. |   Please, refer to section _1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?_ of  [writeup](https://github.com/gtesei/SelfDrivingCars/blob/master/P4-CarND-Advanced-Lane-Lines/writeup.md)|


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original Image | Undistorted Image |
--- | --- | 
<img src="test_images/test2.jpg"/>   | <img src="output_images/test2_undistort.png" />  | 

Other examples can be found here. 

Original Image | Undistorted Image |
--- | --- | 
[test1](test_images/test1.jpg) | [test1 undistorted](output_images/test1_undistort.png) | 
[test2](test_images/test2.jpg) | [test1 undistorted](output_images/test2_undistort.png) | 
[test3](test_images/test3.jpg) | [test1 undistorted](output_images/test3_undistort.png) | 
[test4](test_images/test4.jpg) | [test1 undistorted](output_images/test4_undistort.png) | 
[test5](test_images/test5.jpg) | [test1 undistorted](output_images/test5_undistort.png) | 
[test6](test_images/test6.jpg) | [test1 undistorted](output_images/test6_undistort.png) | 

Related code can be found in cells #5-#7 of Advanced_Lane_Finding_Notebook.ipynb. 


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Original Image | Undistorted Image |
--- | --- | 
<img src="test_images/test2.jpg"/>   | <img src="output_images/test2_undistort.png" />  | 

Other examples can be found here. 

Original Image | Undistorted Image |
--- | --- | 
[test1](test_images/test1.jpg) | [test1 undistorted](output_images/test1_undistort.png) | 
[test2](test_images/test2.jpg) | [test1 undistorted](output_images/test2_undistort.png) | 
[test3](test_images/test3.jpg) | [test1 undistorted](output_images/test3_undistort.png) | 
[test4](test_images/test4.jpg) | [test1 undistorted](output_images/test4_undistort.png) | 
[test5](test_images/test5.jpg) | [test1 undistorted](output_images/test5_undistort.png) | 
[test6](test_images/test6.jpg) | [test1 undistorted](output_images/test6_undistort.png) | 

Related code can be found in cells #5-#7 of Advanced_Lane_Finding_Notebook.ipynb. 

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.  Related code can be found in cells from #8 to #17 of Advanced_Lane_Finding_Notebook.ipynb. 


I combined two pipelines:

1. converted to HSV color space and extratced S and L channels, took the derivative in x and its absolute to accentuate lines away from horizontal, thresholded x gradient, thresholded color channel and finally combined the binary thresholds
2. applied Sobel transform and extracted gradient in x an y direction, thresholded magnitude and direction and finally combined the binary thresholds



Here's examples of my output for this step.  



Original Image | Thresholded Binary Image |
--- | --- | 
<img src="test_images/test1.jpg"/> | <img src="output_images/test1_thresholding.png"/>  | 
<img src="test_images/test2.jpg"/> | <img src="output_images/test2_thresholding.png"/>  | 
<img src="test_images/test3.jpg"/> | <img src="output_images/test3_thresholding.png"/>  | 
<img src="test_images/test4.jpg"/> | <img src="output_images/test4_thresholding.png"/>  | 
<img src="test_images/test5.jpg"/> | <img src="output_images/test5_thresholding.png"/>  | 
<img src="test_images/test6.jpg"/> | <img src="output_images/test6_thresholding.png"/>  | 


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Please refer to cells #18-#20 of Advanced_Lane_Finding_Notebook.ipynb. I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points for the image _test_images/straight_lines1.jpg_:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

| Undistorted image with source points drawn        | Warped result with dest.points drawn   | 
|:-------------:|:-------------:| 
| <img src="output_images/straight_lines1_undist.png"/>      | <img src="output_images/straight_lines1_warped.png"/>       | 
| <img src="output_images/straight_lines2_undist.png"/>      | <img src="output_images/straight_lines2_warped.png"/>       | 


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I fit my the perspective transformed and thresholded image with a 2nd order polynomial using a sliding histogram approach. I start at the maximum peaks in the bottom half of the image and move our way up. I then subsequently search for the line with the same approach and finally fit a polynomial (see #21-#25 of Advanced_Lane_Finding_Notebook.ipynb)


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Please refer to cells #26-#27 of Advanced_Lane_Finding_Notebook.ipynb. 

This approach calculates curvature for given polynomial fits. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Please refer to cells #28-#31 of Advanced_Lane_Finding_Notebook.ipynb.  Here is an example of my result on a test image:


|         |    | 
|:-------------:|:-------------:| 
| <img src="output_images/0_annotated.png"/>      | <img src="output_images/1_annotated.png"/>       | 
| <img src="output_images/2_annotated.png"/>      | <img src="output_images/3_annotated.png"/>       | 
| <img src="output_images/3_annotated.png"/>      | <img src="output_images/4_annotated.png"/>       | 
| <img src="output_images/5_annotated.png"/>      | <img src="output_images/6_annotated.png"/>       | 


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/Y94e3LvBfyM)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach here adopted is based on the following steps: 

1. Camera calibration 
2. Distortion correction 
3. Color & gradient threshold 
4. Perspective transform  
5. Fit lane-line pixels with a polinomial 
6. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center

As a conseguence, each time main assumptions of such approach are not satisfied the pipeline might fail, e.g. 

* There are no lane lines 
* Lane lines are not clearly visible on the road because of obsolescence or other reasons 
* Bad wheater might introduce noise we did not test our pipeline with 
 




