## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  


Contents 
---

* [writeup](https://github.com/gtesei/SelfDrivingCars/blob/master/P4-CarND-Advanced-Lane-Lines/writeup.md) 
* [code (or a Jupyter notebook)](https://github.com/gtesei/SelfDrivingCars/blob/master/P4-CarND-Advanced-Lane-Lines/Advanced_Lane_Finding_Notebook.ipynb) 
* [example output images](https://github.com/gtesei/SelfDrivingCars/tree/master/P4-CarND-Advanced-Lane-Lines/output_images) 
* [output video](https://youtu.be/Y94e3LvBfyM)


Further Readings  
---

* [Real-time illumination invariant lane detection for lane departure
warning system](http://diml.yonsei.ac.kr/papers/Real-time%20Illumination%20Invariant%20Lane%20Detection%20%20for%20Lane%20Departure%20Warning%20System.pdf)
* [Robust lane finding using advanced computer vision techniques: Mid project update](https://medium.com/towards-data-science/robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3)
* [ROBUST AND REAL TIME DETECTION OF CURVY LANES (CURVES) HAVING DESIRED SLOPES FOR DRIVING ASSISTANCE AND AUTONOMOUS VEHICLES](http://airccj.org/CSCP/vol5/csit53211.pdf) 
* [Real time Detection of Lane Markers in Urban Streets](http://www.vision.caltech.edu/malaa/publications/aly08realtime.pdf)
