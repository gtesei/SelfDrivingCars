## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


Contents
---

1. [Jupyter Notebook with code](https://github.com/gtesei/SelfDrivingCars/blob/master/P2-CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb) - [Jupyter Notebook with code - HTML](https://github.com/gtesei/SelfDrivingCars/blob/master/P2-CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.html)
2. [writeup report (md file)](https://github.com/gtesei/SelfDrivingCars/blob/master/P2-CarND-Traffic-Sign-Classifier-Project/writeup.md) - [writeup report (HTML)](https://github.com/gtesei/SelfDrivingCars/blob/master/P2-CarND-Traffic-Sign-Classifier-Project/writeup.html)

__NOTE:__ sometime GitHub is not able to open [Jupyter Notebook with code](https://github.com/gtesei/SelfDrivingCars/blob/master/P2-CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb) as it is a big file. If this happens to you, please use [Jupyter Notebook with code - HTML version](https://github.com/gtesei/SelfDrivingCars/blob/master/P2-CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.html) instead.  


Post-Evaluation Notes  
---

* For preprocessing techniques, another idea is applying Contrast Limited Adaptative Histogram Equalization, or [CLAHE](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) 
* Instead of a fixed number of epochs, one alternative is implementing __early termination__
* To learn more about convolutional networks I recommend this [book](http://www.deeplearningbook.org/contents/convnets.html) 
* Another architecture with good results here is [this example](https://www.tensorflow.org/tutorials/deep_cnn) for CIFAR-10
* For increased robustness, you can use the image augmentation technique to further rebalance the number of examples for each class and to expose your model to a wider variety of image qualities; check out [this article](https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3) for a great explanation with examples


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.