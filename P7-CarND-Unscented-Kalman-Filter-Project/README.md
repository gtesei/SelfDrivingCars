# Unscented Kalman Filter Project Starter Code
Self-Driving Car Engineer Nanodegree Program

In this project I will utilize a **unscented kalman filter** to estimate the state of a moving object of interest with noisy lidar and radar measurements. 

## Important Dependencies

This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems. For windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see [this concept in the classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77) for the required version and installation scripts.

Here is the main protcol that **main.cpp** uses for **uWebSocketIO** in communicating with the simulator.

* **INPUT**: values provided by the simulator to the c++ program
["sensor_measurement"] => the measurment that the simulator observed (either lidar or radar)

* **OUTPUT**: values provided by the c++ program to the simulator
["estimate_x"] <= kalman filter estimated position x
["estimate_y"] <= kalman filter estimated position y
["rmse_x"]
["rmse_y"]
["rmse_vx"]
["rmse_vy"]

## Other Important Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

---

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./UnscentedKF` Previous versions use i/o from text files.  The current state uses i/o
from the simulator.

#  Project Rubric

<img src="final_screen.png" /> 

CRITERIA | MEETS SPECIFICATIONS | HOW I ADDRESSED THE POINT | 
--- | --- | --- |
Your code should compile. | Code must compile without errors with cmake and make. Given that we've made CMakeLists.txt as general as possible, it's recommended that you do not change it unless you can guarantee that your changes will still compile on any platform. | After installing all dependencies, executing `mkdir build && cd build` and `cmake .. && make`  you can see the code in `src` compile | For the new version of the project, there is now only one data set "obj_pose-laser-radar-synthetic-input.txt". px, py, vx, vy output coordinates must have an RMSE <= [.09, .10, .40, .30] when using the file: "obj_pose-laser-radar-synthetic-input.txt" | For the new data set, your algorithm will be run against "obj_pose-laser-radar-synthetic-input.txt". We'll collect the positions that your algorithm outputs and compare them to ground truth data. Your px, py, vx, and vy RMSE should be less than or equal to the values [.09, .10, .40, .30]. | RMSE is [.0704, .0821, .3282, .2981] |
Your Sensor Fusion algorithm follows the general processing flow as taught in the preceding lessons. | While you may be creative with your implementation, there is a well-defined set of steps that must take place in order to successfully build a Kalman Filter. As such, your project should follow the algorithm as described in the preceding lesson. | Project follows the algorithm as described in the lesson. Please, see *src/ukf.cpp* |
Your Kalman Filter algorithm handles the first measurements appropriately. | Your algorithm should use the first measurements to initialize the state vectors and covariance matrices. | Please, see *src/ukf.cpp* |
Your Kalman Filter algorithm first predicts then updates. | Upon receiving a measurement after the first, the algorithm should predict object position to the current timestep and then update the prediction using the new measurement. | Please, see *src/ukf.cpp* |
Your Kalman Filter can handle radar and lidar measurements. | Your algorithm sets up the appropriate matrices given the type of measurement and calls the correct measurement function for a given sensor type. | Please, see *src/ukf.cpp* |
Your algorithm should avoid unnecessary calculations. | This is mostly a "code smell" test. Your algorithm does not need to sacrifice comprehension, stability, robustness or security for speed, however it should maintain good practice with respect to calculations. Here are some things to avoid. This is not a complete list, but rather a few examples of inefficiencies. Running the exact same calculation repeatedly when you can run it once, store the value and then reuse the value later. Loops that run too many times. Creating unnecessarily complex data structures when simpler structures work equivalently. Unnecessary control flow checks. | Please, see *src/ukf.cpp* and *src/tools.cpp* |


# Editor Settings, Code Style and Additional Data

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html) as much as possible.

## Generating Additional Data

If you'd like to generate your own radar and lidar data, see the
[utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) for
Matlab scripts that can generate additional data.

