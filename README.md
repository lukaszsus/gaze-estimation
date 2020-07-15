# Gaze Estimation
Gaze Estimation project for master thesis at Wroclaw University of Science and Technology.

## Abstract
The work aimed to propose a method of controlling the mouse cursor using eye movement based on the image from a video camera. The solution to the task required the development of a multi-stage input image processing pipeline, the main element of which was to predict the viewpoint using an artificial neural network model. The effectiveness of the method was evaluated experimentally. In addition to the benchmark set, the study used a self-collected data set. Analysis of the test results showed that the proposed method allows us to move the mouse cursor to the desired place, but it was not possible to move it exactly to the place where the user was looking at.

## Technical details
Project implemented in Python 3. It includes deep convolutiona neural network implementation in Tensorflow 2.0 and some classical methods of image processing.
