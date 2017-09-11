#**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./examples/center_sample.png "Sample image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The model in model.py uses Keras 2 and may not run with the Udacity-provided Docker image.

####3. Submission code is usable and readable

I have, for the most part, re-used code from the classes and added comments where required.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model is based on the model Nvidia model referenced in the class. It consits of several convolutional layers followed by several dense layers. The model includes relu-activations to add non-linearity as well
as lambda and cropping layers for normalizing and preprocessing the images on the GPU. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers. The validation set is approx. a third of the sample data. Approximately 27k steering angles with three pictures each were recorded. Difficult parts of the map as well as recovery parts were recorded several times. 
The number of epochs was limited to 10 as the validation loss stayed more or less constant while training loss kept decreasing with more epochs.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The correction paramter was tuned by trying various values and recording validation accuracy, a value of 0.1 seems to work well.
Different values for batch size were used, but the results did not differ dramatically.

####4. Appropriate training data

Data recording included several laps of center lane driving (clock and counter-clockwise) as well as recovery driving and recording problematic areas (e.g. bridge) several times. The dataset contains 27k samples.
The second map was also recorded, but not as extensively.


###Model Architecture and Training Strategy

####1. Solution Design Approach

Initially I started out with a simple model and small dataset to test the implementation. I then implemented the conv-net mentioned in class. I tested the model with the simulator and found some problematic areas (such as bridge and dirt road in first map) and so recorded more training data for those sections of the map.
Data was split into validation and train sets and shuffled.
Validation loss decreased mostly during the first few epochs, so training for more than 10 epochs did not seem useful.

I tested the simulator with different speeds, the final model was robust to changes in the speed setting for the first map but not for the second.


####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:
-Lambda layer for normalization of input
-Dropout layer (20%)
-Cropping layer for cutting off top 60 and bottom 20 pixels
-5 convolutional layers with increasing filter size (24, 26, 48, 64, 64), decreasing window sizes (5 by 5 for first 3 layers, then 3 by 3), strides of 2 vertical and 2 horizontal pixels, relu activation and valid padding
-a flatten layer 
-3 dense layers with 100, 50 and 10 neurons, each followed by a dropout layer (50%)
-output layer (dense, 1 neuron) for the predicted steering angle


####3. Creation of the Training Set & Training Process

On the first map two center lane driving laps were combined with recovery laps (recording only when correcting back towards center) as well as additional recording of curves and tricky parts of the map to avoid an imbalanced
training set. Speed during training on the first map was 30mp/h. Several laps from the second map were also added, but at much lower speed.

Here is an example input picture (center camera):
![alt text][image0]

All three pictures were used and flipped to augment the dataset. 

The dataset currently has about 27k steering angles with three pictures each.
Data were preprocessed by normalizing and cropping as described above.
The data was shuffled and 30% used as validation set.

Training involved an iterative approach of training the model, tuning paramters to improve validation accuracy, testing the simulator and recording additional training data for problematic areas. 
The model was trained for 10 epochs as validation accuracy did not improve very much after that and the model began to overfit. Learning rate was adapted by the Adam optimizer. The correction factor for 
the off-center images was tested with several values around the starting value proposed in class.

The model works well for the first map, but not yet for the second.
[video](https://github.com/janreerink/CarND-Behavioral-Cloning-P3/blob/master/examples/run1.mp4)

