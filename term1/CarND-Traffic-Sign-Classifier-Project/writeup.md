#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

It was important to see how the data is structured and how the distribution between training, validation and testing data looks like.

####2. Include an exploratory visualization of the dataset.

I visualized the count of the different traffic signs in the training data set. The intention was to check if every traffic sign has enough examples to train the CNN good.

[image9]: ./writeup/data_visualization.png "Examples for every feature in the training data set"

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I converted the images to grayscale. This makes the CNN more robust against different color values, it then just use the contrast of the colors to detect lines.
As a second step, I normalized the image data around 0. It was told during the videos that this will help to train the NN a lot.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	    	| 2x2 stride,  outputs 5x5x64   				|
| Fully connected		| Input = 120. Output = 84     					|
| Fully connected		| Input = 84. Output = 43        				|
 
I increased the output depth of both ConvNets to make the CNN deeper. This increased the computing time, but increased also the accuracy.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I had to decrease the batch size because my computer wasn't able to compute a batch size of 128. It noticed that you could already see a first indication how good your CNN is, after the first epoch.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 95.1%
* test set accuracy of 92.9%

I started with the LeNet architecture of the previous lab. First I tried to improve the accurancy by the pre-processing of the data. As a second step, I increased the depth of my CNN, this decreased the training time but gave better accurancy. As a last step, I introduced dropouts during the maxpool which prevented an early overfitting of the CNN.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
[image4]: ./german-signs/sign00.png "Traffic Sign 1"
[image5]: ./german-signs/sign01.png "Traffic Sign 2"
[image6]: ./german-signs/sign02.png "Traffic Sign 3"
[image7]: ./german-signs/sign03.png "Traffic Sign 4"
[image8]: ./german-signs/sign04.png "Traffic Sign 5"

I though that the second image could be hard to solve for the CNN, because it's very similiar to the other speed limit signs.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image										| Prediction									| 
|:-----------------------------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection		| Right-of-way at the next intersection 		| 
| Speed limit (70km/h)     					| Speed limit (70km/h) 							|
| Road work									| Road work										|
| Ahead only	      						| Ahead only					 				|
| Priority road								| Priority road      							|

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The signs I've tested are apparently very easy. 3 of 5 times, the confidence of the network is at 100%. Even at the speed limit signs, the has a confidence of 98%, really impressive!


