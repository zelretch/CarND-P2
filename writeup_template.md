#**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/bar_chart.png "Visualization"
[image2]: ./writeup_images/before_normalizing.png "before normalizing"
[image3]: ./writeup_images/normalized_image.png "after normalizing"
[image4]: ./new_images/1.jpg "Traffic Sign 1"
[image5]: ./new_images/2.jpg "Traffic Sign 2"
[image6]: ./new_images/3.jpg "Traffic Sign 3"
[image7]: ./new_images/4.jpg "Traffic Sign 4"
[image8]: ./new_images/5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/zelretch/CarND-P2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across different labels

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided normalize the date because the brightness shouldn't affect the classification of traffic signs 

Here are examples of traffic signs before and after normalization

![alt text][image2]
![alt text][image3]

I didn't do grayscale because color is obviously important for recognizing traffic signs

Further Improvement could be made if fake data to be generated for the labels with less training examples.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model is basically the solution for LeNet with demension doubled

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Flatten           |output 800 |
| Fully connected		| input 800, output 240       									|
| RELU					|												|
| Fully connected		| input 240, output 168     |
| RELU					|												|
| Fully connected		| input 168, output 43       									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I didn't change much from the LeNet solution besides changing the number of Epochs to 30, which is the number of Epochs I observed the error to be stablized around

Optimizer: Adam

Batch Size: 128

Epochs: 30

learning rate: 0.001


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.957
* test set accuracy of 0.946

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I used the LeNet solution as the starting point. Because the two problems are very similar

* What were some problems with the initial architecture?

The initial architecture is expecting grayscale images, so we might need more parameters for colored images because there are more information. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I doubled the number of parameters in each layer to better account for the extra information of colored image

I also normalized the input to exclude the difference in brightness

I also tried to do dropout, though it didn't help much and I commented it out. 

* Which parameters were tuned? How were they adjusted and why?
Number of filters were doubled because the input color images are 3 times larger than grayscale images 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Because cov net is good at preseving space invariance at image recognizing. Dropout layer should reduce overfitting, although on this particular problem it doesnt seem to help much.

If a well known architecture was chosen:
* What architecture was chosen?
LeNet

* Why did you believe it would be relevant to the traffic sign application?
Because LeNet is used to recognize digits which is pretty similar to recognize traffic sign

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

0.957 is reasonable good validation accuracy 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution      		| No passing  									| 
| Children crossing    			| Children crossing 										|
| Yield					| Yield											|
| Roundabout mandatory	      		| Roundabout mandatory				 				|
| Road work			| Road work      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares badly to the accuracy on the test set of 0.946

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is pretty sure that this is a no passing (probability of 0.965), but the image is a general caution. 
I think the text below the sign probably confused the model. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .965         			| no passing   									| 
| .0144     				|  		end of no passing								|
| .0135					| Dangerous curve to the right											|
| .0067	      			| no entry					 				|
| .00001				    | Priority road      							|


For the second image the model is on the edge that this is a Children crossing (probability of 0.52), and the image is a  Children crossing. I think this can be understood because the top 3 predications are all XXX crosing sign.  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .521         			| Children crossing  									| 
| .438     				|  	Wild animals crossing								|
| .04					|  Bicycles crossing											|
| .00005	      			| Slippery road					 				|
| .00002				    | Beware of ice/snow      							|


For the third image the model is certain that this is a Yield (probability of 1), and the image is a  Yield. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Yield  									| 
| 5 10-14     				|  	Slippery road								|
| 1 10-14					|  Ahead only										|
| 1.5 10-15	      			| No passing for vehicles over 3.5 metric tons					 				|
| 5 10-17				    | Road work      							|


For the fourth image the model is almost certain  that this is a Roundabout mandatory (probability of 0.996), and the image is a  Roundabout mandatory. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.996         			| Roundabout mandatory  									| 
| 0.003    				|  	Keep left								|
| 2.8 10-11					|  Speed limit (70km/h)									|
| 3.4 10-13	      			| Go straight or left					 				|
| 1 10-13				    | Wild animals crossing    							|

For the fifth image the model is certain that this is a Road work (probability of 1), and the image is a  Road work. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1        			| Road work  									| 
| 6.8 10-9    				|  	Keep left								|
| 6.2 10-14					|  No passing for vehicles over 3.5 metric tons									|
| 7 10-17	      			| Stop				 				|
| 3.4 10-19				    |Beware of ice/snow    							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


