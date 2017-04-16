#**Traffic Sign Recognition** 

##Project Writeup
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

[image1]: ./examples/dataset_visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./GTS_images/2.ppm "Traffic Sign 1"
[image5]: ./GTS_images/12.ppm "Traffic Sign 2"
[image6]: ./GTS_images/22.ppm "Traffic Sign 3"
[image7]: ./GTS_images/32.ppm "Traffic Sign 4"
[image8]: ./GTS_images/42.ppm "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/caje731/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43
* Dataset ratios-> train:valid:test = 1:0.13:0.36

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed for the different classes in all three categories.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to preprocess the data by performing the following steps:

* Shuffling:
    This step randomises the order of the data samples, such that the neural network doesn't face the same kind of samples one after another. This allows a better model generalization.
    
* Normalization:
    I decided to normalize the images by dividing each pixel by 256. Since the images were RGB, the highest value for a pixel in a channel would be 255, and so dividing each pixel by 256 is a quick method to get every pixel value to fall within the 0 to 1 range.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1 (RELU   | 5X5 filter, same padding, outputs 28x28x18 	|
| Convolution 2 (RELU   | 5X5 filter, same padding, outputs 24x24x48 	|
| Max pooling	      	| 2x2 kernel, outputs 12x12x48              	|
| Convolution 3 (RELU   | 4X4 filter, same padding, outputs 8x8x96   	|
| Max pooling	      	| 2x2 kernel, outputs 4x4x96                 	|
| Flatten               | Down to one-dimension of size 1536            |
| Fully connected 1		| Linear combinations - feature-space to 360    |
| Fully connected 2		| Linear combinations - feature-space to 252    |
| Softmax				| Estimates for the 43 classes of traffic signs |


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the learning rate of 0.001, the batch size of 128, and the number of epochs to 10.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of 96.8% 
* test set accuracy of 94.3%

I started with the the standard LeNet model and adjusted it for the RGB space. This produced similar results to the grayscale images. To adjust for the additional color channels, all I did was multiply the output sizes of each layer by three.

Preprocessing proved to be important since without it I was able to achieve only 60% validation accuracy. I went for two convolutional layers first, and got a validation accuracy of 87%. Then I adjusted the network to accomodate another convolutional layer, and the accuracy increased to 96%.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Speed limit (50km/h)][image4] 
![Priority road	][image5] 
![Bumpy road][image6] 
![End of speed limits][image7] 
![End of no passing by vehicles over 3.5 metric tons][image8]

The fourth image might be difficult to classify because even I'm having trouble identifying it - just looks like a dark blob.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)  | Speed limit (50km/h)                          | 
| Priority road			| Priority road                                 |
| Bumpy road			| Bumpy road									|
| End of speed limits   | Keep right					 	            |
| End of no passing by vehicles over 3.5 metric tons| End of no passing by vehicles over 3.5 metric tons|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is way lower than the over 94% test accuracy we saw earlier, but that's simply due to the very-small number of test images here (just 5).

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For all the images, the softmax probabilities I saw were 100%-confident of the prediction.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Speed limit (50km/h)							| 
| 1     				| Priority road 								|
| 1 					| Bumpy road       								|
| 1 	      			| Keep right					 				|
| 1 				    |End of no passing by vehicles over 3.5 metric tons|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


