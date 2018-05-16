# **Project 2 - Traffic Sign Recognition Classifier** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
---

[//]: # (Image References)

[image1]: ./writeup_images/1_traffic_sign_examples.png "Examples of each traffic sign"
[image2]: ./writeup_images/2_original_dataset_histogram.png "Original training dataset histogram"
[image3]: ./writeup_images/3_data_augmentation_techniques.png "Data augmentation techniques"
[image4]: ./writeup_images/4_augmented_dataset_histogram.png "Augmented trainig dataset histogram"
[image5]: ./writeup_images/5_new_test_images_easy.png "New test images easy"
[image6]: ./writeup_images/6_new_test_images_hard.png "New test images hard"
[image7]: ./writeup_images/7_new_test_images_32x32.png "New test images 32x32"
[image8]: ./writeup_images/8_traffic_sign_predictions.png "Traffic sign predictions"
[image9]: ./writeup_images/9_softmax_probabilities.png "Softmax probabilities"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You are reading the Writeup / README for the Traffic Sign Classifier project and here is a link to my project code [`P2_TrafficSignClassifier.ipynb`](https://github.com/viralzaveri12/CarND-P2-TrafficSignClassifier/blob/master/P2_TrafficSignClassifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

In **Step 0** of IPython notebook `P2_TrafficSignClassfier.ipynb`, I load the training, validation, and testing datasets.

In **Step 1 a)** of IPython notebook `P2_TrafficSignClassfier.ipynb`, I used python and numpy libraries to calculate summary statistics of the traffic signs dataset as shown below:

```python
# Number of training examples
n_train = len(X_train)

# Number of validation examples
n_valid = len(X_valid)

# Number of testing examples
n_test = len(X_test)

# Shape of traffic sign image
image_shape = X_train[0].shape

# Number of unique classes/labels in the dataset
n_classes = len(np.unique(y_train))

print('---------------------------------------------')
print("Number of training examples    = ", n_train)
print("Number of validation examples  = ", n_valid)
print("Number of testing examples     = ", n_test)
print("Image data shape               = ", image_shape)
print("Number of classes              = ", n_classes)
print('---------------------------------------------')
```
```python
---------------------------------------------
Number of training examples    =  34799
Number of validation examples  =  4410
Number of testing examples     =  12630
Image data shape               =  (32, 32, 3)
Number of classes              =  43
---------------------------------------------
```

#### 2. Include an exploratory visualization of the dataset.

In **Step 1 b)** of IPython notebook `P2_TrafficSignClassfier.ipynb`, I plot example images of each traffic sign class as shown below.

![alt text][image1]

In **Step 1 c)** of IPython notebook `P2_TrafficSignClassfier.ipynb`, I illustrate an exploratory visualization of the original training dataset. It is a bar chart showing how the traffic sign data is distributed for different classes.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

**Step 2 a)** of IPython notebook `P2_TrafficSignClassfier.ipynb` discusses, illustrates, and implements various data augmentation techniques for training dataset.

From the above distribution plot it is clear that the data is unbalanced. Some classes have very few examples compared to other classes. This is a problem for a machine learning algorithm, because the lack of balance in the data will lead it to become biased toward the classes with more data points. We need to generate more data points for the less represented classes. Therefore, implementing different Data Augmentation techniques to increase number of examples for the less represented classes in the the training dataset.

Training dataset contains some traffic signs like right turn, left turn, etc. that are either mirror images of other upon horizontal flip or signs like roundabouts, ahead only, etc. have no effect or mean entirely a different traffic sign after a 90&deg; rotation. Such data augmentation techniques would only increase the number of examples in the dataset but not make the training effective for the classes for which examples are less in the first place. Thus, data augmentation techniques need to be chosen with care.

Few examples of data augmentation techniques are shown below:

![alt text][image3]

Mentioned below are the three data augmentation techniques I implemented to increase number of examples for the less represented classes in the training dataset:

1. Scaled -> Salt and Pepper Noise -> Translated
2. Rotation between -15&deg; to 15&deg;
3. Affine Transformation


Below shown are the characteristics of the augmented training dataset like number of images in the set and a histogram showing number of images for each class:
```python
print(X_train_aug.shape)
print(y_train_aug.shape)
```
```python
(62156, 32, 32, 3)
(62156,)
```
![alt text][image4]

**Step 2 b)** of IPython notebook `P2_TrafficSignClassfier.ipynb` performs various dataset preprocessing steps.

Steps in dataset preprocessing include:

1. Converting an RGB to image to grayscale image reduces number of channel, thus reducing learning / processing time.
```python
# Convert to Grayscale
X_train_gry = np.sum(X_train_aug/3, axis=3, keepdims=True)
X_valid_gry = np.sum(X_valid/3, axis=3, keepdims=True)
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)

print('RGB training dataset shape:', X_train_aug.shape)
print('Grayscale training dataset shape:', X_train_gry.shape)
```
```python
RGB training dataset shape: (62156, 32, 32, 3)
Grayscale training dataset shape: (62156, 32, 32, 1)
```

2. Normalizing the images in the dataset in the range [-1.0,1.0] to ensure that all features will have equal contribution and the algorithm converges faster.
```python
print('Mean of training dataset:', np.mean(X_train_gry))

# Normalize training dataset
X_train_normalized = (X_train_gry - 128)/128
X_valid_normalized = (X_valid_gry - 128)/128
X_test_normalized = (X_test_gry - 128)/128

print('Mean of training dataset after normalization:', np.mean(X_train_normalized))
```
```python
Mean of training dataset: 80.0314092358
Mean of training dataset after normalization: -0.374754615345
```

3. Shuffling the images in dataset to reduce variance and ensure that models remain general and avoid overfitting.
```python
print('Dataset before shuffling:\n\n', y_train_aug[100:300])
```
```python
Dataset before shuffling:

 [41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31]
```
```python
X_train_shuf, y_train_shuf = shuffle(X_train_normalized, y_train_aug)
print('Dataset after shuffling:\n\n', y_train_shuf[100:300])
```
```python
Dataset after shuffling:

 [15  9 34 18 34 30  1 17 13  7 31 36 42 41 25 21  5  1 31 28 34 29  1 39 10
 15 10 34  8 23 32 22 40  5 31  2 38 12  3 14  7 31 15 13 26 35 12 16 40 38
 24  1  7 10  2 17 37 22 34  7 37  4 12 28 38 34 16 32 21  4 14 37 14  3 28
 26  6 13 38 15 24 34 26 33 10  3  6 33 36 14 14  6 34 12 38 10 18 36 14  3
 33  5 36  9 30 23  2 25 35  5  2  2 26  6 26 35 15 23 35 27 30 28  0 27 35
 26 25 12  0 23  3  6 21  8 33 34 15 14  5 29 32 13 42 12 23 33 13 14 28 16
  1 25 30 20 15  4 11 37 11 25 31 42 28 15 32  2 14  2  5 14 14 15  2 31 26
 21 39  1 33 11  9 23 25 42  9  2 25 19  5 31 39 36  8 32 31 26 26  5 28  2]
```

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

**Step 2 c)** of IPython notebook `P2_TrafficSignClassfier.ipynb` discusses the final model architecture implemented for traffic sign classification.

I began by implementing the same architecture from the LeNet Lab, with no changes since my dataset is in grayscale. This model worked well enough to begin with (~93% validation accuracy).

Testing my model on new images:  
I used test images from web of three difficulty levels - Easy (very clear, well-centered, straight facing, and background noise-free signs), Medium, and Hard (signs with dirt on them, not in center of the image, perspective view, background noise, and even multiple signs in the image).  
For Easy images, the above model gave 100% on test accuracy. However, the model gave only 20% for Medium and 0% for Hard images.

I explored various combinations - adding convolution layers, increasing filters, running more epochs, and varying batch sizes. I was able to get higher validation accuracy but the model still gave test accuracy of only 60% for Medium and 20% for Hard images.

Thus, to increase the test accuracy for Hard, I decided to go with the model a little more advanced than LeNet that is VGGNet. My architecture is based and derived from the VGGNet. Model architecture consists of six convolution layers and max pooling on alternate layers.

The complete architecture of the model is described in the table below:

| Layer # | Layer Name        	     | Description   	        					 | 
|:-------:|--------------------------|-----------------------------------------------| 
|         | Input         		     | 32x32x1 Grayscale image   					 | 
|  **1**  | Convolution 3x3     	 | 1x1 stride, same padding, output 32x32x64 	 |
|         | RELU					 |												 |
|  **2**  | Convolution 3x3     	 | 1x1 stride, input 32x32x32, output 32x32x32 	 |
|         | RELU					 |												 |
|         | Max pooling	      	     | 2x2 stride, input 32x32x32, output 16x16x32   |
|  **3**  | Convolution 3x3     	 | 1x1 stride, input 16x16x32, output 16x16x64 	 |
|         | RELU					 |												 |
|  **4**  | Convolution 3x3     	 | 1x1 stride, input 16x16x64, output 16x16x64 	 |
|         | RELU					 |												 |
|         | Max pooling	      	     | 2x2 stride, input 16x16x64, output 8x8x64     |
|  **5**  | Convolution 3x3     	 | 1x1 stride, input 8x8x64, output 8x8x128 	 |
|         | RELU					 |												 |
|  **6**  | Convolution 3x3     	 | 1x1 stride, input 8x8x128, output 8x8x128 	 |
|         | RELU					 |												 |
|         | Max pooling	      	     | 2x2 stride, input 8x8x128, output 4x4x128     |
|         |  Flatten	      	     | Input 4x4x128, output 2048                    |
|  **7**  | Fullly Connected	     | Input 2084, Output 120                        |
|         | RELU					 |												 |
|         | Dropout				     |												 |
|  **8**  | Fullly Connected	     | Input 120, Output 120                         |
|         | RELU					 |												 |
|         | Dropout				     |												 |
|  **9**  | Fullly Connected	     | Input 120, Output 43                          |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Steps to run the training data through the training pipeline to train the model:

* Before each epoch, shuffle the training set.
* After each epoch, measure the loss and accuracy of the validation set.

Validation dataset is used to assess how well the model is performing. A low accuracy on the training and validation sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

To train the model, I used the following parameters (After hundreds of tweaks!!!)

* Batch Size = 64
* Learning rate = 0.0009
* Epochs = 20
* mu = 0
* sigma = 0.1
* dropout keep prob = 0.6

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of **99.2%**
* test set accuracy of **97.5%**

My selection of the final model architecture is based on both iterative process and choosing (and modifying) a well known architecture. 

**Began with Iterative approach:**
* What was the first architecture that was tried and why was it chosen?
	> The first architecture I started was based off Lenet Lab, with no changes since my dataset is in grayscale. This model worked well enough to begin with (~93% validation accuracy).

* What were some problems with the initial architecture?
	> I used test images from web of three difficulty levels - Easy (very clear, well-centered, straight facing, and background noise-free signs), Medium, and Hard (signs with dirt on them, not in center of the image, perspective view, background noise, and even multiple signs in the image).  

	> For Easy images, the above model gave 100% on test accuracy. However, the model gave only 20% for Medium and 0% for Hard images.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
	> As the model seemed to be underfitting since it could not generalize and classify new traffic signs with slightly more noise in new test images, I explored various combinations - adding convolution layers, increasing filters, running more epochs, and varying batch sizes. I was able to get higher validation accuracy but the model still gave test accuracy of only 60% for Medium and 20% for Hard images.

* Which parameters were tuned? How were they adjusted and why?
	> Based on several iterations, adding convolution layers and adding number of filters to generate more activation maps were the primary parameters that seemed to improve the model performance.

**Finally modifying a well-known architecture:**
* What architecture was chosen?
	> As the model seemed to be underfitting since it could not generalize and classify new traffic signs with slightly more noise in new test images, and adding convolution layers, increasing filters, running more epochs, and varying batch sizes was only able to yield higher validation accuracy and test accuracy of only 60% for Medium and 20% for Hard images.

	> Thus, to increase the test accuracy for the Hard images, I finally decided to go with the model a little more advanced than LeNet that is VGGNet. My architecture is based and derived from the **VGGNet**. Model architecture consists of six convolution layers and max pooling on alternate layers. The complete architecture of the model is described in the table above in Question 2.

* Why did you believe it would be relevant to the traffic sign application?
	> The Hard images have signs with dirt on them, not in center of the image, perspective view, plenty of background noise, and even multiple signs in the image. Adding more convolution layers and adding number of filters to generate more feature maps per convolution layer would help identify and distinguish between signs and noise.

	> VGGNet is exactly the architecture that consists of several convolution layers with max pooling on alternate layers and dropouts on fully connected layer that can help learn and distinguish between features and noise.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
	> The modified VGGNet model architecture yielded a validation dataset accuracy of **99.2%** and a test dataset accuracy of **97.5%.**

	> With the modified VGGNet model architecture, I could get **100% accuracy on Medium images** and **60% accuracy on Hard images**.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

**Step 3** of IPython notebook `P2_TrafficSignClassfier.ipynb` tests the model on new new images found on web.

Here are five German traffic signs that I found on the web:

**Easy** - The traffic signs in these images are very clear, well-centered, straight facing, and background noise-free.

![alt text][image5]

**Hard** - These images have signs with dirt on them, are not in center of the image, have perspective view, contains plenty of background noise, and the last image even has multiple signs in the image (Stop and Turn Right Ahead signs).

![alt text][image6]

After resizing them to 32x32:

![alt text][image7]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

In **Step 3 - Visualize the Softmax Probabilities** section of IPython notebook `P2_TrafficSignClassfier.ipynb`, trained model is tested on the new traffic sign test images.

Here are the results of the traffic sign predictions for new test images:

![alt text][image8]

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60% for Hard images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

After testing the trained model on the new traffic sign test images in **Step 3 - Visualize the Softmax Probabilities** section of IPython notebook, below is th resulting top 5 softmax probabilities for each of the test images:

![alt text][image9]

**Image 1: Keep Right**  
Model correctly predicts Keep Right sign with **100%** softmax probability.

**Image 2: No Passing**  
Model prediction only has **13%** softmax probability for No Passing sign, while incorrectly has 83% softmax probability for End of No Passing.

**Image 3: Road Work**  
Model correctly predicts Road Work sign with **99%** softmax probability.

**Image 4: Speed Limit 30 kmh**  
Model prediction is way off for this image. Despite the sign for Speed Limit 30kmh being very clear, the background noise (vehicle and speed limit painted on road in the background, another partial traffic showing) is probably triggering softmax probabilities for all classes of the traffic signs, and softmax probability for Speed Limit 30kmh is nowhere near the top 5 predictions.

**Image 5: Stop Sign and Turn Right Ahead**  
I was both most skeptical and curious for this image as it has multiple signs in the image (Stop and Turn Right Ahead signs). I labeled this image as Stop Sign. Stop sign is correctly predicted, however, with softmax probability of only **40%**. Interestingly, the next four predictions does not include prediction for Turn Right Ahead sign.