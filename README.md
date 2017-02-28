#**Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./images/center_lane_driving.jpg "Center Lane Driving"
[image2]: ./images/recovery_1.jpg "Recovery Driving"
[image3]: ./images/recovery_2.jpg "Recovery Driving"
[image4]: ./images/recovery_3.jpg "Recovery Driving"
[image5]: ./images/flip.png "Flip"
[image6]: ./images/epoch_vs_loss.png "Epochs vs Loss"
[image7]: ./images/epoch_vs_loss.png "Epochs vs Loss"
[image8]: ./images/epoch_vs_loss.png "Epochs vs Loss"
[image9]: ./images/track_2_thumpnail.jpg "Track 2 Thumbnail"

###Model Architecture and Training Strategy

####1. Solution Design Approach

My model was inspired by the NVidia architecture. I thought this model might be appropriate because of the similarity between the dataset used by the NVidia model and the dataset I am working with.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set in the ratio of 80% and 20%. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I added 'dropout' layers to my model (model.py lines 58-89).

The final step was to run the simulator to see how well the car was driving around track one. I collected data by driving two center-lane driving laps, one recovery lap and 1 lap of smooth driving around the curves for track 1 and track 2. On track 1 there were a few spots around the curves where the vehicle fell off the track. To improve the driving behavior in these cases, I captured some more recovery driving and driving around the curves. After this, my model could drive smoothly around the track 1. On track 2 the vehicle could drive smoothly upto the first shadow region and then fell off the track. To generalize the model and make it invariant to shadows I captured more data by driving around the second track and capturing only the shadow regions.

At the end of the process, the vehicle is able to drive autonomously around both the tracks without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 55-94) consisted of a convolution neural network with the following layers and layer sizes.

Layer 1, 5x5 Convolution, size=24,  dropout=0.2<br>
Layer 2, 5x5 Convolution, size=36,  dropout=0.2<br>
Layer 3, 3x3 Convolution, size=48,  dropout=0.2<br>
Layer 4, 3x3 Convolution, size=64,  dropout=0.2<br>
Layer 5, Fully Connected, size=100, dropout=0.5<br>
Layer 6, Fully Connected, size=50,  dropout=0.2<br>
Layer 7, Fully Connected, size=10

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to get back to the center if it goes to the side of the road during autonomous driving. The following images show what a recovery looks like:

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

The first track has a left turn bias. To combat the left turn bias and to generalize the model I applied a left-right flip to 50% of the dataset at random. For example, here is an image that has then been flipped:

![alt text][image5]

I have used all three camera images to augment my dataset and to generalize my model. (model.py lines 153-165). The angle correction used by the left and right camera images are [0.25, -0.25].

After the data collection process, I had a total of 30086 data points. As part of the preprocessing I shuffled and normalized ([-0.5, 0.5]) the data. I used a lambda layer in my architecture to parallelize the normalization process. I split the dataset to put 80% into a training set and 20% into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 85 as evidenced by the epochs vs loss graph below. But I used 100 epochs for fine-tuning. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image6]

###4. Video
The result of the car driving around track 1 and track 2 can be seen in the following videos uploaded onto YouTube.

Track 1

[![Track 1](./images/center_lane_driving.jpg)](https://youtu.be/0Ap-CEtEQig "Track 1")

Track 2
[![Track 2](./images/track_2_thumpnail.jpg)](https://youtu.be/XtIj0nkb5S8 "Track 2")

