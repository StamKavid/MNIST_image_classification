# MNIST: Image Classification | Overview

In this project you will find Image Clasiifcation problem using MNIST datasource. More specifically:
1. Problem Statement 
2. Data source | Overview
3. EDA | Data Augmentation
4. CNN | DL

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Resources used

**Python Version:** 3.6.9

**Pandas Version:** 1.3.5

**NumPy Version:** 1.21.6

**Packages:** numpy, pandas, matplotlib, plotly, sklearn, keras, dask

**MNIST Architectures | Papers:** https://bit.ly/3xVXCSL

**MNIST Data source:** https://bit.ly/3McrUEq

**Paper for LeNet5:** https://bit.ly/3EyHQOH

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 1. Problem Statement | Image Classification

**Image classification** is the process of categorizing and labeling groups of pixels or vectors within an image based on specific rules.

Computers can perform computations on numbers. There are two common ways to do this in Image Processing:

**Using Greyscale**

The image will be converted to greyscale (range of gray shades from white to black) the computer will assign each pixel a value based on how dark it is. All the numbers are put into an array and the computer does computations on that array. This is how the number 8 is seen on using Greyscale:

![](https://github.com/StamKavid/MNIST_image_classification/blob/main/Images/1_zY1qFB9aFfZz66YxxoI2aw.gif)

Source: https://bit.ly/36v55wx

**Using RGB Values**

Images could be represented as RGB values (a combination of red, green and blue ranging from 0 - black - to 255 - white-).
When the computer interprets a new image, it will convert the image to an array by using the same technique, which then compares the patterns of numbers against the already-known objects. The class with the highest confidence score is usually the predicted one.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 2. Data Source | Overview

The MNIST database (Modified National Institute of Standards and Technology database) is a large collection of handwritten digits. It has a training set of 60,000 examples, and a test set of 10,000 examples.

The digits have been size-normalized and centered in a fixed-size image. The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm.

The images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

![](https://github.com/StamKavid/MNIST_image_classification/blob/main/Images/mnist-3.0.1.png)


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 3. EDA | Data Augmentation

**Exploratory Data Analysis** refers to the critical process of performing initial investigations on data so as to discover patterns, to spot anomalies, to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.

Data augmentation is a technique to artificially create new training data from existing training data. This is done by applying domain-specific techniques to examples from the training data that create new and different training examples.

Image data augmentation is perhaps the most well-known type of data augmentation and involves creating transformed versions of images in the training dataset that belong to the same class as the original image.

Transforms include a range of operations from the field of image manipulation, such as shifts, flips, zooms, and much more.

The intent is to expand the training dataset with new, plausible examples. This means, variations of the training set images that are likely to be seen by the model.

![](https://github.com/StamKavid/MNIST_image_classification/blob/main/Images/MNIST_Data_aug.jpg)


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 4. CNN | DL

A Convolutional Neural Network (CNN) is a Deep Learning algorithm which can take in an input image, assign importance to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a CNN is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

The architecture of a ConvNet is analogous to that of the connectivity pattern of Neurons in the Human Brain and was inspired by the organization of the Visual Cortex. Individual neurons respond to stimuli only in a restricted region of the visual field known as the Receptive Field. A collection of such fields overlap to cover the entire visual area.

The role of the ConvNet is to reduce the images into a form which is easier to process, without losing features which are critical for getting a good prediction.

There are so many different CNN architectures (AlexNet, LeNet-5, Inception-v1-3, VGG-16 etc.) that were used for experimentation. 

But, how do we choose the most accurate and efficient CNN architecture for classifying MNIST handwritten digits?

A typical CNN design begins with feature extraction (alternating convolution layers with subsampling layers) and finishes with classification (dense layers followed by a final softmax layer).

### Introduction to LeNet5 architecture


![](https://github.com/StamKavid/MNIST_image_classification/blob/main/Images/LeNet5.jpg)

*First Layer (C1)*:

The input for LeNet-5 is a 32×32 grayscale image which passes through the first convolutional layer with 6 feature maps or filters having size 5×5 and a stride of one. The image dimensions changes from 32x32x1 to 28x28x6.

*Second Layer (S2)*:

Then the LeNet-5 applies Average Pooling layer (sub-sampling layer) with a filter size 2×2 and stride = 2. The resulting image dimensions will be reduced to 14x14x6.

*Third Layer (C3)*:

Next, there is a second convolutional layer with 16 feature maps having size 5×5 and a stride of 1. In this layer, only 10 out of 16 feature maps are connected to 6 feature maps of the previous layer as shown below.

*Fourth Layer (S4)*:

The fourth layer is again an Average Pooling layer with filter size 2×2 and a stride = 2. This layer is the same as the second layer except it has 16 feature maps so the output will be reduced to 5x5x16.

*Fifth Layer (C5)*:

The fifth layer is a fully connected convolutional layer with 120 feature maps each of size 1×1. Each of the 120 units is connected to all the 400 nodes (5x5x16) in the fourth layer.

*Sixth Layer (F6)*:

The sixth layer is a fully connected layer with 84 units.

*Output Layer*:

Finally, there is a fully connected softmax output layer ŷ with 10 possible values corresponding to the digits from 0 to 9.

## Architecture Used for MNIST classification

![](https://github.com/StamKavid/MNIST_image_classification/blob/main/Images/MNIST_model_arch.jpg)


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
