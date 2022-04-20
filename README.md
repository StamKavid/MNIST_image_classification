# MNIST: Image Classification | Overview

In this project you will find Image Clasiifcation problem using MNIST datasource. More specifically:
* Problem Statement 
* Data source | Overview
* EDA | Data Augmentation
* ML / DL 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Resources used

**Python Version:** 3.6.9

**Pandas Version:** 1.3.5

**NumPy Version:** 1.21.6

**Packages:** numpy, pandas, matplotlib, plotly, sklearn, keras, dask

**MNIST Architectures | Papers:** https://bit.ly/3xVXCSL

**Paper for LeNet5:** https://bit.ly/3EyHQOH

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Problem Statement | Image Classification

**Image classification** is the process of categorizing and labeling groups of pixels or vectors within an image based on specific rules.

The MNIST database (Modified National Institute of Standards and Technology database) is a large collection of handwritten digits. It has a training set of 60,000 examples, and a test set of 10,000 examples.

The digits have been size-normalized and centered in a fixed-size image. The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm.

The images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Facebook Prophet

It’s a tool intended to help you to do time series forecasting at a scale with ease. It uses decomposable time series model with 3 main components: **seasonal**, **trends**, **holidays** or **events** effect and error which are combined into this equation:

***f(x) = g(x) + s(x) + h(x) + e(t)***

where,

**g(x)** is a trend function which models the non-periodic changes. It can be either a linear function or a logistic function.

**s(x)** represents a periodic changes i.e weekly, monthly, yearly. A yearly seasonal component is modeled using Fourier series and weekly seasonal component using dummy variables.

**h(x)** is a function that represents the effect of holidays which occur on irregular schedules.(n ≥ 1 days)

**e(x)** represents error changes that are not accommodated by the model.

## Visualizations

![](https://github.com/StamKavid/COVID_19_simple_analysis/blob/main/Images/Daily%20COVID-19%20Cases%20(Globally).png)

**Figure 1.1:** Daily COVID-19 cases (Globally)

![](https://github.com/StamKavid/COVID_19_simple_analysis/blob/main/Images/Daily%20COVID-19%20Cases%20(Greece).png)

**Figure 1.2:** Daily COVID-19 cases (Greece)

![](https://github.com/StamKavid/COVID_19_simple_analysis/blob/main/Images/COVID-19%20prediction%20(Greece).png)

**Figure 1.3:** Daily COVID-19 prediction cases (Greece)


Prophet plots the observed values of time series (black dots), the forecasted values (blue lines) and the uncertainty intervals of our forecasts (blue shaded region).


