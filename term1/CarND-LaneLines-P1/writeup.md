# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I convert the image to grayscale, then I use a Gaussian blur for filtering some noise. Afterwards, I use the Canny edge detection to detect edges in the image which represent the lines of the lane. This step is very important because we have to detect small lines, but we have to ensure that noise is not recognized as a line. In the fourth step, I cut of parts of the image which we know that they don't belong to the lane. The Canny edge detection gives us individual points which represent the edges. In the last step, we transform them into the Hough space to robustly detect lines.

With those five steps, we can draw lines around the lane lines in the actual image.

Because lanes are also seperated by dashed lines, I have to extrapolate the left and right line. First, I group the lines into "left lines" and "right lines" by using the lines slope. Then I average starting and end points of each group. Based on that averaged left and right lines, I can calculate the lines slope and intercept. In a last step, I calculate for both lines the starting and end points at y-levels imageHeight & imageHeight * 0.6. With this information, I can draw the averaged lines.

### 2. Identify potential shortcomings with your current pipeline

My pipeline is very fragile regarding to noise. If we filter too much and set the thresholds high, then also the lines aren't recognized anymore. If we filter to less and the thresholds are low, we detect noise as lines. The right values are very specific to the light, environment, etc. in that image.

### 3. Suggest possible improvements to your pipeline

## 3.1. Optimized drawing of dashed lines

Instead of just drawing one line, we could draw lines from one dash to the other and just interpolate the gaps at the top and bottom.

## 3.2. Using information of the past

There will be the sitation where the actual image contains a lot of noise, e.g. sun blinding or other environmental influences. Most of the time, those influences are limited by time, so we could use the detected line from the ~5 images and interpolate it. With additional steps, like we could again filter noise with different statistical methods.

