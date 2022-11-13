
# Image Mosaics

## 1 - Getting Correspondences

#### Use SIFT descriptor

- Scale-space peak selection: Potential location for finding features.

---------

  

![Scale-space peak](img1.jpg)

  

- Keypoint Localization: Accurately locating the feature keypoints.

---------

  

![Keypoint Localization](img2.png)

  

- Orientation Assignment: Assigning orientation to keypoints.

---------

  

- Keypoint descriptor: Describing the keypoints as a high dimensional vector.

---------

  

![Keypoint descriptor](img3.png)

  

- Keypoint Matching

---------

  

![Matching](img4.png)

  

### Matching Paramter

- 2 KNN Matching

- Applied ratio 0.75

- 50 correspondences

  

## 2 - Homography Parameters

### 2D transformations in heterogeneous coordinates

#### Determining the homography matrix

##### $P^{`}$ = $H$ . $P$

$$  \begin{bmatrix}

x_{n}\\

y_{n}\\

1

\end{bmatrix}

= \alpha

\begin{bmatrix}

h_{1} & h_{2} & h_{3} \\

h_{4} & h_{5} & h_{6}\\

h_{7} & h_{8} & h_{9}

\end{bmatrix}

\begin{bmatrix}

x_{}\\

y_{}\\

1

\end{bmatrix}

$$

  

### Solving for H using DLT

#### 1 - For each correspondence, create 2x9 matrix $A_{i}$

#### 2 - Concatenate into single 2n x 9 matrix $A$

#### 3 - Compute SVD of $A = U \Sigma V^{T}$

#### 4 - Store singular vector of the smallest singular value $h_{i} = v_{\hat{i}}$

#### 5 - Reshape to get $H$

## 3 - Stitching Images
    In this step we are using the homography matrix to do the perspective
    transformation, for one of the images, and finally stitch the first
    image with the wrapped one.

#### 1- Multiply every pixel coordination in the (x', y') space with the homography matrix  to get the new coordination in the (x, y) space
#### 2- For every pixel in the (x', y') space that doesn't correspond to a single pixel in the (x, y) space, we will splat it's intensity with all the potential pixels, and finally averaging the intensities for every pixel in the (x, y) space
#### 3- Now for every black pixel in the transformed image, there are two possibilities:
  - It's original value is black
  - it has a fractional correspondence in the (x', y') space
 #### In both cases we will calculate the homography inverse matrix and get the (x', y') correspondence, and finally do a bilinear interpolation (weighted average) using the following formula:
 ![weighted average formula](https://latex.codecogs.com/svg.image?W&space;=&space;%5Cfrac%7B%5Csum_%7Bi=1%7D%5E%7Bn%7Dw_%7Bi%7DX_%7Bi%7D%7D%7B%5Csum_%7Bi=1%7D%5E%7Bn%7D&space;w_%7Bi%7D%7D)
 #### 4- Stitch the first image with the wrapped one


# Examples
## Example 1
### Inputs:
![example1-1](../img/image1.jpg)
![example1-2](../img/image2.jpg)
### Output:
![example1 output](ex1output.png)
## Example 2
### Inputs:
![example2-1](../img/example2-1.png)
![example2-2](../img/example2-2.png)
### Output:
![example2 output](ex2output.png)
## Example 3
### Inputs:
![example3-1](../img/shanghai-21.jpg)
![example3-2](../img/shanghai-22.jpg)
![example3-3](../img/shanghai-23.jpg)
### Output:
![example3 output](ex3output.png)