

# stellarlin’s Edge Detection Plugin for Krita

![Title Screenshot](Other/peppers_result.jpg)

---
**Author:** Sofiia Prykhach

 **Version:** 1.0.0 
 
 **Krita Compatibility:** Krita 5.2.6 or later
 
---
 
## Overview

This plugin was originally developed as a university project to study the process of convolution in image processing. The main goal was to explore and implement edge detection techniques

This plugin provides an interactive way to apply edge detection using Prewitt, Sobel, or custom kernels. It allows users to experiment with convolutional matrices. The plugin works by converting an image to grayscale using the Luma (ITU-R BT.601) formula, applying convolution with selected kernels, and generating a new layer with the detected edges.

> _Future improvements could include support for additional filters (e.g., Canny edge detection) and real-time previews._

---
## Installation & Setup

* Step 1: Install the Plugin
	*	Download the plugin ZIP file.
	*	Open Krita and navigate to: ` Sources → Manage Resources → Open Resource Folder` .
	*	Extract the plugin files into the pykrita folder.

* Step 2: Enable the Plugin
	*	Open ` Preferences → Python Plugin Manager` .
	*	Find ` stellarlin_edge_detection`  and enable it by clicking the checkbox.
	*	Ensure NumPy is installed in Krita’s Python environment. If not, install it manually.

## Usage

* Step 1: Open the Plugin
	* Open an image in Krita.
	* Go to ` Tools → Scripts → Stellarlin’s Edge Detection` .
	* A dialog window will appear.

* Step 2: Select Edge Detection Mode
	* Choose a detection method
		* Prewitt – Highlights edges with less sensitivity to noise.
		* Sobel – Detects stronger edges by emphasizing gradient changes.
		* Custom – Allows defining custom edge detection kernels.
	* If choosing Custom, enter values for Kernel X and Kernel Y in matrix form.

* Step 3: Apply Edge Detection
	* Click Apply to generate a new layer with the edge-detected result.
	* The original image remains unchanged, and the new layer contains the processed output.

--- 
# Technical Documentation

Edge detection is a fundamental operation in image processing and computer vision. It is widely used for feature extraction, object detection, and image segmentation. 

The plugin allows users to apply Prewitt, Sobel, or custom convolution kernels to detect edges in an image. By providing an interactive interface for experimenting with filters, the plugin serves as both a practical tool and an educational resource for understanding edge detection algorithms.

##  Convolution in Image Processing

Convolution is a mathematical operation that involves sliding a filter (or kernel) over an image and computing the weighted sum of pixel intensities. It is defined as:
```math
\begin{flalign*}
G(i,j) &= \sum_{m=-k}^{k} \: \sum_{n=-k}^{k} \: I(i-m, j-n) \: K(m,n) &&
\end{flalign*}
```
where:  

-  $` G(i,j) `$ is the output pixel value,  
-  $` I(i-m, j-n)`$ is the  image pixel values,  
-  $` K(m,n) `$ is the convolution kernel,  
-  $` h,w `$  is half the kernel height and width.


##  Edge Detection

Edges represent significant changes in pixel intensity, which often correspond to object boundaries in an image. Edge detection algorithms apply gradient-based methods to identify these regions. The gradient of an image $`\nabla I`$ is computed using convolution with directional kernels:
```math
G_x = I \ast K_x, \quad G_y = I \ast K_y 
```
where ￼$`G_x`$ and ￼$`G_y`$ are the gradients in the x and y directions, respectively. The final edge magnitude is calculated using the Euclidean norm:

```math
\begin{aligned} G &= \sqrt{G_x^2 + G_y^2} \end{aligned} 
```
## Grayscale Conversion

Before applying convolution, the image is converted to grayscale using the **_Luma (ITU-R BT.601) formula_**. This formula accounts for human perception, where green contributes the most to brightness, followed by red and blue.

```math
 I = 0.2989 R + 0.5870 G + 0.1140 B 
```

## Implementation

### Predefined Edge Detection Kernels

The plugin provides two widely used edge detection operators:

**Prewitt Operator:**
```math
K_x =
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}
, \quad
K_y =
\begin{bmatrix}
1 & 1 & 1 \\
0 & 0 & 0 \\
-1 & -1 & -1
\end{bmatrix}
```
Example of usage of Prewitt Operator on high-quality photos by [Pascal van de Vendel](https://unsplash.com/@pascalvendel)

**Sobel Operator:**
```math
K_x =
\begin{bmatrix}
1 & 0 & -1 \\
2 & 0 & -2 \\
1 & 0 & -1
\end{bmatrix}
, \quad
K_y =
\begin{bmatrix}
1 & 2 & 1 \\
0 & 0 & 0 \\
-1 & -2 & -1
\end{bmatrix}
```
Example of usage of Sobel Operator on high-quality photos by [Pascal van de Vendel](https://unsplash.com/@pascalvendel):

### Custom Kernel Support

Users can define their own convolution kernels of different height and width. The plugin validates the input format and applies the custom kernel pair.

Example of  usage of custom kernels (Roberts Cross Edge Detection Gx = [[1,0].[0,-1]], Gy = [[0,1].[-1,0]]) on high-quality photos by [Pascal van de Vendel](https://unsplash.com/@pascalvendel):


## Algorithm and Workflow
> 1.	User selects an edge detection mode (Prewitt, Sobel, or Custom).
> 2.	Image is converted to grayscale using the ITU-R BT.601 formula.
> 3.	Selected kernels are applied using convolution to compute $`G_x`$ and $`G_y`$.
> 4.	Gradient magnitude is computed as $`G=\sqrt{G_x^2 + G_y^2}`$.
> 5.	The resulting image is normalized to the range $`[0,255]`$.
> 6.	Thresholding is applied to enhance edges, setting values below a threshold to zero.
> 7.	The processed image is stored in a new layer in Krita.


--- 
# Developer Documentation 

## Project Structure
*	stellarlin_edge_detection
	*	` stellarlin_edge_detection.py`  – Main plugin script
		*	` ModeSelectionDialog`  – GUI window for selecting edge detection mode
		*	` stellarlin_edge_detection`  – Krita plugin class
			*	` apply_filter()`  – Applies edge detection
			*	` convolution()`  – Convolves the image with a kernel
			*	` to_greyscale()`  – Converts image to grayscale
			*	` normalize()`  – Scales pixel values
	*   ` \_\_init\_\_.py`  : Registers the plugin as an extension in Krita.
 *   ` stellarlin_edge_detection.desktop` 

##   Core Functions

### Graphical User Interface (GUI)
The Mode Selection Dialog allows users to choose between _Prewitt_, _Sobel_, or _Custom kernels_. It is created using ` _PyQt5_` , Krita’s preferred GUI toolkit, and consists of several key components:

* Mode Dropdown: The user selects between the predefined edge detection methods

* Kernel Input Fields: When a custom kernel is selected, the user is presented with two text fields (one for the X kernel and one for the Y kernel). The default placeholder texts are provided as examples:

	* Kernel X: [[1,0,-1], [1,0,-1], [1,0,-1]]
	* Kernel Y: [[1,1,1], [0,0,0], [-1,-1,-1]]
* Validation: The plugin automatically checks whether the entered custom kernels are valid matrices. It ensures that the matrices:

	* Are properly formatted as 2D arrays.
	* Have equal row lengths across the matrix.
 If the input is incorrect, the status label for each kernel displays an error message ` ("❌ Invalid input")` , and the user is prompted to correct the input.

This validation is handled by the ` validate_custom_matrix()`  method in the `ModeSelectionDialog`  class. If the validation fails, the plugin provides immediate feedback to the user, preventing incorrect kernel application.


### Convolution Function

This function applies a discrete 2D convolution with a given kernel, extracting edge features.
``` 
def convolution(self, image, kernel):
    image_height, image_width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape[:2]
   
    out = np.zeros_like(image, dtype=np.float32)
    
    filter_side_h = (kernel_height - 1) // 2
    filter_side_w = (kernel_width - 1) // 2

    for i in range(filter_side_h, image_height - filter_side_h):
        for j in range(filter_side_w, image_width - filter_side_w):
            region = image[i - filter_side_h : i + filter_side_h + 1,
                           j - filter_side_w : j + filter_side_w + 1]
            out[i, j] = np.sum(region * kernel)

    return out
```


### Edge Detection Pipeline

This function:
	1.	Converts the image to grayscale.
	2.	Computes the x and y gradients.
	3.	Calculates the gradient magnitude.
	4.	Normalizes and applies thresholding.
	5.	Outputs an RGBA image.
 ```
def apply_filter(self, image, kernel_x, kernel_y):
    greyscale = self.to_greyscale(image)
    Gx = self.convolution(greyscale, kernel_x)
    Gy = self.convolution(greyscale, kernel_y)
    
    gradient_magnitude = np.hypot(Gx, Gy)
    gradient_magnitude = self.normalize(gradient_magnitude)
    gradient_magnitude = np.where(gradient_magnitude >= 50, 255, 0)

    out = np.zeros_like(image, dtype=np.uint8)
    out[:, :, 0] = gradient_magnitude
    out[:, :, 1] = gradient_magnitude
    out[:, :, 2] = gradient_magnitude
    out[:, :, 3] = 255

    return out
```


3. Krita Integration
* Loads the active document and layer.
* Duplicates the layer to preserve the original.
* Applies edge detection using the selected mode.
* Updates the layer with the processed image.
  
```
def action_triggered(self):
    dialog = ModeSelectionDialog()
    if dialog.exec_():
        selected_mode = dialog.get_selected_mode()
        doc = self.app.activeDocument()
        layer = doc.activeNode()
        
        if selected_mode == 'Prewitt':
            kernel_x, kernel_y = self.prewitt_kernels()
        elif selected_mode == 'Sobel':
            kernel_x, kernel_y = self.sobel_kernels()
        else:
            kernel_x, kernel_y = dialog.get_custom_kernels()
        
        image_np = self.apply_filter(self.extract_pixels(layer), kernel_x, kernel_y)
        self.set_pixels(layer, image_np)
        doc.refreshProjection()
```
	
