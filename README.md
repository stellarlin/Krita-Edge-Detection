

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

## Installation & Setup

Step 1: Install the Plugin
1.	Download the plugin ZIP file.
2.	Open Krita and navigate to: Sources → Manage Resources → Open Resource Folder.
3.	Extract the plugin files into the pykrita folder.

Step 2: Enable the Plugin
1.	Open Preferences → Python Plugin Manager.
2.	Find stellarlin_edge_detection and enable it by clicking the checkbox.
3.	Ensure NumPy is installed in Krita’s Python environment. If not, install it manually.

## Usage

Step 1: Open the Plugin
1. Open an image in Krita.
2. Go to Tools → Scripts → Stellarlin’s Edge Detection.
3. A dialog window will appear.

Step 2: Select Edge Detection Mode
1. Choose a detection method
* Prewitt – Highlights edges with less sensitivity to noise.
* Sobel – Detects stronger edges by emphasizing gradient changes.
* Custom – Allows defining custom edge detection kernels.
2.	If choosing Custom, enter values for Kernel X and Kernel Y in matrix form.

Step 3: Apply Edge Detection
1.	Click Apply to generate a new layer with the edge-detected result.
2.	The original image remains unchanged, and the new layer contains the processed output.

--- 
# Technical Documentation

Scientific Documentation: Edge Detection Plugin for Krita

1. Introduction

Edge detection is a fundamental operation in image processing and computer vision. It is widely used for feature extraction, object detection, and image segmentation. This plugin was originally developed as a university project to study the mathematical principles and implementation of convolution in digital image processing.

The plugin allows users to apply Prewitt, Sobel, or custom convolution kernels to detect edges in an image. By providing an interactive interface for experimenting with different filters, the plugin serves as both a practical tool and an educational resource for understanding edge detection algorithms.

2. Theoretical Background

2.1 Convolution in Image Processing

Convolution is a mathematical operation that involves sliding a filter (or kernel) over an image and computing the weighted sum of pixel intensities. It is defined as:

￼

where:
	•	￼ is the output pixel value,
	•	￼ represents the input image pixel values,
	•	￼ is the convolution kernel,
	•	￼ is half the kernel size.

2.2 Edge Detection

Edges represent significant changes in pixel intensity, which often correspond to object boundaries in an image. Edge detection algorithms apply gradient-based methods to identify these regions. The gradient of an image ￼ is computed using convolution with directional kernels:

￼

where ￼ and ￼ are the gradients in the x and y directions, respectively. The final edge magnitude is calculated using the Euclidean norm:

￼

2.3 Grayscale Conversion

Before applying convolution, the image is converted to grayscale using the Luma (ITU-R BT.601) formula:

￼

This formula accounts for human perception, where green contributes the most to brightness, followed by red and blue.

3. Implementation

3.1 Predefined Edge Detection Kernels

The plugin provides two widely used edge detection operators:

Prewitt Operator

The Prewitt operator emphasizes horizontal and vertical edges using simple averaging:

￼

Sobel Operator

The Sobel operator improves upon Prewitt by giving more weight to central pixels:

￼

3.2 Custom Kernel Support

Users can define their own 3×3 convolution matrices, allowing experimentation with different filters beyond Prewitt and Sobel. The plugin validates the input format and applies the custom kernel pair.

4. Algorithm and Workflow
	1.	User selects an edge detection mode (Prewitt, Sobel, or Custom).
	2.	Image is converted to grayscale using the ITU-R BT.601 formula.
	3.	Selected kernels are applied using convolution to compute ￼ and ￼.
	4.	Gradient magnitude is computed as ￼.
	5.	The resulting image is normalized to the range ￼.
	6.	Thresholding is applied to enhance edges, setting values below a threshold to zero.
	7.	The processed image is stored in a new layer in Krita.

5. Applications
	•	Artistic effects: Enhancing or extracting line work from digital paintings.
	•	Computer vision: Studying edge detection techniques for feature extraction.
	•	Educational purposes: Learning convolution, gradient computation, and kernel effects.

6. Conclusion

This plugin demonstrates the practical application of convolution for edge detection while serving as a learning tool for image processing concepts. By providing an interactive way to experiment with predefined and custom kernels, it bridges the gap between theoretical understanding and practical implementation.

Future improvements could include support for additional filters (e.g., Canny edge detection) and real-time previews.
--- 
# Developer Documentation 

## Project Structure
	•	stellarlin_edge_detection.py – Main plugin script
	•	ModeSelectionDialog – GUI window for selecting edge detection mode
	•	stellarlin_edge_detection – Krita plugin class
	•	apply_filter() – Applies edge detection
	•	convolution() – Convolves the image with a kernel
	•	to_greyscale() – Converts image to grayscale
	•	normalize() – Scales pixel values

## Core Functions

1. Convolution Function

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

This function applies a discrete 2D convolution with a given kernel, extracting edge features.

2. Edge Detection Pipeline

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

This function:
	1.	Converts the image to grayscale.
	2.	Computes the x and y gradients.
	3.	Calculates the gradient magnitude.
	4.	Normalizes and applies thresholding.
	5.	Outputs an RGBA image.

3. Krita Integration

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

	•	Loads the active document and layer.
	•	Duplicates the layer to preserve the original.
	•	Applies edge detection using the selected mode.
	•	Updates the layer with the processed image.
