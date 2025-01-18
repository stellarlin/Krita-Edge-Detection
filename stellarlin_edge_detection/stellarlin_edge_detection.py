
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton,
    QLineEdit, QPushButton, QButtonGroup
)

from krita import * # pylint: disable=import-error
import numpy as np

EXTENSION_ID = "pykrita_stellarlin_edge_detection"
MENU_ENTRY = "stellarlin's Edge Detection"


class stellarlin_edge_detection(Extension):
    """The main class of the plugin."""

    def __init__(self, parent):
        self.app = parent
        # Always initialise the superclass.
        # This is necessary to create the underlying C++ object
        super().__init__(parent)

    def setup(self):
        """This method is called when the plugin is first loaded."""

    def createActions(self, window):
        """Add your action to the menu and other actions."""
        action = window.createAction(EXTENSION_ID, MENU_ENTRY, "tools/scripts")
        # parameter 1 = the name that Krita uses to identify the action
        # parameter 2 = the text to be added to the menu entry for this script
        # parameter 3 = location of menu entry
        action.triggered.connect(self.action_triggered)

    def to_greyscale(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def convolution(self, image, kernel):
        # height and width of the image
        # Get image dimensions
        image_height, image_width = image.shape[:2]
        kernel_height, kernel_width = kernel.shape[:2]

        filter_side_h = kernel_height // 2
        filter_side_w = kernel_width // 2


        # Initialize out
        out = np.zeros_like(image, dtype=np.float32)

        # Loop through the image
        for i in np.arange(filter_side_h, image_height - filter_side_h):
            for j in np.arange(filter_side_w, image_width - filter_side_w):
                # Extract the region of interest (the local neighborhood in the image)
                region = image[(i - filter_side_h):(i + filter_side_h + 1), (j - filter_side_w):(j + filter_side_w + 1)]

                # Compute the sum of element-wise multiplication between the region and the filter
                out[i - 1, j - 1] = np.sum(region * kernel)

        # return convolution
        return out

    def prewitt3x3(self, image):

        # greyscale
        greyscale = self.to_greyscale(image)

        # Define Prewitt kernels for x and y directions
        kernel_x = np.array([[1, 0, -1],
                             [1, 0, -1],
                             [1, 0, -1]])
        kernel_y = np.array([[1, 1, 1],
                             [0, 0, 0],
                             [-1, -1, -1]])

        out = np.zeros_like(image, dtype=np.uint8)

        # Combine Gx, Gy, and 1 into a gradient vector
        Gx = self.convolution(greyscale,kernel_x)  # Gradient in x direction
        Gy = self.convolution(greyscale,kernel_y) # Gradient in y direction

        # calculate the gradient magnitude of vectors
        gradient_magnitude = np.hypot(Gx , Gy)

        # normalize to the range [0, 255]
        gradient_magnitude = self.normalize(gradient_magnitude)
        out[:,:,0] = gradient_magnitude
        out[:, :, 1] = gradient_magnitude
        out[:, :, 2] = gradient_magnitude
        out[:, :, 3] = 255

        return out

    def normalize(self, chanel):
        return ((chanel - chanel.min()) / (chanel.max() - chanel.min()) * 255).astype(np.uint8)


    def action_triggered(self):
        """This method is called when the action is triggered."""
        # your active code goes here:

        #get image
        doc = self.app.activeDocument()
        width, height = doc.width(), doc.height()
        layer = doc.activeNode()
        if not isinstance(layer, krita.Node):
            return

        #filter = self.app.filter("desaturate")
        #filter.apply(layer, 0, 0, width, height)
        #doc.rootNode().addChildNode(layer, None)
        #doc.refreshProjection()

        #copy previos state
        new_layer = layer.duplicate()
        new_layer.setName("Duplicate Layer")
        root_node = doc.rootNode()
        root_node.addChildNode(new_layer, None)

        # extract image
        image_data = layer.pixelData(0, 0, width, height) # shape=(height, width, 4) channels = BRAG
        image_np = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))


        # get gradients using Perwitt 3x3
        image_np = self.prewitt3x3(image_np)

        # set image
        # Assuming shape=(height, width, 4)

        new_layer.setPixelData(image_np.tobytes(), 0, 0, width, height)