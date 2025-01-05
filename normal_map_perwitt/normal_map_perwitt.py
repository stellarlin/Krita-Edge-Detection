
# pylint: disable=C0301:line-too-long, C0103:invalid-name

from krita import * # pylint: disable=import-error
import numpy as np

EXTENSION_ID = "pykrita_normal_map_perwitt"
MENU_ENTRY = "Normal map using Perwitt 3x3"


class Normal_map_Perwitt(Extension):
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

    def to_greyscale(self, image):
        # Ensure the image is in float format for accurate calculations
        image_float = image.astype(np.float32)

        # Apply the NTSC formula for each pixel
        return (0.299 * image_float[:, :, 0] +  # Red channel
                     0.587 * image_float[:, :, 1] +  # Green channel
                     0.114 * image_float[:, :, 2])  # Blue channel

    def apply_prewitt_operator(self, image):

        # greyscale
        greyscale = self.to_greyscale(image)

        # Define Prewitt kernels for x and y directions
        kernel_x = np.array([[1, 0, -1],
                             [1, 0, -1],
                             [1, 0, -1]])
        kernel_y = np.array([[1, 1, 1],
                             [0, 0, 0],
                             [-1, -1, -1]])

        # Get image dimensions
        height, width = greyscale.shape[:2]

        # Initialize gradients
        Gx = np.zeros_like(greyscale, dtype=np.float32)
        Gy = np.zeros_like(greyscale, dtype=np.float32)

        # Padding the grayscale image to handle edges
        padded = np.pad(greyscale, pad_width=1, mode='constant', constant_values=0)

        # Apply the Prewitt operator manually
        for i in range(1, height + 1):  # Loop over padded image
            for j in range(1, width + 1):
                # Extract the 3x3 region
                region = padded[i - 1:i + 2, j - 1:j + 2]
                # Compute gradients
                Gx[i - 1, j - 1] = np.sum(region * kernel_x)
                Gy[i - 1, j - 1] = np.sum(region * kernel_y)

        # Combine Gx, Gy, and 1 into a gradient vector
        gradients = np.zeros((height, width, 2), dtype=np.float32)
        gradients[:, :, 0] = Gx  # Gradient in x direction
        gradients[:, :, 1] = Gy  # Gradient in y direction

        return gradients

    def normalize (self, chanel):
        return ((chanel - chanel.min()) / (chanel.max() - chanel.min()) * 255).astype(np.uint8)

    def gradient_to_rgb(self, gradients):

        # Extract Gx and Gy
        Gx = gradients[:, :, 0]
        Gy = gradients[:, :, 1]

        # Normalize Gx and Gy to range [0, 255]
        Gx_normalized = self.normalize(Gx)
        Gy_normalized = self.normalize(Gy)

        # Create RGB mapping
        rgb_image = np.zeros((*gradients.shape[:2], 4), dtype=np.uint8)
        rgb_image[:, :, 0] = Gx_normalized  # Map Gx to Red
        rgb_image[:, :, 1] = Gy_normalized  # Map Gy to Green
        rgb_image[:, :, 2] = 255  # Set Blue to constant 255 (from the 1 in [Gx, Gy, 1])
        rgb_image[:, :, 3] = 255 # Set Alpha to constant 255
        return rgb_image

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

        # extract image
        image_data = layer.pixelData(0, 0, width, height) # shape=(height, width, 4) channels = BRAG
        image_np = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))

        # get gradients using Perwitt 3x3
        gradients = self.apply_prewitt_operator(image_np)

        # transform to RGB
        image_np = self.gradient_to_rgb(gradients)

        # set image
        # Assuming shape=(height, width, 4)

        layer.setPixelData(image_np.tobytes(), 0, 0, width, height)