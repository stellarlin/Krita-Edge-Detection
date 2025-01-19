import ast

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton,
    QLineEdit, QPushButton, QButtonGroup, QComboBox, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)

from krita import * # pylint: disable=import-error
import numpy as np

EXTENSION_ID = "pykrita_stellarlin_edge_detection"
MENU_ENTRY = "stellarlin's Edge Detection"


class ModeSelectionDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Select Edge Detection Mode")
        self.setGeometry(100, 100, 300, 300)


        layout = QVBoxLayout()

        # Dropdown (ComboBox)
        self.mode_label = QLabel("Select Mode:")
        self.mode_box = QComboBox()
        self.mode_box.addItems(["Prewitt", "Sobel", "Custom"])

        # Custom matrix input (initially hidden)
        self.kernel_x_label = QLabel("Enter Kernel X:")
        self.kernel_x = QLineEdit()
        self.kernel_x.setPlaceholderText("[[1,0,-1],[1,0,-1],[1,0,-1]]")
        self.kernel_x.hide()
        self.kernel_x_label.hide()

        # Status label
        self.statusLabel_x = QLabel()

        self.kernel_y_label = QLabel("Enter Kernel Y:")
        self.kernel_y = QLineEdit()
        self.kernel_y.setPlaceholderText("[[1,1,1],[0,0,0],[-1,-1,-1]]")
        self.kernel_y.hide()
        self.kernel_y_label.hide()

        # Status label
        self.statusLabel_y = QLabel()

        # Buttons
        self.apply_button = QPushButton("Apply")
        self.cancel_button = QPushButton("Cancel")

        # Add widgets to layout
        layout.addWidget(self.mode_label)
        layout.addWidget(self.mode_box)
        layout.addWidget(self.kernel_x_label)
        layout.addWidget(self.kernel_x)
        layout.addWidget(self.statusLabel_x)
        layout.addWidget(self.kernel_y_label)
        layout.addWidget(self.kernel_y)
        layout.addWidget(self.statusLabel_y)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

        # Connect button signals
        self.kernel_x.textChanged.connect(self.validate_kernel_x)
        self.kernel_y.textChanged.connect(self.validate_kernel_y)
        self.mode_box.currentTextChanged.connect(self.toggle_custom_input)
        self.apply_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)


    def toggle_custom_input(self):
        """Show/hide custom matrix input based on mode selection."""
        if self.mode_box.currentText() == "Custom":
            self.kernel_x_label.show()
            self.kernel_x.show()
            self.kernel_y_label.show()
            self.kernel_y.show()
        else:
            self.kernel_x_label.hide()
            self.kernel_x.hide()
            self.statusLabel_x = ""
            self.kernel_y_label.hide()
            self.kernel_y.hide()
            self.statusLabel_y = ""

    def validate_kernel_x (self):
        return self.validate_custom_matrix(self.kernel_x.text(), self.statusLabel_x)

    def validate_kernel_y (self):
        return self.validate_custom_matrix(self.kernel_y.text(), self.statusLabel_y)

    def validate_custom_matrix(self, text, status):
        """Validate if the input is a properly formatted matrix."""
        try:
            matrix = ast.literal_eval(text)  # Convert string to Python list
            if isinstance(matrix, list) and all(isinstance(row, list) for row in matrix):
                row_lengths = set(len(row) for row in matrix)
                if len(row_lengths) == 1:  # Ensure all rows have same length
                    status.setText("✅ Matrix format is correct")
                    return True
            status.setText("❌ Invalid matrix format")
        except (SyntaxError, ValueError):
            status.setText("❌ Invalid input")

        return False

    def get_selected_mode(self):
        """Returns selected mode or custom matrix."""
        mode = self.mode_box.currentText()
        if mode == "Custom" and (not self.validate_kernel_x() or not self.validate_kernel_y()):
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid matrix format.")
            return None
        return mode

    def get_custom_kernels(self):
        return  np.array(ast.literal_eval(self.kernel_x.text())), np.array(ast.literal_eval(self.kernel_y.text()))


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

        filter_side_h = (kernel_height - 1 )// 2
        filter_side_w = (kernel_width - 1) // 2

        # Initialize out
        out = np.zeros_like(image, dtype=np.float32)

        # Loop through the image
        for i in np.arange(filter_side_h, image_height - filter_side_h - (kernel_height % 2 == 0)):
            for j in np.arange(filter_side_w, image_width - filter_side_w - (kernel_width % 2 == 0)):
                # Extract the region of interest (the local neighborhood in the image)

                region = image[(i - filter_side_h):(i + filter_side_h + (kernel_height % 2 == 0) * 1 + 1),
                         (j - filter_side_w):(j + filter_side_w + (kernel_width % 2 == 0) * 1 + 1)]

                # Compute the sum of element-wise multiplication between the region and the filter
                out[i - 1, j - 1] = np.sum(region * kernel)

        # return convolution
        return out

    def normalize(self, chanel):
        return ((chanel - chanel.min()) / (chanel.max() - chanel.min()) * 255).astype(np.uint8)

    def apply_filter (self, image, kernel_x, kernel_y):
        # greyscale
        greyscale = self.to_greyscale(image)
        out = np.zeros_like(image, dtype=np.uint8)

        # Combine Gx, Gy, and 1 into a gradient vector
        Gx = self.convolution(greyscale, kernel_x)  # Gradient in x direction
        Gy = self.convolution(greyscale, kernel_y)  # Gradient in y direction

        # calculate the gradient magnitude of vectors
        gradient_magnitude = np.hypot(Gx, Gy)

        # normalize to the range [0, 255]
        gradient_magnitude = self.normalize(gradient_magnitude)

        # Apply thresholding: keep only values greater than or equal to threshold
        gradient_magnitude = np.where(gradient_magnitude >= 50, 255, 0)

        out[:, :, 0] = gradient_magnitude
        out[:, :, 1] = gradient_magnitude
        out[:, :, 2] = gradient_magnitude
        out[:, :, 3] = 255

        return out

    def action_triggered(self):
        """This method is called when the action is triggered."""
        dialog = ModeSelectionDialog()
        if dialog.exec_():
            selected_mode = dialog.get_selected_mode()
            if selected_mode is None:
                return

            #get image
            doc = self.app.activeDocument()
            width, height = doc.width(), doc.height()
            layer = doc.activeNode()
            if not isinstance(layer, krita.Node):
                return

            #copy previos state
            new_layer = layer.duplicate()
            new_layer.setName("Duplicate Layer")
            root_node = doc.rootNode()
            root_node.addChildNode(new_layer, None)

            # extract image
            image_data = layer.pixelData(0, 0, width, height) # shape=(height, width, 4) channels = BRAG
            image_np = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))


            if selected_mode == 'Prewitt':
                kernel_x = np.array([[1, 0, -1],
                                     [1, 0, -1],
                                     [1, 0, -1]])
                kernel_y = np.array([[1, 1, 1],
                                     [0, 0, 0],
                                     [-1, -1, -1]])
                image_np = self.apply_filter(image_np, kernel_x, kernel_y)
            elif selected_mode == 'Sobel':
                kernel_x = np.array([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]])
                kernel_y = np.array([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]])
                image_np = self.apply_filter(image_np, kernel_x, kernel_y)
            elif selected_mode == 'Custom':
                kernel_x, kernel_y = dialog.get_custom_kernels()
                image_np = self.apply_filter(image_np, kernel_x, kernel_y)

            # set image
            # Assuming shape=(height, width, 4)

            new_layer.setPixelData(image_np.tobytes(), 0, 0, width, height)
            doc.refreshProjection()