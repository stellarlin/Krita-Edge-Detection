
# pylint: disable=C0301:line-too-long, C0103:invalid-name

from krita import * # pylint: disable=import-error

EXTENSION_ID = "pykrita_normal_map_perwitt"
MENU_ENTRY = "normal_map_perwitt"


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

    def action_triggered(self):
        """This method is called when the action is triggered."""
        # your active code goes here:
