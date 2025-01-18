"""Init file for the plugin folder."""

from .stellarlin_edge_detection import stellarlin_edge_detection

# And add the extension to Krita's list of extensions:
app = Krita.instance()
# Instantiate your class:
extension = stellarlin_edge_detection(parent=app)
app.addExtension(extension)
