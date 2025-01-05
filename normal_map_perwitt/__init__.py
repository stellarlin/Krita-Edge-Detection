"""Init file for the plugin folder."""

from .normal_map_perwitt import Normal_map_Perwitt

# And add the extension to Krita's list of extensions:
app = Krita.instance()
# Instantiate your class:
extension = Normal_map_Perwitt(parent=app)
app.addExtension(extension)
