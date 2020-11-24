"""
Object for data visualization
"""
from pathlib import Path
import os
import subprocess
from EqEngineering.seismicMaster import SeismicMaster


class VIZ:
    def __init__(self, export=False, exportMiser=False):
        """
        Initialize visualization
        :param export: bool                 Export figures or not?
                                            (exports in .emf via inkscape, modify function if need be)
        :param exportMiser: bool            Export png images or not? (It's a trap...)
        """
        self.export = export
        self.exportMiser = exportMiser
        # Main directory
        self.directory = Path(os.getcwd())
        # Default color patterns
        self.color_grid = ['#840d81', '#6c4ba6', '#407bc1', '#18b5d8', '#01e9f5',
                           '#cef19d', '#a6dba7', '#77bd98', '#398684', '#094869']
        
    @staticmethod
    def create_folder(directory):
        """
        Creates a figure if it does not exist
        :param directory: str                       Directory to create
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)
        
    @staticmethod
    def plot_as_emf(figure, **kwargs):
        """
        Saves figure as .emf
        :param figure: fig handle
        :param kwargs: filepath: str                File name, e.g. '*\filename'
        :return: None
        """
        inkscape_path = kwargs.get('inkscape', "C://Program Files//Inkscape//inkscape.exe")
        filepath = kwargs.get('filename', None)
        if filepath is not None:
            path, filename = os.path.split(filepath)
            filename, extension = os.path.splitext(filename)
            svg_filepath = os.path.join(path, filename + '.svg')
            emf_filepath = os.path.join(path, filename + '.emf')
            figure.savefig(svg_filepath, bbox_inches='tight', format='svg')
            subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
            os.remove(svg_filepath)

    @staticmethod
    def plot_as_png(figure, **kwargs):
        """
        Saves figure as .png
        :param figure: fig handle
        :param kwargs: filepath: str                File name, e.g. '*\filename'
        :return: None
        """
        inkscape_path = kwargs.get('inkscape', "C://Program Files//Inkscape//inkscape.exe")
        filepath = kwargs.get('filename', None)
        if filepath is not None:
            path, filename = os.path.split(filepath)
            filename, extension = os.path.splitext(filename)
            svg_filepath = os.path.join(path, filename + '.svg')
            png_filepath = os.path.join(path, filename + '.png')
            figure.savefig(svg_filepath, bbox_inches='tight', format='svg')
            subprocess.call([inkscape_path, svg_filepath, '--export-png', png_filepath])
            os.remove(svg_filepath)


if __name__ == "__main__":

    """Most of the input manipulation and method calling happens here for now, until a GUI 
    (even a simple one) is created"""
    # Calling the master object
    viz = VIZ()
