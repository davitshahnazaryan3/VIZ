import os
import subprocess


class Plotter:
    def __init__(self):
        # Export with v1.0.1 of inkscape does not seem to work
        pass

    @staticmethod
    def plot_as_emf(figure, **kwargs):
        """
        Saves figure as .emf
        :param figure: fig handle
        :param kwargs: filepath: str                File name, e.g. '*\filename'
        :return: None
        """
        inkscape_path = kwargs.get('inkscape', "C://Program Files//Inkscape//bin//inkscape.exe")
        filepath = kwargs.get('filename', None)
        if filepath is not None:
            path, filename = os.path.split(filepath)
            filename, extension = os.path.splitext(filename)
            svg_filepath = os.path.join(path, filename + '.svg')
            emf_filepath = os.path.join(path, filename + '.emf')
            figure.savefig(svg_filepath, bbox_inches='tight', format='svg')
            subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
            # os.remove(svg_filepath)

    @staticmethod
    def plot_as_png(figure, **kwargs):
        """
        Saves figure as .png
        :param figure: fig handle
        :param kwargs: filepath: str                File name, e.g. '*\filename'
        :return: None
        """
        inkscape_path = kwargs.get('inkscape', "C://Program Files//Inkscape//bin//inkscape.exe")
        filepath = kwargs.get('filename', None)
        if filepath is not None:
            path, filename = os.path.split(filepath)
            filename, extension = os.path.splitext(filename)
            svg_filepath = os.path.join(path, filename + '.svg')
            png_filepath = os.path.join(path, filename + '.png')
            figure.savefig(svg_filepath, bbox_inches='tight', format='svg')
            subprocess.call([inkscape_path, svg_filepath, '--export-png', png_filepath])
            # os.remove(svg_filepath)