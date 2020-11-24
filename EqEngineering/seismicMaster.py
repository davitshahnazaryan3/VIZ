from EqEngineering.hazard import Hazard
from EqEngineering.slf import Slf
from EqEngineering.spo import Spo
from EqEngineering.ida import Ida
from EqEngineering.loss import Loss
import pickle
from plotter import Plotter


class SeismicMaster:
    def __init__(self, export=False, exportMiser=False, exportDir=None):
        """
        Initialize visualization
        :param export: bool                 Export figures or not?
                                            (exports in .emf via inkscape, modify function if need be)
        :param exportMiser: bool            Export png images or not? (It's a trap...)
        :param exportDir: str               Export directory
        """
        self.export = export
        self.exportMiser = exportMiser
        self.exportDir = exportDir
        if export or exportMiser:
            self.plotter = Plotter()

    def exportFigure(self, fig, name):
        """
        Exports figures if necessary
        :param fig: figure object
        :param name: str                    Base name of the file
        :return: None
        """
        if self.export:
            self.plotter.plot_as_emf(fig, filename=self.exportDir / name)
        if self.exportMiser:
            self.plotter.plot_as_png(fig, filename=self.exportDir / name)

    def hazard(self, path, true=True, pathFitted=None, fitted=False, period=None):
        """
        Calls a Hazard object
        :param path: str                    Path to file
        :param true: bool                   Plot true hazard function
        :param pathFitted: str              Path to fitted function
        :param fitted: bool                 Plot 2-nd order fitted function
        :param period: float                Period of interest to highlight
        :return: None
        """
        # Call the Hazard object
        hazard = Hazard(period)
        if true:
            with open(path, "rb") as f:
                h = pickle.load(f)
                fig = hazard.true_hazard(h)
                self.exportFigure(fig, "trueHazard")

        if fitted:
            with open(pathFitted, "rb") as f:
                h = pickle.load(f)
                fig = hazard.fitted_hazard(h)
                self.exportFigure(fig, "fittedHazard")
